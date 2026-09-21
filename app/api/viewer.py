"""Viewer endpoints: image slices, segmentation slices, and in-place editing.

Painting writes into a per-reviewer working copy of the label volume instead of
creating an annotation version, so unsaved work survives a reload and is only
committed by an explicit save.
"""

import io
import json
from uuid import UUID

import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException, Path, Query, Request, Response
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from starlette.datastructures import UploadFile

from app.api.cases import annotation_out, current_annotation, require_case
from app.core.security import CurrentUser
from app.db import Db
from app.models import AnnotationFormat, Case, Dataset, ImageFormat, Review
from app.schemas import AnnotationOut, LabelNames, ViewerInfo
from app.services.annotation import create_correction, read_json
from app.services.storage import get_file_path, save_json
from app.services.volume import (
    AXIS_NAMES,
    annotation_volume,
    coco_categories,
    coco_payload,
    discard_work,
    extract,
    face,
    image_volume,
    insert,
    nifti_payload,
    open_work,
    plane_spacing,
    png_bytes,
    work_dir,
    work_state,
)

router = APIRouter(prefix="/viewer/cases/{case_id}", tags=["viewer"])
MAX_SLICE_BYTES = 8 * 1024 * 1024


def case_dataset(db: Db, case_id: UUID) -> tuple[Case, Dataset]:
    case = require_case(db, case_id)
    return case, db.get(Dataset, case.dataset_id)


def label_source(db: Db, case: Case, user, shape):
    """Return the label volume the viewer should display: the draft if one exists."""
    draft = open_work(case, user.id, shape, None)
    if draft is not None:
        return draft
    return annotation_volume(case, current_annotation(db, case.id), shape)


def axis_info(meta: dict, axis: int) -> dict:
    rows, columns = plane_spacing(meta["shape"], axis)
    row_mm, column_mm = plane_spacing(meta["spacing"], axis)
    return {
        "index": axis,
        "name": AXIS_NAMES[axis],
        "count": meta["shape"][axis],
        "rows": int(rows),
        "columns": int(columns),
        "row_mm": round(row_mm, 4),
        "column_mm": round(column_mm, 4),
    }


def label_names(case: Case, draft: dict | None = None) -> dict[int, str]:
    """Names for every label value this case can hold, defaulting to 'Label N'."""
    values = [int(x) for x in case.metadata_json.get("labels", []) if int(x) > 0]
    categories = {
        value: str(item.get("name") or f"Category {item['id']}")
        for value, item in coco_categories(case.metadata_json.get("categories", [])).items()
    }
    stored = {
        int(value): str(name)
        for value, name in case.metadata_json.get("label_names", {}).items()
        if str(value).isdigit()
    }
    values = sorted(set(values) | set(categories) | set(stored)) or [1, 2, 3]
    deleted = set(case.metadata_json.get("deleted_labels", []))
    deleted.update((draft or {}).get("deleted_labels", []))
    return {
        value: stored.get(value) or categories.get(value) or f"Label {value}"
        for value in values
        if value not in deleted
    }


@router.get("", response_model=ViewerInfo)
def viewer_info(case_id: UUID, db: Db, user: CurrentUser):
    case, dataset = case_dataset(db, case_id)
    _, meta = image_volume(case, dataset)
    annotation = current_annotation(db, case.id)
    draft = work_state(case, user.id)
    return ViewerInfo(
        shape=meta["shape"],
        spacing=[round(value, 4) for value in meta["spacing"]],
        axes=[axis_info(meta, axis) for axis in ((2, 1, 0) if meta["shape"][2] > 1 else (2,))],
        level=meta["level"],
        width=meta["width"],
        range=meta["range"],
        labels=[{"value": value, "name": name} for value, name in label_names(case, draft).items()],
        annotation_format=dataset.annotation_format,
        editable=dataset.annotation_format != AnnotationFormat.NONE,
        has_mask=annotation is not None,
        has_draft=draft is not None,
        draft=draft,
        current_annotation=annotation_out(annotation) if annotation else None,
    )


@router.get("/slice/{axis}/{index}")
def image_slice(
    case_id: UUID,
    axis: int,
    index: int,
    db: Db,
    user: CurrentUser,
    level: float = Query(0),
    width: float = Query(400, gt=0),
):
    case, dataset = case_dataset(db, case_id)
    if dataset.image_format in (ImageFormat.PNG, ImageFormat.JPEG):
        # Keep raster cases lossless and in colour; there is nothing to window.
        if axis != 2 or index != 0:
            raise HTTPException(404, "Slice index out of range")
        image_format = case.metadata_json.get("image_format", dataset.image_format.value)
        return Response(
            get_file_path(case.image_path).read_bytes(),
            media_type=f"image/{image_format.lower()}",
            headers={"Cache-Control": "private, max-age=600"},
        )
    volume, _ = image_volume(case, dataset)
    payload = png_bytes(extract(volume, axis, index), level, width)
    return Response(
        payload, media_type="image/png", headers={"Cache-Control": "private, max-age=600"}
    )


@router.get("/mask/{axis}/{index}")
def mask_slice(case_id: UUID, axis: int, index: int, db: Db, user: CurrentUser):
    case, dataset = case_dataset(db, case_id)
    _, meta = image_volume(case, dataset)
    labels = label_source(db, case, user, meta["shape"])
    if labels is None:
        # No segmentation yet: answer with an empty slice of the right size.
        face(np.empty(meta["shape"], dtype=bool), axis, index)
        plane = np.zeros(plane_spacing(meta["shape"], axis), dtype=np.uint8)
    else:
        plane = extract(labels, axis, index)
    return Response(png_bytes(plane), media_type="image/png", headers={"Cache-Control": "no-store"})


@router.post("/paint/{axis}/{index}", status_code=204)
async def paint_slice(
    case_id: UUID, axis: int, index: int, db: Db, user: CurrentUser, request: Request
):
    """Apply one edited label slice, sent as a grayscale PNG, to the working copy."""
    case, dataset = case_dataset(db, case_id)
    if dataset.annotation_format == AnnotationFormat.NONE:
        raise HTTPException(422, "This case does not support segmentation editing")
    body = await request.body()
    if not body or len(body) > MAX_SLICE_BYTES:
        raise HTTPException(413, "Slice payload is empty or too large")
    try:
        with Image.open(io.BytesIO(body)) as decoded:
            # Canvas uploads are RGBA; label IDs live in the red channel unchanged.
            pixels = np.asarray(decoded)
            plane = np.ascontiguousarray(
                pixels if pixels.ndim == 2 else pixels[..., 0], dtype=np.uint8
            )
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(422, "Slice must be a PNG image") from exc
    _, meta = image_volume(case, dataset)
    draft = open_work(case, user.id, meta["shape"], None)
    if draft is None:
        annotation = current_annotation(db, case.id)
        draft = open_work(
            case,
            user.id,
            meta["shape"],
            annotation_volume(case, annotation, meta["shape"]),
            {
                "base_annotation_id": str(annotation.id) if annotation else None,
                "base_version": annotation.version if annotation else None,
            },
        )
    insert(draft, axis, index, plane)
    draft.flush()
    return Response(status_code=204)


@router.post("/save", response_model=AnnotationOut, status_code=201)
def save_draft(case_id: UUID, db: Db, user: CurrentUser, cleanup: BackgroundTasks):
    """Commit the working copy as a new annotation version in the case's own format."""
    case = require_case(db, case_id, lock=True)
    dataset = db.get(Dataset, case.dataset_id)
    _, meta = image_volume(case, dataset)
    draft = open_work(case, user.id, meta["shape"], None)
    if draft is None:
        raise HTTPException(409, "There are no unsaved segmentation changes")
    parent = current_annotation(db, case.id)
    state = work_state(case, user.id) or {}
    names = label_names(case, state)
    if dataset.annotation_format == AnnotationFormat.COCO:
        if parent is None:
            raise HTTPException(422, "COCO corrections need the original annotation")
        payload = coco_payload(draft, read_json(get_file_path(parent.annotation_path)), names)
        filename = "instances.json"
    else:
        payload = nifti_payload(draft, case, dataset, meta)
        filename = "segmentation.nii.gz"
    del draft
    upload = UploadFile(file=io.BytesIO(payload), filename=filename, size=len(payload))
    annotation = create_correction(db, case, user, upload)
    if state.get("deleted_labels"):
        metadata = dict(case.metadata_json)
        metadata["deleted_labels"] = sorted(
            set(metadata.get("deleted_labels", [])) | set(state["deleted_labels"])
        )
        case.metadata_json = metadata
    if dataset.annotation_format != AnnotationFormat.COCO:
        # A NIfTI mask only stores numbers, so the names travel in a sidecar file.
        save_json(
            db,
            {
                "case": case.case_uid,
                "format": dataset.annotation_format.value,
                "labels": {str(value): name for value, name in names.items()},
            },
            f"{annotation.annotation_path.rsplit('/', 1)[0]}/labels.json",
        )
    # Runs after the request transaction commits, so a failed save keeps the draft.
    cleanup.add_task(discard_work, case, user.id)
    return annotation_out(annotation)


@router.post("/discard", status_code=204)
def discard_draft(case_id: UUID, db: Db, user: CurrentUser):
    case = require_case(db, case_id)
    discard_work(case, user.id)
    return Response(status_code=204)


@router.post("/labels", response_model=LabelNames)
def rename_labels(case_id: UUID, body: LabelNames, db: Db, user: CurrentUser):
    """Name the numeric label values; the names are saved with the next version."""
    case = require_case(db, case_id, lock=True)
    if db.get(Dataset, case.dataset_id).annotation_format == AnnotationFormat.NONE:
        raise HTTPException(422, "This case does not support segmentation editing")
    if len({item.value for item in body.labels}) != len(body.labels):
        raise HTTPException(422, "Label values must be unique")
    draft = db.scalar(
        select(Review).where(Review.case_id == case.id, Review.submitted_at.is_(None))
    )
    if draft and draft.reviewer_id != user.id:
        raise HTTPException(409, "Another reviewer is reviewing this case")
    metadata = dict(case.metadata_json)
    metadata["label_names"] = {
        **metadata.get("label_names", {}),
        **{str(item.value): item.name for item in body.labels},
    }
    metadata["deleted_labels"] = [
        value
        for value in metadata.get("deleted_labels", [])
        if value not in {item.value for item in body.labels}
    ]
    case.metadata_json = metadata  # A new dict marks the JSON column as changed.
    db.flush()
    state = work_state(case, user.id)
    if state and set(state.get("deleted_labels", [])) & {item.value for item in body.labels}:
        state["deleted_labels"] = [
            value
            for value in state["deleted_labels"]
            if value not in {item.value for item in body.labels}
        ]
        get_file_path(f"{work_dir(case, user.id)}/meta.json").write_text(json.dumps(state))
    return LabelNames(labels=[{"value": v, "name": n} for v, n in label_names(case, state).items()])


@router.delete("/labels/{value}", response_model=LabelNames)
def delete_label(case_id: UUID, db: Db, user: CurrentUser, value: int = Path(ge=1, le=255)):
    """Clear a label on every slice in the private draft; save/discard applies as usual."""
    case = require_case(db, case_id, lock=True)
    dataset = db.get(Dataset, case.dataset_id)
    if dataset.annotation_format == AnnotationFormat.NONE:
        raise HTTPException(422, "This case does not support segmentation editing")
    review = db.scalar(
        select(Review).where(Review.case_id == case.id, Review.submitted_at.is_(None))
    )
    if review and review.reviewer_id != user.id:
        raise HTTPException(409, "Another reviewer is reviewing this case")
    state = work_state(case, user.id)
    if value not in label_names(case, state):
        raise HTTPException(404, "Label not found")
    _, meta = image_volume(case, dataset)
    annotation = current_annotation(db, case.id)
    state = state or {
        "base_annotation_id": str(annotation.id) if annotation else None,
        "base_version": annotation.version if annotation else None,
    }
    draft = open_work(case, user.id, meta["shape"], None)
    if draft is None:
        draft = open_work(
            case, user.id, meta["shape"], annotation_volume(case, annotation, meta["shape"]), state
        )
    # Process one plane at a time to avoid allocating a whole-volume boolean mask.
    for index in range(draft.shape[2]):
        plane = draft[:, :, index]
        plane[plane == value] = 0
    draft.flush()
    state["deleted_labels"] = sorted(set(state.get("deleted_labels", [])) | {value})
    get_file_path(f"{work_dir(case, user.id)}/meta.json").write_text(json.dumps(state))
    return LabelNames(labels=[{"value": v, "name": n} for v, n in label_names(case, state).items()])
