import json
from datetime import UTC, datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from fastapi import HTTPException, UploadFile
from pycocotools import mask as coco_mask
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import AnnotationFormat, AnnotationVersion, Case, CaseStatus, Review, User
from app.services.storage import get_file_path, save_file


def nifti_metadata(image_path: Path, mask_path: Path) -> dict:
    try:
        image = nib.load(image_path)
        mask = nib.load(mask_path)
        if len(image.shape) != 3:
            raise ValueError("Expected a 3D NIfTI image")
        if image.shape != mask.shape:
            raise ValueError("NIfTI image and segmentation shapes must match")
        # Read both payloads, not only their headers, to catch truncated files.
        np.asanyarray(image.dataobj)
        values = np.asanyarray(mask.dataobj)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError("Segmentation labels must be finite integers")
        if not np.isfinite(image.affine).all():
            raise ValueError("NIfTI affine must be finite")
        spacing = [float(x) for x in image.header.get_zooms()[:3]]
        if any(not np.isfinite(x) or x <= 0 for x in spacing):
            raise ValueError("NIfTI spacing must be positive and finite")
        return {
            "shape": list(image.shape),
            "spacing": spacing,
            "affine": image.affine.tolist(),
            "labels": [int(x) for x in np.unique(values)],
        }
    except (OSError, EOFError, nib.filebasedimages.ImageFileError) as exc:
        raise ValueError("Unreadable NIfTI file") from exc


def create_correction(db: Session, case: Case, user: User, upload: UploadFile) -> AnnotationVersion:
    """Caller holds the case row lock through commit, serializing version numbers."""
    draft = db.scalar(
        select(Review).where(Review.case_id == case.id, Review.submitted_at.is_(None))
    )
    if draft and draft.reviewer_id != user.id:
        raise HTTPException(409, "Another reviewer is reviewing this case")
    parent = db.scalar(
        select(AnnotationVersion)
        .where(AnnotationVersion.case_id == case.id)
        .order_by(AnnotationVersion.version.desc())
        .limit(1)
    )
    if parent is None or parent.format not in (AnnotationFormat.NIFTI, AnnotationFormat.COCO):
        raise HTTPException(422, "This case does not support segmentation corrections")
    filename = upload.filename or ""
    if parent.format == AnnotationFormat.NIFTI and not filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(422, "Upload a .nii or .nii.gz segmentation")
    suffix = ".nii.gz" if filename.endswith(".nii.gz") else ".nii"
    if parent.format == AnnotationFormat.COCO:
        suffix = ".json"
    version = parent.version + 1
    path = f"{case.dataset_id}/cases/{case.id}/annotations/v{version}/segmentation{suffix}"
    save_file(db, upload.file, path, max_bytes=get_settings().max_upload_bytes)
    try:
        if parent.format == AnnotationFormat.NIFTI:
            nifti_metadata(get_file_path(case.image_path), get_file_path(path))
        else:
            original = read_json(get_file_path(parent.annotation_path))
            corrected = read_json(get_file_path(path))
            validate_coco_case(corrected, original["image"], original["categories"])
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    annotation = AnnotationVersion(
        case_id=case.id,
        version=version,
        parent_id=parent.id,
        annotation_path=path,
        format=parent.format,
        created_by=user.id,
    )
    db.add(annotation)
    case.status = CaseStatus.IN_REVIEW
    case.updated_at = datetime.now(UTC)
    db.flush()
    # A doctor's correction becomes the version targeted by their open draft.
    if draft:
        draft.annotation_version_id = annotation.id
    return annotation


def read_json(path: Path) -> dict:
    def invalid_constant(value):
        raise ValueError(f"Invalid JSON number: {value}")

    try:
        with path.open() as stream:
            value = json.load(stream, parse_constant=invalid_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Unreadable or invalid COCO JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("COCO JSON must be an object")
    return value


def coco_ids(items, name: str) -> set[int]:
    if not isinstance(items, list) or any(
        not isinstance(item, dict) or type(item.get("id")) is not int for item in items
    ):
        raise ValueError(f"COCO {name} must be a list of objects with integer IDs")
    ids = {item["id"] for item in items}
    if len(ids) != len(items):
        raise ValueError(f"Duplicate COCO {name} IDs")
    return ids


def rle_counts(counts, pixels: int) -> list[int]:
    if isinstance(counts, str):
        # Validate compressed COCO runs before passing them to the native decoder.
        runs, pos = [], 0
        while pos < len(counts):
            value, shift = 0, 0
            while True:
                if pos >= len(counts) or shift > 60:
                    raise ValueError("Invalid compressed RLE counts")
                code = ord(counts[pos]) - 48
                pos += 1
                if not 0 <= code <= 63:
                    raise ValueError("Invalid compressed RLE counts")
                value |= (code & 31) << shift
                shift += 5
                if not code & 32:
                    if code & 16:
                        value |= -1 << shift
                    break
            if len(runs) > 2:
                value += runs[-2]
            runs.append(value)
    elif isinstance(counts, list):
        runs = counts
    else:
        raise ValueError("RLE counts must be a list or compressed string")
    if (
        not runs
        or any(type(x) is not int or x < 0 or x > pixels for x in runs)
        or sum(runs) != pixels
    ):
        raise ValueError("RLE counts must cover the image exactly")
    return runs


def validate_coco_case(document: dict, expected_image: dict, categories: list[dict]) -> None:
    image = document.get("image")
    if not isinstance(image, dict) or any(
        type(image.get(key)) is not int or image[key] != expected_image[key]
        for key in ("id", "width", "height")
    ):
        raise ValueError("COCO image ID and dimensions must match the source image")
    width, height = image["width"], image["height"]
    if width <= 0 or height <= 0:
        raise ValueError("COCO image dimensions must be positive")
    category_ids = coco_ids(categories, "categories")
    if coco_ids(document.get("categories"), "categories") != category_ids:
        raise ValueError("COCO category IDs must match the dataset categories")
    annotations = document.get("annotations")
    coco_ids(annotations, "annotations")
    for annotation in annotations:
        if type(annotation.get("image_id")) is not int or annotation["image_id"] != image["id"]:
            raise ValueError("Annotation image_id does not match the case")
        if (
            type(annotation.get("category_id")) is not int
            or annotation["category_id"] not in category_ids
        ):
            raise ValueError("Unknown COCO category_id")
        segmentation = annotation.get("segmentation")
        try:
            if isinstance(segmentation, list) and segmentation:
                for polygon in segmentation:
                    if (
                        not isinstance(polygon, list)
                        or len(polygon) < 6
                        or len(polygon) % 2
                        or any(
                            type(x) not in (int, float) or not np.isfinite(x) or abs(x) > 1e8
                            for x in polygon
                        )
                    ):
                        raise ValueError("Invalid COCO polygon")
                encoded = coco_mask.frPyObjects(segmentation, height, width)
            elif isinstance(segmentation, dict):
                if segmentation.get("size") != [height, width]:
                    raise ValueError("RLE dimensions do not match image dimensions")
                runs = rle_counts(segmentation.get("counts"), width * height)
                encoded = coco_mask.frPyObjects(
                    {"size": [height, width], "counts": runs}, height, width
                )
            else:
                raise ValueError("COCO segmentation is required")
            coco_mask.area(encoded)
        except (TypeError, OverflowError, KeyError) as exc:
            raise ValueError("Invalid COCO segmentation structure") from exc
