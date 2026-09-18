from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.api.cases import require_case
from app.core.security import CurrentUser
from app.db import Db
from app.models import AnnotationVersion, Dataset, ImageFormat
from app.services.storage import get_file_path

router = APIRouter(prefix="/files", tags=["files"])


def file_response(path: str, media_type: str | None = None):
    resolved = get_file_path(path)
    if not resolved.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(resolved, media_type=media_type, filename=resolved.name)


@router.get("/cases/{case_id}/image")
def image_file(case_id: UUID, db: Db, user: CurrentUser):
    case = require_case(db, case_id)
    if db.get(Dataset, case.dataset_id).image_format == ImageFormat.DICOM:
        raise HTTPException(400, "Use the individual DICOM slice URLs")
    return file_response(case.image_path)


@router.get("/cases/{case_id}/dicom/{slice_index}")
def dicom_file(case_id: UUID, slice_index: int, db: Db, user: CurrentUser):
    case = require_case(db, case_id)
    slices = case.metadata_json.get("slices", [])
    if slice_index < 0 or slice_index >= len(slices):
        raise HTTPException(404, "DICOM slice not found")
    return file_response(slices[slice_index]["path"], "application/dicom")


@router.get("/annotations/{annotation_id}")
def annotation_file(annotation_id: UUID, db: Db, user: CurrentUser):
    annotation = db.get(AnnotationVersion, annotation_id)
    if not annotation:
        raise HTTPException(404, "Annotation not found")
    return file_response(annotation.annotation_path)


@router.get("/annotations/{annotation_id}/labels")
def annotation_labels(annotation_id: UUID, db: Db, user: CurrentUser):
    annotation = db.get(AnnotationVersion, annotation_id)
    if not annotation:
        raise HTTPException(404, "Annotation not found")
    return file_response(
        f"{annotation.annotation_path.rsplit('/', 1)[0]}/labels.json", "application/json"
    )
