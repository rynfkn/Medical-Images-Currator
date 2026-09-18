from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, UploadFile
from sqlalchemy import select

from app.api.cases import annotation_out, current_annotation, require_case
from app.core.security import CurrentUser
from app.db import Db
from app.models import AnnotationVersion
from app.schemas import AnnotationOut
from app.services.annotation import create_correction

router = APIRouter(prefix="/cases/{case_id}/annotations", tags=["annotations"])


@router.get("", response_model=list[AnnotationOut])
def list_annotations(case_id: UUID, db: Db, user: CurrentUser):
    require_case(db, case_id)
    return [
        annotation_out(a)
        for a in db.scalars(
            select(AnnotationVersion)
            .where(AnnotationVersion.case_id == case_id)
            .order_by(AnnotationVersion.version)
        )
    ]


@router.get("/current", response_model=AnnotationOut)
def get_current_annotation(case_id: UUID, db: Db, user: CurrentUser):
    require_case(db, case_id)
    annotation = current_annotation(db, case_id)
    if annotation is None:
        raise HTTPException(404, "Case has no annotation")
    return annotation_out(annotation)


@router.post("", response_model=AnnotationOut, status_code=201)
def upload_annotation(
    case_id: UUID, db: Db, user: CurrentUser, file: Annotated[UploadFile, File()]
):
    case = require_case(db, case_id, lock=True)
    return annotation_out(create_correction(db, case, user, file))
