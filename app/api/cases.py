from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import Admin, CurrentUser
from app.db import Db
from app.models import AnnotationVersion, Case, CaseStatus, Dataset, ImageFormat
from app.schemas import AnnotationOut, CaseOut, CaseSummary
from app.services.deletion import delete_cases, remove_managed_directory

router = APIRouter(tags=["cases"])


@router.delete("/cases/{case_id}", status_code=204)
def delete_case(case_id: UUID, db: Db, user: Admin, cleanup: BackgroundTasks):
    case = require_case(db, case_id)
    # Use the same dataset -> case lock order as project deletion and ingestion.
    db.scalar(select(Dataset).where(Dataset.id == case.dataset_id).with_for_update())
    case = require_case(db, case_id, lock=True)
    directory = f"{case.dataset_id}/cases/{case.id}"
    delete_cases(db, [case.id])
    db.flush()
    cleanup.add_task(remove_managed_directory, directory)
    return Response(status_code=204)


def require_case(db: Session, case_id: UUID, *, lock: bool = False) -> Case:
    query = select(Case).where(Case.id == case_id)
    if lock:
        query = query.with_for_update().execution_options(populate_existing=True)
    case = db.scalar(query)
    if not case:
        raise HTTPException(404, "Case not found")
    return case


def current_annotation(db: Session, case_id: UUID) -> AnnotationVersion | None:
    return db.scalar(
        select(AnnotationVersion)
        .where(AnnotationVersion.case_id == case_id)
        .order_by(AnnotationVersion.version.desc())
        .limit(1)
    )


def annotation_out(annotation: AnnotationVersion) -> AnnotationOut:
    return AnnotationOut(
        **{key: getattr(annotation, key) for key in AnnotationOut.model_fields if key != "url"},
        url=f"/api/v1/files/annotations/{annotation.id}",
    )


def case_out(db: Session, case: Case) -> CaseOut:
    dataset = db.get(Dataset, case.dataset_id)
    annotation = current_annotation(db, case.id)
    dicom = dataset.image_format == ImageFormat.DICOM
    urls = (
        [
            f"/api/v1/files/cases/{case.id}/dicom/{i}"
            for i in range(len(case.metadata_json["slices"]))
        ]
        if dicom
        else [f"/api/v1/files/cases/{case.id}/image"]
    )
    return CaseOut(
        id=case.id,
        dataset_id=case.dataset_id,
        case_uid=case.case_uid,
        dimension=dataset.dimension,
        image_format=case.metadata_json.get("image_format", dataset.image_format),
        annotation_format=dataset.annotation_format,
        status=case.status,
        image_url=None if dicom else urls[0],
        image_urls=urls,
        annotation_url=f"/api/v1/files/annotations/{annotation.id}" if annotation else None,
        current_annotation=annotation_out(annotation) if annotation else None,
        metadata=case.metadata_json,
        created_at=case.created_at,
        updated_at=case.updated_at,
    )


@router.get("/datasets/{dataset_id}/cases", response_model=list[CaseOut])
def list_cases(
    dataset_id: UUID,
    db: Db,
    user: CurrentUser,
    status: CaseStatus | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    if db.get(Dataset, dataset_id) is None:
        raise HTTPException(404, "Dataset not found")
    query = select(Case).where(Case.dataset_id == dataset_id)
    if status:
        query = query.where(Case.status == status)
    cases = db.scalars(query.order_by(Case.case_uid, Case.id).offset(offset).limit(limit))
    return [case_out(db, case) for case in cases]


@router.get("/cases/{case_id}", response_model=CaseOut)
def get_case(case_id: UUID, db: Db, user: CurrentUser):
    return case_out(db, require_case(db, case_id))


@router.get("/datasets/{dataset_id}/cases/index", response_model=list[CaseSummary])
def case_index(dataset_id: UUID, db: Db, user: CurrentUser):
    """Every case ID in display order, so the viewer can step through the dataset."""
    if db.get(Dataset, dataset_id) is None:
        raise HTTPException(404, "Dataset not found")
    return db.scalars(
        select(Case).where(Case.dataset_id == dataset_id).order_by(Case.case_uid, Case.id)
    ).all()
