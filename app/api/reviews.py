from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, UploadFile
from sqlalchemy import select

from app.api.cases import current_annotation, require_case
from app.core.security import CurrentUser
from app.db import Db
from app.models import AnnotationVersion, CaseStatus, Decision, Review
from app.schemas import ReviewCreate, ReviewOut
from app.services.annotation import create_correction
from app.services.storage import get_file_path

router = APIRouter(tags=["reviews"])


@router.post("/cases/{case_id}/reviews", response_model=ReviewOut, status_code=201)
def create_review(case_id: UUID, body: ReviewCreate, db: Db, user: CurrentUser):
    case = require_case(db, case_id, lock=True)
    if db.scalar(select(Review.id).where(Review.case_id == case.id, Review.submitted_at.is_(None))):
        raise HTTPException(409, "Case already has an open review")
    annotation = (
        db.get(AnnotationVersion, body.annotation_version_id)
        if body.annotation_version_id
        else current_annotation(db, case_id)
    )
    if body.annotation_version_id and (not annotation or annotation.case_id != case_id):
        raise HTTPException(422, "Annotation does not belong to this case")
    if body.decision == Decision.MODIFIED and annotation is None:
        raise HTTPException(422, "Case has no compatible annotation to modify")
    review = Review(
        case_id=case_id,
        reviewer_id=user.id,
        decision=body.decision,
        comment=body.comment,
        annotation_version_id=annotation.id if annotation else None,
    )
    db.add(review)
    case.status = CaseStatus.IN_REVIEW
    db.flush()
    return review


@router.patch("/reviews/{review_id}", response_model=ReviewOut)
def update_review(review_id: UUID, body: ReviewCreate, db: Db, user: CurrentUser):
    """Save an open draft's decision and comment; submitted reviews stay immutable."""
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(404, "Review not found")
    case = require_case(db, review.case_id, lock=True)
    db.refresh(review)
    if review.reviewer_id != user.id:
        raise HTTPException(403, "Only the review author can edit it")
    if review.submitted_at:
        raise HTTPException(409, "Review already submitted")
    annotation = (
        db.get(AnnotationVersion, body.annotation_version_id)
        if body.annotation_version_id
        else current_annotation(db, case.id)
    )
    if body.annotation_version_id and (not annotation or annotation.case_id != case.id):
        raise HTTPException(422, "Annotation does not belong to this case")
    review.decision = body.decision
    review.comment = body.comment
    review.annotation_version_id = annotation.id if annotation else None
    db.flush()
    return review


@router.get("/cases/{case_id}/reviews", response_model=list[ReviewOut])
def list_reviews(case_id: UUID, db: Db, user: CurrentUser):
    require_case(db, case_id)
    return db.scalars(
        select(Review).where(Review.case_id == case_id).order_by(Review.created_at, Review.id)
    ).all()


@router.post("/reviews/{review_id}/submit", response_model=ReviewOut)
def submit_review(
    review_id: UUID, db: Db, user: CurrentUser, file: Annotated[UploadFile | None, File()] = None
):
    review = db.get(Review, review_id)
    if not review:
        raise HTTPException(404, "Review not found")
    case = require_case(db, review.case_id, lock=True)
    db.refresh(review)  # Another submit may have committed while acquiring the case lock.
    if review.reviewer_id != user.id:
        raise HTTPException(403, "Only the review author can submit it")
    if review.submitted_at:
        raise HTTPException(409, "Review already submitted")
    if file:
        if review.decision != Decision.MODIFIED:
            raise HTTPException(422, "An attached correction requires a MODIFIED decision")
        annotation = create_correction(db, case, user, file)
        review.annotation_version_id = annotation.id
    latest = current_annotation(db, case.id)
    if latest and review.annotation_version_id != latest.id:
        raise HTTPException(409, "Review targets an outdated annotation version")
    if review.decision == Decision.MODIFIED:
        if not latest or latest.version == 0 or latest.created_by != user.id:
            raise HTTPException(422, "MODIFIED requires a correction uploaded by this reviewer")
        if not get_file_path(latest.annotation_path).is_file():
            raise HTTPException(409, "Corrected annotation file is missing")
    review.submitted_at = datetime.now(UTC)
    case.status = CaseStatus(review.decision.value)
    db.flush()
    return review
