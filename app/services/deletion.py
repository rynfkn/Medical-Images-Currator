"""Delete database dependents first; remove managed files only after commit."""

import logging
import shutil

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from app.models import AnnotationVersion, Case, Review
from app.services.storage import get_file_path

logger = logging.getLogger(__name__)


def delete_cases(db: Session, case_ids: list) -> None:
    if not case_ids:
        return
    # Lock in a stable order to serialize against corrections and reviews.
    list(db.scalars(select(Case).where(Case.id.in_(case_ids)).order_by(Case.id).with_for_update()))
    db.execute(delete(Review).where(Review.case_id.in_(case_ids)))
    db.execute(
        update(AnnotationVersion)
        .where(AnnotationVersion.case_id.in_(case_ids))
        .values(parent_id=None)
    )
    db.execute(delete(AnnotationVersion).where(AnnotationVersion.case_id.in_(case_ids)))
    db.execute(delete(Case).where(Case.id.in_(case_ids)))


def remove_managed_directory(relative: str) -> None:
    path = get_file_path(relative)
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except OSError:
        # The database deletion has committed. Report cleanup failures for retry
        # without turning a successful deletion into a misleading API failure.
        logger.exception("Could not remove deleted dataset storage: %s", path)
