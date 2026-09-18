import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class Role(str, enum.Enum):
    ADMIN = "ADMIN"
    REVIEWER = "REVIEWER"


class Dimension(str, enum.Enum):
    TWO_D = "2D"
    THREE_D = "3D"


class ImageFormat(str, enum.Enum):
    NIFTI = "NIFTI"
    DICOM = "DICOM"
    PNG = "PNG"
    JPEG = "JPEG"


class AnnotationFormat(str, enum.Enum):
    NIFTI = "NIFTI"
    COCO = "COCO"
    NONE = "NONE"


class CaseStatus(str, enum.Enum):
    PENDING = "PENDING"
    IN_REVIEW = "IN_REVIEW"
    APPROVED = "APPROVED"
    MODIFIED = "MODIFIED"
    NEEDS_CORRECTION = "NEEDS_CORRECTION"
    REJECTED = "REJECTED"


class Decision(str, enum.Enum):
    APPROVED = "APPROVED"
    MODIFIED = "MODIFIED"
    NEEDS_CORRECTION = "NEEDS_CORRECTION"
    REJECTED = "REJECTED"


def enum_column(kind):
    return Enum(
        kind,
        native_enum=False,
        create_constraint=True,
        values_callable=lambda cls: [item.value for item in cls],
    )


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(100), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(200))
    role: Mapped[Role] = mapped_column(enum_column(Role))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    dimension: Mapped[Dimension] = mapped_column(enum_column(Dimension))
    image_format: Mapped[ImageFormat] = mapped_column(enum_column(ImageFormat))
    annotation_format: Mapped[AnnotationFormat] = mapped_column(enum_column(AnnotationFormat))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Case(Base):
    __tablename__ = "cases"
    __table_args__ = (
        UniqueConstraint("dataset_id", "case_uid"),
        Index("ix_cases_dataset_status", "dataset_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("datasets.id"))
    case_uid: Mapped[str] = mapped_column(String(255))
    image_path: Mapped[str] = mapped_column(Text)
    # SQLAlchemy reserves 'metadata'; keep the actual DB column/API key as metadata.
    metadata_json: Mapped[dict] = mapped_column(
        "metadata", JSON().with_variant(JSONB, "postgresql")
    )
    status: Mapped[CaseStatus] = mapped_column(enum_column(CaseStatus), default=CaseStatus.PENDING)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class AnnotationVersion(Base):
    __tablename__ = "annotation_versions"
    __table_args__ = (
        UniqueConstraint("case_id", "version"),
        CheckConstraint("version >= 0", name="annotation_version_nonnegative"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id"))
    version: Mapped[int]
    annotation_path: Mapped[str] = mapped_column(Text, unique=True)
    format: Mapped[AnnotationFormat] = mapped_column(enum_column(AnnotationFormat))
    parent_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("annotation_versions.id"))
    created_by: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Review(Base):
    __tablename__ = "reviews"
    __table_args__ = (Index("ix_reviews_case_id", "case_id"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cases.id"))
    reviewer_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    annotation_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("annotation_versions.id")
    )
    decision: Mapped[Decision] = mapped_column(enum_column(Decision))
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
