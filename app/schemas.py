import uuid
from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.models import AnnotationFormat, CaseStatus, Decision, Dimension, ImageFormat, Role


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class UserOut(ORMModel):
    id: uuid.UUID
    username: str
    full_name: str
    role: Role
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None
    dimension: Dimension
    image_format: ImageFormat
    annotation_format: AnnotationFormat

    @model_validator(mode="after")
    def supported_combination(self) -> Self:
        supported = {
            (Dimension.THREE_D, ImageFormat.NIFTI, AnnotationFormat.NIFTI),
            (Dimension.THREE_D, ImageFormat.DICOM, AnnotationFormat.NONE),
            (Dimension.THREE_D, ImageFormat.DICOM, AnnotationFormat.NIFTI),
            (Dimension.TWO_D, ImageFormat.PNG, AnnotationFormat.COCO),
            (Dimension.TWO_D, ImageFormat.JPEG, AnnotationFormat.COCO),
        }
        if (self.dimension, self.image_format, self.annotation_format) not in supported:
            raise ValueError("Unsupported dimension/image/annotation format combination")
        return self


class DatasetOut(DatasetCreate, ORMModel):
    id: uuid.UUID
    created_at: datetime


class IngestRequest(BaseModel):
    type: Literal["NIFTI", "COCO", "DICOM"]
    path: str = Field(min_length=1)


class AnnotationOut(ORMModel):
    id: uuid.UUID
    case_id: uuid.UUID
    version: int
    annotation_path: str
    format: AnnotationFormat
    parent_id: uuid.UUID | None
    created_by: uuid.UUID
    created_at: datetime
    url: str


class CaseOut(BaseModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    case_uid: str
    dimension: Dimension
    image_format: ImageFormat
    annotation_format: AnnotationFormat
    status: CaseStatus
    image_url: str | None
    image_urls: list[str]
    annotation_url: str | None
    current_annotation: AnnotationOut | None
    metadata: dict
    created_at: datetime
    updated_at: datetime


class ReviewCreate(BaseModel):
    decision: Decision
    comment: str | None = Field(default=None, max_length=10000)
    annotation_version_id: uuid.UUID | None = None


class ReviewOut(ORMModel):
    id: uuid.UUID
    case_id: uuid.UUID
    reviewer_id: uuid.UUID
    annotation_version_id: uuid.UUID | None
    decision: Decision
    comment: str | None
    created_at: datetime
    submitted_at: datetime | None


class ViewerAxis(BaseModel):
    index: int
    name: str
    count: int
    rows: int
    columns: int
    row_mm: float
    column_mm: float


class ViewerLabel(BaseModel):
    value: int = Field(ge=1, le=255)
    name: str = Field(min_length=1, max_length=60)

    @field_validator("name", mode="before")
    @classmethod
    def trim_name(cls, value):
        return value.strip() if isinstance(value, str) else value


class LabelNames(BaseModel):
    """Names for the numeric label values stored in the segmentation mask."""

    labels: list[ViewerLabel] = Field(max_length=255)


class ViewerInfo(BaseModel):
    shape: list[int]
    spacing: list[float]
    axes: list[ViewerAxis]
    level: float
    width: float
    range: list[int]
    labels: list[ViewerLabel]
    annotation_format: AnnotationFormat
    editable: bool
    has_mask: bool
    has_draft: bool
    draft: dict | None
    current_annotation: AnnotationOut | None


class CaseSummary(ORMModel):
    id: uuid.UUID
    case_uid: str
    status: CaseStatus
