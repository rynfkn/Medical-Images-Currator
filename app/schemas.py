import uuid
from datetime import datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
