import os

os.environ.setdefault("JWT_SECRET", "test-only-secret-key-at-least-32-characters")
os.environ.setdefault("DATABASE_URL", "sqlite://")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, event  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app import db as database  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.security import make_token, password_hasher  # noqa: E402
from app.db import Base  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Role, User  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "data_dir", tmp_path / "datasets")
    monkeypatch.setattr(settings, "import_dir", tmp_path / "import")
    settings.import_dir.mkdir()
    url = os.environ.get("TEST_DATABASE_URL", "sqlite://")
    options = (
        {"connect_args": {"check_same_thread": False}, "poolclass": StaticPool}
        if url == "sqlite://"
        else {}
    )
    engine = create_engine(url, **options)
    if url == "sqlite://":

        @event.listens_for(engine, "connect")
        def foreign_keys(connection, record):
            connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(database, "SessionLocal", sessions)
    users = {}
    with sessions.begin() as db:
        hashed = password_hasher.hash("test-password-123")
        for name, role in [
            ("admin", Role.ADMIN),
            ("doctor", Role.REVIEWER),
            ("other", Role.REVIEWER),
        ]:
            user = User(username=name, full_name=name, role=role, password_hash=hashed)
            db.add(user)
            db.flush()
            users[name] = {"Authorization": f"Bearer {make_token(user)}"}
    with TestClient(app) as client:
        yield client, settings, users, sessions
    Base.metadata.drop_all(engine)
    engine.dispose()


def create_dataset(client, headers, kind="NIFTI"):
    formats = {
        "NIFTI": ("3D", "NIFTI", "NIFTI"),
        "COCO": ("2D", "PNG", "COCO"),
        "DICOM": ("3D", "DICOM", "NONE"),
    }
    dimension, image, annotation = formats[kind]
    response = client.post(
        "/api/v1/datasets",
        headers=headers,
        json={
            "name": "Test dataset",
            "dimension": dimension,
            "image_format": image,
            "annotation_format": annotation,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def ingest(client, headers, dataset_id, root, kind="NIFTI"):
    return client.post(
        f"/api/v1/datasets/{dataset_id}/ingest",
        headers=headers,
        json={"type": kind, "path": str(root)},
    )
