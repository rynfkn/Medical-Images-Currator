from pathlib import Path
from uuid import UUID

from conftest import create_dataset
from test_coco import coco_input
from test_reviews import setup_case

from app.models import Case


def test_authentication_and_admin_permissions(env):
    client, settings, users, _ = env
    assert client.get("/api/v1/datasets").status_code == 401
    assert (
        client.post(
            "/api/v1/auth/login", data={"username": "doctor", "password": "incorrect"}
        ).status_code
        == 401
    )
    response = client.post(
        "/api/v1/auth/login", data={"username": "doctor", "password": "test-password-123"}
    )
    assert response.status_code == 200
    response = client.get(
        "/api/v1/auth/me", headers={"Authorization": f"Bearer {response.json()['access_token']}"}
    )
    assert response.json()["role"] == "REVIEWER"
    assert "password_hash" not in response.json()
    assert (
        client.post(
            "/api/v1/datasets",
            headers=users["doctor"],
            json={
                "name": "Denied",
                "dimension": "3D",
                "image_format": "NIFTI",
                "annotation_format": "NIFTI",
            },
        ).status_code
        == 403
    )
    dataset_id = create_dataset(client, users["admin"])
    assert (
        client.post(
            f"/api/v1/datasets/{dataset_id}/ingest",
            headers=users["doctor"],
            json={"type": "NIFTI", "path": str(settings.import_dir)},
        ).status_code
        == 403
    )
    assert (
        client.post(
            f"/api/v1/datasets/{dataset_id}/ingest",
            headers=users["admin"],
            json={"type": "NIFTI", "path": "/etc"},
        ).status_code
        == 400
    )


def test_file_path_containment_and_symlinks(env, tmp_path):
    client, settings, users, sessions = env
    case = setup_case(env)
    outside = tmp_path / "secret"
    outside.write_text("not a dataset file")
    symlink = settings.data_dir / "escape"
    symlink.symlink_to(outside)
    for path in ("../secret", str(outside), "escape"):
        with sessions.begin() as db:
            db.get(Case, UUID(case["id"])).image_path = path
        response = client.get(case["image_url"], headers=users["doctor"])
        assert response.status_code == 400
        assert outside.read_text() not in response.text


def test_import_symlink_cannot_escape(env, tmp_path):
    client, settings, users, _ = env
    root = settings.import_dir / "unsafe"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    outside = tmp_path / "outside.nii.gz"
    outside.write_bytes(b"secret")
    (root / "images/a.nii.gz").symlink_to(outside)
    (root / "labels/a.nii.gz").symlink_to(outside)
    dataset_id = create_dataset(client, users["admin"])
    response = client.post(
        f"/api/v1/datasets/{dataset_id}/ingest",
        headers=users["admin"],
        json={"type": "NIFTI", "path": str(root)},
    )
    assert response.status_code == 422
    assert not [p for p in settings.data_dir.rglob("*") if Path(p).is_file()]


def test_coco_images_directory_symlink_cannot_escape(env, tmp_path):
    client, settings, users, _ = env
    root = settings.import_dir / "coco"
    coco_input(root)
    outside = tmp_path / "outside_images"
    (root / "images").rename(outside)
    (root / "images").symlink_to(outside, target_is_directory=True)
    dataset_id = create_dataset(client, users["admin"], "COCO")
    response = client.post(
        f"/api/v1/datasets/{dataset_id}/ingest",
        headers=users["admin"],
        json={"type": "COCO", "path": str(root)},
    )
    assert response.status_code == 422
    assert not [p for p in settings.data_dir.rglob("*") if p.is_file()]
