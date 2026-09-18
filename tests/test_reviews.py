import uuid

import nibabel as nib
import numpy as np
import pytest
from conftest import create_dataset, ingest
from sqlalchemy import select
from test_nifti import nifti_input

from app.models import Case, Review


def setup_case(env):
    client, settings, users, _ = env
    root = settings.import_dir / "nifti"
    nifti_input(root)
    dataset_id = create_dataset(client, users["admin"])
    assert ingest(client, users["admin"], dataset_id, root).status_code == 201
    return client.get(f"/api/v1/datasets/{dataset_id}/cases", headers=users["doctor"]).json()[0]


def correction_file(settings, shape=(4, 5, 6)):
    path = settings.import_dir / "corrected.nii.gz"
    nib.save(nib.Nifti1Image(np.full(shape, 2, dtype=np.int16), np.eye(4)), path)
    return {"file": (path.name, path.read_bytes(), "application/octet-stream")}


@pytest.mark.parametrize("decision", ["APPROVED", "NEEDS_CORRECTION", "REJECTED"])
def test_review_decisions(env, decision):
    client, _, users, _ = env
    case = setup_case(env)
    response = client.post(
        f"/api/v1/cases/{case['id']}/reviews",
        headers=users["doctor"],
        json={"decision": decision, "comment": "Reviewed."},
    )
    assert response.status_code == 201
    review_id = response.json()["id"]
    assert (
        client.get(f"/api/v1/cases/{case['id']}", headers=users["doctor"]).json()["status"]
        == "IN_REVIEW"
    )
    assert (
        client.post(f"/api/v1/reviews/{review_id}/submit", headers=users["other"]).status_code
        == 403
    )
    assert (
        client.post(f"/api/v1/reviews/{review_id}/submit", headers=users["doctor"]).status_code
        == 200
    )
    assert (
        client.get(f"/api/v1/cases/{case['id']}", headers=users["doctor"]).json()["status"]
        == decision
    )
    assert (
        client.post(f"/api/v1/reviews/{review_id}/submit", headers=users["doctor"]).status_code
        == 409
    )


@pytest.mark.parametrize("atomic", [False, True])
def test_modified_preserves_original(env, atomic):
    client, settings, users, _ = env
    case = setup_case(env)
    original_path = settings.data_dir / case["current_annotation"]["annotation_path"]
    original = original_path.read_bytes()
    if not atomic:
        response = client.post(
            f"/api/v1/cases/{case['id']}/annotations",
            headers=users["doctor"],
            files=correction_file(settings),
        )
        assert response.status_code == 201, response.text
        assert response.json()["version"] == 1
    review = client.post(
        f"/api/v1/cases/{case['id']}/reviews",
        headers=users["doctor"],
        json={"decision": "MODIFIED", "comment": "Tumor boundary corrected."},
    ).json()
    kwargs = {"files": correction_file(settings)} if atomic else {}
    response = client.post(
        f"/api/v1/reviews/{review['id']}/submit", headers=users["doctor"], **kwargs
    )
    assert response.status_code == 200, response.text
    versions = client.get(f"/api/v1/cases/{case['id']}/annotations", headers=users["doctor"]).json()
    assert [v["version"] for v in versions] == [0, 1]
    assert versions[1]["parent_id"] == versions[0]["id"]
    assert original_path.read_bytes() == original
    assert (
        client.get(f"/api/v1/cases/{case['id']}", headers=users["doctor"]).json()["status"]
        == "MODIFIED"
    )
    assert client.get(versions[0]["url"], headers=users["doctor"]).content == original
    assert client.get(versions[0]["url"]).status_code == 401


def test_invalid_correction_keeps_draft_and_original(env):
    client, settings, users, sessions = env
    case = setup_case(env)
    review = client.post(
        f"/api/v1/cases/{case['id']}/reviews",
        headers=users["doctor"],
        json={"decision": "MODIFIED"},
    ).json()
    response = client.post(
        f"/api/v1/reviews/{review['id']}/submit",
        headers=users["doctor"],
        files=correction_file(settings, shape=(1, 2, 3)),
    )
    assert response.status_code == 422
    with sessions() as db:
        assert db.get(Case, uuid.UUID(case["id"])).status == "IN_REVIEW"
        assert db.scalar(select(Review)).submitted_at is None
    assert not list(settings.data_dir.rglob("v1/*.nii.gz"))
    assert (
        client.post(f"/api/v1/reviews/{review['id']}/submit", headers=users["doctor"]).status_code
        == 422
    )


def test_failed_commit_removes_new_file(env, monkeypatch):
    client, settings, users, sessions = env
    case = setup_case(env)
    review = client.post(
        f"/api/v1/cases/{case['id']}/reviews",
        headers=users["doctor"],
        json={"decision": "MODIFIED"},
    ).json()

    def fail_commit(self):
        raise RuntimeError("Simulated commit failure")

    with monkeypatch.context() as patch:
        patch.setattr(sessions.class_, "commit", fail_commit)
        with pytest.raises(RuntimeError, match="Simulated"):
            client.post(
                f"/api/v1/reviews/{review['id']}/submit",
                headers=users["doctor"],
                files=correction_file(settings),
            )
    versions = client.get(f"/api/v1/cases/{case['id']}/annotations", headers=users["doctor"]).json()
    assert [v["version"] for v in versions] == [0]
    assert not list(settings.data_dir.rglob("v1/*.nii.gz"))
    with sessions() as db:
        assert db.get(Review, uuid.UUID(review["id"])).submitted_at is None


def test_upload_limit_and_reviewer_ownership(env, monkeypatch):
    client, settings, users, _ = env
    case = setup_case(env)
    client.post(
        f"/api/v1/cases/{case['id']}/reviews",
        headers=users["doctor"],
        json={"decision": "MODIFIED"},
    )
    response = client.post(
        f"/api/v1/cases/{case['id']}/annotations",
        headers=users["other"],
        files=correction_file(settings),
    )
    assert response.status_code == 409
    monkeypatch.setattr(settings, "max_upload_bytes", 5)
    response = client.post(
        f"/api/v1/cases/{case['id']}/annotations",
        headers=users["doctor"],
        files=correction_file(settings),
    )
    assert response.status_code == 413
    assert not list(settings.data_dir.rglob("v1/*.nii.gz"))


def test_review_cannot_reference_another_case(env):
    client, _, users, sessions = env
    case = setup_case(env)
    with sessions.begin() as db:
        second = Case(
            dataset_id=uuid.UUID(case["dataset_id"]),
            case_uid="second",
            image_path="unused",
            metadata_json={},
        )
        db.add(second)
        db.flush()
        second_id = str(second.id)
    response = client.post(
        f"/api/v1/cases/{second_id}/reviews",
        headers=users["doctor"],
        json={
            "decision": "APPROVED",
            "annotation_version_id": case["current_annotation"]["id"],
        },
    )
    assert response.status_code == 422
