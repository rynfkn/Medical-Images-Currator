import json

import nibabel as nib
import numpy as np
from sqlalchemy import func, select
from test_nifti import nifti_input

from app.models import AnnotationVersion, Case
from app.validate_nifti import main


def test_preflight_reports_all_failing_cases_without_writing(env, monkeypatch, capsys):
    _, settings, _, sessions = env
    root = settings.import_dir / "check"
    nifti_input(root)
    image_bytes = (root / "images/case_0001.nii.gz").read_bytes()
    for name in ("fractional", "missing", "shape", "unreadable"):
        (root / f"images/{name}.nii.gz").write_bytes(image_bytes)
    nib.save(nib.Nifti1Image(np.full((4, 5, 6), 1.5), np.eye(4)), root / "labels/fractional.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)), root / "labels/shape.nii.gz")
    (root / "labels/unreadable.nii.gz").write_bytes(b"invalid NIfTI payload")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr("sys.argv", ["validate_nifti", str(root)])
    assert main() == 1
    report = json.loads(capsys.readouterr().out)
    assert report["cases_checked"] == 5
    assert report["valid_cases"] == 1
    assert report["invalid_cases"] == 4
    assert {error["file"] for error in report["errors"]} == {
        "fractional.nii.gz",
        "missing.nii.gz",
        "shape.nii.gz",
        "unreadable.nii.gz",
    }
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before
    assert not list(settings.data_dir.rglob("*"))
    with sessions() as db:
        assert db.scalar(select(func.count()).select_from(Case)) == 0
        assert db.scalar(select(func.count()).select_from(AnnotationVersion)) == 0


def test_preflight_accepts_scaled_labels(env, monkeypatch, capsys):
    _, settings, _, _ = env
    root = settings.import_dir / "check"
    nifti_input(root)
    raw = np.resize(np.array([-32768, 0, 32767], dtype=np.int16), (4, 5, 6))
    mask = nib.Nifti1Image(raw, np.eye(4))
    mask.header.set_slope_inter(3.051804378628731e-05, 1.0000152587890625)
    nib.save(mask, root / "labels/case_0001.nii.gz")
    monkeypatch.setattr("sys.argv", ["validate_nifti", str(root)])
    assert main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["valid_cases"] == 1
    assert report["errors"] == []
