import pytest
from conftest import create_dataset, ingest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


def dicom_slice(path, study, series, instance, position=None):
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.StudyInstanceUID = study
    ds.SeriesInstanceUID = series
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = CTImageStorage
    ds.Modality = "CT"
    ds.Rows, ds.Columns = 4, 5
    ds.InstanceNumber = instance
    ds.PixelSpacing = [0.8, 0.8]
    ds.SliceThickness = 2
    if position is not None:
        # Sagittal series: sorting by Z alone would be wrong.
        ds.ImageOrientationPatient = [0, 1, 0, 0, 0, 1]
        ds.ImagePositionPatient = [position, 0, 0]
    ds.save_as(path, enforce_file_format=True)


@pytest.mark.parametrize("geometry", [True, False])
def test_dicom_groups_sorts_and_serves(env, geometry):
    client, settings, users, _ = env
    root = settings.import_dir / "dicom"
    root.mkdir()
    study, series = generate_uid(), generate_uid()
    dicom_slice(root / "a.dcm", study, series, 2, 1 if geometry else None)
    dicom_slice(root / "b.dcm", study, series, 1, 3 if geometry else None)
    dicom_slice(root / "c.dcm", study, generate_uid(), 1)
    dataset_id = create_dataset(client, users["admin"], "DICOM")
    response = ingest(client, users["admin"], dataset_id, root, "DICOM")
    assert response.status_code == 201, response.text
    assert response.json()["cases_ingested"] == 2
    cases = client.get(f"/api/v1/datasets/{dataset_id}/cases", headers=users["doctor"]).json()
    case = next(case for case in cases if case["case_uid"] == series)
    assert [s["InstanceNumber"] for s in case["metadata"]["slices"]] == (
        [2, 1] if geometry else [1, 2]
    )
    assert case["annotation_url"] is None
    assert len(case["image_urls"]) == 2
    response = client.get(case["image_urls"][0], headers=users["doctor"])
    assert response.status_code == 200
    assert response.content == (root / ("a.dcm" if geometry else "b.dcm")).read_bytes()
    assert (
        client.post(
            f"/api/v1/cases/{case['id']}/annotations",
            headers=users["doctor"],
            files={"file": ("mask.nii.gz", b"invalid")},
        ).status_code
        == 422
    )


def test_invalid_dicom(env):
    client, settings, users, _ = env
    root = settings.import_dir / "dicom"
    root.mkdir()
    (root / "invalid.dcm").write_text("Not DICOM")
    dataset_id = create_dataset(client, users["admin"], "DICOM")
    assert ingest(client, users["admin"], dataset_id, root, "DICOM").status_code == 422
