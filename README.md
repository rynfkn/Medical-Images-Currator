# Medical dataset curation backend

FastAPI, SQLAlchemy 2, PostgreSQL, and local files. Supports NIfTI images and masks,
PNG/JPEG with COCO annotations, and basic single-frame DICOM series. No frontend.

## Start with Docker Compose

Run from `backend/`:

```bash
cp .env.example .env
python3 -c 'import secrets; print(secrets.token_hex(32))'
# Put the generated value in JWT_SECRET in .env.
mkdir -p data/import data/datasets
  docker compose up --build
```

The backend waits for PostgreSQL and runs `alembic upgrade head` before starting.
Open <http://localhost:8000/docs>. Only the backend and PostgreSQL run. Files persist
in `./data`; the database persists in the `postgres_data` volume. Both published
ports bind to localhost. `API_PORT` and `POSTGRES_PORT` can override their defaults.

Use a URL-safe `POSTGRES_PASSWORD` (for example, a generated hex string); update
`DATABASE_URL` to match when running Python locally. To apply migrations manually:

```bash
docker compose exec backend alembic upgrade head
```

## Local Python installation

Requires Python 3.12 or newer and PostgreSQL. From `backend/`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
cp .env.example .env
# Set JWT_SECRET, DATABASE_URL, DATA_DIR, and IMPORT_DIR for your machine.
docker compose up -d postgres
alembic upgrade head
uvicorn app.main:app --reload
```

`.env` is read from the working directory. `DATA_DIR` defaults to `data/datasets`;
`IMPORT_DIR` defaults to `data/import`. Uploads default to a 1 GiB limit configured
by `MAX_UPLOAD_BYTES`. Ingestion runs synchronously in the request; large datasets
need adequate request timeouts and memory for validation.

## Create users and log in

There is no public registration or default account. The CLI prompts for a password:

```bash
docker compose exec backend python -m app.cli admin --full-name 'Dataset Admin' --role ADMIN
docker compose exec backend python -m app.cli doctor --full-name 'Doctor' --role REVIEWER
```

For a local installation, use `python -m app.cli ...` with the same arguments.
Administrators can create and ingest datasets. All authenticated users can read
datasets, access files, upload corrections, and review cases. Only a review's
author can submit it.

Login uses OAuth2 form fields (also supported by the **Authorize** button in `/docs`):

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d 'username=admin' --data-urlencode 'password=YOUR_PASSWORD'
```

Set `TOKEN` to the returned `access_token`. Subsequent requests need
`Authorization: Bearer $TOKEN`, including file downloads. `GET /api/v1/auth/me`
returns the current user. JWTs expire after `ACCESS_TOKEN_MINUTES` (default 480).

## Ingest datasets

Copy input datasets under `backend/data/import/`. The API accepts **server-side**
paths inside `IMPORT_DIR`, not client filesystem paths. With Compose, use
`/app/data/import/...`. Managed copies are kept under `DATA_DIR`; source files are
never edited. Traversal and symlinks pointing outside the allowed root are rejected.

Create a dataset first, then use its returned ID:

```bash
curl http://localhost:8000/api/v1/datasets \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"name":"Kidney", "dimension":"3D", "image_format":"NIFTI", "annotation_format":"NIFTI"}'

DATASET_ID=REPLACE_WITH_RETURNED_ID
curl "http://localhost:8000/api/v1/datasets/$DATASET_ID/ingest" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"type":"NIFTI", "path":"/app/data/import/kidney"}'
```

Supported combinations:

| Input | dimension | image_format | annotation_format | Ingest type |
| --- | --- | --- | --- | --- |
| NIfTI | 3D | NIFTI | NIFTI | NIFTI |
| COCO + PNG/JPEG | 2D | PNG or JPEG | COCO | COCO |
| DICOM | 3D | DICOM | NONE | DICOM |

**NIfTI:** put matched `.nii.gz` or `.nii` filenames in `images/` and `labels/`,
such as `images/case_0001.nii.gz` and `labels/case_0001.nii.gz`. Files remain NIfTI.
Validation checks readable 3D payloads, identical image/mask shapes, and finite
integer mask labels. Floating-point storage uses an absolute tolerance of `1e-6`
from the nearest integer. For integer storage with a nonzero absolute scaling
step below 1, the tolerance is the larger of `1e-6` and half the scaling step plus
`1e-6`. This allows integer classes encoded with different quantization steps,
including scaled int16 labels decoding to `1.0000152587890625`, without a fixed
`1e-4` cap. Scaling steps of 1 or larger retain the strict `1e-6` tolerance;
half-integer values are always rejected because their nearest class is ambiguous.
Relative tolerance is zero. This assumes the input represents categorical integer
labels: the header alone cannot establish the intended meaning of voxel values.
Values outside the encoding tolerance, NaN, and infinity are rejected. Metadata
label IDs are rounded to the nearest integer; source and stored annotation files
remain byte-for-byte unchanged.
Metadata includes shape, voxel spacing, affine, and labels.
Shape agreement does not establish that a mask is anatomically aligned; spatial
registration is outside this MVP.

Check every NIfTI pair before importing a large dataset:

```bash
docker compose exec backend python -m app.validate_nifti /app/data/import/RCC-AID
```

This read-only command uses the same file validation as ingestion and reports all
failing cases together, including filenames. It exits with status 0 when all pairs
pass, or 1 when validation fails. It does not create records or modify files.
It does not check existing database case IDs, disk capacity, or anatomical alignment;
ingestion still validates the files again and may fail if those conditions change.

**COCO:** put PNG/JPEG files in `images/` and the full COCO file at
`annotations/instances.json`. Create a 2D/PNG/COCO (or 2D/JPEG/COCO) dataset and ingest:

```json
{"type":"COCO", "path":"/app/data/import/coco_dataset"}
```

Mixed PNG/JPEG datasets are supported; case details report the actual image format.
Dimensions must agree with the image file. Image, category, and annotation IDs must
be unique integers; references and segmentation structures are validated. Polygon,
uncompressed RLE, and compressed RLE are supported. The full original JSON is copied
byte-for-byte to `original/`, while each case gets a v0 JSON with `image`,
`annotations`, and `categories`. Annotation fields, including bbox/area/iscrowd,
are preserved without recalculation. An empty annotation list is allowed.

**DICOM:** place `.dcm` files anywhere below the import directory. Create a
3D/DICOM/NONE dataset and ingest:

```json
{"type":"DICOM", "path":"/app/data/import/dicom_dataset"}
```

Files are grouped by `SeriesInstanceUID`, sorted by position projected onto the
slice normal when consistent orientation/position data exists, otherwise by
`InstanceNumber`. Case metadata includes the required study/series/instance IDs,
modality, description, geometry, and per-slice metadata. `image_urls` lists protected
slice downloads in that order. No pixel decoding, conversion, DICOM segmentation,
or multi-frame support is provided.

An ingest request is all-or-nothing. Duplicate case identifiers within a dataset
are rejected. Additional distinct cases can be ingested later. A failed request
rolls back its rows and removes the files it created, leaving existing data intact.

## Cases, corrections, and reviews

List cases at `GET /api/v1/datasets/{id}/cases?status=PENDING&limit=50&offset=0`.
`GET /api/v1/cases/{id}` returns format, dimension, metadata, status, the current
annotation, and protected file URLs. Reading a case does not change its status.
Starting a review or uploading a correction sets `IN_REVIEW`.

Use a doctor's token for these examples. Create a review with one decision:
`APPROVED`, `MODIFIED`, `NEEDS_CORRECTION`, or `REJECTED`.

```bash
CASE_ID=REPLACE_WITH_CASE_ID
curl "http://localhost:8000/api/v1/cases/$CASE_ID/reviews" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"decision":"MODIFIED", "comment":"Tumor boundary corrected."}'
```

Creation saves a draft. There is one open review per case; another reviewer cannot
upload a correction during that review. `annotation_version_id` is optional and
defaults to the latest version. Corrections uploaded by the author update their
open draft's annotation reference. Submitted reviews are immutable.

For a modified review, upload the corrected mask with submission to save the file,
new annotation version, review submission, and case status in one transaction:

```bash
REVIEW_ID=REPLACE_WITH_REVIEW_ID
curl -X POST "http://localhost:8000/api/v1/reviews/$REVIEW_ID/submit" \
  -H "Authorization: Bearer $TOKEN" -F 'file=@corrected_segmentation.nii.gz'
```

Alternatively, upload the correction first, then create/submit the review:

```bash
curl "http://localhost:8000/api/v1/cases/$CASE_ID/annotations" \
  -H "Authorization: Bearer $TOKEN" -F 'file=@corrected_segmentation.nii.gz'

curl -X POST "http://localhost:8000/api/v1/reviews/$REVIEW_ID/submit" \
  -H "Authorization: Bearer $TOKEN"
```

The standalone upload is a committed version even if no review is submitted.
Versions are `v0` (original), `v1`, `v2`, etc., with parent and author references.
No endpoint overwrites or deletes an annotation. PostgreSQL row locks serialize
concurrent changes to a case. `MODIFIED` requires the latest correction to have
been uploaded by the submitting reviewer; stale review references are rejected.
Other decisions submit without a file. DICOM cases can receive general reviews
but cannot receive corrected annotations or a `MODIFIED` decision.

For COCO corrections, upload a JSON file with this per-case structure:

```json
{
  "image": {"id": 1, "file_name": "image_0001.png", "width": 512, "height": 512},
  "annotations": [{
    "id": 1, "image_id": 1, "category_id": 1,
    "segmentation": [[10, 10, 50, 10, 50, 50, 10, 50]],
    "bbox": [10, 10, 40, 40], "area": 1600, "iscrowd": 0
  }],
  "categories": [{"id": 1, "name": "tumor"}]
}
```

The image ID/dimensions and category ID set must match the existing annotation.
List versions at `GET /api/v1/cases/{id}/annotations`, get the current version at
`GET /api/v1/cases/{id}/annotations/current`, and list reviews at
`GET /api/v1/cases/{id}/reviews`. File routes accept record IDs or slice indices,
never user-supplied filesystem paths.

## Tests

```bash
pip install -e '.[test]'
pytest -q
ruff check .
ruff format --check .
```

The default suite uses isolated SQLite databases and generated images. It covers
NIfTI/COCO validation, original preservation, review decisions, file/auth boundaries,
DICOM ordering, and rollback after invalid uploads or simulated commit failure.

To also test real PostgreSQL locking, use a **disposable** database:

```bash
TEST_DATABASE_URL=postgresql+psycopg://USER:PASSWORD@localhost:5432/curator_test pytest -q
```

The integration fixture creates and drops the application's tables. Never point it
at a database containing real data. Migrations can be checked separately with
`alembic upgrade head` and `alembic check` against a configured PostgreSQL database.

Filesystem cleanup covers normal request failures, including database commit
failures. A process kill or machine failure between file creation and database
commit may leave an unreferenced file; this MVP does not implement crash recovery.
Back up the database and `data/datasets` together.
