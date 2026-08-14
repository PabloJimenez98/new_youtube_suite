# Decision: M0 Backend Skeleton — Storage, Stack, Auth

**Date:** 2026-08-14
**Status:** Decided (M0 planning sub-phase)

## Context

M0 stands up the backend skeleton: FastAPI app, Postgres schema for
`users`/`uploads`/`jobs`, single-user auth, and a storage abstraction
for uploaded video files. `docs/project-description.md` §3 left
several infrastructure choices explicitly open at the end of Phase 3:
job queue technology, blob/object storage technology, the ML training
stack, the scheduler mechanism, and where models are served. M0's
planning bullet resolves the blob-storage question; job queue stays
open for M2, and the ML/scheduler/model-serving questions stay open
for their respective later milestones. M0 also forces a few
implementation-level choices (dependency manager, ORM, auth mechanism,
enum-column representation) that project-description.md didn't specify
at the architecture level, since they only matter once code is being
written.

## Decision

**Blob storage:** local filesystem, behind a `StorageBackend`
interface (`save`/`get`/`delete`) so a future S3-compatible swap
doesn't touch calling code. The local implementation writes under a
configurable base path (`settings.storage_base_path`, defaulting to
`<repo-root>/uploads/`, which `.gitignore` already excludes). Storage
keys are generated (`{uuid4}_{original_filename}`), not derived
directly from the client-supplied filename, to avoid collisions and
path-traversal. The `uploads.storage_path` column stores this
backend-relative key rather than an absolute path or URL.

**Dependency manager:** plain `pip` with `requirements.txt` (runtime)
and `requirements-dev.txt` (test/lint tooling).

**ORM / migrations:** SQLAlchemy 2.0 in async mode (`asyncpg` driver)
+ Alembic for migrations.

**Auth:** OAuth2 password flow with JWT access tokens
(`fastapi.security.OAuth2PasswordBearer`, tokens signed via
`python-jose`, ~12h expiry, HS256 with a server-side secret).
Passwords hashed with `passlib[bcrypt]`. No refresh tokens or session
revocation infrastructure, and no `/auth/register` endpoint — the
single user row is created by a one-off seed command
(`backend/app/cli.py`), not a public signup route.

**Enum-shaped columns** (`uploads.status`, `jobs.tool_type`,
`jobs.status`): plain `TEXT` with a `CHECK` constraint, not a native
Postgres `ENUM` type.

## Alternatives considered

- **S3-compatible storage (e.g. MinIO) instead of local filesystem** —
  rejected for v1: adds a service to run locally and a dependency, for
  no benefit on a single-machine deployment. The `StorageBackend`
  interface exists specifically so this can be revisited later without
  touching callers.
- **Poetry / uv instead of pip** — rejected: `.claude/settings.json`
  already allowlists bare `pytest`/`ruff`/`black`/`mypy`/`alembic`/
  `uvicorn`, i.e. assumes a normal venv+pip workflow. Neither tool adds
  clear value for a solo-operator v1 at this stage.
- **Sync SQLAlchemy instead of async** — rejected: FastAPI's request
  handling and M2's future async job worker both favor an async DB
  layer; no reason to start sync and migrate later.
- **Session-based auth / refresh tokens** — rejected as overbuilt for
  a single-operator tool with no multi-tenancy; a stateless JWT with a
  generous-but-finite expiry is the boring, sufficient choice.
- **Native Postgres `ENUM` type for status/tool_type columns** —
  rejected: adding a value later requires `ALTER TYPE ... ADD VALUE`
  with transactional restrictions, whereas `TEXT + CHECK` is a trivial
  migration. `jobs.tool_type` already lists 8 values and will keep
  growing as M3–M14 land.

## Consequences

- Swapping to S3-compatible storage later is a new `StorageBackend`
  implementation plus a config change — no data migration, since
  `storage_path` is already backend-relative.
- Adding a new `jobs.tool_type` or status value later (all but
  certain, given M3–M14) is a plain `ALTER TABLE ... DROP CONSTRAINT /
  ADD CONSTRAINT` migration, not a Postgres enum-alteration dance.
- Because there's no `/auth/register` endpoint, provisioning the
  (single) user account is an operational step (`python -m app.cli
  create-user`) rather than a UI flow — acceptable given v1 is
  explicitly not multi-tenant.
