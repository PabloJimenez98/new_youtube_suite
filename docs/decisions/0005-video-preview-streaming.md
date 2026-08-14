# Decision: Video Preview Streaming — Query-Param Auth, Manual Range Support

**Date:** 2026-08-14
**Status:** Decided

## Context

The Manage page's video list needed a real `<video>` preview/player
per the video listed there (replacing a decorative placeholder box).
Two problems stood in the way:

1. The backend never exposed a way to read back an uploaded file's
   bytes — `StorageBackend`/`LocalStorageBackend` only had
   `save`/`get`/`delete`, and no router endpoint used `get()` at all.
2. Browsers don't let a `<video src="...">` request carry a custom
   `Authorization` header — that's set by our `authInterceptor` for
   Angular `HttpClient` calls, but a `<video>` element's request goes
   straight from the browser's native network stack, bypassing
   Angular entirely.

## Decision

**New endpoint**: `GET /uploads/{id}/file`, streaming the file's bytes
with a guessed `Content-Type` (from the upload's original filename).

**Auth**: a new dependency, `get_current_user_for_media`, accepts the
JWT either in the `Authorization` header (like every other endpoint)
*or* as a `?token=` query parameter — query-param only for this one
media-serving route, not adopted anywhere else. This is a deliberate,
narrow trade-off: the token becomes visible in the URL (server access
logs, browser history), which is weaker than header-only auth. Accepted
for the same reasoning `docs/decisions/0003-frontend-skeleton.md`
already applied to storing the JWT in `localStorage` — single-operator
v1 tool, not a multi-tenant/public-facing product. `oauth2_scheme` was
changed to `auto_error=False` so both `get_current_user` (header-only,
raises its own 401) and `get_current_user_for_media` (header-or-query)
can share it without the header-only path firing before the query
fallback gets a chance.

**Range requests, implemented by hand**: the pinned Starlette version
(0.38.6)'s `FileResponse` does not implement HTTP Range requests — a
`Range` header is silently ignored and the full file is always sent
with `200`. Range support is what lets a `<video>` element seek without
downloading the whole file first, so `stream_upload_file` parses the
`Range` header itself and returns a `StreamingResponse` reading just
the requested byte span (206 Partial Content, `Content-Range` header),
falling back to streaming the whole file (200) when no `Range` header
is present. `LocalStorageBackend.path_for(key)` was added (local-only,
not part of the abstract `StorageBackend` interface) to get a real
filesystem path for this — the existing `get()` returns a bare
`BinaryIO`, which doesn't support the seek-by-byte-offset streaming
this needs as cleanly.

## Alternatives considered

- **Fetch the video as a Blob via `HttpClient` (proper header auth),
  then `URL.createObjectURL`** — rejected: forces the entire file to
  download before any playback can start, which is bad for anything
  but very short clips. The query-param approach preserves real
  streaming/seeking.
- **Short-lived signed URLs** (mint a one-time token per video, expire
  quickly) — rejected as overbuilt for a single-operator internal
  tool; revisit if this app ever needs to be exposed beyond one
  trusted operator.
- **Upgrade Starlette for native Range support** — not attempted: this
  version's `FileResponse` doesn't have it regardless of pinning
  choices this project controls without a wider dependency bump, and
  the hand-rolled version is small and fully under test.

## Consequences

- Any URL containing `?token=...` should be treated as sensitive —
  don't log full request URLs anywhere those logs might be shared.
- If a token expires (12h, per M0), an in-progress video preview will
  start failing with 401s until the user logs in again; acceptable for
  now, no refresh-token flow exists to smooth this over.
- Future milestones adding other media (e.g. exported short clips,
  M8) can reuse the same `get_current_user_for_media` dependency and
  range-streaming helper rather than re-solving this.
