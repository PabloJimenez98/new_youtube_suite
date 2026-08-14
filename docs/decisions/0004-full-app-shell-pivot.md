# Decision: Pivot to a Full App Shell Before M1's PR

**Date:** 2026-08-14
**Status:** Decided

## Context

M0 (backend skeleton) is merged. M1 (frontend skeleton: login, an
upload form, a bare uploads list) is implemented but its PR was not
yet open. Before opening any PRs, the project owner asked to
deliberately jump ahead of the milestone-by-milestone order that
`CLAUDE.md` and `docs/development-plan.md` otherwise mandate:

- **Backend**: add REST endpoints to trigger and inspect jobs for all
  8 v1 tools (`transcribe`, `diarize`, `chapters`, `relevance`,
  `shorts`, `shownotes`, `ctr`, `hook`), plus the remaining CRUD the
  frontend needs (upload detail, delete) — even though none of these
  tools do real work until their owning milestones (M3–M14) land.
- **Frontend**: build every screen the finished v1 app will use —
  a library/landing page, a KPI dashboard, a market-analytics preview,
  and a video-management page — using real data where it already
  exists (uploads, jobs) and explicitly-labeled dummy/preview data
  everywhere the real pipeline doesn't exist yet.
- **Visual design**: a full vaporwave-minimalist-Y2K design system,
  replacing the ad-hoc hardcoded-color styling M1 shipped with.

`CLAUDE.md`'s working rules say not to build a milestone's scope
beyond what it specifies "without discussing it first." This document
*is* that discussion, and the owner's explicit, detailed instruction
is the authorization to proceed. It's recorded here so the plan and
decision log stay truthful about what actually happened and why,
rather than this scope expansion being invisible in git history.

## Decision

Proceed with the full expansion described above, on the existing
`m1-frontend-skeleton` branch (no PR was open yet to conflict with).

**Backend**: no new tables — the `jobs` table from M0 already has the
exact shape needed (`tool_type` CHECK-constrained to the 8 tools,
`status` CHECK-constrained to `queued/running/succeeded/failed`).
New endpoints only create `queued` job rows; nothing processes them
(that's still M2's job — the queue/worker — and each tool's own
milestone after that). `GET/DELETE /uploads/{id}` and
`POST/GET /uploads/{id}/jobs` and `GET /jobs` round out what the new
frontend screens need.

**Frontend**: four authenticated routes (`/library`, `/dashboard`,
`/analytics`, `/manage`) under a new `Shell` component, replacing the
single combined upload+list page M1 shipped. Real data is used
wherever it exists (the uploads list, job statuses); everywhere else
(predicted CTR, hook scores, market-channel trends, retention/
analytics) renders from dedicated, clearly-commented dummy-data
constant files with visible "preview"/"coming soon" UI treatment —
never presented as if it were real.

## Consequences

- **`docs/development-plan.md` is not rewritten** — M0 and M1 stay
  marked `Done`, because what they originally specified was in fact
  delivered. This document is the record of what was added *on top*.
- **Future milestones' frontend "Development" bullets change in
  practice**: M3 (transcript), M5 (chapters), M6 (show notes), M7
  (relevance), M8 (shorts), M13 (CTR), M14 (hook), and M10+
  (market-analysis) will now mean *wiring real data into an
  already-existing screen* (Manage's per-tool status, Dashboard's
  preview tiles, the Analytics page) rather than building that screen
  from scratch. Each of those milestones should update the relevant
  dummy-data constant file and remove its "preview" badge once real
  data is available — not build a parallel new view.
  Also flagged in `docs/brainstorm.md` so it isn't lost.
- **Test debt is not being deferred**: new backend endpoints and new
  frontend components/services still get real tests as part of this
  same change, matching the rigor M0/M1 established — this pivot
  changes *what* gets built when, not the project's testing standard.
- The `jobs` created by the new "run tool" buttons will sit at
  `status='queued'` forever until M2 (the async job pipeline) exists
  to actually process them. This is expected, not a bug.
