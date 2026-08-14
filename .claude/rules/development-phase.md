# Development-phase rules

- Work is organized into milestones (`docs/development-plan.md`, M0–M15).
  Implement them **in order** — don't start a later milestone's scope
  before an earlier one is done, and don't silently expand a
  milestone's scope beyond what its entry describes.
- Every milestone has three sub-phases, in order:
  1. **Planning** — write a short design/test-plan note before writing
     any implementation code. If this resolves a previously-open
     architecture question (job queue, storage, etc.), record it in
     `docs/decisions/` with date/reasoning/alternatives-rejected.
  2. **Development** — implement the milestone's scope only.
  3. **Testing** — write and run tests for the new functionality, and
     re-run existing tests as a regression check. A milestone isn't
     done until its tests pass, including the regression pass.
- Don't scaffold code for a future milestone "while you're in there."
  If you notice something a later milestone will need, note it in
  `docs/brainstorm.md` or the relevant milestone's entry rather than
  building it early.
- Risk notes in the development plan (M8, M12, M15) are real —
  budget iteration time for those rather than treating estimates as
  fixed-cost.
- Keep the `docs/decisions/` log current: it's the source of truth for
  "why did we choose X" once an implementation detail diverges from or
  fills in a gap left by `docs/project-description.md`.
