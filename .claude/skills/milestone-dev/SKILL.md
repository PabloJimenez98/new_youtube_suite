---
name: milestone-dev
description: Work the next milestone in docs/development-plan.md following the planning/development/testing workflow. Use when the user asks to start, continue, or work on development for YouTube Suite (e.g. "let's start M0", "next milestone", "continue development").
---

# Milestone-driven development

This project builds strictly milestone-by-milestone from
`docs/development-plan.md`. Never work outside the current milestone's
declared scope.

## Steps

1. **Identify the current milestone.** Read the Progress table near the
   top of `docs/development-plan.md`. The current milestone is the
   first row still `Not started` or `In progress`. If the user names a
   specific milestone out of order, confirm that's intentional before
   proceeding — the plan assumes sequential order for good reasons
   (each milestone's dependencies are listed in its entry).

2. **Planning sub-phase.** Read that milestone's full entry (depends
   on / estimate / planning / development / testing / risk notes).
   Produce a short planning note covering:
   - Any design decisions the milestone's "Planning" bullet calls for
     — if this resolves a previously-open architecture question, write
     a dated entry to `docs/decisions/` (what was decided, why,
     alternatives considered).
   - A concrete test plan for this milestone's "Testing" bullet
     (what will be tested, with what fixtures/data).
   Do not write implementation code in this step.

3. **Development sub-phase.** Implement exactly the milestone's
   "Development" scope — no more. If you notice something a later
   milestone will need, note it in `docs/brainstorm.md` rather than
   building it now.

4. **Testing sub-phase.** Write and run the tests from the plan,
   including a regression pass over prior milestones' tests (not just
   new tests for this milestone). A milestone is not done until this
   passes.

5. **Mark progress.** Update the Progress table row for this milestone
   (`Not started` → `In progress` when starting, → `Done` once tests
   pass). Update `docs/brainstorm.md` if new open questions surfaced.

6. **Stop at the milestone boundary.** Report what shipped and ask
   before starting the next milestone — don't chain multiple milestones
   in one pass without checking in, even if the next one looks quick.
