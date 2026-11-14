# Decision: Frontend UX simplification + explainability layer (out-of-plan pass)

**Date:** 2026-11-08
**Status:** Implemented (2026-11-08). Outside the milestone plan, like the hardware pass
(0016); M12's live phase 2 and M15 are unchanged.

## Context

- **Manage was all noise.** `/manage` rendered every episode (63) at once, each with its video, all
  8 tools with their Run buttons and every result panel open (chapters, relevance, shorts, show
  notes, CTR). A disabled Run only said why in a `title` tooltip.
- **Nothing explained the data science.** The project is a data-science master's thesis (TFM).
  The UI should show *which data process* is behind every result: extraction, windows, features,
  model, aggregation, metric. Software-engineering detail can stay out.
- **Dashboard mixed real and fake data.** It still had the M1-era dummy tiles (average predicted CTR,
  top hook clip, relevance sparkline), and Library cards had a ▶ placeholder instead of a frame.
- **Decided with the owner (2026-11-08):**
  - **Language:** the UI and the explanations stay in **English**.
  - **Manage:** a compact list plus one page per episode, with tabs.
  - **Process:** an out-of-plan pass, with this record and a regression run.
  - **Extras, all four:** a *How it works* page, a "suggested next step" guide, model cards with a
    glossary, and a Dashboard/Library clean-up.
  - **Flowcharts (owner's request):** every data-science process gets a flow diagram; the custom
    models show both inference and training.

**Frontend only.** There are no backend or API changes; everything uses endpoints that already exist.

## Decisions

### 1. One content source: `src/app/explain/`

- **`tool-docs.ts`:** one `ToolDoc` per tool (8) plus the platform nodes `upload`, `market`,
  `analytics`, `reporting` and `m15`. Each has:
  - a one-liner, inputs and outputs
  - numbered data-process steps, each tagged with glossary terms
  - a model card: name, kind, where it runs
  - a metric note, with `means` and `doesNotMean`. Relative scores say explicitly that they are
    not a CTR % or a retention %.
  - limitations, dependencies, and a `source` naming the decision records the content is taken
    from
- **Flows:** every doc has an inference `flow`. The custom models (chapters, relevance, CTR, hook)
  also have a `training` flow:
  - **Chapters:** real. VidChapters-7M → filter → split → units → frozen encoder → BiLSTM vs
    TextTiling → F1@5s → bundle.
  - **Relevance, CTR, hook:** marked `planned`, with dashed nodes, since that is M15's training.
- **`SYSTEM_FLOW`:** the whole platform, from the episode through the tools to the market and
  analytics feedback loop into the M15 models. Every node links to its doc.
- **`glossary.ts`:** 33 terms, defined as they apply on this platform (VAD, diarization, TF-IDF,
  BiLSTM, F1@5s, zero-shot, GGUF, pseudo-label, relative score…).
- **`pipeline.ts`:** pure tool-state logic, moved out of the old `VideoRow`, so every view agrees:
  - `toolStates` gives each tool one of `done | running | queued | failed | ready | blocked`, with a
    reason and the tool that unblocks it
  - `nextSuggested` puts a failed tool first, then the first never-run tool in dependency order,
    and suggests nothing while a prerequisite is in flight
  - `displayName` drops the extension and the trailing `[videoId]`; `formatDuration` formats
    durations

### 2. Flowchart component: `FlowDiagram`

- Inline SVG, top to bottom, with no library.
- **Layout:** each node's layer is its longest path from a source; `minLayer` is a hint for late
  data sources.
- **Long edges:** an edge that skips a layer goes through a side lane and is drawn dashed, so it
  never passes behind a node and reads as "feeds into" the wrong box.
- **Node kinds:** data, transformation, model, rule/scoring, stored result and planned. Each differs
  in stroke style as well as colour, with a legend.
- Linked nodes are keyboard-focusable buttons.

### 3. Explanation popup: `ExplainDialog` + `InfoButton` (ⓘ) + `Term`

- **One dialog in the shell, opened through the `Explain` service.** Its sections:
  1. lead
  2. input / output
  3. data flow, with an *Inference / Training* toggle for the custom models
  4. step by step
  5. model card
  6. how to read the score
  7. limitations
- **Behaviour:** Escape, the close button and a backdrop click close it; Tab stays inside it.
- **`Term`:** a glossary chip with its definition in a hover/focus tooltip.

### 4. Manage: list + episode page

- **`/manage`:**
  - a search box
  - one row per episode: clean name, duration, date, 8 status dots, `n/8`, and "Next: X" (or
    "Working…")
  - upload in a modal
  - still the 3 s jobs poll (0006)
- **`/manage/:id` (`EpisodeDetail` → `EpisodeWorkspace`):**
  - **Top:** a sticky player, so relevance and shorts can still seek, plus the **pipeline
    stepper**: 8 nodes, and a "Suggested next step" card with its Run and ⓘ.
    - Shorts opens its tab instead of running, because it has options.
    - A tool blocked on its own input (CTR candidates) opens its tab.
  - **Six tabs:** Transcript & speakers · Chapters · Relevance · Shorts & hooks · Show notes ·
    Titles & thumbnails.
  - **A shared tool bar in each tab:** name, ⓘ, model-kind chip, status badge, the blocked reason
    with a link to the unblocking tab, and Run / Re-run.
  - **Empty tabs** explain the tool and what it is waiting for.
  - The page polls `GET /uploads/{id}/jobs` only.
- **Inactive tabs stay mounted (`[hidden]`), not destroyed.** This differs from the plan, which
  said only the active tab renders: destroying them would lose unsaved show-notes edits and the
  CTR panel's candidate count, which gates CTR's Run. One episode's panels are cheap to keep.
- **Delete** moved from the list to the episode page.
- **Optimistic jobs are deduplicated** against the polled list by id.
- **`VideoRow` is removed.** Its spec is ported to `episode-workspace.spec.ts`, with every former
  case kept.
- **The blocked text for Hook** is now "Needs shorts" (it was "Run Shorts first"), consistent with
  the other reasons.

### 5. *How it works* page (`/how-it-works`, in the nav)

- The `SYSTEM_FLOW` diagram, where clicking a node opens its popup.
- A **model status table**: today (pretrained / trained / heuristic v0 / LLM), next step (M15) and
  training label.
- The full glossary.

### 6. Dashboard and Library

- **Dashboard:**
  - The dummy tiles and `dashboard/dummy-stats.ts` are removed.
  - New **Pipeline coverage**: per tool, how many episodes have results, computed from real jobs.
  - Recent activity shows clean names, the job outcome, and links to the episode.
- **Library:**
  - Cards show a real frame through the existing token-authenticated stream (0005), with
    `#t=min(60, duration/2)` and `preload="metadata"`.
  - The frame loads only once the card is near the viewport (IntersectionObserver).
  - The card links to `/manage/:id`.
- **Analytics** keeps its own preview data; it is not redesigned here.

### 7. Performance

- `manage/:id`, `how-it-works` and `analytics` are lazy routes (`loadComponent`).
- The initial bundle is 437 kB, under the 500 kB budget. Without lazy routes it was 513 kB.

## Alternatives rejected

- **An accordion on `/manage`** (collapsed episodes, tabs inside): still one long page, and the
  owner preferred a page per episode.
- **Bilingual UI (Angular i18n):** more work and twice the copy to maintain; the owner chose English.
- **A diagram library** (Mermaid, dagre, D3): another dependency and bundle weight for a dozen small
  DAGs, and harder to theme with the design tokens. The layered layout is about 100 lines and tested.
- **Content in the backend:** the explanations are static copy tied to the UI; keeping them typed in
  the frontend lets the tests check them (every tool has a doc, every flow is a DAG, every glossary
  key exists).

## Test plan (run 2026-11-08)

- **New specs:**
  - `pipeline.spec.ts`: states, blockers, `nextSuggested`, display helpers
  - `explain-content.spec.ts`:
    - every tool has a doc, and docs are complete
    - every flow is a DAG and lays out without overlaps
    - long edges are routed clear of nodes
    - glossary keys exist
    - custom models have a training flow
  - `explain-dialog.spec.ts`
  - `how-it-works.spec.ts`
  - `episode-detail.spec.ts`
  - `episode-workspace.spec.ts`: the ported `VideoRow` cases, plus tabs, the stepper and ⓘ
  - `manage.spec.ts`: rewritten for the list
- **Updated specs:** Dashboard (no dummy tiles, coverage), VideoCard/Library (router, clean name,
  frame URL), Shell (5 nav links).
- **Results:**
  - **Frontend:** 36 files, 276 tests passed. `ng build` has no warnings.
  - **Backend regression:** 672 passed, 10 deselected. There are no backend changes.
- **Manual check:** in Chrome against the dev stack and the real 63 episodes: Manage list, episode
  page, tabs, popups (inference and training), How it works, Dashboard, Library. It found and
  fixed three issues:
  - all tab panels showed at once: `display: flex` overrode `[hidden]`
  - a horizontal scrollbar in the popup
  - edges that skip a layer passed behind nodes

## Consequences

- **Content source:** the popup text is the content the owner will defend in the TFM. When a tool
  changes, its `ToolDoc` must change with it. The `source` field names the decision record to
  re-check.
- **Review (closed 2026-11-14):** the owner reviewed and approved the popup copy.
- **M15:** when M15 trains a model, its `training.planned` flips to `false` and its flow is replaced
  with the real pipeline.
