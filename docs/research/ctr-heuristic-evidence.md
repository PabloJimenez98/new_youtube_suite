# Evidence for a v0 title/thumbnail CTR heuristic (M13)

**Date:** 2026-11-08 · research-analyst pass, requested in M13 planning
(`docs/decisions/0019-ctr-heuristic-v0.md`).

## Headline

- **No peer-reviewed or arXiv study was found that measures the CTR effect** of any feature in
  the M13 set.
- **The evidence that does exist is weak:**
  - mostly vendor analyses;
  - often measured on **views** or "breakout" status rather than CTR;
  - often without a non-breakout baseline, which makes it survivorship-biased;
  - confounded by niche and channel size.
- **No public (title, thumbnail, CTR) dataset exists.** The realistic CTR source is our own
  channel's Reporting API data (M12, `analytics_reach_snapshots`).
- **Design consequence:** the v0 weights are priors. The ranking is relative within one
  upload, which removes video-level confounders. That is a design advantage.

## Per feature

| Feature | Claimed effect | Strength | Source |
|---|---|---|---|
| Title length | Conflicting. Median 58 chars for both breakout and normal videos (2,622 titles). Shorter titles win for long videos (120,703 videos). Another blog says 70–100. No agreed sweet spot; "40–70" appears only in blogs. | weak | overseeros; subsub |
| Title hard cap / truncation | 100-char cap. Search truncation reported at roughly 50–70 chars depending on device. | moderate (cap) / weak (truncation) | Influencer Marketing Hub |
| Numbers in title | Vendor claim that they help. | weak/anecdotal | subsub |
| `?`, `!`, brackets, ALL-CAPS, "you" | No CTR evidence. Upper-case count appears as a views feature with no reported direction. | none | arXiv 1611.00687 (2016) |
| Face count | Multi-face covers did best vs single-face (observational). Covers: 71 % have faces, 46 % single, 25 % two-plus (descriptive, views/day). | weak | 1of10 via SEJ (2025); Net Influencer |
| Face presence | 69 % of 500 breakouts have faces (no baseline). Faces vs none similar overall; lift only above a subscriber threshold, and niche-dependent. | weak, conflicting | vidIQ (2026); 1of10 |
| Face size | No effect evidence. "About a third of the frame" is a schematic. | none | vidIQ |
| Text coverage | No effect evidence. 72 % of breakouts have overlay text, median 5 words (descriptive). | none | vidIQ |
| Brightness/contrast/saturation | High contrast in 56 % of breakouts and 46 % of the top 50 (descriptive). Nothing on saturation. | none/weak | vidIQ |

**Podcasts/interviews:**
- No study isolates two-host or interview thumbnails.
- Face effects are niche-dependent.
- "Real faces beat audiograms 2–3×" is anecdotal and confounded.

**Pitfalls:**
- views ≠ CTR
- survivorship (breakout-only samples)
- niche and channel-size confounding
- YouTube's own Test & Compare reports watch time, not CTR

## Suggested shapes (all priors unless noted)

- **Title length:** zero above 100 (platform cap). Full credit across ~40–70; ramps toward ~15
  and ~100. Consider tapering after ~50–60 because of truncation.
- **Word count:** redundant with length; tiebreaker at most.
- **Digits, `?`, `!`, brackets, caps, "you":** weight 0 in v0; log them for later
  calibration.
- **Face count:** 0 low; 1–2 high; 3+ flat (no evidence that it declines).
- **Largest face area:** rises to a plateau around 10–35 % of the frame.
- **Text coverage:** 0 neutral; mild penalty above a high threshold.
- **Brightness/contrast/saturation:** keep out of the v0 score; log them.

## Sources

- vidIQ, *YouTube Thumbnail Study: 500 Breakout Thumbnails* (2026): https://vidiq.com/research/youtube-thumbnail-study/
- Search Engine Journal, *Do Faces Help YouTube Thumbnails?* (1of10 report, 2025 data): https://searchenginejournal.com/do-faces-help-youtube-thumbnails
- arXiv 1611.00687, *Engagement dynamics and sensitivity analysis of YouTube videos* (2016): https://arxiv.org/abs/1611.00687
- Net Influencer, thumbnail study of 100k channels (undated): https://www.netinfluencer.com/youtube-thumbnail-study-of-100k-channels-finds-shocked-faces-on-just-1-in-10-covers/
- overseeros, *Best YouTube Title Length: 2,622 Titles Analyzed* (undated): https://www.overseeros.com/blog/best-youtube-title-length-study
- subsub, *YouTube Titles That Get Clicks: 120,703 Videos* (undated): https://www.subsub.io/blog/youtube-title-formulas-that-work
- Influencer Marketing Hub, character limits (undated): https://influencermarketinghub.com/youtube-character-limits/ ; Test & Compare: https://influencermarketinghub.com/youtube-test-compare/
- Datasets (no CTR): YouTubeSet, https://gts.ai/dataset-download/youtubeset ; `l3afai/youtube-thumbnails` on Hugging Face.
- Unverified leads, not located: a PACIS 2017 study of ~400k thumbnails; a claimed inverted-U
  visual-complexity result on 22,958 thumbnails.
