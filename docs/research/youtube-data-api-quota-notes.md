# YouTube Data API — Quota Notes for the Market-Analysis Pipeline

Quick reference research (verified Aug 2026), not a deep literature review —
supports the market-analysis pipeline design agreed in `docs/decisions/v1-scope.md`.

## Quota facts
- **Default daily quota**: 10,000 units/day per project.
- **`videos.list`**: 1 unit per call, and accepts **up to 50 video IDs per
  request** (batched) — so fetching stats/metadata for 500K videos/day is
  quota-feasible if IDs are batched in chunks of 50. This should be the
  backbone of the daily pull.
- **`search.list`**: 100 units per call — roughly **100 calls/day max** if
  it were the only thing consuming quota. Expensive; should be used
  sparingly (e.g. periodically discovering new channels to add to the
  curated list), not as the main daily mechanism.
- Write operations (create/update/delete) cost ~50 units — not relevant
  here since the pipeline is read-only.

## Implication for pipeline design
- Maintain the curated channel list as stored state (channel IDs), not
  rediscovered via `search.list` each day.
- Daily job: batch-fetch video stats via `videos.list` (50 IDs/call) for
  all known videos across the curated channels — cheap and scales well
  past what a single-person project needs.
- **Free, quota-free discovery alternative**: every YouTube channel
  publishes a public RSS/Atom feed at
  `https://www.youtube.com/feeds/videos.xml?channel_id=UCxxxxxxxxxxxxx`
  with no API key and no quota cost — usable to detect new uploads from
  curated channels without spending `search.list` budget. Limitation:
  only surfaces the **last ~15 videos** per channel, so it's a
  "what's new" signal, not a backfill mechanism.
- `search.list` budget can be reserved for periodic (not daily)
  discovery of new candidate channels to add to the curated list.

## Sources
- [YouTube API Quota: 100 Searches Burn 10,000 Units (2026) — SocialCrawl](https://www.socialcrawl.dev/blog/youtube-data-api-2026)
- [YouTube's API Quota Is 10,000 Units/Day: Tracking 100K Videos Without Hitting It — DEV Community](https://dev.to/siyabuilt/youtubes-api-quota-is-10000-unitsday-heres-how-i-track-100k-videos-without-hitting-it-5d8h)
- [Videos: list — YouTube Data API — Google for Developers](https://developers.google.com/youtube/v3/docs/videos/list)
- [YouTube Data API v3 videos.list ID Parameter Limit — TechNetExperts](https://www.technetexperts.com/youtube-api-videos-list-id-limit/)
- [Determine Quota Cost — Google for Developers](https://developers.google.com/youtube/v3/determine_quota_cost)
- [Convert YouTube Channel to RSS Feed](https://linkmyprompt.com/convert-youtube-channel-to-rss-feed/)
