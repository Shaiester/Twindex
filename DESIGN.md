# Design decisions

Each entry: what I chose, the alternatives, and why.

## 1. Chat collection: anonymous IRC, stdlib only
- **Chose:** a read-only IRC connection over TLS (`irc.chat.twitch.tv:6697`) using the anonymous `justinfanNNNNN` nick, written with `socket` + `ssl`. One connection joins all channels.
- **Alternatives:** EventSub `channel.chat.message` over WebSocket (Twitch's recommended path); TwitchIO 3.
- **Why:**
  - EventSub needs a user access token with `user:read:chat` from a logged-in account, plus OAuth refresh handling. Anonymous IRC needs no credentials, so collection could start right away. That matters because chat history can't be fetched afterwards.
  - IRC tags include `tmi-sent-ts`, a server-side per-message timestamp. The EventSub message event has no per-message timestamp field (only the envelope metadata).
  - TwitchIO 3's quickstart is built around conduits and the broadcaster granting `channel:bot`. We can't get that grant for channels we don't own, and the library would add a dependency to replace about 100 lines.
- **Risk:** Twitch calls IRC "historical" and doesn't document anonymous login (it works today). If it breaks, switching to EventSub WebSocket with a user token only touches `run_once`. `parse_line` would need a second shape.

## 2. Raw storage: daily JSONL, one record per message
- **Chose:** `data/chat/YYYY-MM-DD.jsonl` by UTC date, appended and flushed per line. Records keep `ts, channel, room_id, id, user, type, text, emotes`. USERNOTICEs (subs, raids) are kept with `type = msg-id`, because raids and sub waves are failure-analysis targets.
- **Alternatives:** write straight into SQLite; one file per channel.
- **Why:** append-only text survives crashes (at worst one partial line), is easy to inspect, and keeps collection decoupled from index format, so the index can be rebuilt from scratch at any time.

## 3. Privacy: salted HMAC of user ID at ingest
- **Chose:** `HMAC-SHA256(salt, user_id)[:16]`, with the salt generated once into `.env`. Login names are never written to disk. `data/` and `.env` are gitignored.
- **Alternatives:** plain SHA-256 of the login; dropping users entirely.
- **Why:** Twitch user IDs and logins are public and enumerable, so an unsalted hash can be reversed with a lookup table. We still need a stable per-user token to count unique chatters (an activity feature). 64 bits makes collisions negligible at this scale.
- **Limit:** message text can itself contain usernames (@mentions). Raw logs are never published, and result cards are only shown locally.

## 4. Helix: app access token, urllib, on-disk cache
- **Chose:** a client-credentials app token (Get Videos/Clips/Games/Streams accept app tokens). Responses are cached as JSON files keyed by URL hash with a max-age (users/games forever, lists 6 h). The client sleeps until `Ratelimit-Reset` when `Ratelimit-Remaining` < 5 or on 429.
- **Alternatives:** `requests`/`httpx`; TwitchIO's HTTP client; caching in SQLite.
- **Why:** a few GET endpoints don't justify a dependency. A file cache makes reruns free and reproducible offline.

## 5. Stream poller for category timeline
- **Chose:** poll Get Streams every 5 minutes for all channels in one call, logging `game_name`, stream id and `started_at`.
- **Why:** windows need a game/category filter, and nothing in chat or VOD metadata records what category was live at a given minute. The poll also confirms a channel was live, which helps map windows to VOD offsets.

## 6. Record unit: 30 s tumbling windows aligned to the epoch, >= 3 messages
- **Chose:** windows `[floor(ts/30)*30, +30)` per channel. The id is `channel:start_epoch`. Windows with fewer than 3 messages are dropped.
- **Alternatives:** single messages; sliding/overlapping windows (e.g. 30 s every 10 s); adaptive segmentation on activity changes.
- **Why:** single messages are 1–5 tokens, too short for BM25 or embeddings. Epoch alignment makes ids deterministic, so a full rebuild produces the same ids and **frozen qrels stay valid**. Overlapping windows would handle moments that straddle a boundary better, but they'd triple the corpus and create near-duplicate results that eat the top 10. The cost: a reaction split across two windows shows up as two weaker windows. 1–2 message windows are mostly idle chatter and would make up most of the corpus by count.

## 7. Activity score: robust z-score of log counts, per channel-day
- **Chose:** for messages, unique chatters and emotes, compute `(log1p(x) - median) / (1.4826 * MAD)` within the window's channel and UTC day, then average the three.
- **Alternatives:** raw counts; z-score with mean/std; per-stream baselines; an EWMA "surprise" relative to the last few minutes.
- **Why:** jynxzi gets about 10x the messages of dantes, so raw counts would rank every jynxzi window above every dantes reaction. Log + median/MAD resists the heavy-tailed spikes we're trying to detect, where a mean/std would be dragged up by them. Unique chatters stop one spammer from looking like a crowd. Emotes capture "KEKW"-style reactions.
- **Limit:** a stream that crosses UTC midnight gets two baselines. Per-stream baselines from the poller's `started_at` would fix this.

## 8. FTS5 configuration and query handling
- **Chose:** `tokenize='porter unicode61'`. Query terms are lowercased, stopwords removed (including meta words like "chat", "react", "moment", "when"), each term quoted, and terms joined with **OR**. Ranking uses FTS5's built-in `bm25()`. `fts.rowid = windows.rowid`, and metadata stays in a normal table with an index on `(channel, start)`.
- **Alternatives:** implicit AND (FTS5 default); trigram tokenizer; no stemming.
- **Why:** with AND, a plain-language query like "streamer died and chat laughed" needs every word in one window and returns almost nothing. With OR, BM25's IDF does the weighting. Quoting each term defuses FTS5 syntax (`NOT`, `"`, `*`) from user input. Meta words appear in nearly every query but never in the chat that matches it, so they only add noise. Porter stemming helps paraphrases ("clutched"/"clutch"). It can't help slang or emotes, which is a measured failure mode.

## 9. Atomic rebuild and incremental upsert
- **Chose:** a full rebuild writes `index.db.tmp`, then `os.replace` swaps it in (atomic on the same filesystem). Incremental mode re-windows every chat file from the newest indexed UTC day onward and upserts by id. The dense store follows the same pattern (`.tmp` + replace).
- **Why:** a reader never sees a half-built index. Re-processing whole days is simple and correct: the last window may have been partial, and activity baselines are per channel-day, so earlier windows on that day can change too.
- **Cost:** incremental dense re-embeds the whole newest day (about 3–5 min for a full stream-day on CPU). Hashing doc text to skip unchanged windows would fix this if it starts to matter.

## 10. Result cards
- **Chose:** channel · UTC time · game · activity · score, then id, VOD link at the offset, events, the top repeated messages with counts ("what chat spammed"), messages containing query terms, and "shown N of M messages". Output is capped by a character budget (`--max-chars`, default 6000 ≈ 1500 tokens). `--json` gives the same fields for agents.
- **Why:** the repeated-message summary compresses a 200-message window into its signal, and the budget keeps an agent's context bounded.

## 11. Dense retrieval: bge-small-en-v1.5 + exact FAISS search
- **Chose:** embed each window's **unique messages sorted by frequency**, truncated at bge's 512 tokens. Queries get bge's retrieval instruction prefix. Search uses `IndexFlatIP` on normalized vectors (cosine). Filters (channel, time) run as SQL, and the allowed rows pass to FAISS as an `IDSelectorBatch`. Vectors live beside their SQLite file (`index.db` → `index.npz`).
- **Alternatives:** truncating raw text in time order (cuts off the end of busy windows); chunking and max-pooling (several times the compute); HNSW/IVF (approximate); larger models.
- **Why:** a busy window has more than 512 tokens of mostly repeats. Deduping by frequency keeps both the dominant reaction and its variety. Exact search over tens of thousands of 384-d vectors takes milliseconds, so approximate search would only add recall loss and tuning knobs. Filtering inside FAISS gives exactly k results that satisfy the filter. Over-fetching and post-filtering can return fewer.
- **Environment:** the current `transformers` needs a newer torch than the global one (2.5.1, pinned by torchvision), so the project uses its own `.venv` instead of upgrading global packages.

## 12. Hybrid: reciprocal rank fusion, k = 60, depth 100
- **Why RRF:** BM25 scores are unbounded and cosine scores sit in a narrow band. RRF uses only ranks, so no score calibration is needed. k = 60 is the value from Cormack et al. (2009) and the common default. I fixed it rather than tuning it, which leaves one less knob to overfit the dev split. Depth 100 lets a document that one method ranks around #50 still contribute.

## 13. Chat-activity boost: additive rerank over retrieved candidates
- **Chose:** take the method's top 100, min-max scale relevance and activity within that set, and score `relevance + λ·activity`. λ is the only parameter tuned, by grid search on the **dev** split.
- **Alternatives:** multiplicative boost; adding activity as a third RRF list; a learned ranker.
- **Why:** reranking only retrieved candidates means a loud but off-topic window can't enter the results. It can only move up among windows that already match. Additive with scaled terms makes λ interpretable ("how many relevance-ranges is one activity-range worth"). A learned ranker would need far more labels than one person can produce.

## 14. Eval set construction
- **Frozen corpus.** `evalset sample` records a cutoff timestamp. Every eval run rebuilds a separate index from raw chat strictly before it (`data/eval/index.db`), so later collection can't shift windows, activity baselines or the candidate set.
- **Sampling:** equal share per channel, split evenly over within-channel activity terciles. Otherwise the busiest channel and the quiet majority of windows would dominate.
- **Queries:** I draft them in-session (no API calls in project code), one per sampled window. They mix exact-term, paraphrase, event and slang types, and avoid reusing the window's own wording. Otherwise BM25 would win by construction.
- **Pooling:** the top 5 from each of the 6 configs plus the source window. Documents outside the pool count as non-relevant (standard TREC practice). `judged@10` is reported per method to show how much that assumption is being leaned on.
- **Labels:** 0/1/2 drafted by the LLM (me). The human then labels **blind** (the LLM's label is never shown, which avoids anchoring): every LLM 2, plus a seeded 20% of the 0s and 1s. The final label is the human one where it exists. Agreement is reported as exact %, binary (rel ≥ 1) %, Cohen's κ, quadratic-weighted κ, per-LLM-label accuracy and the confusion matrix. Because the verified subset over-samples 2s, the per-label rows are the honest view.
- **Split:** 30/70 dev/test by query, seeded and stratified by query type. Qrels are in TREC format. `freeze` stores a SHA-256 of queries + qrels, and eval/tune refuse to run if they've changed.

## 15. Metrics
- **Chose:** nDCG@10 with graded gain `2^rel − 1`, Recall@20 and MRR at relevance ≥ 1, judged@10, p50/p95 latency (per query, after a warmup that loads the model), and build time and size for FTS and dense. Also a 95% bootstrap CI on nDCG@10 and a paired randomization test against BM25 (about 100 test queries is small enough that differences need a significance check).
- **Implemented myself (about 25 lines, unit-tested against hand calculations)** rather than `pytrec_eval` (painful to build on Windows) or `ranx` (pulls in numba). Every formula is visible and defensible. Runs are also written in TREC format, so anyone can cross-check with `trec_eval`.
- **Recall caveat:** recall is relative to the pool, not the true number of relevant windows.

## 16. Clip-alignment sanity check
- A clip spans `[VOD created_at + vod_offset, + duration]`. A window "aligns" if it overlaps that span extended by 30 s, since chat reacts after the stream delay. The check is limited to the period the fetched clips cover. It reports the base rate, the rate for the top 2% windows by activity (and the lift), and the rate for each method's top-1 result.
- **Why:** it's independent evidence. Clips are viewers voting on which moments mattered, so if the activity score means anything, loud windows should be clipped far more often than chance.

## 17. CI
- GitHub Actions runs pytest with only `faiss-cpu`, `numpy` and `pytest`. The embedding model and matplotlib are imported lazily and never touched by tests, which keeps CI under a minute instead of installing torch.

## 18. Pilot instead of waiting a week
- **Chose:** run the whole eval on about 1.5 h of chat (42 queries, 14 per channel) instead of 150 queries after a week of collection.
- **Why:** it exercises every stage end to end and produces defensible-if-wide numbers now. Every result is labeled "pilot". Rerunning the same commands on more data replaces them with no code changes.

## 19. First human labeling pass rejected
- **What happened:** the first pass had 144 of 154 labels as "1" and no "2"s, including obvious matches labeled 0. κ was −0.01. The labeler confirmed they had misread the scale.
- **Chose:** keep that file (`eval/human_labels.rejected.tsv`), clear the freeze, add a worked-example guide plus a scale reminder on every prompt, and relabel the same 154 pairs. Nothing else changed (same corpus, queries, pool and seed).
- **Why:** labels that don't vary with content can't measure anything. Rejecting them is a data-quality call, not tuning: the methods hadn't been scored yet.

## 20. Boost weight: tuned value recorded, prior default evaluated
- **What happened:** the dev grid search (12 queries) picked λ = 0. The boost hurt at every λ > 0 on these mostly topical queries.
- **Chose:** keep the default chosen before any labels existed (λ = 0.3) for the test run, and record the tuning result in `results/tuning.json`.
- **Why:** λ = 0 makes "+boost" identical to the base method, which would hide what the boost does. Evaluating a pre-set value on test is legitimate because it wasn't chosen by looking at test. The honest conclusion is "the boost doesn't help topical queries", and the clip check suggests it does favor big moments.

## 21. Helix clip fetching: one query per day, always with ended_at
- **Bug 1:** without `ended_at`, Get Clips ends the range one week after `started_at`, so "the last 30 days" silently returned days 30–23.
- **Bug 2:** a range query returns about 1,000 clips sorted by views, so popular old clips crowded out the night we indexed (3 jynxzi clips instead of 122).
- **Fix:** one query per day, with bounds rounded to the hour so the cached URL is stable within an hour.

## 22. Known bug, deliberately not fixed in the pilot
- `STOPWORDS` contains "stream", so the query "W stream" was searched as just "w" (`results/failure_analysis.md`).
- I found this after scoring the test set. Fixing it now would be changing the method based on test results, so it gets fixed before the next eval round on new data.

## 23. MCP server: three read-only tools over the existing code
- **Chose:** the official `mcp` SDK (v2, `MCPServer`) over stdio, with `search`, `get_window` and `describe`. Each tool is a thin wrapper around the CLI's `card()` / `describe()` and `search.search()`, so no retrieval logic is duplicated. All tools are annotated `read_only_hint`.
- **Defaults:** MCP `search` defaults to `hybrid`, the best method in the eval, because an agent pays the one-off model load once per session. The CLI keeps `bm25` for instant answers. Output is bounded for agent contexts: cards stop once their JSON exceeds `max_chars`, and `get_window` caps at 200 messages with "shown N of M".
- **Return format:** plain dicts, which the v2 SDK sends as JSON text content. Every client can read that. Structured output would add a schema nobody consumes yet.
- **Alternatives:** HTTP transport (no need, it's local-only); a separate service layer (would only add an abstraction over two functions).
- **Tested** end to end over stdio with the SDK's own client: tool listing and annotations, all three tools, the budget, and the error path for bad ids and methods.

## 24. Pre-publication privacy audit and one redaction
- **Checked before the first commit:**
  - every file to be committed was scanned for the actual `.env` values (none found) and for token, hash, email and local-path patterns
  - the files were cross-checked against all 1,011 usernames mentioned in the raw chat
  - `.gitignore` was hardened (`.env.*`, `*.db`, `*.npz`)
- **One finding:** query p39 ("alen drew", an exact-term query) contained the start of a chatter's handle.
- **Chose:** redact it in the published `eval/queries.jsonl` and `results/per_query.csv`, and keep the real text in the gitignored `data/eval/query_overrides.json`. `read_queries()` overlays it for searching, and results only ever write `public_query`. The freeze checksum was recomputed, with a note in `meta.json`. A full rerun gave identical quality metrics.
- **Why not delete the query:** its labels are part of the scored test set. Removing it would change the results, while redacting the text changes nothing measurable.
- **Lesson for the next round:** exact-term queries must not use chatter names. That's now a rule for query drafting.
