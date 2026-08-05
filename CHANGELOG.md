# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Benchmark methodology — the judge is measured, and it is not clean

The CodeMemEval accuracies were graded by an LLM judge that nothing had ever
validated. It is now measured, against 24 hand-written plausible-but-false
answers covering all 17 facts, plus 24 correct-answer controls:

| Judge | False accept | 95% CI | False reject |
|---|:-:|:-:|:-:|
| **`gpt-5.4-mini`** — grades the published numbers | **1/24 (4.2%)** | [0.7%, 20.2%] | 0/24 |
| `claude-sonnet-5` — different model family | 0/24 (0.0%) | [0.0%, 13.8%] | 0/24 |

The accepted answer was on `loc_protos`: gold says generated stubs are
**committed** and never hand-edited, the candidate said they are **gitignored
and regenerated per developer**. Right paths, inverted policy, graded correct.

At a 4.2% false-accept rate over 27 questions, about one graded answer is
expected to be wrong-but-accepted — the entire margin between the 96.3% and
92.6% rows. Both remain **upper bounds**, now for a measured reason rather than
an unexamined one. README and REPORT say so.

- **The adversarial set grew from 10 to 24** and now covers all 17 facts rather
  than 7. This is the finding that justifies the module's own thesis: at n = 10
  *both* judges scored a flawless 0/10 and the 95% upper bound was still 27.8%.
  The defect appeared only after widening the set, which is exactly the small-n
  error the module was written to name.
- **A cross-family judge was added as a check on the "generator, reader and
  judge share a lineage" objection.** It is *stricter*, not more lenient: the
  two judges agree on 23 of 24 adversarial cases and on all 24 controls, so on
  this evidence shared lineage does not appear to be inflating the score.

### Benchmark methodology — the ranking suite could not select a ranking weight

`benchmark/tune_ranking.py` exists to choose the ranking constants. It could
not: as shipped in 2.0.0 it returned an identical 80.0% top-1, with identical
per-category results, for **every** recency weight in [0.25, 1.5]. It scored the
2.0.1 supersession regression as 100% and let it ship. Library code is
unchanged; this is the instrument, not the scorer.

- **Every supersession case gave the newer memory a lexical score at least as
  good as the stale one**, so all three passed at any weight. Real superseded
  facts usually look the opposite way: the stale row repeats the query's own
  wording while its replacement describes the change. Two cases with that shape
  are added — including the exact one `memory_stress_bench.py` caught and this
  suite missed.

- **`age-spread/car` retrieved nothing at all.** It asked "what car did I buy"
  against "Bought a Honda Civic." — no token matches ("car" is in neither
  memory, and "buy" is not "Bought"), so FTS returned an empty set and the case
  was scored as a ranking failure. It held `age_spread` at 50% for every weight
  in the grid while measuring nothing. The query now shares vocabulary with both
  memories, and `evaluate` reports an empty result set as a **broken fixture**
  rather than a scorer failure, so the next dead case is visible instead of
  quietly deflating a category forever.

- **Known defects are separated from regressions.** `relevance-wins/port` is a
  real, characterised defect — the row that literally answers the question loses
  to a fresh row that does not, because age both suppresses recency *and* decays
  confidence, so a 2x lexical advantage cannot recover. It fails identically at
  0.5 and 0.75, predating and surviving the 2.0.1 fix. It is now reported as a
  known defect with its decomposition instead of being averaged into the
  headline, so the top-1 number moves when something actually breaks.

- **`--explain`** prints the score decomposition for each failure. Reconstructing
  it by hand is what made the 2.0.1 bug expensive to find. Where a scorer emits
  no decomposition (`legacy`), it says so rather than printing zeros.

- **`--sweep`** scores a list of weights and states plainly whether the suite
  can separate them. Run it before quoting any tuned constant.

Repaired, the suite does its job:

| recency weight | top-1 | regressions |
|---|:-:|---|
| 0.25 (what the old cross-validation preferred) | 83.3% | 2 supersession |
| 0.50 (shipped in 2.0.0) | **75.0%** | **2 supersession** |
| 0.75 (shipped in 2.0.1) | **91.7%** | none |

Cross-validation now independently selects **0.75** in four of five folds — the
value 2.0.1 arrived at by measuring entirely different suites. The fifth fold
holds out supersession itself, picks 0.25 from what remains, and then scores
60% on the held-out cases, which is the tradeoff stated explicitly.

## [2.0.1] — 2026-08-04

### Fixed

- **A superseded fact could outrank the fact that replaced it.** 2.0.0 shipped
  `rank_weight_recency = 0.5`, and at that weight a stale memory won whenever it
  matched one more query token than its replacement: lexical ratio-to-best
  handed the older row a 0.53 edge while recency returned only 0.42 after
  weighting. Two memories 947 and 795 days old, asked "frontend framework team"
  — the row saying the team *migrated away* from React lost to the row saying it
  uses React. The weight is now **0.75**; the inversion flips below 0.63.

  This is the knowledge-update case the project is built around, and 2.0.0
  regressed it against 1.5.0. It was caught by `memory_stress_bench.py`, whose
  result 2.0.0 published as 23/23 while the shipped code scored 22/23.

  | | 2.0.0 (0.5) | 2.0.1 (0.75) |
  |---|:-:|:-:|
  | memory stress suite | 22/23 | **23/23** |
  | pooled LongMemEval R@5 | 79.1% | 79.1% |
  | CodeMemEval R@5, questions as written | 100% | 100% |
  | CodeMemEval R@5, questions restated | 75.0% | **79.2%** |
  | temporal ranking suite top-1 | 80.0% | 80.0% |

  Nothing regressed, including the two cases the temporal suite already fails.
  That suite scores an identical 80.0% with identical per-category results for
  every weight in [0.25, 1.5], so it cannot select this value and the choice
  rests on the two suites that can see the difference.

- **`RankingWeights` carried a second, independent default.** `from_settings`
  reads the config value, but a bare `RankingWeights()` used the dataclass
  default, so changing only `config.py` left every caller that skips settings
  scoring against the old weights. Both are now asserted equal by a test.

- **`ranking.py`'s module docstring described an algorithm the module does not
  implement** — Reciprocal Rank Fusion for the lexical signal and a bare
  hyperbolic map for recency. The code has used ratio-to-best and a
  span-weighted blend since the rewrite. The docstring is the first thing
  anyone reads when a ranking result looks wrong, and it sent two separate
  investigations down the wrong path before the decomposition in `_explain`
  showed what was actually happening.

- Removed a duplicated `tier` assignment in `fuse()`.

## [2.0.0] — 2026-08-04

Architecture review remediation. **This is a major version because four defaults
change in ways that break existing deployments** — an upgrade that does not read
the Changed section will find a Postgres-backed server silently running on
SQLite and no longer reachable off localhost. Several published benchmark
numbers are also withdrawn or restated; see Removed and Benchmark methodology.

### PostgreSQL backend brought to parity

Every memory-model fix below had originally landed on SQLite only, leaving the
PostgreSQL backend carrying all of the same defects — plus one that was worse.
It now passes the same conformance suite as SQLite.

- **`memories_updated` was an *unconditional* `BEFORE UPDATE` trigger** setting
  `NEW.updated_at = now()`. SQLite's equivalent at least had a
  `WHEN old.updated_at = new.updated_at` guard; this one fired on every UPDATE,
  so any write reset the column the time-decay ranking reads. Dropped
  (migration 3).
- **Schema bootstrap and a migration runner.** `sql/schema.sql` had to be
  applied by hand with psql, and `init()` ran `information_schema` probes whose
  failure was indistinguishable from success. `opendb_core/storage/_pg_migrations.py`
  now creates the schema on first run and records every step in a
  `schema_version` table, each step committing atomically with its version row.
  Verified on both a fresh database and one created from the legacy
  `schema.sql`.
- **Non-destructive supersede** via a `memory_revisions` table, plus
  `memory_history()`. The old path set `superseded_id = id` — a self-reference,
  not provenance — and destroyed the prior content.
- **Conflict detection is lexical.** It used to take the 20 most recently
  updated same-type memories with no relevance filter at all, so the true
  candidate was almost never considered and 20 arbitrary rows were eligible for
  destruction. Now FTS-ranked, with the same dual thresholds and distinct-event
  guard as SQLite.
- **Query semantics aligned.** `plainto_tsquery` ANDs every lexeme; SQLite ORs
  its terms. The same query against the same data returned different result
  sets per backend. `build_pg_or_tsquery()` produces OR semantics and reduces
  caller input to alphanumerics so no query text can reach `to_tsquery`'s
  parser as syntax.
- **Identifier expansion**, weighted `'D'` in the combined tsvector so a
  derived match cannot outrank a literal one.
- Recall is read-only; `total` is the match count; pagination reaches past the
  first window; `memory_type` is constrained by a CHECK.
- **Workspace isolation.** `workspace_id` existed only on `memories`, so two
  workspaces sharing one database saw each other's documents. Added to `files`
  (everything else reaches a file through it), and the checksum uniqueness
  index is now per workspace.
- **`idx_pages_trgm` dropped** — a GIN trigram index over full page text that
  no query read, paid for on every insert.

### Operability, lifecycle and resource bounds

- **`opendb doctor`** — executable invariant checks: schema version, durability
  pragmas, FTS/base-table agreement, orphaned rows, stuck ingestions, tokenizer
  skew, recall latency, and PostgreSQL index usage. Exits non-zero on failure so
  it can gate a deploy.
- **Tokenizer fingerprinting.** Text is tokenized at index time and again at
  query time; if the rules change in between, queries silently stop matching.
  The fingerprint is recorded in `opendb_meta` and `doctor` reports skew.
- **Deletions and renames converge.** The watcher had no `on_deleted` or
  `on_moved` handler at all, so a removed file stayed searchable forever and a
  rename produced a duplicate.
- **Partially-written files are no longer indexed.** The debounce was
  leading-edge — the first event was ingested immediately and follow-ups were
  dropped — so a file still being written was indexed truncated and never
  corrected. The consumer now waits for the file to quiesce.
- **Parser resource bounds** at the single dispatch point: zip expansion
  ratio/size (DOCX/PPTX/XLSX are zips), a hard Pillow decompression-bomb
  threshold, and a page cap.
- **Uploads stream.** The endpoint did `await file.read()` and *then* checked
  the size, so the whole body was resident before being rejected.
- **Request-size and rate limiting** middleware, and asyncpg now has an acquire
  timeout, `statement_timeout` and `idle_in_transaction_session_timeout` — pool
  exhaustion and runaway queries both used to present as an unbounded hang.
- **Japanese and Korean are segmented.** Both were routed to jieba, a Chinese
  segmenter, which returned kana and Hangul runs essentially whole — a Japanese
  sentence was indexed as one enormous token. They now fall back to character
  bigrams.
- **Retrieved memories are fenced and labelled as untrusted data.** Memories are
  agent-writable and replay into every later session, which makes them a durable
  prompt-injection channel; stored content cannot forge the fence.

### Fixed

- **Reading a memory no longer resets its time-decay clock.** The
  `memories_updated` trigger fired `WHEN old.updated_at = new.updated_at`, so any
  UPDATE that did not name the column silently set it to `now()`. Recall's
  inline reinforcement did exactly that, and `updated_at` is what the decay
  ranking reads — a 400-day-old fact that had been read once outranked the fresh
  fact that replaced it. The trigger is dropped (migration 4) and `updated_at` is
  now set explicitly by the paths that genuinely modify a fact.
- **Supersede no longer destroys data.** It was an in-place
  `UPDATE memories SET content = ?` chosen by an English regex plus a flat 0.3
  Jaccard threshold; three separate dated bug-fix records about one component
  collapsed into one, with `superseded_id` left pointing at a row that no longer
  existed. Now: prior versions are archived to `memory_revisions` and readable
  via `memory_history()`; differently-dated records are never treated as
  versions of one fact; and without an explicit update phrase supersede requires
  a near-duplicate (0.65) rather than 0.3. The `sim >= 0.05` "most recent
  same-type memory" fallback is removed outright.
- **`total` and pagination in `memory_recall`.** `total` was overwritten with the
  post-gate candidate-window size — it reported 60 on a 200-memory store and
  swung to 1 as the corpus grew. It is now the real match count, alongside
  `ranked` and `truncated`. Any `offset` at or beyond 60 previously returned
  nothing regardless of how many memories matched.
- **FSRS decay is no longer a no-op.** Recall set `confidence = 1.0` on every
  hit, so the curve could only ever demote memories nobody read. Recall is now
  read-only; reinforcement is the explicit `reinforce_memories()`, and it
  preserves `updated_at`.
- **camelCase identifiers are searchable by their parts.** `CreateInvoice` was a
  single unicode61 token, so "invoice", "create invoice" and "where do we create
  invoices" all returned zero results — on the project's own flagship example.
  Both FTS tables gain a down-weighted `expansion` column.
- **Write-path correctness.** `mark_file_failed`, `delete_file` and
  `log_eval_capture` ran unlocked and called `commit()` on the shared
  connection, committing whatever transaction another coroutine had open and
  turning its later `rollback()` into a no-op. `persist_ingestion` caught only
  `IntegrityError`, so any other exception left an open transaction on the
  shared connection and every subsequent write in the process failed. All writes
  now go through a single `write_txn()` primitive.
- **`FILEDB_AUTH_API_KEY` is enforced.** `ApiKeyMiddleware` was implemented,
  documented and unit-tested — and never mounted on the app.
- **Workspace confinement.** `_resolve_workspace` computed `relative_to()`,
  swallowed the `ValueError` and returned the path anyway; the only containment
  check in the codebase was a no-op. It now raises `WorkspaceViolation`, and
  rejects symlink escapes.
- **ReDoS in grep.** The per-file deadline was checked once every 5000 lines,
  which cannot interrupt a single exponential `search()`. Patterns with nested
  quantifiers are now rejected, the deadline is checked per line, and the
  subject handed to the engine is length-bounded.
- **SQL injection surface in the PostgreSQL backend.** `sort_field`/`sort_dir`
  were interpolated into `ORDER BY` unvalidated; they are now allowlisted, as
  they already were on SQLite.

### Changed

- **Default backend is now `sqlite`** (was `postgres`). The documented three-line
  install pointed at a PostgreSQL server the user was never told to run.
- **Default bind address is `127.0.0.1`** (was `0.0.0.0`) and **CORS defaults to
  no origins** (was `["*"]`). With auth unenforced, the previous defaults made
  the workspace reachable from the local network and scriptable by any web page.
- **`FILEDB_VISION_ENABLED` defaults to `false`.** Enabling it POSTs the bytes of
  every indexed image to OpenRouter; it used to be on by default and activated
  off an ambient `OPENROUTER_API_KEY`.
- **`memory_type` is validated** against `episodic|semantic|procedural`. It
  silently flips storage semantics, so a typo previously meant duplicates or
  data loss.
- **Schema migrations are versioned.** `PRAGMA user_version` replaces
  `PRAGMA table_info` sniffing wrapped in `except: pass`; each step commits
  atomically with its version bump. The code-symbol backfill is migration 3
  rather than an unbounded join re-run on every `init()`.
- Readers and writers use separate SQLite connections, so a reader can no longer
  observe another coroutine's uncommitted rows.

### Benchmark methodology — CodeMemEval

`benchmark/codemem_hard.py` answers, with numbers, the four objections that sink
CodeMemEval as previously published. None of it needs an API key: `gen_codemem`
imports the OpenAI client at module scope, so the facts are read with
`ast.literal_eval` instead — a methodology check that requires a paid credential
is a methodology check nobody runs.

- **Every score is published with its interval.** 96.3% is 26/27, whose 95%
  Wilson interval is [81.7%, 99.3%] — 17.6 points wide. It overlaps the cheap
  reader's 92.6% [76.6%, 97.9%] across almost its whole width, so the benchmark
  cannot actually distinguish the two readers, and the README and REPORT no
  longer imply that it can. Per-category results rest on n = 3–6 and are now
  reported as counts rather than percentages. Reaching ±2pp needs n = 457.
- **The questions restated the evidence.** Across the 24 questions that have
  evidence, a mean 53.4% of question tokens appear verbatim in the fact being
  asked about, so a lexical retriever won part of the set by construction.
  `PARAPHRASE_QUESTIONS` now covers **every** question with a 0.0%-overlap
  restatement, and `--emit-paraphrase` writes them as a dataset with identical
  haystacks and gold sessions, so the retrieval harness scores the hard split
  directly:

  | | R@1 | R@5 |
  |---|:-:|:-:|
  | questions as written | 100% (24/24) | 100% (24/24) |
  | restated | 45.8% (11/24) | 75.0% (18/24) |

  One question returns **zero** results. This is the honest boundary of a
  pure-lexical retriever, and it was previously averaged away.
- **The judge is unvalidated, and now says so.** Both accuracy figures come from
  an LLM judge sharing a model family with the reader it grades, and nothing had
  measured how often it accepts a wrong answer. `--run-judge` scores 10
  hand-written plausible-but-false answers, 7 correct answers restated in other
  words, and the 17 gold answers verbatim. Both halves are required: a judge that
  answers INCORRECT to everything scores a flawless 0% false-accept rate, and
  only the correct-answer controls expose it. It refuses to report a rate if any
  call errored, since `judge_answer` scores a failed call as a rejection — an
  expired credential would otherwise print a perfect result. **The rate has not
  yet been measured**, so the E2E figures stand as upper bounds.
- **Baselines.** Random 5-of-18 is a 27.6% chance floor and the oracle is 100%,
  so a headline can be read against something.
- Two categories the flat question/answer shape could not express: `temporal`
  (the same fact asked about now and then — a store that overwrites on update
  cannot answer the second) and `staleness` (scored on whether the system flags
  a memory a later commit made false, not merely on whether it answers).

### Removed

- **The "100% R@5" retrieval claim.** It was measured on
  `benchmark/longmemeval_oracle.json`, where all 500 questions satisfy
  `set(haystack_session_ids) == set(answer_session_ids)` with a mean of 1.896
  sessions and zero distractors — R@5 is arithmetically guaranteed for 497/500.
  `longmemeval_bench.py --pooled` indexes every question's sessions into one
  882-session corpus and reports the discriminative figure: **R@5 = 79.1%**,
  with `single-session-preference` at 30.0%. The default mode now warns when a
  dataset contains no distractors. Latency claims are restated with their corpus
  size (7.5 ms median at 882 sessions).

## [1.5.0] — 2026-04-11

### Added
- **Runtime workspace management.** An agent (or a human) can now list, add,
  switch, and remove workspaces at runtime without restarting the server.
  Already-opened workspaces switch in sub-millisecond time — the storage-layer
  registry keeps each workspace's SQLite connection warm, so switching is just
  a pointer flip plus a settings patch under an `asyncio.Lock`.
- New global workspace registry persisted at `~/.opendb/workspaces.json`
  (overridable via the `FILEDB_STATE_DIR` environment variable). Each entry
  records `id`, `name`, `root`, `backend`, `created_at`, and `last_used_at`.
- Five new REST endpoints under `/workspaces`:
  - `GET /workspaces` — list all registered workspaces (active one first)
  - `POST /workspaces` — register (and optionally switch to) a new workspace
  - `GET /workspaces/active` — return the currently active workspace
  - `PUT /workspaces/active` — switch by id or root path
  - `DELETE /workspaces/{id}` — unregister a workspace (files untouched;
    use `?force=true` to remove the currently-active one)
- Five new MCP tools exposed by the server:
  - `opendb_list_workspaces`
  - `opendb_current_workspace`
  - `opendb_use_workspace` — accepts either a workspace id or a root path
  - `opendb_add_workspace`
  - `opendb_remove_workspace`
- `opendb_info` / `GET /info` now include a `workspace` block with the active
  workspace's identity (id, name, root, backend, last-used time), so agents
  can answer "which workspace am I in?" in a single call.
- New CLI subcommand group: `opendb workspace list | add | use | current | remove`.

### Changed
- `opendb_core/workspace.py` — factored the settings-patching block and the
  parser registration block out of `Workspace.init()` into reusable helpers
  (`apply_workspace_config`, `_ensure_parsers_registered`) so the runtime
  switch path and the embedded-mode `Workspace` class apply workspace config
  identically.
- `opendb serve` (SQLite mode) now auto-registers the startup workspace into
  the global registry during lifespan setup, so runtime `/workspaces`
  endpoints can see and switch to it from the first request.
- README: workspace management section, updated tool count (7 → 12),
  updated REST endpoint table, new `FILEDB_STATE_DIR` configuration entry.

## [1.4.0] — 2025-10

### Added
- LongMemEval benchmark improvements — **93.6% E2E accuracy**, #3 on the
  leaderboard, beating MemMachine, Vectorize, Emergence AI, Supermemory,
  and Zep.
- Comprehensive benchmarks and improved memory conflict detection.
- GitHub Actions workflows for CI tests and PyPI publishing.

### Changed
- Skip episodic conflict detection during memory store for better
  LongMemEval accuracy.
- Added `skip-existing` flag to the PyPI publish workflow.
- Added return type annotations across the codebase (52% → 100%).

### Fixed
- FastAPI `response_model` for the glob and read endpoints.

### Removed
- Obsolete benchmark experiment results.

## [1.3.0]

### Added
- Custom while-loop agent example in the README.
- `CONTRIBUTING.md` with contributor guidelines.
- Quality improvements: PostgreSQL CJK search, expanded test coverage,
  deduplication, authentication middleware, and more.

### Changed
- Rebranded from museDB to **openDB** with a new logo, banner, and the
  "AI-native database" tagline.
- Rewrote docs: logo brand guide, architecture doc with a Mermaid diagram,
  removed dead tool-definitions link.
- Refactored: removed the legacy `app/` package, narrowed exceptions, split
  the storage layer, added tests.
- License switched from AGPL-3.0 → Apache 2.0 → **MIT**.
- Fixed PyPI package name to `open-db`.

[Unreleased]: https://github.com/wuwangzhang1216/openDB/compare/v1.5.0...HEAD
[1.5.0]: https://github.com/wuwangzhang1216/openDB/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/wuwangzhang1216/openDB/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/wuwangzhang1216/openDB/releases/tag/v1.3.0
