from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://opendb:opendb@localhost:5432/opendb"
    db_pool_min: int = 5
    db_pool_max: int = 20
    db_command_timeout: float = 60.0
    # Waiting for a free connection must fail, not hang: without an acquire
    # timeout, pool exhaustion looks identical to a dead server.
    db_acquire_timeout: float = 10.0
    # Server-side backstop so PostgreSQL aborts a runaway query itself.
    db_statement_timeout_ms: int = 60_000

    # Request limits (server mode). A body larger than this is rejected before
    # it is buffered.
    max_request_bytes: int = 128 * 1024 * 1024
    # Simple per-client request cap; 0 disables.
    rate_limit_per_minute: int = 600

    # File storage
    file_storage_path: Path = Path("./data")
    max_file_size: int = 100 * 1024 * 1024  # 100 MB

    # OCR
    ocr_enabled: bool = True
    ocr_languages: str = "eng+chi_sim+chi_tra"

    # Vision (LLM-based image description — replaces Tesseract for images).
    #
    # OFF by default: enabling it POSTs the bytes of every indexed image to a
    # third-party inference API (OpenRouter). It used to default to True and
    # activate off an ambient OPENROUTER_API_KEY, so a developer who had that
    # variable set for unrelated reasons silently uploaded their documents by
    # indexing a folder. Images fall back to local Tesseract OCR.
    vision_enabled: bool = False
    vision_api_key: str = ""  # OpenRouter API key; falls back to env OPENROUTER_API_KEY

    # Directory indexing
    index_max_concurrent: int = 4
    index_exclude_patterns: list[str] = []

    # Directory watching
    watch_max_watchers: int = 10

    # Agent memory
    memory_decay_halflife_days: float = 30.0  # time-decay half-life for recall scoring
    memory_stability_days: float = 30.0  # FSRS stability: days for confidence to drop to 0.9
    memory_confidence_threshold: float = 0.3  # below this, memory fades out of recall

    # Ranking. "rrf" fuses lexical/recency/confidence in rank space; "legacy"
    # is the original score-space product of BM25 x decay x pin x confidence,
    # kept so the two can be compared on the same corpus.
    ranking_mode: str = "rrf"
    rank_weight_lexical: float = 1.0
    # 0.5, not the 0.25 the grid search preferred. Cross-validation ties the two
    # on the temporal suite, but 0.25 is not enough to break a lexical near-tie
    # that BM25 decided on document length alone (see the metadata-date case in
    # tests/test_sqlite_backend.py). Ties go to the more conservative value.
    rank_weight_recency: float = 0.5
    rank_weight_confidence: float = 0.3
    rank_rrf_k: float = 60.0

    # Evaluation capture (opt-in): records real search/recall traffic for offline analysis
    eval_capture_enabled: bool = False

    # Lightweight link graph. Links are indexed deterministically; backlink
    # search boosting is opt-in so default ranking stays unchanged.
    link_extraction_enabled: bool = True
    backlink_boost_enabled: bool = False
    backlink_boost_weight: float = 0.05

    # Authentication (optional — if set, all requests require X-API-Key header)
    auth_api_key: str = ""

    # Storage backend: "sqlite" (embedded, zero-config) or "postgres" (server).
    #
    # The default is the embedded backend, which is what `pip install open-db`
    # advertises and what every quickstart in the README assumes. It used to
    # default to "postgres", so the documented three-line install pointed at a
    # PostgreSQL server the user had not been told to run.
    backend: str = "sqlite"

    # SQLite embedded mode — path to the .opendb directory
    opendb_dir: Path = Path(".opendb")

    # Server.
    #
    # Loopback and no cross-origin by default. The old defaults were
    # host="0.0.0.0" + cors_origins=["*"] with auth unenforced, which meant an
    # OpenDB server was reachable from the local network *and* scriptable by
    # any web page the developer happened to visit — including read and index
    # of the whole workspace. Set FILEDB_HOST / FILEDB_CORS_ORIGINS explicitly
    # to expose it.
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = []

    model_config = {"env_prefix": "FILEDB_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
