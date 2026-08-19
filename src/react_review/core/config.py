"""Application configuration: YAML loading + Pydantic validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from react_review.core.exceptions import ConfigError

#: Steps that may appear under ``routing``. Unknown names are a hard error.
BACKEND_STEPS = (
    "review_lens",
    "evidence_localize",
    "table_capture",
    "forest_ocr_vision",
    "forest_ocr_text",
    "claim_origin",
    "unpivot",
    "references",
    "field_resolution",
    "extract_locate",
    "extract_transcribe",
    "semantic_compare",
)

#: Providers that actually honour a ``reasoning`` knob. Others must not pretend.
REASONING_PROVIDERS = frozenset({"openai", "glm", "zhipu", "deepseek"})


class LLMSettings(BaseModel):
    """Settings for the LLM backend."""

    provider: str = "mock"
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4096
    api_key: str = ""
    base_url: str = ""
    # Extra fields merged verbatim into the OpenAI-compatible request body.
    # Provider-specific knobs go here, e.g. GLM-4.5 reasoning control:
    #   extra_body: {"thinking": {"type": "disabled"}}
    # (reasoning models otherwise spend the token budget "thinking" and get
    # truncated before emitting the answer).
    extra_body: dict = Field(default_factory=dict)
    # Gemini-only: controls reasoning budget for 2.5-series models.
    #   None → auto (disable for 2.5 to avoid truncation surprises)
    #   0    → explicitly disable thinking
    #   -1   → dynamic (model decides)
    #   N>0  → cap thinking at N tokens
    thinking_budget: int | None = None
    # Maximum number of concurrent in-flight requests this backend
    # instance allows. Enforced by an asyncio.Semaphore inside
    # ``LLMBackend``. Lower this to match an organisation-level
    # concurrency cap (e.g. Moonshot allows 3) so the pipeline does not
    # get throttled with HTTP 429 when Step 0 fires its 4 parallel
    # ingestion sub-tasks or Step 3 processes multiple papers in
    # parallel. The semaphore is per-backend instance, so ``llm`` and
    # ``llm2`` get independent caps when they point at different
    # providers.
    max_concurrency: int = 3
    # Maximum number of retry attempts when the provider returns HTTP 429
    # (rate limited). Each attempt reads the response's ``Retry-After``
    # header when present and falls back to exponential backoff using
    # ``retry_base_delay`` as the base. Set to 0 to disable retries.
    max_retries: int = 5
    # Base delay (seconds) for exponential backoff when no ``Retry-After``
    # header is supplied. Effective delay for attempt N is
    # ``retry_base_delay * 2 ** N`` (so 2.0 → 2s / 4s / 8s / 16s / 32s).
    retry_base_delay: float = 2.0


class PubMedSettings(BaseModel):
    """Settings for PubMed E-utilities API."""

    api_key: str = ""
    base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    rate_limit: float = 3.0  # requests per second (10 with api_key)
    email: str = ""  # recommended by NCBI; also used for Unpaywall API


class UnpaywallSettings(BaseModel):
    """Settings for Unpaywall API (free OA full-text discovery)."""

    email: str = ""  # required by Unpaywall TOS (your email address)
    enabled: bool = True  # set False to skip Unpaywall tier


class CrossRefSettings(BaseModel):
    """Settings for CrossRef API."""

    base_url: str = "https://api.crossref.org"
    mailto: str = ""  # enter email to join the polite pool (faster)
    timeout: float = 30.0


class ThresholdSettings(BaseModel):
    """Numeric thresholds used in verification / comparison."""

    title_similarity: float = 0.85
    author_match_ratio: float = 0.8


class PathSettings(BaseModel):
    """File-system paths used by the application."""

    data_dir: Path = Path("./data")
    output_dir: Path = Path("./output")
    log_file: Path = Path("./logs/react_review.log")


class BackendProfile(BaseModel):
    """One named backend gear: provider + model + reasoning. Optional extras inherit from ``llm``."""

    provider: str
    model: str
    reasoning: str | None = None  # "on" | "off" | None (do not inject)
    temperature: float | None = None
    max_tokens: int | None = None
    api_key: str | None = None
    base_url: str | None = None
    extra_body: dict | None = None
    max_concurrency: int | None = None
    max_retries: int | None = None
    retry_base_delay: float | None = None
    thinking_budget: int | None = None

    @field_validator("reasoning", mode="before")
    @classmethod
    def _coerce_reasoning(cls, value):
        if value is None or value == "":
            return None
        if value is True:
            return "on"
        if value is False:
            return "off"
        text = str(value).strip().lower()
        if text in {"on", "true", "yes"}:
            return "on"
        if text in {"off", "false", "no"}:
            return "off"
        raise ValueError("reasoning must be 'on' or 'off'")


class AppConfig(BaseModel):
    """Top-level application configuration."""

    app_name: str = "react-review"
    environment: str = "development"
    mock_mode: bool = True
    enabled_steps: list[str] = Field(
        default_factory=lambda: [
            "search_validation",
            "paper_verification",
            "data_extraction",
            "table_comparison",
        ]
    )
    llm: LLMSettings = Field(default_factory=LLMSettings)
    llm2: LLMSettings | None = None  # optional second LLM for cross-validation
    vision: LLMSettings | None = None  # optional vision model; not a second judge
    backend_profiles: dict[str, BackendProfile] = Field(default_factory=dict)
    routing: dict[str, str] = Field(default_factory=dict)
    pubmed: PubMedSettings = Field(default_factory=PubMedSettings)
    unpaywall: UnpaywallSettings = Field(default_factory=UnpaywallSettings)
    crossref: CrossRefSettings = Field(default_factory=CrossRefSettings)
    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    paths: PathSettings = Field(default_factory=PathSettings)

    @model_validator(mode="after")
    def _validate_backend_routing(self) -> "AppConfig":
        known = set(BACKEND_STEPS)
        for step, name in self.routing.items():
            if step not in known:
                raise ValueError(
                    f"routing names unknown step {step!r}; "
                    f"known steps: {', '.join(BACKEND_STEPS)}")
            if name not in self.backend_profiles:
                raise ValueError(
                    f"routing[{step!r}] points at profile {name!r} which is "
                    "not in backend_profiles")
        for name, profile in self.backend_profiles.items():
            if profile.reasoning is not None:
                provider = profile.provider.lower()
                if provider not in REASONING_PROVIDERS:
                    raise ValueError(
                        f"backend_profiles.{name}: provider "
                        f"{profile.provider!r} does not support the reasoning "
                        "field")
        return self


def load_config(path: Path) -> AppConfig:
    """Load application config from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated AppConfig instance.

    Raises:
        ConfigError: If the file cannot be read or parsed.
    """
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return AppConfig(**data)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"Failed to load config from {path}: {exc}") from exc


def apply_profile_all(config: AppConfig, name: str | None) -> AppConfig:
    """Pin every routed step to one named profile. No-op when ``name`` is empty."""
    if not name:
        return config
    if name not in config.backend_profiles:
        raise ConfigError(
            f"--profile-all {name!r} is not a backend_profiles entry")
    return AppConfig(**{
        **config.model_dump(),
        "routing": {step: name for step in BACKEND_STEPS},
    })


def settings_from_profile(config: AppConfig, name: str) -> LLMSettings:
    """Build LLMSettings for a named profile, inheriting keys/limits from ``llm``.

    ``extra_body`` is NOT inherited from ``llm``: a global thinking flag on the
    default model must not leak into a transcribe gear. Reasoning injection is
    MeteredBackend's job.
    """
    profile = config.backend_profiles[name]
    data = config.llm.model_dump()
    data["provider"] = profile.provider
    data["model"] = profile.model
    data["extra_body"] = dict(profile.extra_body or {})
    for field in ("temperature", "max_tokens", "api_key", "base_url",
                  "max_concurrency", "max_retries", "retry_base_delay",
                  "thinking_budget"):
        value = getattr(profile, field)
        if value is not None and value != "":
            data[field] = value
    return LLMSettings(**data)
