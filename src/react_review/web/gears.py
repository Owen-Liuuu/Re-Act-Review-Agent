"""Parse homepage gear fields from POST /runs. Keys never go in the dump."""
from __future__ import annotations

from typing import Any

from react_review.core.config import AppConfig, apply_run_gears
from react_review.llm.catalog import DEFAULTS, ResolvedGear, resolve


def _field(form: Any, name: str) -> str:
    value = form.get(name) if form is not None else None
    if value is None:
        return ""
    return str(value).strip()


def parse_gears(form: Any) -> tuple[ResolvedGear, ResolvedGear, ResolvedGear]:
    """Read vendor/model pairs. Missing fields fall back to catalog defaults."""
    cv, cm = DEFAULTS["complex"]
    sv, sm = DEFAULTS["simple"]
    vv, vm = DEFAULTS["visual"]
    return (
        resolve(_field(form, "complex_vendor") or cv,
                _field(form, "complex_model") or cm),
        resolve(_field(form, "simple_vendor") or sv,
                _field(form, "simple_model") or sm),
        resolve(_field(form, "visual_vendor") or vv,
                _field(form, "visual_model") or vm, vision=True),
    )


def parse_keys(form: Any) -> tuple[str, str, str]:
    return (
        _field(form, "complex_key"),
        _field(form, "simple_key"),
        _field(form, "visual_key"),
    )


def overlay_from_form(config: AppConfig, form: Any) -> AppConfig:
    """Host config + homepage gears. Raises ConfigError on bad keys/slugs."""
    complex, simple, visual = parse_gears(form)
    c_key, s_key, v_key = parse_keys(form)
    return apply_run_gears(
        config, complex=complex, simple=simple, visual=visual,
        complex_key=c_key, simple_key=s_key, visual_key=v_key,
    )


def redacted_gears(
    overlay: AppConfig,
    *,
    complex: ResolvedGear,
    simple: ResolvedGear,
    visual: ResolvedGear,
    name: str = "",
    email: str = "",
) -> dict[str, Any]:
    """Safe-to-write record: vendors, models, identity. No api_key."""
    return {
        "name": name,
        "email": email,
        "complex": {
            "vendor": complex.vendor,
            "slug": complex.slug,
            "provider": overlay.llm.provider,
            "model": overlay.llm.model,
            "base_url": overlay.llm.base_url,
        },
        "simple": {
            "vendor": simple.vendor,
            "slug": simple.slug,
            "provider": overlay.backend_profiles["transcribe"].provider,
            "model": overlay.backend_profiles["transcribe"].model,
            "base_url": overlay.backend_profiles["transcribe"].base_url,
        },
        "visual": {
            "vendor": visual.vendor,
            "slug": visual.slug,
            "provider": overlay.vision.provider if overlay.vision else "",
            "model": overlay.vision.model if overlay.vision else "",
            "base_url": overlay.vision.base_url if overlay.vision else "",
        },
    }


def missing_customer_keys(form: Any) -> bool:
    c_key, s_key, v_key = parse_keys(form)
    return not (c_key and s_key and v_key)


def partial_customer_keys(form: Any) -> bool:
    keys = parse_keys(form)
    n = sum(1 for k in keys if k)
    return 0 < n < 3
