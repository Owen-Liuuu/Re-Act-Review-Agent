"""What the model is ASKED, pinned — separately from what its answer becomes.

The batch prompt's SHA-256 is not decoration. It goes into the replay cache key
and into the `BatchQuestionId` recorded on every batched answer, so a single
character changed anywhere in the template silently invalidates every recording
ever made under it. The symptom is `ExtractionCacheMiss: attempt 1`, which reads
as "the recording is incomplete" — not as "somebody edited the prompt".

Nothing pinned it. `legacy_v3` has had a golden SHA since Phase 7; the batch
contract was written, routed to in a frozen benchmark profile, and left
unpinned, and the benchmark profile does not close the gap: it pins the run
profile, which names route STRINGS, not the words those routes send.

So the pin is here, and it is a pin on the RENDERED PROMPT, not on the file that
produces it. A comment, a rename, a helper extracted — none of those change what
the model is asked and none of them may fail this. Only a different question
does.

The inputs live in the contract file beside the hash they produce. A pin whose
inputs live in the test can be made to pass by editing the inputs, which is the
same hole one directory over: the hash and the thing hashed have to travel
together, and the file carries its own frozen SHA in
`tests/test_contract_pins.py` so moving either one is a visible decision.

**Governance.** A rendered prompt that changes is a NEW PROMPT VERSION — a new
profile in `tools/extraction_profile.py`, not a new number typed into this file.
Updating a pinned hash so that an edited prompt passes converts every existing
recording into a replay miss with nothing to say why. See
`docs/acceptance/gate_versions.md`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from react_review.contracts import ContractError, read_json_object, repo_root
from react_review.tools.batch_prompt import build_batch_prompt
from react_review.tools.extraction_cache import extraction_cache_key

#: The contract governing `targeted_v5_batch`. One file per prompt contract, so
#: a second contract cannot be added by widening this one.
BATCH_V5 = "configs/prompt_contracts/batch_v5.json"

#: Renderers by contract id. A contract file names the prompt it pins rather
#: than being assumed to pin this one — a loader that always called the batch
#: builder would happily "verify" a contract for some other prompt.
_RENDERERS = {"batch_v5": build_batch_prompt}


def sha256_text(text: str) -> str:
    """The hash of what was ASKED, in the encoding it is sent in."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


@dataclass(frozen=True)
class PromptCase:
    """One rendered branch of a prompt, and the bytes it must produce."""

    case_id: str
    why: str
    inputs: dict[str, Any]
    rendered_sha256: str
    cache_key: str

    def render(self, renderer) -> str:
        return renderer(**self.inputs)


@dataclass(frozen=True)
class PromptContract:
    """Every branch of one prompt, pinned."""

    contract_id: str
    prompt_version: str
    extraction_profile: str
    pin_model_id: str
    pin_attempt: int
    cases: tuple[PromptCase, ...]
    path: Path

    @classmethod
    def load(cls, path: Path | str = BATCH_V5) -> "PromptContract":
        full = Path(path)
        if not full.is_absolute():
            full = repo_root() / full
        body = read_json_object(full, kind="prompt contract")
        contract_id = str(body.get("contract_id") or "")
        if contract_id not in _RENDERERS:
            raise ContractError(
                f"prompt contract {contract_id!r} in {full} names no renderer "
                f"(known: {', '.join(sorted(_RENDERERS))})")
        cases = tuple(
            PromptCase(case_id=str(c.get("case_id") or ""), why=str(c.get("why") or ""),
                       inputs=dict(c.get("inputs") or {}),
                       rendered_sha256=str(c.get("rendered_sha256") or ""),
                       cache_key=str(c.get("cache_key") or ""))
            for c in body.get("cases") or ())
        if not cases:
            raise ContractError(f"prompt contract {full} pins no cases at all")
        return cls(contract_id=contract_id,
                   prompt_version=str(body.get("prompt_version") or ""),
                   extraction_profile=str(body.get("extraction_profile") or ""),
                   pin_model_id=str(body.get("pin_model_id") or ""),
                   pin_attempt=int(body.get("pin_attempt") or 1),
                   cases=cases, path=full)

    @property
    def renderer(self):
        return _RENDERERS[self.contract_id]

    def key_for(self, prompt: str) -> str:
        """The cache key this prompt would be recorded under, under the pins."""
        return extraction_cache_key(
            model_id=self.pin_model_id, prompt_version=self.prompt_version,
            prompt=prompt, attempt=self.pin_attempt)

    def drifts(self) -> list[str]:
        """Every branch whose rendered prompt is no longer what was published.

        Named one by one. "A hash changed" sends the reader to diff a whole
        template; "the STUDY branch with aggregation changed" says which
        question moved, which is the thing they have to decide about.
        """
        found: list[str] = []
        for case in self.cases:
            try:
                rendered = case.render(self.renderer)
            except Exception as exc:                               # noqa: BLE001
                found.append(f"{case.case_id}: could not be rendered at all ({exc})")
                continue
            digest = sha256_text(rendered)
            if digest != case.rendered_sha256.upper():
                found.append(
                    f"{case.case_id}: the rendered prompt is {digest[:16]}, "
                    f"published as {case.rendered_sha256[:16].upper()} — "
                    f"{case.why}")
            key = self.key_for(rendered)
            if key.upper() != case.cache_key.upper():
                found.append(
                    f"{case.case_id}: the recording key is {key[:16].upper()}, "
                    f"published as {case.cache_key[:16].upper()} — every "
                    "recording made under the old key is now a replay miss")
        return found
