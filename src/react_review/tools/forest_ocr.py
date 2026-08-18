"""Extract stage: read a forest plot from the PDF text layer into a study grid.

Vector RevMan figures often have a selectable text layer; this path is valid
only then. Raster plots have no per-study grid in the dump — those need a
vision path. This module never invents Events/Total from unreadable pixels.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any

import structlog

from react_review.contracts import ContractError, read_json_object, repo_root
from react_review.core.exceptions import LLMError
from react_review.llm.base import LLMBackend, parse_llm_response
from react_review.parser.table_capture import _parse_tables
from react_review.schemas.table import CapturedTable
from react_review.tools.base import Tool, ToolStage
from react_review.tools.models import ForestOcrInput, ForestOcrResult

logger = structlog.get_logger(__name__)

FOREST_OCR_V1 = Path("configs/prompt_contracts/forest_ocr_v1.json")
FOREST_OCR_VISION_V1 = Path("configs/prompt_contracts/forest_ocr_vision_v1.json")
RENDERER_IDENTITY = "react_review.review_extraction.render.v1"
PROMPT_ID = "forest_ocr_v1"
PROMPT_VERSION = "forest-ocr-v1"
VISION_PROMPT_ID = "forest_ocr_vision_v1"
VISION_PROMPT_VERSION = "forest-ocr-vision-v1"

_FOREST = """You are reading the TEXT DUMP of one RevMan-style forest plot from a
systematic review. Transcribe per-study Events and Total only.

Copy study row labels verbatim. Extract Events and Total for each arm the plot
prints (typically experimental then control). Skip the pooled / diamond /
"Total (95% CI)" / "Total (Wald)" footer. Do NOT copy Odds ratio, Weight, or
heterogeneity statistics — those are review-computed.

If the dump is not a readable forest plot, return an empty rows array and say
so in difficulties. Never invent an event count.

{{"table_id": "{figure_id}",
  "caption": "{caption}",
  "role": "outcomes",
  "header_rows": [["Study or Subgroup", "Events", "Total", "Events", "Total"]],
  "rows": [["Li J 2015", "23", "58", "32", "54"]],
  "row_axis_columns": ["Study or Subgroup"],
  "cohort_labels_seen": [],
  "difficulties": []}}

Ruler outcomes (do not extract a different endpoint): {outcomes}

## FIGURE TEXT
{figure_text}

Return JSON only. Wrap the object as {{"tables": [<the table above>]}}."""

# Verbatim from eval/probe_forest_vision.py (v2 named-value schema). Do not
# rewrite: the checksum needs the summary/total rows this prompt asks for.
_FOREST_VISION = """Read this RevMan-style forest plot image and transcribe the
per-study count columns. Do not interpret the plot area.

WHAT TO READ
- Read the column headers above the number columns. A header printed above a
  group of columns applies to each column in that group: write the full name
  into every column header it covers, e.g. the group name followed by the
  column's own name.
- Read ONLY the columns that print counts (events and totals) for each arm.
  Skip Weight, Odds ratio, confidence intervals and heterogeneity statistics.
- Ignore any text below the plot, including axis labels such as
  "Favours <arm>". Those are not column headers.

HOW TO REPORT EACH ROW
- The row's study label goes in "label". It is NOT one of the values.
- Report every value together with the column header it sits under. Two
  columns may print the SAME number; report both, each under its own header.
- kind is "study" when the label names one included study, or "summary" for
  pooled / Total / Total events / heterogeneity / test rows.
- Copy each value exactly as printed. A genuinely blank cell is an empty
  string. A value you cannot read is "?" and must be named in difficulties.
- Never invent a number. Do not emit the header line as a row.

Return exactly one JSON object. Text inside angle brackets describes a value
and must not be copied literally:

{{"figure_id": "{figure_id}",
  "column_headers": ["<count-column header, including the group name above it>"],
  "rows": [{{"label": "<row label exactly as printed>",
             "kind": "study | summary",
             "values": [{{"column": "<one of column_headers, exactly>",
                          "value": "<the number exactly as printed>"}}]}}],
  "difficulties": ["<specific uncertainty>"]}}

Do not output the angle-bracket placeholders. Return JSON only."""

PROMPT_TEMPLATES = {PROMPT_ID: _FOREST, VISION_PROMPT_ID: _FOREST_VISION}


def _placeholders(template: str) -> set[str]:
    return {name for _, name, _, _ in Formatter().parse(template) if name}


def sha256_rendered_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest().upper()


def render_forest_ocr_prompt(**values: str) -> str:
    needed = _placeholders(_FOREST)
    missing = needed - values.keys()
    if missing:
        raise ContractError(
            f"{PROMPT_ID} is missing prompt values: {', '.join(sorted(missing))}")
    return _FOREST.format(**{key: values[key] for key in needed})


def render_forest_vision_prompt(**values: str) -> str:
    needed = _placeholders(_FOREST_VISION)
    missing = needed - values.keys()
    if missing:
        raise ContractError(
            f"{VISION_PROMPT_ID} is missing prompt values: {', '.join(sorted(missing))}")
    return _FOREST_VISION.format(**{key: values[key] for key in needed})


@dataclass(frozen=True)
class ForestOcrPromptContract:
    prompt_id: str
    prompt_version: str
    rendered_prompt_sha256: str
    renderer_identity: str
    hash_algorithm: str
    fixture_inputs: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, profile_or_path: str | Path = PROMPT_ID) -> "ForestOcrPromptContract":
        if isinstance(profile_or_path, str) and profile_or_path == PROMPT_ID:
            path = repo_root() / FOREST_OCR_V1
        else:
            path = Path(profile_or_path)
            if not path.is_absolute():
                path = repo_root() / path
        body = read_json_object(path, kind="forest OCR prompt contract")
        fixture_inputs = body.get("fixture_inputs")
        needed = _placeholders(_FOREST)
        if not isinstance(fixture_inputs, dict) or set(fixture_inputs) != needed:
            raise ContractError(
                f"forest OCR contract {path} must pin fixture_inputs {sorted(needed)}")
        return cls(
            prompt_id=str(body.get("prompt_id") or ""),
            prompt_version=str(body.get("prompt_version") or ""),
            rendered_prompt_sha256=str(body.get("rendered_prompt_sha256") or "").upper(),
            renderer_identity=str(body.get("renderer_identity") or ""),
            hash_algorithm=str(body.get("hash_algorithm") or ""),
            fixture_inputs=dict(fixture_inputs),
            path=path,
        )

    def drifts(self) -> list[str]:
        found: list[str] = []
        if self.prompt_id != PROMPT_ID:
            found.append(f"prompt_id is {self.prompt_id!r}")
        if self.prompt_version != PROMPT_VERSION:
            found.append(f"version is {self.prompt_version!r}")
        if self.renderer_identity != RENDERER_IDENTITY:
            found.append(f"renderer is {self.renderer_identity!r}")
        if self.hash_algorithm != "sha256-rendered-utf8-v1":
            found.append("unsupported hash algorithm")
        rendered = render_forest_ocr_prompt(**self.fixture_inputs)
        actual = sha256_rendered_prompt(rendered)
        if actual != self.rendered_prompt_sha256:
            found.append(
                f"rendered prompt is {actual[:16]}, published as "
                f"{self.rendered_prompt_sha256[:16]}")
        return found


@dataclass(frozen=True)
class ForestVisionPromptContract:
    prompt_id: str
    prompt_version: str
    rendered_prompt_sha256: str
    renderer_identity: str
    hash_algorithm: str
    fixture_inputs: dict[str, Any]
    path: Path

    @classmethod
    def load(cls, profile_or_path: str | Path = VISION_PROMPT_ID) -> "ForestVisionPromptContract":
        if isinstance(profile_or_path, str) and profile_or_path == VISION_PROMPT_ID:
            path = repo_root() / FOREST_OCR_VISION_V1
        else:
            path = Path(profile_or_path)
            if not path.is_absolute():
                path = repo_root() / path
        body = read_json_object(path, kind="forest OCR vision prompt contract")
        fixture_inputs = body.get("fixture_inputs")
        needed = _placeholders(_FOREST_VISION)
        if not isinstance(fixture_inputs, dict) or set(fixture_inputs) != needed:
            raise ContractError(
                f"forest vision contract {path} must pin fixture_inputs {sorted(needed)}")
        return cls(
            prompt_id=str(body.get("prompt_id") or ""),
            prompt_version=str(body.get("prompt_version") or ""),
            rendered_prompt_sha256=str(body.get("rendered_prompt_sha256") or "").upper(),
            renderer_identity=str(body.get("renderer_identity") or ""),
            hash_algorithm=str(body.get("hash_algorithm") or ""),
            fixture_inputs=dict(fixture_inputs),
            path=path,
        )

    def drifts(self) -> list[str]:
        found: list[str] = []
        if self.prompt_id != VISION_PROMPT_ID:
            found.append(f"prompt_id is {self.prompt_id!r}")
        if self.prompt_version != VISION_PROMPT_VERSION:
            found.append(f"version is {self.prompt_version!r}")
        if self.renderer_identity != RENDERER_IDENTITY:
            found.append(f"renderer is {self.renderer_identity!r}")
        if self.hash_algorithm != "sha256-rendered-utf8-v1":
            found.append("unsupported hash algorithm")
        rendered = render_forest_vision_prompt(**self.fixture_inputs)
        actual = sha256_rendered_prompt(rendered)
        if actual != self.rendered_prompt_sha256:
            found.append(
                f"rendered prompt is {actual[:16]}, published as "
                f"{self.rendered_prompt_sha256[:16]}")
        return found


_MIN_IMAGE_WIDTH = 200
_MIN_IMAGE_HEIGHT = 80
_HEADING = re.compile(r"^\s*\d+\.\d+(?:\.\d+)*\s+(.+)$")
_FIGURE_NO = re.compile(r"(?i)\bfigure\s+(\d+)\b")


@dataclass(frozen=True)
class _LocatedImage:
    page: int
    xref: int
    bbox: tuple[float, float, float, float]
    width: int
    height: int


@dataclass(frozen=True)
class _ForestPair:
    image: _LocatedImage
    outcome_key: str
    figure_no: str


@dataclass(frozen=True)
class _FigurePick:
    image: _LocatedImage | None
    reason: str
    review_required: bool
    status: str  # ok | mismatch | missing


def _empty_table(
    payload: ForestOcrInput,
    difficulties: list[str],
    *,
    capture_path: str = "",
    image: bytes = b"",
    pick: _FigurePick | None = None,
) -> CapturedTable:
    table = CapturedTable(
        table_id=payload.figure_id or "forest",
        caption=payload.caption,
        page_hint=payload.page_hint,
        role="outcomes",
        display_kind="forest_plot",
        capture_method="figure_ocr",
        outcome=payload.caption,
        row_axis_columns=["Study or Subgroup"],
        difficulties=difficulties,
    )
    return _stamp_capture(table, capture_path, image=image, pick=pick)


def _stamp_capture(
    table: CapturedTable,
    capture_path: str,
    *,
    image: bytes = b"",
    pick: _FigurePick | None = None,
    extra: str = "",
) -> CapturedTable:
    """Record which path produced the table, and which embedded image it used."""
    if capture_path:
        table.capture_path = capture_path
    if image:
        table.image_bytes = len(image)
    if pick is not None:
        table.review_required = pick.review_required
        if pick.image is not None:
            table.image_page = pick.image.page
            table.image_xref = pick.image.xref
            if not (table.page_hint or "").strip().isdigit():
                table.page_hint = str(pick.image.page)
    bits = [f"capture path: {capture_path}"] if capture_path else []
    if extra:
        bits.append(extra)
    if image:
        bits.append(f"{len(image)} bytes")
    if pick is not None and pick.reason:
        bits.append(pick.reason)
    if pick is not None and pick.image is not None:
        bits.append(f"page {pick.image.page} xref {pick.image.xref}")
    if bits:
        table.difficulties.append("; ".join(bits))
    return table


def _outcome_key(text: str) -> str:
    blob = (text or "").lower()
    if "anastomotic" in blob or re.search(r"\bleak\b", blob):
        return "anastomotic_leak"
    if "mortal" in blob:
        return "30_day_mortality"
    if "pulmonary" in blob:
        return "pulmonary"
    if "overall" in blob or "postoperative complication" in blob:
        return "overall_complications"
    return ""


def _figure_no(text: str) -> str:
    match = _FIGURE_NO.search(text or "")
    return match.group(1) if match else ""


def _figure_text(pdf_path: str, caption: str, page_hint: str) -> str:
    """Nearby text layer. Figures often have none — that is a difficulty, not a guess."""
    if not pdf_path:
        return ""
    try:
        import fitz
    except ImportError:
        return ""
    try:
        doc = fitz.open(pdf_path)
    except Exception:  # noqa: BLE001
        return ""
    try:
        pages: list[str] = []
        hint = page_hint.strip()
        index = None
        if hint.isdigit():
            index = max(0, int(hint) - 1)
            if index < len(doc):
                pages.append(doc[index].get_text() or "")
        needle = (caption or "").strip().lower()
        if needle:
            for i, page in enumerate(doc):
                if index is not None and i == index:
                    continue
                text = page.get_text() or ""
                if needle[:48] in text.lower():
                    pages.append(text)
                    break
        # No page-1 dump. An empty dump is a named gap, not a guess.
        return "\n\n".join(pages)
    finally:
        doc.close()


def _locate_figure_images(pdf_path: str) -> list[_LocatedImage]:
    """Embedded images in document order. Tiny decorations are dropped."""
    if not pdf_path:
        return []
    try:
        import fitz
    except ImportError:
        return []
    try:
        doc = fitz.open(pdf_path)
    except Exception:  # noqa: BLE001
        return []
    try:
        return _locate_in_doc(doc)
    finally:
        doc.close()


def _locate_in_doc(doc: Any) -> list[_LocatedImage]:
    found: list[_LocatedImage] = []
    for index, page in enumerate(doc):
        for img in page.get_images(full=True):
            xref = int(img[0])
            width, height = int(img[2] or 0), int(img[3] or 0)
            if width < _MIN_IMAGE_WIDTH or height < _MIN_IMAGE_HEIGHT:
                continue
            try:
                rects = list(page.get_image_rects(xref) or [])
            except Exception:  # noqa: BLE001
                rects = []
            if not rects:
                continue
            rect = rects[0]
            found.append(_LocatedImage(
                page=index + 1,
                xref=xref,
                bbox=(float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)),
                width=width,
                height=height,
            ))
    return found


def _iter_text_lines(doc: Any) -> list[tuple[int, float, float, str]]:
    lines: list[tuple[int, float, float, str]] = []
    for index, page in enumerate(doc):
        for block in page.get_text("dict").get("blocks") or []:
            if block.get("type") != 0:
                continue
            for line in block.get("lines") or []:
                text = "".join(
                    span.get("text", "") for span in line.get("spans") or [])
                if not text.strip():
                    continue
                bbox = line.get("bbox") or (0, 0, 0, 0)
                lines.append((index + 1, float(bbox[1]), float(bbox[3]), text))
    return lines


def _clip_nearby(page: Any, y0: float, y1: float) -> str:
    try:
        import fitz
        clip = fitz.Rect(0, max(0.0, y0 - 15), page.rect.width, min(page.rect.height, y1 + 40))
        return page.get_text("text", clip=clip) or ""
    except Exception:  # noqa: BLE001
        return ""


def _pick_image_below(
    images: list[_LocatedImage],
    page: int,
    y0: float,
    page_heights: dict[int, float],
    used: set[tuple[int, int]],
) -> _LocatedImage | None:
    below = [
        im for im in images
        if im.page == page and im.bbox[1] >= y0 - 8
        and (im.page, im.xref) not in used
    ]
    if below:
        return min(below, key=lambda im: im.bbox[1])
    height = page_heights.get(page, 0.0)
    if height and y0 < 0.70 * height:
        return None
    nxt = [
        im for im in images
        if im.page == page + 1 and (im.page, im.xref) not in used
    ]
    if not nxt:
        return None
    return min(nxt, key=lambda im: (im.bbox[1], im.bbox[0]))


def _paired_forest_images(pdf_path: str) -> list[_ForestPair]:
    """Pair each 'forest plot' sentence with the nearest image below it."""
    if not pdf_path:
        return []
    try:
        import fitz
    except ImportError:
        return []
    try:
        doc = fitz.open(pdf_path)
    except Exception:  # noqa: BLE001
        return []
    try:
        images = _locate_in_doc(doc)
        heights = {i + 1: float(doc[i].rect.height) for i in range(len(doc))}
        last_heading = ""
        anchors: list[tuple[int, float, str, str]] = []
        for page_no, y0, y1, text in _iter_text_lines(doc):
            heading = _HEADING.match(text.strip())
            if heading:
                key = _outcome_key(heading.group(1))
                if key:
                    last_heading = key
            if "forest plot" not in text.lower():
                continue
            nearby = _clip_nearby(doc[page_no - 1], y0, y1)
            anchors.append((
                page_no, y0, last_heading or _outcome_key(nearby or text),
                _figure_no(nearby) or _figure_no(text),
            ))
        used: set[tuple[int, int]] = set()
        pairs: list[_ForestPair] = []
        for page_no, y0, outcome, figure_no in anchors:
            image = _pick_image_below(images, page_no, y0, heights, used)
            if image is None:
                continue
            used.add((image.page, image.xref))
            pairs.append(_ForestPair(image, outcome, figure_no))
        return pairs
    finally:
        doc.close()


def _resolve_figure(
    pdf_path: str,
    caption: str = "",
    page_hint: str = "",
    figure_ordinal: int = 0,
    image_index: int = 0,
) -> _FigurePick:
    """Pick one forest image. Caption outcome beats hit order; never silent swap."""
    if not pdf_path:
        return _FigurePick(None, "no forest image located", True, "missing")
    pairs = _paired_forest_images(pdf_path)
    want = _outcome_key(caption)
    fig_want = _figure_no(caption)
    page = int(page_hint.strip()) if (page_hint or "").strip().isdigit() else None

    if want:
        hits = [p for p in pairs if p.outcome_key == want]
        if page is not None and len(hits) > 1:
            hits = [p for p in hits if p.image.page == page]
        if len(hits) == 1:
            return _FigurePick(hits[0].image, "caption-outcome", False, "ok")
        if len(hits) > 1:
            return _FigurePick(
                None, "caption matched multiple forest images", True, "mismatch")
        return _FigurePick(
            None, f"no forest image matches caption outcome {want}", True, "mismatch")

    if fig_want:
        hits = [p for p in pairs if p.figure_no == fig_want]
        if len(hits) == 1:
            return _FigurePick(hits[0].image, "caption-figure-number", False, "ok")
        if len(hits) > 1:
            return _FigurePick(
                None, "figure number matched multiple forest images", True, "mismatch")

    if page is not None:
        on_page = [p for p in pairs if p.image.page == page]
        if on_page and 0 <= image_index < len(on_page):
            return _FigurePick(
                on_page[image_index].image, "page-paired-index", True, "ok")
        if on_page and 0 <= figure_ordinal < len(on_page):
            return _FigurePick(
                on_page[figure_ordinal].image, "page-paired-ordinal", True, "ok")

    if pairs and 0 <= figure_ordinal < len(pairs):
        chosen = pairs[figure_ordinal]
        return _FigurePick(chosen.image, "forest-ordinal", True, "ok")
    return _FigurePick(None, "no forest image located", True, "missing")


def _read_png(pdf_path: str, xref: int) -> bytes:
    try:
        import fitz
    except ImportError:
        return b""
    try:
        doc = fitz.open(pdf_path)
    except Exception:  # noqa: BLE001
        return b""
    try:
        pix = fitz.Pixmap(doc, xref)
        if pix.n - pix.alpha >= 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        return pix.tobytes("png")
    except Exception:  # noqa: BLE001
        return b""
    finally:
        doc.close()


def _figure_image(
    pdf_path: str,
    page_hint: str = "",
    image_index: int = 0,
    *,
    caption: str = "",
    figure_ordinal: int = 0,
) -> bytes:
    """PNG bytes for one forest plot. Empty if pairing cannot name an image."""
    pick = _resolve_figure(
        pdf_path, caption=caption, page_hint=page_hint,
        figure_ordinal=figure_ordinal, image_index=image_index)
    if pick.image is None:
        return b""
    return _read_png(pdf_path, pick.image.xref)


# A forest study row is "Name 2015 23 58 32 54": a word+year, then ≥2 more
# whitespace-delimited integers on the same line. Decimals (3.3.1), ratios
# (0/92), and "95%" are not independent grid cells. No Events/Total lexicon.
_STUDY_YEAR = re.compile(
    r"(?i)(?<![A-Za-z0-9])[A-Za-z][A-Za-z'.-]*"
    r"(?:\s+[A-Za-z]\.?){0,4}"
    r"(?:\s+et\s+al\.?)?"
    r"\s+(?:19|20)\d{2}(?!\d)"
)
_GRID_INT = re.compile(r"(?<!\S)\d+(?!\S)")


def _has_per_study_grid(text: str) -> bool:
    """True when the text dump contains at least one per-study forest row."""
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _STUDY_YEAR.search(line)
        if match is None:
            continue
        if len(_GRID_INT.findall(line[match.end():])) >= 2:
            return True
    return False


_SUMMARY_LABEL = re.compile(
    r"(?i)^\s*(total(?:\s+events)?|subtotal|heterogeneity|test\s+for)\b"
)
_INT_CELL = re.compile(r"^-?\d+$")


def _is_summary_label(label: str) -> bool:
    """True for printed total / heterogeneity / test rows, not study names."""
    return bool(_SUMMARY_LABEL.search((label or "").strip()))


def _parse_int_cell(cell: str) -> int | None:
    text = (cell or "").strip()
    if not _INT_CELL.fullmatch(text):
        return None
    return int(text)


def _is_count_header(header: str) -> bool:
    text = re.sub(r"\s+", " ", (header or "").strip().lower())
    return (
        text == "events" or text.endswith(" events")
        or text == "total" or text.endswith(" total")
    )


def _count_columns(table: CapturedTable) -> list[tuple[int, str]]:
    return [
        (j, path) for j, path in enumerate(table.column_paths())
        if _is_count_header(path)
    ]


def _printed_label(row: list[str]) -> str:
    return (row[0] if row else "").strip()


def _row_is_summary(table: CapturedTable, index: int, row: list[str]) -> bool:
    if len(table.row_kinds) == len(table.rows):
        return table.row_kinds[index] == "summary"
    return _is_summary_label(_printed_label(row))


def _split_forest_rows(table: CapturedTable) -> tuple[list[list[str]], list[list[str]]]:
    study, summary = [], []
    for index, row in enumerate(table.rows):
        if _row_is_summary(table, index, row):
            summary.append(row)
        else:
            study.append(row)
    return study, summary


def _kind_mismatch_note(label: str, kind: str) -> str | None:
    """Deterministic label vs the model's declared kind, when both exist."""
    declared = (kind or "").strip().lower()
    if declared not in {"study", "summary"}:
        return None
    structural = _is_summary_label(label)
    if structural and declared == "study":
        return f"label {label!r} looks like a summary row but kind=study"
    if not structural and declared == "summary":
        return f"label {label!r} looks like a study row but kind=summary"
    return None


def _summary_int_set(summary_rows: list[list[str]]) -> set[int]:
    found: set[int] = set()
    for row in summary_rows:
        for cell in row:
            number = _parse_int_cell(cell)
            if number is not None:
                found.add(number)
    return found


def _forest_checksum(table: CapturedTable) -> list[str]:
    """Count columns whose study-row integer sum is not on any summary row.

    Compares sums against the SET of integers printed on summary rows, not
    against the cell in the same column. A total written under the wrong
    header still validates the number.
    """
    study_rows, summary_rows = _split_forest_rows(table)
    if not summary_rows:
        return []
    printed = _summary_int_set(summary_rows)
    failed: list[str] = []
    for index, header in _count_columns(table):
        numbers = [
            n for row in study_rows
            if index < len(row)
            for n in [_parse_int_cell(row[index])]
            if n is not None
        ]
        if not numbers:
            continue
        if sum(numbers) not in printed:
            failed.append(header)
    return failed


def _missing_count_columns(table: CapturedTable) -> list[str]:
    """Count columns with no parseable integer in any study row."""
    study_rows, _summary = _split_forest_rows(table)
    missing: list[str] = []
    for index, header in _count_columns(table):
        if not any(
            index < len(row) and _parse_int_cell(row[index]) is not None
            for row in study_rows
        ):
            missing.append(header)
    return missing


def _forest_integrity_failures(table: CapturedTable) -> list[str]:
    found: list[str] = []
    for header in _missing_count_columns(table) + _forest_checksum(table):
        if header not in found:
            found.append(header)
    return found


def _column_sums(table: CapturedTable) -> dict[str, int]:
    study_rows, _summary = _split_forest_rows(table)
    sums: dict[str, int] = {}
    for index, header in _count_columns(table):
        numbers = [
            n for row in study_rows
            if index < len(row)
            for n in [_parse_int_cell(row[index])]
            if n is not None
        ]
        if numbers:
            sums[header] = sum(numbers)
    return sums


def _quarantine_failed_columns(table: CapturedTable, failures: list[str]) -> None:
    """Blank study-row cells in failed count columns; leave the rest."""
    if not failures:
        return
    failed = set(failures)
    indexes = [j for j, path in _count_columns(table) if path in failed]
    for i, row in enumerate(table.rows):
        if _row_is_summary(table, i, row):
            continue
        for index in indexes:
            if index < len(row):
                row[index] = ""


_LABEL_COL = "Study or Subgroup"


def _table_from_named_values(
    parsed: dict[str, Any],
    *,
    table_id: str = "",
    caption: str = "",
) -> CapturedTable:
    """Build a CapturedTable from the vision named-value JSON shape."""
    headers = [str(h) for h in (parsed.get("column_headers") or [])]
    has_label_col = bool(headers) and headers[0].strip() == _LABEL_COL
    value_cols = headers[1:] if has_label_col else headers
    cols = [_LABEL_COL, *value_cols]
    rows: list[list[str]] = []
    kinds: list[str] = []
    notes: list[str] = []
    for raw in parsed.get("rows") or []:
        if not isinstance(raw, dict):
            continue
        label = str(raw.get("label") or "").strip()
        values = raw.get("values") if isinstance(raw.get("values"), list) else []
        by_col: dict[str, str] = {}
        for item in values:
            if not isinstance(item, dict):
                continue
            column = str(item.get("column") or "")
            if column:
                by_col[column] = str(item.get("value") or "")
        rows.append([label] + [by_col.get(h, "") for h in value_cols])
        raw_kind = str(raw.get("kind") or "").strip().lower()
        deterministic = "summary" if _is_summary_label(label) else "study"
        kinds.append(raw_kind if raw_kind in ("study", "summary") else deterministic)
        note = _kind_mismatch_note(label, raw_kind)
        if note:
            notes.append(note)
    difficulties = [str(x) for x in (parsed.get("difficulties") or [])] + notes
    return CapturedTable(
        table_id=table_id or str(parsed.get("figure_id") or "forest"),
        caption=caption,
        role="outcomes",
        header_rows=[cols] if headers else [],
        rows=rows,
        row_kinds=kinds,
        row_axis_columns=[_LABEL_COL],
        difficulties=difficulties,
        display_kind="forest_plot",
        capture_method="figure_ocr",
    )


def _apply_forest_integrity(table: CapturedTable) -> CapturedTable:
    """Record checksum audit, quarantine failed count columns, keep the rest."""
    _study_rows, summary_rows = _split_forest_rows(table)
    printed = sorted(_summary_int_set(summary_rows)) if summary_rows else []
    table.checksum_column_sums = _column_sums(table)
    table.checksum_printed_values = printed
    failures = _forest_integrity_failures(table)
    table.checksum_failures = list(failures)
    if failures:
        names = "; ".join(failures)
        table.difficulties.append(
            "column sums disagree with the figure's own total row: "
            f"{names}; study sums {table.checksum_column_sums}; "
            f"printed totals {printed}; these cells were not released")
        _quarantine_failed_columns(table, failures)
    return table


class ForestOcrTool(Tool):
    """Read one forest-plot figure into a CapturedTable.

    Vision is first when a backend is configured. The text-layer path is only
    a fallback when no embedded forest image can be located. Checksum integrity
    applies to both paths. A 429 after a real image is an honest empty table,
    not a cue to invent cells from nearby prose.
    """

    name = "ocr_forest_plot"
    stage = ToolStage.EXTRACT
    input_model = ForestOcrInput
    output_model = ForestOcrResult

    def __init__(
        self,
        backend: LLMBackend | None = None,
        vision_backend: LLMBackend | None = None,
    ) -> None:
        self._backend = backend
        self._vision = vision_backend

    async def run(self, payload: ForestOcrInput) -> ForestOcrResult:
        if payload.injected_table:
            try:
                table = CapturedTable.model_validate(payload.injected_table)
            except Exception as exc:  # noqa: BLE001
                empty = _empty_table(
                    payload, [f"injected table was malformed: {exc}"[:160]],
                    capture_path="injected")
                return ForestOcrResult(table=empty, difficulties=list(empty.difficulties))
            table.display_kind = table.display_kind or "forest_plot"
            table.capture_method = table.capture_method or "figure_ocr"
            if not table.table_id:
                table.table_id = payload.figure_id or table.table_id
            _stamp_capture(table, "injected")
            return ForestOcrResult(table=table, difficulties=list(table.difficulties))

        dump = _figure_text(payload.pdf_path, payload.caption, payload.page_hint)
        text_ok = (
            bool(dump.strip()) and _has_per_study_grid(dump)
            and self._backend is not None)

        if self._vision is not None:
            vision_out = await self._run_vision(payload)
            if vision_out.table.capture_path != "no_image":
                if text_ok:
                    vision_out.table.difficulties.append(
                        "text-layer grid was present but not used")
                    vision_out.difficulties = list(vision_out.table.difficulties)
                return vision_out

        if text_ok:
            return await self._run_text(payload, dump)

        if dump.strip() and not _has_per_study_grid(dump):
            logger.info(
                "forest_text_layer_absent",
                figure_id=payload.figure_id or payload.caption[:40],
                chars=len(dump),
            )
            empty = _empty_table(payload, [
                "the figure has no per-study grid in the PDF text layer; "
                "its numbers exist only as raster pixels"], capture_path="abstain")
            return ForestOcrResult(table=empty, difficulties=list(empty.difficulties))

        empty = _empty_table(payload, [
            "no raster OCR backend; forest-plot cells were not invented "
            "from the text layer"], capture_path="abstain")
        return ForestOcrResult(table=empty, difficulties=list(empty.difficulties))

    async def _run_text(self, payload: ForestOcrInput, dump: str) -> ForestOcrResult:
        prompt = render_forest_ocr_prompt(
            figure_id=payload.figure_id or "forest",
            caption=payload.caption.replace("{", "{{").replace("}", "}}"),
            outcomes="; ".join(payload.outcomes) or "(unspecified)",
            figure_text=dump[:12000],
        )
        try:
            raw = parse_llm_response(
                await self._backend.complete(prompt), self._backend.model_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("forest_ocr_llm_failed", error=str(exc)[:160])
            empty = _empty_table(
                payload, [f"forest OCR parse failed: {exc}"[:160]],
                capture_path="text")
            return ForestOcrResult(table=empty, difficulties=list(empty.difficulties))

        tables = _parse_tables(raw if isinstance(raw, dict) else {"tables": [raw]})
        if not tables:
            empty = _empty_table(payload, [
                "forest OCR returned no table; cells were not invented"],
                capture_path="text")
            return ForestOcrResult(table=empty, difficulties=list(empty.difficulties))
        table = tables[0]
        table.display_kind = "forest_plot"
        table.capture_method = "figure_ocr"
        table.table_id = payload.figure_id or table.table_id
        if not table.row_axis_columns:
            table.row_axis_columns = ["Study or Subgroup"]
        _apply_forest_integrity(table)
        _stamp_capture(table, "text")
        return ForestOcrResult(table=table, difficulties=list(table.difficulties))

    async def _run_vision(self, payload: ForestOcrInput) -> ForestOcrResult:
        pick = _resolve_figure(
            payload.pdf_path, caption=payload.caption,
            page_hint=payload.page_hint, figure_ordinal=payload.figure_ordinal,
            image_index=payload.image_index)
        image = _figure_image(
            payload.pdf_path, payload.page_hint, payload.image_index,
            caption=payload.caption, figure_ordinal=payload.figure_ordinal)
        if not image:
            if pick.status == "mismatch":
                empty = _empty_table(payload, [
                    pick.reason,
                    "caption and figure did not pair; forest-plot cells were "
                    "not invented"], capture_path="mismatch", pick=pick)
                return ForestOcrResult(table=empty, difficulties=list(empty.difficulties))
            empty = _empty_table(payload, [
                "no embedded forest-plot image; "
                "forest-plot cells were not invented"],
                capture_path="no_image", pick=pick)
            return ForestOcrResult(table=empty, difficulties=list(empty.difficulties))
        prompt = render_forest_vision_prompt(figure_id=payload.figure_id or "forest")
        try:
            raw = parse_llm_response(
                await self._vision.complete_vision(prompt, [image]),
                self._vision.model_id)
        except NotImplementedError:
            empty = _empty_table(payload, [
                "the figure has no per-study grid in the PDF text layer; "
                "its numbers exist only as raster pixels"],
                capture_path="vision", image=image, pick=pick)
            return ForestOcrResult(table=empty, difficulties=list(empty.difficulties))
        except LLMError as exc:
            logger.warning("forest_ocr_vision_failed", error=str(exc)[:160])
            empty = _empty_table(payload, [
                f"vision OCR failed after retries ({exc}); "
                "forest-plot cells were not invented"][:240],
                capture_path="vision", image=image, pick=pick)
            return ForestOcrResult(table=empty, difficulties=list(empty.difficulties))
        except Exception as exc:  # noqa: BLE001
            logger.warning("forest_ocr_vision_failed", error=str(exc)[:160])
            empty = _empty_table(payload, [
                f"vision OCR failed: {exc}; "
                "forest-plot cells were not invented"][:240],
                capture_path="vision", image=image, pick=pick)
            return ForestOcrResult(table=empty, difficulties=list(empty.difficulties))

        parsed = raw if isinstance(raw, dict) else {}
        table = _table_from_named_values(
            parsed, table_id=payload.figure_id or "forest", caption=payload.caption)
        table.page_hint = payload.page_hint
        table.outcome = payload.caption
        if not table.rows:
            empty = _empty_table(payload, list(table.difficulties) + [
                "vision OCR returned no rows; cells were not invented"],
                capture_path="vision", image=image, pick=pick)
            return ForestOcrResult(table=empty, difficulties=list(empty.difficulties))
        study_rows, _summary = _split_forest_rows(table)
        if len(study_rows) > 15:
            table.difficulties.append(
                f"this figure has {len(study_rows)} study rows; a short output "
                "budget may truncate the grid")
        _apply_forest_integrity(table)
        _stamp_capture(table, "vision", image=image, pick=pick)
        return ForestOcrResult(table=table, difficulties=list(table.difficulties))
