"""Heading anchors for results/capture windows. Text fixtures, no PDFs."""
from __future__ import annotations

from react_review.parser.review_extraction.windows import (
    _AFTER_RESULTS,
    _INTRO,
    _METHODS,
    _RESULTS,
    capture_window,
    results_window,
)


def test_dotted_section_numbers_match_all_four_headings():
    assert _INTRO.search("2. Introduction\nbody")
    assert _METHODS.search("3. Methods\nbody")
    assert _RESULTS.search("3. Results\n3.1. Study selection\n")
    assert _AFTER_RESULTS.search("4. Discussion\n")


def test_undotted_section_numbers_still_match():
    assert _INTRO.search("2 Introduction\n")
    assert _METHODS.search("3 Methods\n")
    assert _RESULTS.search("3 Results\n")
    assert _AFTER_RESULTS.search("4 Discussion\n")


def test_uppercase_results_heading_matches():
    match = _RESULTS.search("RESULTS \nTable 1\n")
    assert match is not None
    assert match.group(0).strip().upper() == "RESULTS"


def test_abstract_results_label_does_not_match():
    """D1 lock: the first 'Results:' in the abstract must not become the window."""
    abstract = (
        "Abstract\n"
        "Results: Twenty-one observational studies were included.\n"
        "Conclusions: Something.\n"
        "\n"
        "3. Results\n"
        "Table 1 Characteristics.\n"
        "Figure 2 Forest plot.\n"
    )
    assert _RESULTS.search("Results: Twenty-one observational studies…") is None
    match = _RESULTS.search(abstract)
    assert match is not None
    assert match.start() == abstract.index("3. Results")
    window = results_window(abstract)
    assert window.startswith("3. Results")
    assert "Twenty-one observational" not in window
    assert "Table 1" in window and "Figure 2" in window


def test_inline_results_sentence_does_not_match():
    body = (
        "Methods\n"
        "results. Where necessary, unit conversions were applied.\n"
        "3. Results\n"
        "Figure 1 Forest plot of overall survival.\n"
    )
    assert _RESULTS.search("results. Where necessary, unit conversions were applied.") is None
    match = _RESULTS.search(body)
    assert match is not None
    assert match.start() == body.index("3. Results")


def test_two_column_broken_results_heading_still_matches():
    body = "Results \nLiterature \nsearch \nFigure 3 Forest plot.\n"
    match = _RESULTS.search(body)
    assert match is not None
    assert results_window(body).startswith("Results")
    assert "Figure 3" in results_window(body)


def test_unnumbered_results_heading_still_opens_the_window():
    filler = "Study characteristics were tabulated.\n" * 4
    window = results_window(
        "Abstract SECRET\nIntroduction\nMethods\n"
        f"Results\n{filler}Table 1\nDiscussion\n"
    )
    assert window.startswith("Results")
    assert "SECRET" not in window
    assert "Table 1" in window
    assert "Discussion" not in window


def test_a_normal_results_section_is_not_clipped_at_15k():
    body = "Results\n" + ("Figure 1 forest plot. " * 2000) + "\nDiscussion\n"
    window = results_window(body)
    assert len(window) > 15000
    assert "Figure 1" in window
    assert "Discussion" not in window


def test_capture_window_limit_stays_at_20k():
    body = "3. Methods\n" + ("x" * 30000) + "\n4. Discussion\n"
    window = capture_window(body)
    assert len(window) == 20000


def test_capture_window_accepts_dotted_methods_heading():
    filler = "The search strategy is described below.\n" * 4
    body = (
        "Abstract SECRET\n"
        "2. Introduction\nintro\n"
        f"3. Methods\n{filler}"
        "3. Results\nTable 1\n"
        "4. Discussion\nend\n"
    )
    window = capture_window(body)
    assert window.startswith("3. Methods")
    assert "SECRET" not in window
    assert "Table 1" in window
    assert "Discussion" not in window
