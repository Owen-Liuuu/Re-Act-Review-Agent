"""The identities a batched reading has to keep straight.

Each test here is a way the audit has already been wrong, written down before
the code that could repeat it exists. The load-bearing one is the first: a trial
reports 314 allocated to the combination arm and 313 analysed in it, and any
scheme that treats those as two arms — or as one duplicate — cannot fix MA004,
which is the whole point of the batch.
"""
from __future__ import annotations

from react_review.normalize.population import PopulationScope
from react_review.schemas.batch import (
    ARM,
    COMPARISON,
    STUDY,
    BatchEntry,
    BatchRequestId,
    ClaimGroupKey,
    EntryIdentity,
    EvidenceAnchor,
)
from react_review.tools.evidence_binding import (
    SAME_BLOCK,
    SAME_QUOTE,
    UNBOUND,
    binding_verdict,
    bound,
)


def _combo(basis: str, **kw) -> EntryIdentity:
    return EntryIdentity(target_shape=ARM, arm_label="nivolumab-plus-ipilimumab group",
                         population=PopulationScope(basis=basis), **kw)


# --- one arm, several readings -------------------------------------------

def test_the_same_arm_read_two_ways_is_one_target_and_two_readings():
    """MA004: 314 allocated and 313 analysed are both the combination arm."""
    allocated, analysed = _combo("allocated"), _combo("analysed")
    assert allocated.target() == analysed.target()          # one arm to map…
    assert allocated.selection() != analysed.selection()    # …two things to pick from


def test_the_arm_mapping_level_ignores_population_timepoint_and_effect():
    """Folding a reading into the target is what would make one arm look like three."""
    plain = _combo("allocated")
    elaborate = _combo("allocated", timepoint="median_pfs",
                       effect_definition="progression-free survival")
    assert plain.target() == elaborate.target()


def test_two_arms_reporting_the_same_number_are_not_duplicates():
    """Both arms can be 6.9. Only the identities decide sameness, never the value."""
    left = BatchEntry(identity=EntryIdentity(arm_label="nivolumab group"), value="6.9")
    right = BatchEntry(identity=EntryIdentity(arm_label="ipilimumab group"), value="6.9")
    assert left.selection_key() != right.selection_key()


def test_a_selection_key_is_not_an_entry_id():
    """Same reading, two response rows: a contradiction to surface, not a duplicate."""
    first = BatchEntry(identity=_combo("allocated"), value="314", raw_index=0)
    second = BatchEntry(identity=_combo("allocated"), value="313", raw_index=4)
    assert first.selection_key() == second.selection_key()
    assert first.entry_id("R") != second.entry_id("R")


def test_a_comparison_identity_keeps_its_direction():
    forwards = EntryIdentity(target_shape=COMPARISON,
                             comparison_pair=("combo", "ipilimumab"))
    backwards = EntryIdentity(target_shape=COMPARISON,
                              comparison_pair=("ipilimumab", "combo"))
    assert forwards.target() != backwards.target()


# --- which claims may be asked together ----------------------------------

def test_a_different_timepoint_is_a_different_question():
    baseline = ClaimGroupKey(study_id="larkin", field_type="progression_free_survival",
                             column_header="PFS", timepoint="baseline")
    median = baseline.model_copy(update={"timepoint": "median_pfs"})
    assert baseline.key() != median.key()


def test_arm_identity_and_value_are_not_asked_together():
    """One asks what the arm IS, the other what it reports. Different answers."""
    value = ClaimGroupKey(study_id="larkin", field_type="treatment_arm",
                          target_kind="value")
    identity = value.model_copy(update={"target_kind": "arm_identity"})
    assert value.key() != identity.key()


def test_shapes_are_asked_separately():
    arms = ClaimGroupKey(study_id="larkin", field_type="hazard_ratio",
                         target_shape=ARM)
    comparisons = arms.model_copy(update={"target_shape": COMPARISON})
    totals = arms.model_copy(update={"target_shape": STUDY})
    assert len({arms.key(), comparisons.key(), totals.key()}) == 3


# --- what was actually asked ---------------------------------------------

def _request(**kw) -> BatchRequestId:
    body = dict(group=ClaimGroupKey(study_id="larkin", field_type="cohort_n"),
                claim_targets=["combo", "ipilimumab"],
                requested_scopes=["allocated"], extraction_profile="targeted_v5_batch",
                research_context="melanoma trials", document_sha256="ABC")
    body.update(kw)
    return BatchRequestId(**body)


def test_the_same_group_with_different_members_is_a_different_request():
    """A key that stayed the same while its membership changed would hide this."""
    two = _request()
    three = _request(claim_targets=["combo", "ipilimumab", "nivolumab"])
    assert two.group.key() == three.group.key()
    assert two.identity() != three.identity()


def test_request_identity_ignores_the_order_members_arrive_in():
    assert _request(claim_targets=["a", "b"]).identity() == \
        _request(claim_targets=["b", "a"]).identity()


def test_a_different_document_or_contract_is_a_different_request():
    assert _request().identity() != _request(document_sha256="DEF").identity()
    assert _request().identity() != _request(extraction_profile="targeted_v4").identity()


# --- evidence binding -----------------------------------------------------

PAPER = ("A total of 945 patients underwent randomization: 316 patients were "
         "assigned to the nivolumab group, 314 to the nivolumab-plus-ipilimumab "
         "group, and 315 to the ipilimumab group.\n\n"
         "Table 3. Analysis population. Nivolumab plus Ipilimumab (N = 313)")


def test_a_phrase_inside_the_same_quote_binds():
    verdict, _ = binding_verdict(
        "were assigned to", "314",
        quote="316 patients were assigned to the nivolumab group, 314 to the "
              "nivolumab-plus-ipilimumab group", document=PAPER)
    assert verdict == SAME_QUOTE and bound(verdict)


def test_a_real_sentence_from_another_block_does_not_bind():
    """Both halves are genuine text; only their relationship is invented.

    This is the failure the rule exists for: the allocation sentence from the
    Results attached to a number read out of an analysis-set table.
    """
    verdict, reason = binding_verdict(
        "underwent randomization", "313",
        quote="Nivolumab plus Ipilimumab (N = 313)", document=PAPER)
    assert verdict == UNBOUND and not bound(verdict)
    assert "different blocks" in reason or "not a relationship" in reason


def test_occurring_in_the_same_paper_is_never_enough():
    far = PAPER + ("\n\n" + "filler. " * 200) + "\n\nper-protocol population"
    verdict, reason = binding_verdict("per-protocol population", "314",
                                      quote="314 to the nivolumab-plus-ipilimumab "
                                            "group", document=far)
    assert verdict == UNBOUND
    assert "characters from" in reason or "different blocks" in reason


def test_a_phrase_the_document_does_not_contain_does_not_bind():
    verdict, reason = binding_verdict("intention-to-treat", "314",
                                      quote="314 to the group", document=PAPER)
    assert verdict == UNBOUND and "does not occur" in reason


def test_binding_needs_a_document_when_the_quote_does_not_carry_the_phrase():
    verdict, reason = binding_verdict("allocated", "314", quote="314 patients",
                                      document="")
    assert verdict == UNBOUND and "no document" in reason


def test_an_anchor_knows_whether_it_was_located():
    assert EvidenceAnchor(quote="x").located is False
    assert EvidenceAnchor(quote="x", start=10).located is True
