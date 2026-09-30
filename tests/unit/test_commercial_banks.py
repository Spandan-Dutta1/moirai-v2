"""Tests for the commercial banking layer.

Two things get most of the attention.

The asymmetry, because it is the mechanism the layer exists to produce and
the one prediction that survived out-of-sample testing. Lending rates must
rise faster than they fall, deposit rates the reverse, and stress must
make both worse.

And the distinction between calibration and validation. The easing-cycle
targets were fitted to, so a test that they pass is a test of the fitting.
The tightening figures were held out, and two of three predictions against
them fail. Those failures are pinned here so that a later change cannot
quietly make them disappear: if someone adds a liquidity state and the
deposit check starts passing, the test should be updated deliberately
rather than silently going green.
"""

import pytest

from moirai.core.exceptions import EngineError
from moirai.engine.financial.commercial_banks import (
    CAPITAL_MINIMUM,
    EASING_TARGETS,
    INDIAN_BANKING_SYSTEM,
    TIGHTENING_OBSERVED,
    BankGroup,
    BankingSystem,
    CommercialBank,
    evaluate_transmission,
    validate_out_of_sample,
)
from moirai.engine.financial.game import Confidence


def make_bank(name: str = "Test Bank", **overrides) -> CommercialBank:
    defaults = dict(
        name=name,
        group=BankGroup.PRIVATE,
        assets=1_000_000.0,
        loans=620_000.0,
        deposits=780_000.0,
        capital=620_000.0 * 0.16,
        eblr_share=0.70,
        retail_deposit_share=0.55,
        npa_ratio=0.02,
    )
    return CommercialBank(**{**defaults, **overrides})


@pytest.fixture
def system() -> BankingSystem:
    return INDIAN_BANKING_SYSTEM


# --- balance sheet ---------------------------------------------------------

def test_capital_ratio_is_capital_over_loans():
    bank = make_bank(loans=100_000.0, capital=15_000.0)
    assert bank.capital_ratio == pytest.approx(0.15)


def test_a_thinly_capitalised_bank_is_flagged():
    """Within two points of the regulatory minimum."""
    assert make_bank(loans=100_000.0, capital=12_000.0).is_capital_constrained is True


def test_a_well_capitalised_bank_is_not_flagged():
    assert make_bank(loans=100_000.0, capital=20_000.0).is_capital_constrained is False


def test_the_constraint_threshold_sits_above_the_minimum():
    """A bank exactly at the minimum has no buffer and should be flagged."""
    at_minimum = make_bank(loans=100_000.0, capital=100_000.0 * CAPITAL_MINIMUM)
    assert at_minimum.is_capital_constrained is True


def test_loans_cannot_exceed_assets():
    with pytest.raises(Exception, match="exceed total assets"):
        make_bank(assets=100_000.0, loans=200_000.0)


def test_capital_cannot_exceed_assets():
    with pytest.raises(Exception, match="exceeds total assets"):
        make_bank(assets=100_000.0, loans=50_000.0, capital=200_000.0)


def test_a_bank_is_frozen():
    with pytest.raises(Exception):
        make_bank().eblr_share = 0.9  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("eblr_share", 1.5),
        ("eblr_share", -0.1),
        ("retail_deposit_share", 1.2),
        ("npa_ratio", 0.5),
        ("assets", 0.0),
    ],
)
def test_out_of_range_characteristics_are_rejected(field, value):
    with pytest.raises(Exception):
        make_bank(**{field: value})


# --- stress ----------------------------------------------------------------

def test_stress_rises_with_bad_loans():
    clean = make_bank(npa_ratio=0.01)
    troubled = make_bank(npa_ratio=0.08)
    assert troubled.stress > clean.stress


def test_stress_rises_as_capital_thins():
    strong = make_bank(loans=100_000.0, capital=25_000.0)
    weak = make_bank(loans=100_000.0, capital=12_500.0)
    assert weak.stress > strong.stress


def test_stress_is_bounded():
    worst = make_bank(npa_ratio=0.25, loans=100_000.0, capital=11_600.0)
    best = make_bank(npa_ratio=0.0, loans=100_000.0, capital=30_000.0)
    assert 0.0 <= best.stress <= worst.stress <= 1.0


# --- the asymmetry, which is the point -------------------------------------

def test_lending_rates_rise_faster_than_they_fall():
    """The mechanism the layer exists to produce, and the one prediction
    that survived out-of-sample testing."""
    bank = make_bank()
    assert bank.lending_pass_through(tightening=True) > bank.lending_pass_through(
        tightening=False
    )


def test_deposit_rates_fall_faster_than_they_rise():
    """The mirror image, and the other half of the wedge."""
    bank = make_bank()
    assert bank.deposit_pass_through(tightening=False) > bank.deposit_pass_through(
        tightening=True
    )


def test_the_wedge_favours_the_bank_in_both_directions():
    """Margin widens either way, though not in the way one might expect.

    Tightening: lending rises more than deposits, so the bank captures
    part of the increase. Easing: lending falls more than deposits, which
    looks like the bank losing out until you notice that externally
    benchmarked loans reprice mechanically in both directions while
    deposits do not. The bank's discretion is on the deposit side, and it
    uses it to cut less than it has to.
    """
    bank = make_bank()
    tightening_wedge = bank.lending_pass_through(
        tightening=True
    ) - bank.deposit_pass_through(tightening=True)
    easing_deposit_retention = 1.0 - bank.deposit_pass_through(tightening=False)

    assert tightening_wedge > 0
    assert easing_deposit_retention > 0.5


# --- the three drivers -----------------------------------------------------

def test_a_higher_benchmark_share_raises_pass_through():
    """Externally benchmarked loans reprice mechanically; the rest reprice
    at the bank's discretion."""
    low = make_bank(eblr_share=0.30)
    high = make_bank(eblr_share=0.90)
    assert high.lending_pass_through(tightening=True) > low.lending_pass_through(
        tightening=True
    )


def test_full_benchmark_linkage_gives_near_complete_pass_through():
    assert make_bank(eblr_share=1.0).lending_pass_through(tightening=True) > 0.9


def test_stress_lowers_pass_through_in_both_directions():
    """The intended mechanism is that stress bites harder when easing,
    since a bank needing margin simply declines to pass a cut through.

    The implementation applies a proportional penalty to an already-lower
    easing base, so the absolute cost comes out marginally smaller on the
    way down: 10.0 basis points of pass-through against 10.7 when
    tightening. That mismatch between the stated intent and the arithmetic
    is documented in the module rather than re-tuned, because the layer
    has since been validated out of sample and changing it now would
    compromise that test.
    """
    clean = make_bank(eblr_share=0.3, npa_ratio=0.005)
    troubled = make_bank(eblr_share=0.3, npa_ratio=0.09)

    for tightening in (True, False):
        assert troubled.lending_pass_through(
            tightening=tightening
        ) < clean.lending_pass_through(tightening=tightening)




def test_retail_deposit_dependence_constrains_cutting():
    """Retail term depositors can move to administered small savings
    schemes, which puts a floor under how far deposit rates can fall."""
    wholesale = make_bank(retail_deposit_share=0.25)
    retail = make_bank(retail_deposit_share=0.85)
    assert retail.deposit_pass_through(tightening=False) < wholesale.deposit_pass_through(
        tightening=False
    )


@pytest.mark.parametrize("tightening", [True, False])
def test_pass_through_is_a_valid_share(tightening):
    for eblr in (0.0, 0.5, 1.0):
        for npa in (0.0, 0.15, 0.30):
            bank = make_bank(eblr_share=eblr, npa_ratio=npa)
            assert 0.0 <= bank.lending_pass_through(tightening=tightening) <= 1.0
            assert 0.0 <= bank.deposit_pass_through(tightening=tightening) <= 1.0


# --- rates -----------------------------------------------------------------

def test_lending_exceeds_deposit_rates():
    rates = make_bank().rates_at(0.06, 0.06)
    assert rates["lending_rate"] > rates["deposit_rate"]
    assert rates["net_interest_margin"] > 0


def test_a_policy_rise_raises_both_rates():
    bank = make_bank()
    before = bank.rates_at(0.05, 0.05)
    after = bank.rates_at(0.06, 0.05)
    assert after["lending_rate"] > before["lending_rate"]
    assert after["deposit_rate"] > before["deposit_rate"]


def test_lending_rises_by_more_than_deposits():
    bank = make_bank()
    before = bank.rates_at(0.05, 0.05)
    after = bank.rates_at(0.06, 0.05)
    assert (after["lending_rate"] - before["lending_rate"]) > (
        after["deposit_rate"] - before["deposit_rate"]
    )


def test_deposit_rates_cannot_go_negative():
    rates = make_bank(deposit_spread=-0.015).rates_at(0.005, 0.005)
    assert rates["deposit_rate"] >= 0.0


# --- the system ------------------------------------------------------------

def test_the_indian_system_spans_three_groups(system):
    assert {b.group for b in system.banks} == set(BankGroup)


def test_public_banks_hold_most_of_the_credit(system):
    """Roughly sixty percent, which is the actual Indian structure."""
    assert system.group_share(BankGroup.PUBLIC) > 0.5


def test_group_shares_sum_to_one(system):
    assert sum(system.group_share(g) for g in BankGroup) == pytest.approx(1.0)


def test_public_banks_transmit_less_than_private(system):
    """Lower external benchmark shares and higher stress. This holds in
    the model in both directions, which is where the out-of-sample test
    finds it wrong for tightening."""
    public = system.weighted_lending_pass_through(tightening=True, group=BankGroup.PUBLIC)
    private = system.weighted_lending_pass_through(
        tightening=True, group=BankGroup.PRIVATE
    )
    assert public < private


def test_foreign_banks_transmit_most(system):
    foreign = system.weighted_lending_pass_through(
        tightening=True, group=BankGroup.FOREIGN
    )
    private = system.weighted_lending_pass_through(
        tightening=True, group=BankGroup.PRIVATE
    )
    assert foreign > private


def test_the_system_average_lies_between_its_groups(system):
    values = [
        system.weighted_lending_pass_through(tightening=True, group=g)
        for g in BankGroup
    ]
    overall = system.weighted_lending_pass_through(tightening=True)
    assert min(values) <= overall <= max(values)


def test_weighting_is_by_loans_not_by_count(system):
    """A small foreign bank must not count as much as SBI."""
    equal_weight = sum(
        b.lending_pass_through(tightening=True) for b in system.banks
    ) / len(system.banks)
    loan_weighted = system.weighted_lending_pass_through(tightening=True)
    assert loan_weighted != pytest.approx(equal_weight)


def test_an_empty_group_is_rejected():
    only_public = BankingSystem(banks=(make_bank(group=BankGroup.PUBLIC),))
    with pytest.raises(EngineError, match="no banks in group"):
        only_public.weighted_lending_pass_through(
            tightening=True, group=BankGroup.FOREIGN
        )


def test_duplicate_bank_names_are_rejected():
    with pytest.raises(Exception, match="unique"):
        BankingSystem(banks=(make_bank("A"), make_bank("A")))


def test_the_system_serialises(system):
    payload = system.to_ledger_dict()
    assert payload["n_banks"] == 12
    assert "system_pass_through" in payload
    assert len(payload["banks"]) == 12


# --- what a household faces ------------------------------------------------

def test_effective_rates_report_the_direction(system):
    assert system.effective_rates(0.06, 0.05)["tightening"] is True
    assert system.effective_rates(0.04, 0.05)["tightening"] is False


def test_a_hundred_point_move_reaches_borrowers_partially(system):
    before = system.effective_rates(0.05, 0.05)
    after = system.effective_rates(0.06, 0.05)
    passed = after["lending_rate"] - before["lending_rate"]
    assert 0.0 < passed < 0.01


def test_the_wedge_is_positive(system):
    """Borrowers absorb more of a rise than savers receive, and the
    difference accrues to the banking system as margin."""
    before = system.effective_rates(0.05, 0.05)
    after = system.effective_rates(0.06, 0.05)
    lending_move = after["lending_rate"] - before["lending_rate"]
    deposit_move = after["deposit_rate"] - before["deposit_rate"]
    assert lending_move > deposit_move > 0


def test_no_policy_move_means_no_rate_change(system):
    before = system.effective_rates(0.05, 0.05)
    assert before["lending_rate"] == pytest.approx(
        system.effective_rates(0.05, 0.05)["lending_rate"]
    )


# --- calibration, which is not validation ----------------------------------

def test_the_easing_targets_cite_their_source():
    for target in EASING_TARGETS:
        assert "RBI" in target.source
        assert target.confidence is Confidence.DERIVED


def test_the_model_reproduces_the_easing_cycle(system):
    """It should, because it was tuned to. This tests the fitting, not the
    model: the discretionary coefficients were adjusted until these group
    aggregates landed near the published figures."""
    result = evaluate_transmission(system)
    assert result["n_passed"] >= 5


def test_the_foreign_deposit_channel_is_the_known_in_sample_miss(system):
    """Left failing rather than tuned away. Foreign banks fund wholesale
    and reprice faster than the retail-dependence mechanism allows, and
    they are 2.4 percent of credit so the aggregate barely moves."""
    result = evaluate_transmission(system)
    foreign = next(r for r in result["results"] if r["group"] == "foreign")
    assert foreign["deposit_passed"] is False


# --- out of sample, which is ----------------------------------------------

def test_the_tightening_data_was_held_out():
    """Calibration used the easing cycle only, so these figures are a
    genuine test rather than a second fit."""
    assert TIGHTENING_OBSERVED["repo_change"] == 0.0250
    assert TIGHTENING_OBSERVED["system_lending_bp"] == 189.0


def test_the_asymmetry_survives_out_of_sample(system):
    """The one prediction that holds. The model produces roughly the right
    magnitude on a cycle it never saw."""
    result = validate_out_of_sample(system)
    asymmetry = next(c for c in result["checks"] if c["name"] == "lending asymmetry")
    assert asymmetry["passed"] is True


def test_the_group_ordering_fails_out_of_sample(system):
    """Pinned deliberately. The model predicts private banks transmit more
    in both directions; the data shows the ordering reverses under
    tightening. If a later change makes this pass, the test should be
    updated on purpose rather than going green by accident."""
    result = validate_out_of_sample(system)
    ordering = next(
        c for c in result["checks"] if "ordering" in c["name"]
    )
    assert ordering["passed"] is False


def test_deposit_pass_through_fails_out_of_sample(system):
    """Also pinned. The model has no liquidity state, so it cannot produce
    the late deposit repricing that followed the drain of surplus
    liquidity. Adding one after seeing this figure would convert the test
    into a second in-sample fit."""
    result = validate_out_of_sample(system)
    deposits = next(c for c in result["checks"] if "deposit" in c["name"])
    assert deposits["passed"] is False


def test_out_of_sample_performance_is_worse_than_in_sample(system):
    """The honest measure of how much the calibration was fitting: five of
    six in sample against one of three out of sample."""
    in_sample = evaluate_transmission(system)
    out_of_sample = validate_out_of_sample(system)

    in_rate = in_sample["n_passed"] / in_sample["n_checks"]
    out_rate = out_of_sample["n_passed"] / out_of_sample["n_checks"]
    assert out_rate < in_rate


def test_every_check_records_its_source(system):
    for check in validate_out_of_sample(system)["checks"]:
        assert "RBI" in check["source"]
        assert check["detail"].strip()



# --- bounding results that depend on deposit pass-through (ADR 018) -----------


def test_without_an_override_the_mechanism_is_unchanged(system):
    assert system.deposit_pass_through_override is None
    rebuilt = BankingSystem(banks=system.banks)
    for tightening in (True, False):
        assert rebuilt.weighted_deposit_pass_through(
            tightening=tightening
        ) == system.weighted_deposit_pass_through(tightening=tightening)


def test_the_override_replaces_system_tightening_deposit_pass_through(system):
    bounded = system.model_copy(update={"deposit_pass_through_override": 0.97})
    assert bounded.weighted_deposit_pass_through(tightening=True) == 0.97
    rates = bounded.effective_rates(0.0625, 0.0525)
    assert rates["deposit_pass_through"] == 0.97


def test_the_override_leaves_lending_easing_and_groups_alone(system):
    """It stands in for one observation, a system-wide tightening figure,
    and must not reach anything that observation does not describe."""
    bounded = system.model_copy(update={"deposit_pass_through_override": 0.97})
    assert bounded.weighted_lending_pass_through(
        tightening=True
    ) == system.weighted_lending_pass_through(tightening=True)
    assert bounded.weighted_deposit_pass_through(
        tightening=False
    ) == system.weighted_deposit_pass_through(tightening=False)
    for group in {b.group for b in system.banks}:
        assert bounded.weighted_deposit_pass_through(
            tightening=True, group=group
        ) == system.weighted_deposit_pass_through(tightening=True, group=group)


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_the_override_must_be_a_share(system, value):
    with pytest.raises(ValueError):
        BankingSystem(banks=system.banks, deposit_pass_through_override=value)


def test_the_observed_figure_is_the_held_out_one():
    from moirai.engine.financial.commercial_banks import (
        OBSERVED_TIGHTENING_DEPOSIT_PASS_THROUGH,
    )

    assert pytest.approx(0.972) == OBSERVED_TIGHTENING_DEPOSIT_PASS_THROUGH


def test_the_mechanism_is_far_below_what_was_observed(system):
    """The failure ADR 009 reports, kept visible: if the mechanism were
    retuned to the held-out figure this would fail, and it should."""
    from moirai.engine.financial.commercial_banks import (
        OBSERVED_TIGHTENING_DEPOSIT_PASS_THROUGH,
    )

    mechanism = system.weighted_deposit_pass_through(tightening=True)
    assert mechanism < OBSERVED_TIGHTENING_DEPOSIT_PASS_THROUGH - 0.5
