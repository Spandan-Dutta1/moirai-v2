"""
Commercial banks: the layer between policy and households.

Until now a policy rate reached a household through two constants: 85
percent of a repo move landed on floating loan rates, 45 percent on
deposits. Those numbers were doing real work, since the gap between them
is the transfer from borrowers to the banking system, and they were
assumed.

They need not be. The RBI publishes transmission by bank group, and the
mechanism behind the differences is documented rather than inferred.
During the easing cycle from February 2025 to May 2026, against a
cumulative 125 basis point repo cut, weighted average lending rates on
fresh rupee loans fell by 1.24 points at foreign banks, 1.08 at private
banks and 0.66 at public sector banks. Deposit rates fell 0.90, 0.46 and
0.53 respectively.

Two things in those numbers are worth noticing. Public sector banks
transmit roughly half a repo move to lending rates while private banks
transmit most of it. And private banks cut lending rates by more than
twice what they cut deposits, protecting margins, while public banks cut
the two almost in step.

The RBI attributes the gap to three causes, and all three are modelled
here rather than assumed away:

  * Benchmark mix. A larger share of private bank loans is linked to an
    external benchmark, which reprices mechanically with the repo rate.
    Loans on internal benchmarks reprice at the bank's discretion.

  * Funding mix. Public sector banks depend more on retail term deposits
    and compete with small savings schemes, which constrains how far they
    can cut deposit rates.

  * Stress. Banks carrying higher non-performing assets are slower to
    transmit cuts, because they need the margin to cover provisions.

Pass-through is therefore derived from those three characteristics rather
than set directly, and the published group averages become a calibration
target the model has to reproduce.

Limitations, stated rather than discovered:

  * No default. Loans are repaid, so household distress never feeds back
    into bank capital. That loop is the interesting one for a credit
    stress test and it is not here.
  * No interbank market, so no contagion.
  * Deposits are exogenous. Households do not move money between banks in
    response to rates.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from moirai.core.exceptions import EngineError
from moirai.core.logging import get_logger
from moirai.engine.financial.game import Confidence

log = get_logger(__name__)

#: Regulatory minimum total capital ratio under Basel III as applied in
#: India, including the capital conservation buffer.
CAPITAL_MINIMUM = 0.115


class BankGroup(StrEnum):
    """How the RBI groups scheduled commercial banks in its statistics.

    The grouping is not cosmetic. It is the unit the RBI reports
    transmission by, which makes it the unit this model can be calibrated
    against.
    """

    PUBLIC = "public"
    PRIVATE = "private"
    FOREIGN = "foreign"


class CommercialBank(BaseModel):
    """One bank, its balance sheet and the rates it sets.

    Pass-through is not a field. It is computed from the benchmark mix,
    the funding mix and the stress level, so a bank that transmits weakly
    does so for a stated reason.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    group: BankGroup

    # ---- balance sheet, in crore ----
    assets: float = Field(gt=0, description="Total assets.")
    loans: float = Field(gt=0)
    deposits: float = Field(gt=0)
    capital: float = Field(gt=0)

    # ---- the three drivers of transmission ----
    eblr_share: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Share of the loan book on an external benchmark. These reprice "
            "mechanically with the repo rate; internal-benchmark loans "
            "reprice at the bank's discretion."
        ),
    )
    retail_deposit_share: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Share of deposits from retail term accounts. A high share "
            "constrains deposit rate cuts, because these savers can move to "
            "small savings schemes whose rates are administered."
        ),
    )
    npa_ratio: float = Field(
        ge=0.0,
        le=0.30,
        description="Gross non-performing assets over loans.",
    )

    # ---- pricing ----
    lending_spread: float = Field(
        default=0.030,
        ge=0.0,
        description="Spread over the policy rate charged on loans.",
    )
    deposit_spread: float = Field(
        default=-0.015,
        le=0.0,
        description="Spread under the policy rate paid on deposits.",
    )
    floating_share: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Share of the loan book on floating rates.",
    )

    @model_validator(mode="after")
    def _balance_sheet_is_coherent(self) -> CommercialBank:
        if self.loans > self.assets:
            raise ValueError(f"{self.name}: loans exceed total assets")
        if self.capital > self.assets:
            raise ValueError(f"{self.name}: capital exceeds total assets")
        return self

    # ---- position ----

    @property
    def capital_ratio(self) -> float:
        """Capital over risk-weighted assets, with loans risk-weighted at one."""
        return self.capital / self.loans

    @property
    def is_capital_constrained(self) -> bool:
        """Within two points of the regulatory minimum.

        A constrained bank widens spreads rather than absorbing a shock,
        which makes transmission depend on the state of the banking system
        and not only on the size of the policy move.
        """
        return self.capital_ratio < CAPITAL_MINIMUM + 0.02

    @property
    def loan_to_deposit(self) -> float:
        return self.loans / self.deposits

    @property
    def stress(self) -> float:
        """A scalar in [0, 1] combining bad loans and thin capital.

        Both constrain a bank in the same direction: they raise the margin
        it needs, so it passes less of a cut through and more of a rise.
        """
        npa_stress = min(self.npa_ratio / 0.10, 1.0)
        capital_headroom = max(self.capital_ratio - CAPITAL_MINIMUM, 0.0)
        capital_stress = 1.0 - min(capital_headroom / 0.05, 1.0)
        return float(np.clip(0.6 * npa_stress + 0.4 * capital_stress, 0.0, 1.0))

    # ---- transmission ----

    def lending_pass_through(self, *, tightening: bool) -> float:
        """Share of a policy move reaching lending rates.

        Externally benchmarked loans reprice close to fully and
        mechanically. The rest reprice at the bank's discretion, and
        discretion is asymmetric: a bank raises rates faster than it cuts
        them, more so when it is stressed and needs the margin.

        The asymmetry is the reason a tightening cycle hurts more than an
        easing cycle helps, and a single pass-through constant cannot
        produce it.
        """
        mechanical = self.eblr_share * 0.95

        discretionary_base = 0.75 if tightening else 0.45
        discretionary = discretionary_base * (1.0 - 0.4 * self.stress)
        if not tightening:
            # Stress bites harder on the way down: a bank that needs margin
            # simply declines to pass a cut through.
            discretionary *= 1.0 - 0.3 * self.stress

        return float(
            np.clip(mechanical + (1.0 - self.eblr_share) * discretionary, 0.0, 1.0)
        )

    def deposit_pass_through(self, *, tightening: bool) -> float:
        """Share of a policy move reaching deposit rates.

        The mirror image. A bank is slow to raise deposit rates and quick
        to cut them, except that retail term depositors can move to
        administered small savings schemes, which puts a floor under how
        far deposit rates can fall.
        """
        base = 0.35 if tightening else 0.55
        retail_constraint = 1.0 - 0.45 * self.retail_deposit_share
        if not tightening:
            # Competition from small savings binds only when cutting.
            base *= retail_constraint
        else:
            base *= 0.6 + 0.4 * retail_constraint

        return float(np.clip(base, 0.0, 1.0))

    def rates_at(self, policy_rate: float, baseline_policy_rate: float) -> dict[str, float]:
        """Lending and deposit rates given a policy rate."""
        move = policy_rate - baseline_policy_rate
        tightening = move >= 0

        lending = (
            baseline_policy_rate
            + self.lending_spread
            + move * self.lending_pass_through(tightening=tightening)
        )
        deposit = (
            baseline_policy_rate
            + self.deposit_spread
            + move * self.deposit_pass_through(tightening=tightening)
        )
        return {
            "lending_rate": float(lending),
            "deposit_rate": float(max(deposit, 0.0)),
            "net_interest_margin": float(lending - max(deposit, 0.0)),
        }

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group.value,
            "assets": self.assets,
            "capital_ratio": round(self.capital_ratio, 4),
            "loan_to_deposit": round(self.loan_to_deposit, 4),
            "eblr_share": self.eblr_share,
            "retail_deposit_share": self.retail_deposit_share,
            "npa_ratio": self.npa_ratio,
            "stress": round(self.stress, 4),
            "capital_constrained": self.is_capital_constrained,
            "lending_pass_through_tightening": round(
                self.lending_pass_through(tightening=True), 4
            ),
            "lending_pass_through_easing": round(
                self.lending_pass_through(tightening=False), 4
            ),
        }


class BankingSystem(BaseModel):
    """A collection of banks, aggregated by loan share.

    The system's pass-through is the loan-weighted average of its members,
    which is what the RBI's published group figures measure. That makes
    the aggregate directly comparable to the calibration target rather
    than to a number the model invented.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    banks: tuple[CommercialBank, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _names_are_unique(self) -> BankingSystem:
        names = [b.name for b in self.banks]
        if len(set(names)) != len(names):
            raise ValueError("bank names must be unique")
        return self

    @property
    def total_loans(self) -> float:
        return sum(b.loans for b in self.banks)

    @property
    def total_assets(self) -> float:
        return sum(b.assets for b in self.banks)

    def by_group(self, group: BankGroup) -> tuple[CommercialBank, ...]:
        return tuple(b for b in self.banks if b.group is group)

    def group_share(self, group: BankGroup) -> float:
        """Share of system loans held by one group."""
        return sum(b.loans for b in self.by_group(group)) / self.total_loans

    def weighted_lending_pass_through(
        self, *, tightening: bool, group: BankGroup | None = None
    ) -> float:
        """Loan-weighted pass-through, for the system or for one group."""
        banks = self.banks if group is None else self.by_group(group)
        if not banks:
            raise EngineError(f"no banks in group {group}")

        total = sum(b.loans for b in banks)
        return sum(
            b.loans * b.lending_pass_through(tightening=tightening) for b in banks
        ) / total

    def weighted_deposit_pass_through(
        self, *, tightening: bool, group: BankGroup | None = None
    ) -> float:
        banks = self.banks if group is None else self.by_group(group)
        if not banks:
            raise EngineError(f"no banks in group {group}")

        total = sum(b.deposits for b in banks)
        return sum(
            b.deposits * b.deposit_pass_through(tightening=tightening) for b in banks
        ) / total

    def effective_rates(
        self, policy_rate: float, baseline_policy_rate: float
    ) -> dict[str, float]:
        """System-wide rates a household would face.

        Loan-weighted for lending, deposit-weighted for deposits, because
        a household's borrowing cost depends on where the credit is and
        its deposit return on where the savings are.
        """
        move = policy_rate - baseline_policy_rate
        tightening = move >= 0

        lending = self.weighted_lending_pass_through(tightening=tightening)
        deposit = self.weighted_deposit_pass_through(tightening=tightening)

        spread_lending = sum(b.loans * b.lending_spread for b in self.banks) / self.total_loans
        total_deposits = sum(b.deposits for b in self.banks)
        spread_deposit = (
            sum(b.deposits * b.deposit_spread for b in self.banks) / total_deposits
        )

        return {
            "lending_rate": float(baseline_policy_rate + spread_lending + move * lending),
            "deposit_rate": float(
                max(baseline_policy_rate + spread_deposit + move * deposit, 0.0)
            ),
            "lending_pass_through": lending,
            "deposit_pass_through": deposit,
            "tightening": tightening,
        }

    def stressed_share(self) -> float:
        """Share of system loans held by capital-constrained banks."""
        constrained = sum(b.loans for b in self.banks if b.is_capital_constrained)
        return constrained / self.total_loans

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "n_banks": len(self.banks),
            "total_assets": self.total_assets,
            "group_shares": {
                g.value: round(self.group_share(g), 4)
                for g in BankGroup
                if self.by_group(g)
            },
            "stressed_loan_share": round(self.stressed_share(), 4),
            "system_pass_through": {
                "lending_tightening": round(
                    self.weighted_lending_pass_through(tightening=True), 4
                ),
                "lending_easing": round(
                    self.weighted_lending_pass_through(tightening=False), 4
                ),
                "deposit_tightening": round(
                    self.weighted_deposit_pass_through(tightening=True), 4
                ),
                "deposit_easing": round(
                    self.weighted_deposit_pass_through(tightening=False), 4
                ),
            },
            "banks": [b.to_ledger_dict() for b in self.banks],
        }


# ---- calibration targets --------------------------------------------------

RBI_BULLETIN = (
    "RBI Bulletin, easing cycle February 2025 to May 2026, weighted average "
    "lending rate on fresh rupee loans against a cumulative 125bp repo cut"
)


class TransmissionTarget(BaseModel):
    """A published pass-through figure the system should reproduce."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    group: BankGroup
    lending_pass_through: float
    deposit_pass_through: float
    tightening: bool
    tolerance: float = Field(default=0.15, gt=0)
    source: str
    confidence: Confidence


#: Derived from the published rate cuts divided by the 125bp repo cut.
#: Foreign 1.24/1.25, private 1.08/1.25, public 0.66/1.25 on lending;
#: 0.90, 0.46 and 0.53 respectively on deposits.
EASING_TARGETS: tuple[TransmissionTarget, ...] = (
    TransmissionTarget(
        group=BankGroup.PUBLIC,
        lending_pass_through=0.53,
        deposit_pass_through=0.42,
        tightening=False,
        source=RBI_BULLETIN,
        confidence=Confidence.DERIVED,
    ),
    TransmissionTarget(
        group=BankGroup.PRIVATE,
        lending_pass_through=0.86,
        deposit_pass_through=0.37,
        tightening=False,
        source=RBI_BULLETIN,
        confidence=Confidence.DERIVED,
    ),
    TransmissionTarget(
        group=BankGroup.FOREIGN,
        lending_pass_through=0.99,
        deposit_pass_through=0.72,
        tightening=False,
        source=RBI_BULLETIN,
        confidence=Confidence.DERIVED,
    ),
)


def evaluate_transmission(
    system: BankingSystem, targets: tuple[TransmissionTarget, ...] = EASING_TARGETS
) -> dict[str, Any]:
    """Compare the system's pass-through against the published figures."""
    results = []
    for target in targets:
        if not system.by_group(target.group):
            continue

        lending = system.weighted_lending_pass_through(
            tightening=target.tightening, group=target.group
        )
        deposit = system.weighted_deposit_pass_through(
            tightening=target.tightening, group=target.group
        )
        lending_gap = lending - target.lending_pass_through
        deposit_gap = deposit - target.deposit_pass_through

        results.append(
            {
                "group": target.group.value,
                "lending_target": target.lending_pass_through,
                "lending_modelled": round(lending, 4),
                "lending_gap": round(lending_gap, 4),
                "lending_passed": abs(lending_gap) <= target.tolerance,
                "deposit_target": target.deposit_pass_through,
                "deposit_modelled": round(deposit, 4),
                "deposit_gap": round(deposit_gap, 4),
                "deposit_passed": abs(deposit_gap) <= target.tolerance,
                "source": target.source,
                "confidence": target.confidence.value,
            }
        )

    n_checks = 2 * len(results)
    n_passed = sum(r["lending_passed"] + r["deposit_passed"] for r in results)

    return {
        "n_checks": n_checks,
        "n_passed": n_passed,
        "loss": sum(r["lending_gap"] ** 2 + r["deposit_gap"] ** 2 for r in results),
        "results": results,
    }


# ---- the Indian banking system --------------------------------------------


def _bank(
    name: str,
    group: BankGroup,
    assets: float,
    *,
    eblr: float,
    retail: float,
    npa: float,
    capital_ratio: float,
    loan_ratio: float = 0.62,
    deposit_ratio: float = 0.78,
) -> CommercialBank:
    loans = assets * loan_ratio
    return CommercialBank(
        name=name,
        group=group,
        assets=assets,
        loans=loans,
        deposits=assets * deposit_ratio,
        capital=loans * capital_ratio,
        eblr_share=eblr,
        retail_deposit_share=retail,
        npa_ratio=npa,
    )


#: Twelve banks spanning the three RBI groups. Asset sizes are order of
#: magnitude rather than exact, and the characteristics follow the pattern
#: the RBI documents: public sector banks with lower external benchmark
#: shares, higher retail deposit dependence and higher stress; private
#: banks the reverse; foreign banks at the extreme.
#:
#: Individual bank figures are illustrative. What is calibrated is the
#: group-level pass-through, which is the quantity the RBI publishes and
#: therefore the quantity that can be checked.
INDIAN_BANKING_SYSTEM = BankingSystem(
    banks=(
        # Public sector: roughly sixty percent of system credit.
        _bank("SBI", BankGroup.PUBLIC, 6_200_000, eblr=0.42, retail=0.72, npa=0.022, capital_ratio=0.140),
        _bank("PNB", BankGroup.PUBLIC, 1_600_000, eblr=0.38, retail=0.78, npa=0.048, capital_ratio=0.128),
        _bank("Bank of Baroda", BankGroup.PUBLIC, 1_500_000, eblr=0.40, retail=0.75, npa=0.031, capital_ratio=0.132),
        _bank("Canara Bank", BankGroup.PUBLIC, 1_400_000, eblr=0.36, retail=0.80, npa=0.042, capital_ratio=0.125),
        _bank("Union Bank", BankGroup.PUBLIC, 1_300_000, eblr=0.35, retail=0.79, npa=0.045, capital_ratio=0.126),
        _bank("Bank of India", BankGroup.PUBLIC, 900_000, eblr=0.34, retail=0.81, npa=0.052, capital_ratio=0.121),
        # Private sector: roughly thirty five percent.
        _bank("HDFC Bank", BankGroup.PRIVATE, 3_900_000, eblr=0.72, retail=0.55, npa=0.013, capital_ratio=0.185),
        _bank("ICICI Bank", BankGroup.PRIVATE, 2_400_000, eblr=0.70, retail=0.52, npa=0.018, capital_ratio=0.176),
        _bank("Axis Bank", BankGroup.PRIVATE, 1_500_000, eblr=0.68, retail=0.54, npa=0.021, capital_ratio=0.168),
        _bank("Kotak Mahindra", BankGroup.PRIVATE, 700_000, eblr=0.66, retail=0.50, npa=0.015, capital_ratio=0.208),
        # Foreign: small share, strongest transmission.
        _bank("Citibank India", BankGroup.FOREIGN, 250_000, eblr=0.88, retail=0.28, npa=0.011, capital_ratio=0.196),
        _bank("HSBC India", BankGroup.FOREIGN, 280_000, eblr=0.86, retail=0.30, npa=0.014, capital_ratio=0.190),
    )
)