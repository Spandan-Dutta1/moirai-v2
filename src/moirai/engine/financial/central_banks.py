"""
Central banks as strategic players.

Every central bank in this module carries its actual published mandate,
because those mandates differ in ways that change the game. The Federal
Reserve is the only major central bank with a dual mandate, and only
fifteen to twenty percent of central banks worldwide have one. The ECB has
explicitly rejected adopting one. A model that gave every player the same
loss function would misrepresent the system it claims to describe.

What is sourced and what is assumed:

    inflation target      SOURCED   published, numeric
    tolerance band        SOURCED   where the mandate specifies one
    mandate structure     SOURCED   from statute or published strategy
    output weight         ASSUMED   derived from the mandate structure
    external weight       ASSUMED   except the RBI, which is DERIVED
    spillover magnitude   DERIVED   estimable from a VAR

The weights are the weak point and are marked accordingly. How much a
central bank dislikes a point of inflation relative to a point of output
gap is a preference, not a quantity anyone has measured. What can be
defended is the *ordering*: a bank with a hierarchical price stability
mandate weights inflation more heavily than one with a dual mandate. The
numbers implement that ordering; they do not discover it.

The RBI's external weight is the exception. Its own Report on Currency
and Finance states that in an open economy setting foreign exchange
reserves and associated liquidity management are crucial, and that
sterilisation capacity needs enhancing to deal with surges in capital
flows. That is published evidence for an objective the Federal Reserve
does not share, so the asymmetry in this game is documented rather than
invented.

A warning about the Lucas critique. Spillover magnitudes estimated from a
VAR are estimated under the historical policy regime. This layer is
defensible for marginal strategic questions inside an existing framework
and not for regime change, because private expectations would adapt to a
genuinely new rule and the estimated coefficients would no longer describe
the economy.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from moirai.core.exceptions import EngineError
from moirai.core.logging import get_logger
from moirai.engine.financial.game import (
    Confidence,
    Game,
    Outcome,
    Player,
    cooperation_gain,
    nash_equilibria,
    stackelberg_equilibrium,
)

log = get_logger(__name__)


class MandateType(StrEnum):
    """How a central bank's objectives are legally ordered."""

    #: Price stability and employment as coequal goals. The Federal
    #: Reserve is the only major central bank with this structure.
    DUAL = "dual"
    #: Price stability dominant, other goals subordinate to it.
    HIERARCHICAL = "hierarchical"
    #: Price stability as the sole primary objective.
    PRICE_STABILITY = "price_stability"
    #: Growth, employment and financial stability without a single
    #: numeric inflation target. The PBoC is the main example.
    MULTIPLE_OBJECTIVES = "multiple_objectives"


class CentralBank(BaseModel):
    """A central bank, its published mandate and its assumed preferences."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    jurisdiction: str = Field(min_length=1)

    # ---- sourced ----
    inflation_target: float = Field(description="Annual, as a decimal.")
    tolerance_lower: float | None = None
    tolerance_upper: float | None = None
    mandate: MandateType
    target_measure: str = Field(default="", description="Which index the target uses.")
    mandate_source: str = Field(min_length=1)

    # ---- assumed, ordered by the sourced mandate ----
    inflation_weight: float = Field(default=1.0, gt=0)
    output_weight: float = Field(default=0.5, ge=0)
    external_weight: float = Field(
        default=0.0,
        ge=0,
        description="Weight on the exchange rate or capital flow channel.",
    )
    smoothing_weight: float = Field(
        default=0.1,
        ge=0,
        description="Dislike of large rate changes. Central banks move gradually.",
    )
    weight_confidence: Confidence = Confidence.ASSUMED
    weight_note: str = ""

    # ---- current state ----
    current_rate: float = Field(default=0.0, description="Policy rate as a decimal.")
    current_inflation: float = Field(default=0.0)
    current_output_gap: float = Field(default=0.0)

    @model_validator(mode="after")
    def _band_is_ordered_around_the_target(self) -> CentralBank:
        if self.tolerance_lower is not None and self.tolerance_upper is not None:
            if self.tolerance_lower >= self.tolerance_upper:
                raise ValueError(f"{self.name}: tolerance band is inverted")
            if not self.tolerance_lower <= self.inflation_target <= self.tolerance_upper:
                raise ValueError(
                    f"{self.name}: target {self.inflation_target} lies outside "
                    f"its own band"
                )
        return self

    @property
    def has_band(self) -> bool:
        return self.tolerance_lower is not None and self.tolerance_upper is not None

    def within_band(self, inflation: float) -> bool:
        """Is inflation inside the tolerance band?

        A band changes behaviour, not just presentation. The RBI is
        mandated to explain a breach of two to six percent to the
        government, so the cost of missing rises sharply at the edge
        rather than smoothly around the midpoint.
        """
        if not self.has_band:
            return abs(inflation - self.inflation_target) < 0.005
        return self.tolerance_lower <= inflation <= self.tolerance_upper

    def loss(
        self,
        inflation: float,
        output_gap: float,
        rate: float,
        *,
        external_gap: float = 0.0,
    ) -> float:
        """Quadratic loss over the bank's objectives.

        Quadratic because it is the standard form and because it implies
        symmetric aversion: overshooting the target by a point is as bad as
        undershooting by one. That is a real assumption. Central banks
        arguably dislike overshoots more, and a loss function capturing
        that would change the equilibrium.

        Returns a loss, so a payoff is its negative.
        """
        inflation_gap = inflation - self.inflation_target
        total = (
            self.inflation_weight * inflation_gap**2
            + self.output_weight * output_gap**2
            + self.external_weight * external_gap**2
            + self.smoothing_weight * (rate - self.current_rate) ** 2
        )

        # A band breach carries an explicit accountability cost.
        if self.has_band and not self.within_band(inflation):
            breach = min(
                abs(inflation - self.tolerance_lower),
                abs(inflation - self.tolerance_upper),
            )
            total += self.inflation_weight * breach**2

        return float(total)

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "jurisdiction": self.jurisdiction,
            "mandate": self.mandate.value,
            "inflation_target": self.inflation_target,
            "band": [self.tolerance_lower, self.tolerance_upper] if self.has_band else None,
            "mandate_source": self.mandate_source,
            "weights": {
                "inflation": self.inflation_weight,
                "output": self.output_weight,
                "external": self.external_weight,
                "smoothing": self.smoothing_weight,
            },
            "weight_confidence": self.weight_confidence.value,
            "weight_note": self.weight_note,
        }


# ---- the published mandates ----------------------------------------------

_FED_SOURCE = (
    "Statement on Longer-Run Goals and Monetary Policy Strategy, 2012 onward; "
    "Federal Reserve Reform Act 1977"
)
_ECB_SOURCE = "Treaty on the Functioning of the EU, Article 127; ECB strategy review 2021"
_RBI_SOURCE = (
    "RBI Act as amended 2016, flexible inflation targeting; target reviewed "
    "every five years by the Government of India"
)

FED = CentralBank(
    name="Federal Reserve",
    jurisdiction="US",
    inflation_target=0.02,
    mandate=MandateType.DUAL,
    target_measure="PCE price index",
    mandate_source=_FED_SOURCE,
    inflation_weight=1.0,
    output_weight=1.0,
    external_weight=0.0,
    smoothing_weight=0.15,
    weight_note=(
        "Output weight equal to inflation weight because the mandate is "
        "coequal rather than hierarchical. The Fed publishes no numeric "
        "employment target, holding that maximum employment is not directly "
        "measurable, so this weight cannot be calibrated the way the "
        "inflation target can. External weight is zero: the dollar's reserve "
        "status means the Fed faces little external constraint."
    ),
    current_rate=0.0425,
    current_inflation=0.030,
)

ECB = CentralBank(
    name="European Central Bank",
    jurisdiction="EA",
    inflation_target=0.02,
    mandate=MandateType.PRICE_STABILITY,
    target_measure="HICP",
    mandate_source=_ECB_SOURCE,
    inflation_weight=1.0,
    output_weight=0.35,
    external_weight=0.05,
    smoothing_weight=0.15,
    weight_note=(
        "Output weight well below the inflation weight because price "
        "stability is the primary objective and other goals are subordinate "
        "to it. The ECB has explicitly declined to adopt a dual mandate."
    ),
    current_rate=0.0200,
    current_inflation=0.021,
)

BANK_OF_ENGLAND = CentralBank(
    name="Bank of England",
    jurisdiction="UK",
    inflation_target=0.02,
    mandate=MandateType.HIERARCHICAL,
    target_measure="CPI",
    mandate_source="Bank of England Act 1998; annual remit from HM Treasury",
    inflation_weight=1.0,
    output_weight=0.40,
    external_weight=0.05,
    smoothing_weight=0.15,
    weight_note=(
        "Hierarchical: growth and employment are pursued subject to the "
        "inflation target rather than alongside it."
    ),
    current_rate=0.0375,
    current_inflation=0.034,
)

BANK_OF_JAPAN = CentralBank(
    name="Bank of Japan",
    jurisdiction="JP",
    inflation_target=0.02,
    mandate=MandateType.PRICE_STABILITY,
    target_measure="CPI",
    mandate_source="Bank of Japan Act; price stability target of 2 percent",
    inflation_weight=1.0,
    output_weight=0.45,
    external_weight=0.15,
    smoothing_weight=0.30,
    weight_note=(
        "High smoothing weight reflects decades of gradualism and the "
        "asymmetry of escaping deflation. External weight above the European "
        "banks because yen depreciation feeds import costs directly."
    ),
    current_rate=0.0075,
    current_inflation=0.020,
)

RBI = CentralBank(
    name="Reserve Bank of India",
    jurisdiction="IN",
    inflation_target=0.04,
    tolerance_lower=0.02,
    tolerance_upper=0.06,
    mandate=MandateType.HIERARCHICAL,
    target_measure="CPI Combined",
    mandate_source=_RBI_SOURCE,
    inflation_weight=1.0,
    output_weight=0.60,
    external_weight=0.40,
    smoothing_weight=0.20,
    weight_confidence=Confidence.DERIVED,
    weight_note=(
        "The external weight is derived rather than assumed. The RBI's own "
        "Report on Currency and Finance states that in an open economy "
        "setting foreign exchange reserves and associated liquidity "
        "management are crucial, and that sterilisation capacity needs "
        "enhancing to deal with surges in capital flows. That is published "
        "evidence for an objective the Federal Reserve does not share, and "
        "it is the asymmetry this game exists to model. The magnitude "
        "remains assumed; the presence of the objective does not."
    ),
    current_rate=0.0525,
    current_inflation=0.044,
)

MAJOR_CENTRAL_BANKS: tuple[CentralBank, ...] = (
    FED,
    ECB,
    BANK_OF_ENGLAND,
    BANK_OF_JAPAN,
    RBI,
)


# ---- the game -------------------------------------------------------------


class SpilloverParameters(BaseModel):
    """How one bank's rate decision reaches the other economy.

    The three channels a rate rise abroad transmits through:

        demand      weaker foreign demand lowers the home output gap
        exchange    capital flows out, the currency depreciates, imports
                    cost more, and inflation rises
        financial   global risk appetite tightens, raising domestic
                    borrowing costs regardless of the domestic policy rate

    The signs are not symmetric across the pair. A Fed tightening
    depreciates the rupee and raises Indian inflation; an RBI tightening
    has almost no effect on US prices. That asymmetry is the substance of
    the game rather than a simplification of it.

    Magnitudes here are illustrative defaults. `from_irf` builds them from
    an estimated impulse response instead, which is the honest source.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    demand_spillover: float = Field(
        default=0.25,
        ge=0,
        description="Points of foreign output gap per point of home tightening.",
    )
    exchange_passthrough: float = Field(
        default=0.30,
        ge=0,
        description="Points of foreign inflation per point of rate differential.",
    )
    domestic_output_effect: float = Field(default=1.20, ge=0)
    domestic_inflation_effect: float = Field(default=0.80, ge=0)
    confidence: Confidence = Confidence.ASSUMED
    note: str = ""

    def to_ledger_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def build_policy_game(
    home: CentralBank,
    foreign: CentralBank,
    *,
    spillovers: SpilloverParameters | None = None,
    rate_grid: tuple[float, ...] | None = None,
    title: str = "",
) -> Game:
    """A two-player game over policy rates.

    Each bank chooses a rate. Its inflation and output gap depend on its
    own choice and on the other bank's, through the spillover channels.
    The payoff is the negative of its loss.

    The rate grid is discrete in twenty five basis point steps because that
    is how policy rates actually move, and because a visible payoff surface
    is easier to trust than a continuous optimum whose shape nobody has
    looked at.
    """
    spillovers = spillovers or SpilloverParameters()

    if rate_grid is None:
        rate_grid = tuple(np.round(np.arange(0.0, 0.0825, 0.0025), 6))
    if len(rate_grid) < 2:
        raise EngineError("the rate grid needs at least two actions")

    def outcomes_for(home_rate: float, foreign_rate: float) -> tuple[float, ...]:
        """Inflation and output gap for each bank under a rate pair."""
        home_tightening = home_rate - home.current_rate
        foreign_tightening = foreign_rate - foreign.current_rate
        differential = home_tightening - foreign_tightening

        home_inflation = (
            home.current_inflation
            - spillovers.domestic_inflation_effect * home_tightening
            - spillovers.exchange_passthrough * differential * (home.external_weight > 0)
        )
        home_output = (
            -spillovers.domestic_output_effect * home_tightening
            - spillovers.demand_spillover * foreign_tightening
        )

        foreign_inflation = (
            foreign.current_inflation
            - spillovers.domestic_inflation_effect * foreign_tightening
            + spillovers.exchange_passthrough * differential * (foreign.external_weight > 0)
        )
        foreign_output = (
            -spillovers.domestic_output_effect * foreign_tightening
            - spillovers.demand_spillover * home_tightening
        )

        return home_inflation, home_output, foreign_inflation, foreign_output

    def home_payoff(profile: tuple[float, ...]) -> float:
        home_rate, foreign_rate = profile
        inflation, output, _, _ = outcomes_for(home_rate, foreign_rate)
        external = abs(home_rate - foreign_rate) if home.external_weight > 0 else 0.0
        return -home.loss(inflation, output, home_rate, external_gap=external)

    def foreign_payoff(profile: tuple[float, ...]) -> float:
        home_rate, foreign_rate = profile
        _, _, inflation, output = outcomes_for(home_rate, foreign_rate)
        external = abs(foreign_rate - home_rate) if foreign.external_weight > 0 else 0.0
        return -foreign.loss(inflation, output, foreign_rate, external_gap=external)

    return Game(
        title=title or f"{home.name} and {foreign.name}",
        players=(
            Player(
                name=home.name,
                actions=rate_grid,
                payoff=home_payoff,
                description=f"{home.mandate.value} mandate, target {home.inflation_target:.1%}",
            ),
            Player(
                name=foreign.name,
                actions=rate_grid,
                payoff=foreign_payoff,
                description=(
                    f"{foreign.mandate.value} mandate, "
                    f"target {foreign.inflation_target:.1%}"
                ),
            ),
        ),
    )


def analyse_pair(
    leader: CentralBank,
    follower: CentralBank,
    *,
    spillovers: SpilloverParameters | None = None,
) -> dict[str, Any]:
    """Solve a pair under Nash, Stackelberg and cooperation.

    The Stackelberg case takes the larger bank as leader. That is the
    structure the international system actually has: other central banks
    take Federal Reserve policy as given when setting their own, and the
    reverse is not true to anything like the same degree.
    """
    game = build_policy_game(leader, follower, spillovers=spillovers)

    equilibria = nash_equilibria(game)
    stackelberg = stackelberg_equilibrium(game, leader.name)
    gain = cooperation_gain(game)

    result = {
        "leader": leader.name,
        "follower": follower.name,
        "nash": [o.to_ledger_dict() for o in equilibria],
        "stackelberg": stackelberg.to_ledger_dict(),
        "cooperation": gain,
        "mandates": {
            leader.name: leader.to_ledger_dict(),
            follower.name: follower.to_ledger_dict(),
        },
        "spillovers": (spillovers or SpilloverParameters()).to_ledger_dict(),
    }

    log.info(
        "policy_pair_analysed",
        leader=leader.name,
        follower=follower.name,
        n_nash=len(equilibria),
        stackelberg_actions=stackelberg.actions,
        cooperation_gain=round(gain["gain"], 4),
    )
    return result