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

from collections.abc import Sequence
from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from moirai.core.exceptions import EngineError
from moirai.core.logging import get_logger
from moirai.engine.financial.game import (
    Confidence,
    Game,
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
    current_income_growth: float | None = Field(
        default=None,
        description=(
            "Annual household income growth the economy sits at before a "
            "shock, as a decimal. Needed only when this bank originates the "
            "household-facing path; None refuses rather than guessing."
        ),
    )

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

    def external_gap(self, rate: float, other_rates: Sequence[float]) -> float:
        """The rate differential the external objective penalises.

        Measured against the mean of the other banks' rates. Capital flows
        respond to a differential against a global rate, not to an average
        distance from each rate separately, and the mean is the natural
        reference for an exchange rate channel. It also keeps the loss
        linear-quadratic, so the network's reaction system is its exact
        first order condition (ADR 011).

        Zero for a bank without an external objective. Signed, but only
        its square enters the loss.
        """
        if self.external_weight == 0 or not other_rates:
            return 0.0
        return rate - float(np.mean(other_rates))

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

        `external_gap` must come from `external_gap()`, the differential
        against the mean of the other banks' rates. Any other definition
        makes this loss disagree with the reaction system the solvers
        use, and the solved equilibrium stops being one (ADR 011).

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
    output_weight=0.40,
    external_weight=0.0,
    weight_confidence=Confidence.DERIVED,
    smoothing_weight=0.15,
    weight_note=(
        "Derived rather than assumed. A partial-adjustment policy rule fitted "
        "to the observed funds rate over 1986-2007 at quarterly frequency "
        "gives an output response of 0.372 against an inflation response of "
        "0.153. Three qualifications: the rule explains twelve percent of "
        "quarterly rate changes; the HP-filtered variant did not produce an "
        "interior estimate and the two detrending methods disagree by 0.29; "
        "and the sample is the Great Moderation, when inflation was near "
        "target throughout, which gives the inflation gap little variation "
        "and makes a dual-mandate bank look output-focused. The earlier "
        "assumed value of 1.00 moved the equilibrium by 145 basis points "
        "across its plausible range, which is why it was worth estimating."
    ),
    current_rate=0.0425,
    current_inflation=0.030,
    # Unsourced. Carried over from the scenario runner's former default.
    current_income_growth=0.02,
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
    # Unsourced. Carried over from the baseline in run_pipeline_india.py.
    current_income_growth=0.065,
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
        external = home.external_gap(home_rate, (foreign_rate,))
        return -home.loss(inflation, output, home_rate, external_gap=external)

    def foreign_payoff(profile: tuple[float, ...]) -> float:
        home_rate, foreign_rate = profile
        _, _, inflation, output = outcomes_for(home_rate, foreign_rate)
        external = foreign.external_gap(foreign_rate, (home_rate,))
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
# ---- analytic solution ----------------------------------------------------


class AnalyticSolution(BaseModel):
    """An exact equilibrium, solved rather than searched.

    A quadratic loss in a linear economy has linear first order
    conditions, so the fixed point is the solution of a two by two linear
    system. That is exact, instant, and unlike a grid it can resolve a
    difference smaller than the step size, which is what makes the gain
    from coordination measurable at all.

    It is also more honest about failure. A singular system means the
    banks' reaction functions are parallel, so either no equilibrium
    exists or a continuum does, and a grid search would silently return
    whichever profile the tie-break happened to reach.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rates: dict[str, float]
    losses: dict[str, float]
    concept: str
    condition_number: float = Field(
        description="Of the reaction system. Large means near-parallel responses."
    )
    is_well_conditioned: bool
    note: str = ""

    @property
    def total_loss(self) -> float:
        return sum(self.losses.values())

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "concept": self.concept,
            "rates": {k: round(v, 6) for k, v in self.rates.items()},
            "losses": {k: round(v, 8) for k, v in self.losses.items()},
            "total_loss": round(self.total_loss, 8),
            "condition_number": round(self.condition_number, 2),
            "is_well_conditioned": self.is_well_conditioned,
            "note": self.note,
        }


def _loss_coefficients(
    bank: CentralBank,
    other: CentralBank,
    spillovers: SpilloverParameters,
    *,
    is_home: bool,
) -> tuple[float, float, float]:
    """Quadratic coefficients of one bank's loss in (own_rate, other_rate).

    Returns (a, b, c) such that the loss is

        a * own^2 + b * own * other + c * own + constant

    so the first order condition is 2a*own + b*other + c = 0. Derived by
    substituting the linear transmission equations into the quadratic loss
    and collecting terms. The band penalty is excluded here because it is
    piecewise and would break linearity; it is checked afterwards instead.
    """
    d_inf = spillovers.domestic_inflation_effect
    d_out = spillovers.domestic_output_effect
    demand = spillovers.demand_spillover
    passthrough = spillovers.exchange_passthrough if bank.external_weight > 0 else 0.0

    # The exchange rate term has the same sign for both banks. Each bank's
    # inflation falls when it tightens by more than the other, because its
    # currency appreciates. The pair's orientation is already carried by
    # `differential` in `_losses_at` and `build_policy_game`; flipping the
    # sign here as well counted it twice and gave the foreign bank a
    # reaction function in which relative tightening raised its inflation.
    # `is_home` is kept so callers are unchanged. See ADR 015.
    sign = 1.0

    # inflation = pi0 - d_inf*(r - r0) - sign*passthrough*((r - r0) - (o - o0))
    inf_own = -(d_inf + sign * passthrough)
    inf_other = sign * passthrough
    inf_const = (
        bank.current_inflation
        - bank.inflation_target
        + (d_inf + sign * passthrough) * bank.current_rate
        - sign * passthrough * other.current_rate
    )

    # output_gap = -d_out*(r - r0) - demand*(o - o0)
    out_own = -d_out
    out_other = -demand
    out_const = d_out * bank.current_rate + demand * other.current_rate

    wi, wo, we, ws = (
        bank.inflation_weight,
        bank.output_weight,
        bank.external_weight,
        bank.smoothing_weight,
    )

    a = wi * inf_own**2 + wo * out_own**2 + we + ws
    b = 2 * wi * inf_own * inf_other + 2 * wo * out_own * out_other - 2 * we
    c = (
        2 * wi * inf_own * inf_const
        + 2 * wo * out_own * out_const
        - 2 * ws * bank.current_rate
    )
    return a, b, c


def analytic_nash(
    home: CentralBank,
    foreign: CentralBank,
    *,
    spillovers: SpilloverParameters | None = None,
) -> AnalyticSolution:
    """Solve the Nash equilibrium exactly.

    Each bank's first order condition is linear in both rates, so the
    equilibrium solves

        [2a_h   b_h] [r_h]   [-c_h]
        [b_f   2a_f] [r_f] = [-c_f]

    The condition number is reported because a near-singular system means
    the two reaction functions are almost parallel, and the equilibrium is
    then extremely sensitive to the assumed weights. That is worth seeing
    before quoting the answer to four decimal places.
    """
    spillovers = spillovers or SpilloverParameters()

    a_h, b_h, c_h = _loss_coefficients(home, foreign, spillovers, is_home=True)
    a_f, b_f, c_f = _loss_coefficients(foreign, home, spillovers, is_home=False)

    matrix = np.array([[2 * a_h, b_h], [b_f, 2 * a_f]], dtype=float)
    rhs = np.array([-c_h, -c_f], dtype=float)

    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > 1e10:
        raise EngineError(
            f"the reaction system is singular (condition {condition:.2e}). "
            f"The banks' best responses are parallel, so the equilibrium is "
            f"either non-existent or a continuum. Check the weights."
        )

    solution = np.linalg.solve(matrix, rhs)
    rates = {home.name: float(solution[0]), foreign.name: float(solution[1])}

    losses = _losses_at(home, foreign, solution[0], solution[1], spillovers)

    note = "exact solution of the linear reaction system"
    for bank, rate in ((home, solution[0]), (foreign, solution[1])):
        if rate < 0:
            note += f"; {bank.name} solves to a negative rate, below the usual floor"

    result = AnalyticSolution(
        rates=rates,
        losses=losses,
        concept="nash_analytic",
        condition_number=condition,
        is_well_conditioned=condition < 1e4,
        note=note,
    )

    log.info(
        "analytic_nash_solved",
        home=home.name,
        foreign=foreign.name,
        rates={k: round(v, 5) for k, v in rates.items()},
        condition=round(condition, 1),
    )
    return result


def analytic_cooperative(
    home: CentralBank,
    foreign: CentralBank,
    *,
    spillovers: SpilloverParameters | None = None,
) -> AnalyticSolution:
    """Minimise the sum of the two losses.

    Solved numerically rather than by hand-derived first order conditions.
    An earlier version differentiated the joint objective analytically and
    got the cross terms wrong, which produced a cooperative outcome worse
    than Nash. That is impossible by construction, since the cooperative
    problem minimises over a superset of what each bank controls
    individually, so the error was visible only because the impossibility
    was checked. It is now asserted below rather than assumed.

    The joint loss is smooth and low dimensional, so Nelder-Mead from the
    Nash point converges immediately and does not require the gradient
    that was the source of the mistake.
    """
    from scipy.optimize import minimize

    spillovers = spillovers or SpilloverParameters()
    nash = analytic_nash(home, foreign, spillovers=spillovers)

    def joint_loss(rates: np.ndarray) -> float:
        losses = _losses_at(home, foreign, float(rates[0]), float(rates[1]), spillovers)
        return sum(losses.values())

    start = np.array([nash.rates[home.name], nash.rates[foreign.name]])
    result = minimize(joint_loss, start, method="Nelder-Mead", tol=1e-12)

    if not result.success:
        raise EngineError(f"joint minimisation failed: {result.message}")

    home_rate, foreign_rate = float(result.x[0]), float(result.x[1])
    losses = _losses_at(home, foreign, home_rate, foreign_rate, spillovers)

    # The cooperative solution optimises over both rates jointly, so it
    # cannot be worse than an equilibrium in which each bank optimises over
    # one. A violation means the solver failed or the loss is not what the
    # Nash solver assumed it was.
    if sum(losses.values()) > nash.total_loss + 1e-9:
        raise EngineError(
            f"cooperative joint loss {sum(losses.values()):.9f} exceeds the "
            f"Nash joint loss {nash.total_loss:.9f}, which is impossible. "
            f"The two solvers disagree about the objective."
        )

    return AnalyticSolution(
        rates={home.name: home_rate, foreign.name: foreign_rate},
        losses=losses,
        concept="cooperative_numeric",
        condition_number=nash.condition_number,
        is_well_conditioned=nash.is_well_conditioned,
        note=(
            "minimises the joint loss by direct search; not individually "
            "rational in general, so it is not self-enforcing"
        ),
    )
def _losses_at(
    home: CentralBank,
    foreign: CentralBank,
    home_rate: float,
    foreign_rate: float,
    spillovers: SpilloverParameters,
) -> dict[str, float]:
    """Realised losses at a rate pair, including any band penalty."""
    home_move = home_rate - home.current_rate
    foreign_move = foreign_rate - foreign.current_rate
    differential = home_move - foreign_move

    home_inflation = (
        home.current_inflation
        - spillovers.domestic_inflation_effect * home_move
        - spillovers.exchange_passthrough * differential * (home.external_weight > 0)
    )
    foreign_inflation = (
        foreign.current_inflation
        - spillovers.domestic_inflation_effect * foreign_move
        + spillovers.exchange_passthrough * differential * (foreign.external_weight > 0)
    )
    home_output = (
        -spillovers.domestic_output_effect * home_move
        - spillovers.demand_spillover * foreign_move
    )
    foreign_output = (
        -spillovers.domestic_output_effect * foreign_move
        - spillovers.demand_spillover * home_move
    )

    return {
        home.name: home.loss(
            home_inflation,
            home_output,
            home_rate,
            external_gap=home.external_gap(home_rate, (foreign_rate,)),
        ),
        foreign.name: foreign.loss(
            foreign_inflation,
            foreign_output,
            foreign_rate,
            external_gap=foreign.external_gap(foreign_rate, (home_rate,)),
        ),
    }


def coordination_value(
    home: CentralBank,
    foreign: CentralBank,
    *,
    spillovers: SpilloverParameters | None = None,
) -> dict[str, Any]:
    """The exact gain from coordinating, and who captures it.

    A grid cannot resolve a gain smaller than its step size, which for
    twenty five basis points is most of the gains that actually arise
    between central banks in normal conditions. Solving exactly is what
    makes the question answerable.
    """
    nash = analytic_nash(home, foreign, spillovers=spillovers)
    cooperative = analytic_cooperative(home, foreign, spillovers=spillovers)

    by_bank = {
        name: nash.losses[name] - cooperative.losses[name]
        for name in nash.losses
    }

    return {
        "nash_rates": nash.rates,
        "cooperative_rates": cooperative.rates,
        "total_gain": nash.total_loss - cooperative.total_loss,
        "gain_by_bank": by_bank,
        "someone_loses": any(value < 0 for value in by_bank.values()),
        "well_conditioned": nash.is_well_conditioned and cooperative.is_well_conditioned,
    }

# ---- sensitivity to the assumed weights ------------------------------------


class WeightSensitivity(BaseModel):
    """How an equilibrium moves as an assumed weight varies.

    The mandates in this module are published and the transmission
    parameters are estimable, but the preference weights are neither. How
    much a central bank dislikes a point of output gap relative to a point
    of inflation is not a quantity anyone has measured, and a conclusion
    that depends sharply on it is a conclusion about the assumption.

    This is the same exercise as ordering sensitivity in the causal layer.
    A result that holds across the plausible range is robust. A result that
    changes sign inside it means the weight is doing the work, and the
    honest report is the range rather than the point.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bank: str
    parameter: str
    values: tuple[float, ...]
    rates: tuple[float, ...] = Field(description="Equilibrium rate at each value.")
    moves: tuple[float, ...] = Field(description="Basis points from the current rate.")
    baseline_value: float
    baseline_rate: float

    @property
    def rate_range(self) -> tuple[float, float]:
        return min(self.rates), max(self.rates)

    @property
    def spread_bp(self) -> float:
        return (max(self.rates) - min(self.rates)) * 10_000

    @property
    def sign_flips(self) -> bool:
        """Does the direction of the policy move change across the range?"""
        signs = {np.sign(m) for m in self.moves if abs(m) > 1.0}
        return len(signs) > 1

    @property
    def is_robust(self) -> bool:
        """No sign change and the spread stays under fifty basis points.

        Fifty is two standard policy increments. A conclusion that survives
        that much variation in an unmeasured parameter is worth quoting; one
        that does not should be reported as a range.
        """
        return not self.sign_flips and self.spread_bp < 50.0

    def to_ledger_dict(self) -> dict[str, Any]:
        return {
            "bank": self.bank,
            "parameter": self.parameter,
            "baseline_value": self.baseline_value,
            "baseline_rate": round(self.baseline_rate, 6),
            "rate_range": [round(r, 6) for r in self.rate_range],
            "spread_bp": round(self.spread_bp, 1),
            "sign_flips": self.sign_flips,
            "is_robust": self.is_robust,
            "values": list(self.values),
            "rates": [round(r, 6) for r in self.rates],
        }


def weight_sensitivity(
    home: CentralBank,
    foreign: CentralBank,
    *,
    vary: str,
    on: str,
    values: Sequence[float],
    observe: str | None = None,
    spillovers: SpilloverParameters | None = None,
) -> WeightSensitivity:
    """Re-solve the equilibrium across a range of one assumed weight.

    Parameters
    ----------
    vary
        Which weight to vary, for example "external_weight".
    on
        Which bank's weight is varied, by name.
    observe
        Whose equilibrium rate to report. Defaults to the bank whose weight
        is being varied, but the interesting case is often the other one:
        the question is usually whether *my* assumption about the RBI
        changes what the *Fed* does.
    """
    banks = {home.name: home, foreign.name: foreign}
    if on not in banks:
        raise EngineError(f"no bank named {on!r}; have {sorted(banks)}")
    if not hasattr(banks[on], vary):
        raise EngineError(f"{vary!r} is not a weight on CentralBank")
    if len(values) == 0:
        raise EngineError("at least one value is required")
    observed = observe or on
    if observed not in banks:
        raise EngineError(f"cannot observe {observed!r}; have {sorted(banks)}")

    rates: list[float] = []
    for value in values:
        adjusted = banks[on].model_copy(update={vary: value})
        pair = (
            (adjusted, foreign) if on == home.name else (home, adjusted)
        )
        solution = analytic_nash(*pair, spillovers=spillovers)
        rates.append(solution.rates[observed])

    baseline_value = float(getattr(banks[on], vary))
    baseline = analytic_nash(home, foreign, spillovers=spillovers)
    current = banks[observed].current_rate

    result = WeightSensitivity(
        bank=observed,
        parameter=f"{on}.{vary}",
        values=tuple(float(v) for v in values),
        rates=tuple(rates),
        moves=tuple((r - current) * 10_000 for r in rates),
        baseline_value=baseline_value,
        baseline_rate=baseline.rates[observed],
    )

    log.info(
        "weight_sensitivity_computed",
        varied=result.parameter,
        observed=observed,
        spread_bp=round(result.spread_bp, 1),
        robust=result.is_robust,
    )
    return result


def sensitivity_report(
    home: CentralBank,
    foreign: CentralBank,
    *,
    spillovers: SpilloverParameters | None = None,
    n_points: int = 9,
) -> dict[str, Any]:
    """Vary every assumed weight in turn and report which results survive.

    The ranges are deliberately wide. A narrow range around the chosen
    value would flatter the model: the point is to find out whether the
    conclusion holds across values another analyst might reasonably have
    picked, not to confirm that small perturbations do little.
    """
    ranges = {
        "output_weight": np.linspace(0.1, 1.5, n_points),
        "external_weight": np.linspace(0.0, 0.8, n_points),
        "smoothing_weight": np.linspace(0.0, 0.6, n_points),
    }

    results: list[WeightSensitivity] = []
    for bank in (home, foreign):
        for parameter, values in ranges.items():
            # Skip a weight the bank does not use: varying the Fed's
            # external weight from a baseline of zero is a different
            # question, and it is asked explicitly below.
            if parameter == "external_weight" and getattr(bank, parameter) == 0.0:
                continue
            for observed in (home.name, foreign.name):
                results.append(
                    weight_sensitivity(
                        home,
                        foreign,
                        vary=parameter,
                        on=bank.name,
                        values=values,
                        observe=observed,
                        spillovers=spillovers,
                    )
                )

    fragile = [r for r in results if not r.is_robust]
    return {
        "n_checks": len(results),
        "n_robust": sum(1 for r in results if r.is_robust),
        "fragile": [r.to_ledger_dict() for r in fragile],
        "worst_spread_bp": max((r.spread_bp for r in results), default=0.0),
        "any_sign_flip": any(r.sign_flips for r in results),
        "results": [r.to_ledger_dict() for r in results],
    }
