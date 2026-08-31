"""Monte-Carlo simulation engine for the CUBICS assurance framework.

Copyright©[2026] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html), or (at your option) any later version.

Contact: benjamin.herd@iks.fraunhofer.de

"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import List

import numpy as np
import numpy.typing as npt
from cubics.sl import MultiOp

from cubics.assurance.context import Context
from cubics.assurance.contract import Conditional, Contract

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Captured history from a :func:`run_simulation` call.

    Attributes:
        history_R: Maps each situation label to a list of cumulative success
            counts, one entry per milestone.
        history_S: Maps each situation label to a list of cumulative failure
            counts, one entry per milestone.
        marginal_history: Marginal guarantee :class:`MultiOp` at each milestone.
        milestones: Iteration indices at which snapshots were recorded (e.g.
            ``[0, 50, 500, 5000]``).
        situations: Ordered situation labels matching the keys of
            :attr:`history_R` and :attr:`history_S`.
    """

    history_R: dict[str, List[float]]
    history_S: dict[str, List[float]]
    marginal_history: List[MultiOp]
    milestones: List[int]
    situations: List[str]


def run_simulation(
    context: Context,
    conditional: Conditional,
    situation_probs: List[float],
    failure_rates: List[float],
    num_runs: int = 5000,
    seed: int = 0,
    prior_weight: float = 2.0,
    context_prior_weight: float = 2.0,
) -> SimulationResult:
    """Run the CUBICS Monte-Carlo simulation and return captured milestone history.

    Each iteration:

    1. Samples a situation from the ground-truth distribution
       (``situation_probs``).
    2. Updates the context opinion using cumulative frequency counts.
    3. Simulates a pass/fail outcome for the sampled situation.
    4. Recomputes the per-situation conditional opinion from running counts.

    Snapshots of all counts and the marginal guarantee are recorded at
    milestones ``[0, 50, 500, num_runs]``.

    Args:
        context: Operational context with SL opinions per dimension.  Modified
            in place during the simulation; pass a fresh instance each call.
        conditional: Near-vacuous prior conditional guarantee opinions.  Modified
            in place during the simulation.
        situation_probs: Ground-truth situation sampling weights.  Must sum to
            approximately 1.0 and have the same length as *failure_rates*.
        failure_rates: Ground-truth per-situation failure rates.
        num_runs: Total number of simulation iterations.  Defaults to 5000.
        seed: Random seed for full reproducibility.  Defaults to 0.
        prior_weight: SL prior weight W for conditional opinion updates.
            Defaults to 2.0.
        context_prior_weight: Prior weight used when updating context opinions
            from observed situation frequencies.  Defaults to 2.0.

    Returns:
        A :class:`SimulationResult` containing per-milestone success/failure
        counts, marginal guarantee opinions, and the milestone list.

    Raises:
        ValueError: If ``situation_probs`` and ``failure_rates`` have different
            lengths.

    Example:
        >>> ctx, cond, probs, fails = make_default_scenario()
        >>> result = run_simulation(ctx, cond, probs, fails, num_runs=500, seed=42)
        >>> print(result.milestones)
        [0, 50, 500]
    """
    if len(situation_probs) != len(failure_rates):
        raise ValueError(
            f"situation_probs length ({len(situation_probs)}) must match "
            f"failure_rates length ({len(failure_rates)})"
        )

    random.seed(seed)
    contract = Contract(context, conditional, prior_weight=prior_weight)
    num_situations = len(context.situations)
    W = prior_weight
    W_context = context_prior_weight

    milestones = [0, 50, 500, num_runs]
    # Deduplicate while preserving order (e.g. if num_runs < 500)
    milestones = sorted(set(m for m in milestones if m <= num_runs))

    history_R: dict[str, List[float]] = {lbl: [] for lbl in context.situations}
    history_S: dict[str, List[float]] = {lbl: [] for lbl in context.situations}
    marginal_history: List[MultiOp] = []

    R_counts = np.zeros(num_situations)
    S_counts = np.zeros(num_situations)
    context_evidence = [{state: 0.0 for state in op.states} for op in context.ops]

    logger.info("Simulating %d runs...", num_runs)

    for run_idx in range(num_runs + 1):
        # Snapshot at milestones
        if run_idx in milestones:
            for j, lbl in enumerate(context.situations):
                history_R[lbl].append(float(R_counts[j]))
                history_S[lbl].append(float(S_counts[j]))
            marginal_history.append(contract.get_marginal_guarantee())

        if run_idx == num_runs:
            break

        # Sample a situation
        sampled_idx = random.choices(range(num_situations), weights=situation_probs, k=1)[0]
        sampled_lbl = context.situations[sampled_idx]

        # Update context opinion from cumulative frequency evidence
        for i, op in enumerate(context.ops):
            for state in op.states:
                if state in sampled_lbl:
                    context_evidence[i][state] += 1.0
                    break
        for i, op in enumerate(context.ops):
            total = sum(context_evidence[i].values())
            new_b = [context_evidence[i][state] / (W_context + total) for state in op.states]
            new_u = W_context / (W_context + total)
            updated_op = MultiOp(
                b=new_b, u=new_u, a=op.a, domain=op.domain, states=op.states
            )
            contract.update_context_opinion(updated_op)

        # Simulate pass/fail
        if random.random() < failure_rates[sampled_idx]:
            S_counts[sampled_idx] += 1
        else:
            R_counts[sampled_idx] += 1

        # Recompute conditional from running counts (SL with Beta(1,1) prior)
        r = R_counts[sampled_idx]
        s = S_counts[sampled_idx]
        contract.conditional.b_Z_given_XY[sampled_idx][0] = r / (W + r + s)
        contract.conditional.b_Z_given_XY[sampled_idx][1] = s / (W + r + s)
        contract.conditional.u_Z_given_XY[sampled_idx] = W / (W + r + s)

    # Log final per-situation summaries
    for lbl in context.situations:
        g = contract.get_guarantees([lbl])[0]
        logger.info("  %s: (b=%.2f, d=%.2f, u=%.2f)", lbl, g.b, g.d, g.u)
    final_risk = contract.get_marginal_guarantee()
    logger.info("Final marginal guarantee: %s", final_risk)

    return SimulationResult(
        history_R=history_R,
        history_S=history_S,
        marginal_history=marginal_history,
        milestones=milestones,
        situations=list(context.situations),
    )


def make_default_scenario() -> tuple[Context, Conditional, List[float], List[float]]:
    """Return the canonical Rain × Wind × Time scenario used in the paper.

    Three binary context dimensions (Rain, Wind, Time) yield 8 situations.
    All context opinions start fully uncertain; conditionals start near-vacuous
    so the simulation learns everything from observed data.

    Returns:
        A tuple of ``(context, conditional, situation_probs, failure_rates)``
        ready to be passed directly to :func:`run_simulation`.

    Example:
        >>> ctx, cond, probs, fails = make_default_scenario()
        >>> len(ctx.situations)
        8
    """
    o_rain = MultiOp(b=[0, 0], u=1, a=[0.5, 0.5], domain="Rain", states=["Yes", "No"])
    o_wind = MultiOp(b=[0, 0], u=1, a=[0.5, 0.5], domain="Wind", states=["High", "Low"])
    o_time = MultiOp(b=[0, 0], u=1, a=[0.5, 0.5], domain="Time", states=["Day", "Night"])

    context = Context(ops=[o_rain, o_wind, o_time])

    conditional = Conditional.vacuous(len(context.situations))

    situation_probs = [0.1, 0.2, 0.05, 0.15, 0.1, 0.1, 0.2, 0.1]
    failure_rates = [0.40, 0.85, 0.30, 0.65, 0.15, 0.50, 0.05, 0.40]

    return context, conditional, situation_probs, failure_rates
