"""Safety contract representation for the CUBICS assurance framework.

Copyright©[2026] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html), or (at your option) any later version.

Contact: benjamin.herd@iks.fraunhofer.de


"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import reduce
from typing import List

import numpy as np
import numpy.typing as npt
from cubics.sl import BinOp, MultiOp

from cubics.assurance.context import Context

logger = logging.getLogger(__name__)


@dataclass
class Conditional:
    """Per-situation conditional guarantee opinions stored as parallel arrays.

    Row *i* corresponds to ``context.situations[i]`` in the parent
    :class:`Contract`.

    Args:
        b_Z_given_XY: Belief/disbelief array of shape ``(num_situations, 2)``.
            Column 0 is belief in the guarantee; column 1 is disbelief.
        u_Z_given_XY: Uncertainty array of shape ``(num_situations,)``.

    Example:
        >>> cond = Conditional.vacuous(8)
        >>> cond.b_Z_given_XY.shape
        (8, 2)
    """

    b_Z_given_XY: npt.NDArray[np.float64]  # shape (num_situations, 2)
    u_Z_given_XY: npt.NDArray[np.float64]  # shape (num_situations,)

    @classmethod
    def vacuous(cls, num_situations: int) -> "Conditional":
        """Construct a near-vacuous (maximally uncertain) prior.

        Uses a tiny belief ``1e-4`` so that SL operations that require non-zero
        belief remain well-defined while the opinion is still effectively
        uninformative.

        Args:
            num_situations: Number of situations in the parent :class:`Context`.

        Returns:
            A :class:`Conditional` with near-vacuous opinions for all situations.
        """
        b = np.zeros((num_situations, 2))
        b[:, 0] = 1e-4
        u = np.full(num_situations, 1.0 - 1e-4)
        return cls(b_Z_given_XY=b, u_Z_given_XY=u)


class Contract:
    """Pairs a :class:`Context` with a :class:`Conditional` to represent a safety contract.

    Answers: *given where the system is operating (context), what is the
    probability the system meets its guarantee?*

    Args:
        context: The operational context opinion.
        conditional: Per-situation conditional guarantee opinions.
        prior_weight: SL prior weight W used when updating conditional opinions
            from raw evidence.  Defaults to 2.0 (equivalent to a Beta(1, 1)
            uniform prior).

    Attributes:
        risk: Marginal guarantee opinion; populated by :meth:`get_marginal_guarantee`.
    """

    def __init__(
        self,
        context: Context,
        conditional: Conditional,
        prior_weight: float = 2.0,
    ) -> None:
        self.context = context
        self.conditional = conditional
        self.prior_weight = prior_weight
        self.risk: MultiOp | None = None

    def get_guarantees(self, situation: List[str]) -> List[BinOp]:
        """Return BinOp opinions for all situations matching the given label fragment.

        Args:
            situation: List of dimension-value strings identifying the target
                situations (e.g. ``["Rain", "Day"]``).

        Returns:
            One :class:`BinOp` per matching situation.
        """
        idx_situation = self.context.get_situation_idxs(situation)
        result = []
        for idx in idx_situation:
            cond = self.conditional.b_Z_given_XY[idx]
            o_guarantee = BinOp.from_bda(cond[0], cond[1])
            result.append(o_guarantee)
        return result

    def get_marginal_guarantee(self) -> MultiOp:
        """Compute the system-wide marginal guarantee via SL multinomial deduction.

        Deduction propagates the joint context opinion through the per-situation
        conditionals to produce a marginal opinion over ``[Guarantee, No Guarantee]``.

        Returns:
            A :class:`MultiOp` representing the system-level guarantee opinion.
        """
        omega_joint = reduce(lambda o1, o2: o1 & o2, self.context.ops)
        self.risk = omega_joint.deduce(
            b_Y_given_x=self.conditional.b_Z_given_XY,
            u_Y_given_x=self.conditional.u_Z_given_XY,
            states_Y=["Guarantee", "No Guarantee"],
            name_Y="Guarantee",
        )
        return self.risk

    def update_context_opinion(self, o_new: MultiOp) -> None:
        """Replace the context opinion for the domain of *o_new* and recompute the joint.

        Args:
            o_new: Updated opinion for one context dimension.  Matched to the
                existing dimension by its ``domain`` attribute.
        """
        for i, op in enumerate(self.context.ops):
            if op.domain == o_new.domain:
                self.context.ops[i] = o_new
                self.context.update_situations()
                return

    def add_evidence_to_situational_guarantee(
        self,
        dims: List[str],
        successes: float,
        failures: float,
    ) -> None:
        """Update conditional opinions using fractional evidence allocation.

        When one observation cannot be pinned to a single situation (because only
        a subset of context dimensions is known), evidence is scaled by each
        matching situation's projected probability so the total weight added still
        sums to one observation.

        Args:
            dims: Dimension-value strings identifying the known context (e.g.
                ``["Rain"]`` if only the rain dimension is observed).
            successes: Number of successes in the observation (may be fractional).
            failures: Number of failures in the observation (may be fractional).
        """
        W = self.prior_weight
        omega_joint = reduce(lambda o1, o2: o1 & o2, self.context.ops)
        idx_situations = self.context.get_situation_idxs(dims)
        situations = self.context.get_situations(dims)

        for idx, lbl in zip(idx_situations, situations):
            p_situation = omega_joint.prob(lbl)
            succs = successes * p_situation
            fails = failures * p_situation
            logger.debug(
                "Updating situational guarantee for %s; proj. prob. = %.4f", lbl, p_situation
            )
            b_cond = self.conditional.b_Z_given_XY[idx]
            o_guarantee = BinOp.from_bda(b_cond[0], b_cond[1])
            o_guarantee = o_guarantee % BinOp.from_rsa(succs, fails)
            self.conditional.b_Z_given_XY[idx][0] = o_guarantee.b
            self.conditional.b_Z_given_XY[idx][1] = o_guarantee.d
            self.conditional.u_Z_given_XY[idx] = o_guarantee.u
