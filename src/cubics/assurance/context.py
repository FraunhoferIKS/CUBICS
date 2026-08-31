"""Context dimension management for the CUBICS assurance framework.

Copyright©[2026] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html), or (at your option) any later version.

Contact: benjamin.herd@iks.fraunhofer.de


"""

from __future__ import annotations

import logging
from functools import reduce
from typing import List

from cubics.sl import MultiOp

logger = logging.getLogger(__name__)


class Context:
    """Holds SL opinions over independent context dimensions and derives the joint
    situation opinion via SL conjunction.

    Each *dimension* (e.g. Rain, Wind, Time) is represented by a :class:`MultiOp`
    whose states are the discrete values that dimension can take.  The Cartesian
    product of all dimensions yields a set of *situations*; the joint opinion over
    those situations is computed via repeated SL conjunction (``&``).

    Args:
        ops: One :class:`MultiOp` per context dimension.  The order is preserved
            and used consistently when generating situation labels.

    Attributes:
        ops: List of per-dimension opinions.
        ops_situations: Joint :class:`MultiOp` over the full Cartesian product.
        situations: Ordered string labels for each situation, e.g.
            ``"Yes_._High_._Day"``.

    Example:
        >>> o_rain = MultiOp(b=[0, 0], u=1, a=[0.5, 0.5], domain="Rain", states=["Yes", "No"])
        >>> o_time = MultiOp(b=[0, 0], u=1, a=[0.5, 0.5], domain="Time", states=["Day", "Night"])
        >>> ctx = Context(ops=[o_rain, o_time])
        >>> ctx.situations
        ['Yes_._Day', 'Yes_._Night', 'No_._Day', 'No_._Night']
    """

    ops: List[MultiOp]
    ops_situations: MultiOp
    situations: List[str]

    def __init__(self, ops: List[MultiOp]) -> None:
        self.ops = ops
        self.update_situations()

    def update_situations(self) -> None:
        """Recompute the joint situation opinion from the current per-dimension opinions.

        Called automatically on construction and whenever a dimension opinion is
        updated via :meth:`~cubics.assurance.contract.Contract.update_context_opinion`.
        """
        self.ops_situations = reduce(lambda o1, o2: o1 & o2, self.ops)
        logger.debug("omega_S: %s", self.ops_situations)
        self.situations = self.ops_situations.states

    def get_situations(self, dims: List[str]) -> List[str]:
        """Return all situation labels that contain every value in *dims*.

        Args:
            dims: One or more dimension-value strings (e.g. ``["Rain", "Day"]``).

        Returns:
            Subset of :attr:`situations` whose labels include all of *dims*.
        """
        return [s for s in self.situations if all(dim in s for dim in dims)]

    def get_situation_idxs(self, dims: List[str]) -> List[int]:
        """Return the indices of all situations that contain every value in *dims*.

        Args:
            dims: One or more dimension-value strings.

        Returns:
            List of integer indices into :attr:`situations`.
        """
        return [i for i, s in enumerate(self.situations) if all(dim in s for dim in dims)]
