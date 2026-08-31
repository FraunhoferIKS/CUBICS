"""Subjective Logic opinion types for CUBICS.

Copyright©[2026] Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V. acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.  
This software is subject to the terms and conditions of the GNU GPLv2 (https://www.gnu.de/documents/gpl-2.0.de.html), or (at your option) any later version.

Contact: benjamin.herd@iks.fraunhofer.de

"""
from __future__ import annotations

import math
import numpy as np
from typing import List, Optional


# ---------------------------------------------------------------------------
# Binary opinion
# ---------------------------------------------------------------------------

class BinOp:
    """Binary Subjective Logic opinion (b, d, u, a).

    Attributes:
        b: Belief mass.
        d: Disbelief mass.
        u: Uncertainty mass.
        a: Base rate (prior probability).
    """

    def __init__(self, b: float, d: float, u: float, a: float, ci: float = 0.95) -> None:
        self.b = b
        self.d = d
        self.u = u
        self.a = a
        self.ci = ci

    # --- Factories ---

    @classmethod
    def from_bda(cls, b: float, d: float, a: float = 0.5, ci: float = 0.95) -> "BinOp":
        """Construct from belief + disbelief; uncertainty is derived."""
        return cls(b, d, 1.0 - b - d, a, ci=ci)

    @classmethod
    def from_rsa(cls, r: float, s: float, a: float = 0.5, ci: float = 0.95) -> "BinOp":
        """Construct from success / failure evidence counts."""
        total = r + s + 2.0
        return cls(r / total, s / total, 2.0 / total, a, ci=ci)

    # --- Operator ---

    def prob(self) -> float:
        """Expected probability: b + a * u."""
        return self.b + self.u * self.a

    def __str__(self) -> str:
        return (
            f"BinOp(b={self.b:.5f}, d={self.d:.5f}, u={self.u:.5f}, a={self.a:.5f},"
            f" E={self.prob():.4f})"
        )

    def __mod__(self, other: "BinOp") -> "BinOp":
        """Cumulative belief fusion (CBF)."""
        op1, op2 = self, other
        if math.isclose(op1.u, 0.0, abs_tol=1e-10) and math.isclose(op2.u, 0.0, abs_tol=1e-10):
            b = 0.5 * op1.b + 0.5 * op2.b
            d = 1.0 - b
            u = 0.0
            a = 0.5 * op1.a + 0.5 * op1.b
        else:
            k = op1.u + op2.u - op1.u * op2.u
            b = (op1.b * op2.u + op2.b * op1.u) / k
            d = (op1.d * op2.u + op2.d * op1.u) / k
            u = (op1.u * op2.u) / k
            if op1.u != 1 or op2.u != 1:
                a = (op2.a * op1.u + op1.a * op2.u - (op1.a + op2.a) * op1.u * op2.u) / (
                    op1.u + op2.u - 2.0 * op1.u * op2.u
                )
            else:
                a = (op1.a + op2.a) / 2.0
        assert math.isclose(b + d + u, 1.0), "CBF result does not sum to 1.0"
        return BinOp(b, d, u, a, ci=op1.ci)


# ---------------------------------------------------------------------------
# Multinomial opinion
# ---------------------------------------------------------------------------

class MultiOp:
    """Multinomial Subjective Logic opinion over K mutually exclusive states.

    Attributes:
        b: Belief mass vector of length K.
        u: Uncertainty mass (scalar).
        a: Base rate distribution of length K.
        domain: Name of the domain (e.g. "Rain").
        states: Labels for each state (e.g. ["Yes", "No"]).
        K: Number of states.
    """

    def __init__(
        self,
        b,
        u: float,
        a,
        domain: Optional[str] = None,
        states: Optional[List[str]] = None,
        ci: float = 0.95,
    ) -> None:
        self.b = np.array(b, dtype=float)
        self.u = float(u)
        self.a = np.array(a, dtype=float)
        self.K = len(self.b)
        self.ci = ci
        self.domain = domain or "None"
        self.states = states or [f"s{i + 1}" for i in range(self.K)]

        s = np.sum(self.b) + self.u
        if not np.isclose(s, 1.0, atol=1e-6):
            raise ValueError(f"Belief + uncertainty must sum to 1. Got {s:.6f}.")
        if not np.isclose(np.sum(self.a), 1.0, atol=1e-6):
            raise ValueError("Base rate distribution must sum to 1.")
        if len(self.states) != self.K:
            raise ValueError("Number of states must match belief vector length.")

    # --- Projected probability ---

    def prob(self, state=None):
        """Return projected probability distribution (or a single state's value)."""
        p = self.b + self.a * self.u
        if state is None:
            return p
        if isinstance(state, int):
            return float(p[state])
        if isinstance(state, str):
            if state not in self.states:
                raise KeyError(f"Unknown state '{state}'.")
            return float(p[self.states.index(state)])
        raise TypeError("State must be a string label or integer index.")

    # --- Operators ---

    def __and__(self, other: "MultiOp") -> "MultiOp":
        """Multinomial conjunction (joint opinion over the Cartesian product)."""
        return _MultinomialMultiplication.multiply(self, other)

    def deduce(self, b_Y_given_x, u_Y_given_x, states_Y=None, name_Y: str = "Y") -> "MultiOp":
        """Multinomial deduction: propagate self through per-state conditionals."""
        return _MultinomialDeduction.deduce(
            self, b_Y_given_x, u_Y_given_x, states_Y=states_Y, name_Y=name_Y
        )

    def __str__(self) -> str:
        p = self.prob()
        label_str = ", ".join(f"{s}: {p_i:.2f}" for s, p_i in zip(self.states, p))
        return (
            f"MultiOp({self.domain}, K={self.K}, u={self.u:.3f}, "
            f"beliefs={dict(zip(self.states, np.round(self.b, 3)))}, "
            f"E={{ {label_str} }})"
        )


# ---------------------------------------------------------------------------
# Internal operator implementations
# ---------------------------------------------------------------------------

class _MultinomialDeduction:
    """SL multinomial deduction (Jøsang Eqs. 9.61, 9.68, 9.70–9.77)."""

    @staticmethod
    def _compute_a_Y(a_X, b_Y_given_x, u_Y_given_x):
        a_X = np.asarray(a_X, float)
        b_Y_given_x = np.asarray(b_Y_given_x, float)
        u_Y_given_x = np.asarray(u_Y_given_x, float)
        numerator = a_X @ b_Y_given_x
        denominator = 1.0 - a_X @ u_Y_given_x
        return numerator / denominator

    @staticmethod
    def _compute_P_Y_given_x(b_Y_given_x, a_Y, u_Y_given_x):
        b_Y_given_x = np.asarray(b_Y_given_x, float)
        a_Y = np.asarray(a_Y, float)
        u_Y_given_x = np.asarray(u_Y_given_x, float)
        return b_Y_given_x + np.outer(u_Y_given_x, a_Y)

    @staticmethod
    def _compute_u_Y_given_X_hat(a_X, P_Y_given_x, b_Y_given_x, a_Y):
        a_X = np.asarray(a_X, float)
        P_Y_given_x = np.asarray(P_Y_given_x, float)
        b_Y_given_x = np.asarray(b_Y_given_x, float)
        a_Y = np.asarray(a_Y, float)
        P_Y_given_X_hat = a_X @ P_Y_given_x
        min_b = np.min(b_Y_given_x, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            u_j = (P_Y_given_X_hat - min_b) / a_Y
            u_j = np.where(a_Y == 0, np.inf, u_j)
        return float(np.min(u_j)), u_j

    @staticmethod
    def _compute_u_Y_conditional_on_X(u_Y_given_Xhat, u_Y_given_x, b_X):
        u_Y_given_x = np.asarray(u_Y_given_x, float)
        b_X = np.asarray(b_X, float)
        return float(u_Y_given_Xhat - np.sum((u_Y_given_Xhat - u_Y_given_x) * b_X))

    @staticmethod
    def _projected_PY_given_X(P_X, P_Y_given_x):
        return np.asarray(P_X, float) @ np.asarray(P_Y_given_x, float)

    @staticmethod
    def _belief_Y_cond_X(P_Y_cond_X, a_Y, u_Y_cond_X):
        return np.asarray(P_Y_cond_X, float) - np.asarray(a_Y, float) * float(u_Y_cond_X)

    @staticmethod
    def deduce(omega_X: MultiOp, b_Y_given_x, u_Y_given_x, *, states_Y=None, name_Y="Y") -> MultiOp:
        b_Y_given_x = np.asarray(b_Y_given_x, float)
        u_Y_given_x = np.asarray(u_Y_given_x, float)
        a_X = np.asarray(omega_X.a, float)
        b_X = np.asarray(omega_X.b, float)
        u_X = float(omega_X.u)

        a_Y = _MultinomialDeduction._compute_a_Y(a_X, b_Y_given_x, u_Y_given_x)
        P_Y_given_x = _MultinomialDeduction._compute_P_Y_given_x(b_Y_given_x, a_Y, u_Y_given_x)
        u_Y_given_Xhat, _ = _MultinomialDeduction._compute_u_Y_given_X_hat(
            a_X, P_Y_given_x, b_Y_given_x, a_Y
        )
        u_Y_cond_X = _MultinomialDeduction._compute_u_Y_conditional_on_X(
            u_Y_given_Xhat, u_Y_given_x, b_X
        )
        P_X = b_X + a_X * u_X
        P_Y_cond_X = _MultinomialDeduction._projected_PY_given_X(P_X, P_Y_given_x)
        b_Y_cond_X = _MultinomialDeduction._belief_Y_cond_X(P_Y_cond_X, a_Y, u_Y_cond_X)

        b = np.clip(b_Y_cond_X, 0.0, 1.0)
        u = float(np.clip(u_Y_cond_X, 0.0, 1.0))
        s = b.sum() + u
        if not np.isclose(s, 1.0, atol=1e-8) and s > 0:
            b = b * ((1.0 - u) / max(b.sum(), 1e-15))

        return MultiOp(
            b=b,
            u=u,
            a=a_Y,
            domain=name_Y,
            states=states_Y or [f"y{i + 1}" for i in range(len(b))],
        )


class _MultinomialMultiplication:
    """SL multinomial multiplication (joint opinion over Cartesian product)."""

    @staticmethod
    def _compute_uncertainty(omega_X: MultiOp, omega_Y: MultiOp, b_singletons):
        b_X, u_X, a_X = np.asarray(omega_X.b, float), float(omega_X.u), np.asarray(omega_X.a, float)
        b_Y, u_Y, a_Y = np.asarray(omega_Y.b, float), float(omega_Y.u), np.asarray(omega_Y.a, float)
        k, l = len(b_X), len(b_Y)
        u_matrix = np.zeros((k, l))
        for i in range(k):
            for j in range(l):
                numerator = (b_X[i] + a_X[i] * u_X) * (b_Y[j] + a_Y[j] * u_Y) - b_singletons[i, j]
                denom = a_X[i] * a_Y[j]
                u_matrix[i, j] = numerator / denom if denom != 0 else np.inf
        return float(np.min(u_matrix))

    @staticmethod
    def _compute_belief(omega_X: MultiOp, omega_Y: MultiOp, u_XY: float):
        b_X, u_X, a_X = np.asarray(omega_X.b, float), float(omega_X.u), np.asarray(omega_X.a, float)
        b_Y, u_Y, a_Y = np.asarray(omega_Y.b, float), float(omega_Y.u), np.asarray(omega_Y.a, float)
        P_X = b_X + a_X * u_X
        P_Y = b_Y + a_Y * u_Y
        return np.outer(P_X, P_Y) - np.outer(a_X, a_Y) * u_XY

    @staticmethod
    def multiply(omega_X: MultiOp, omega_Y: MultiOp) -> MultiOp:
        a_XY = np.outer(omega_X.a, omega_Y.a).flatten()
        b_singletons = np.outer(omega_X.b, omega_Y.b)
        u_XY = _MultinomialMultiplication._compute_uncertainty(omega_X, omega_Y, b_singletons)
        b_XY = _MultinomialMultiplication._compute_belief(omega_X, omega_Y, u_XY).flatten()
        total = b_XY.sum() + u_XY
        if not np.isclose(total, 1.0, atol=1e-8) and total > 0:
            b_XY = b_XY / total * (1.0 - u_XY)
        states_XY = [f"{x}_._{y}" for x in omega_X.states for y in omega_Y.states]
        domain_XY = f"{omega_X.domain}×{omega_Y.domain}"
        return MultiOp(b=b_XY, u=u_XY, a=a_XY, domain=domain_XY, states=states_XY)
