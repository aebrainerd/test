#!/usr/bin/env python3
"""
1D silicon abrupt PN diode drift-diffusion + Poisson solver (steady-state DC).

Sign convention:
- Domain x in [0, L], left contact (x=0) is p-ohmic, right contact (x=L) is n-ohmic.
- Electrostatic potential phi(0)=0, phi(L)=Vapp.
- Forward bias corresponds to positive Vapp (reduces built-in barrier), yielding
  increased injection and positive current magnitude.

Run:
  python pn_dd_poisson_1d.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt


@dataclass
class Params:
    # Physical constants
    q: float = 1.602176634e-19
    kB: float = 1.380649e-23
    T: float = 300.0
    eps0: float = 8.8541878128e-12
    eps_si_rel: float = 11.7

    # Transport parameters (constant mobility)
    mu_n: float = 0.135  # m^2/V/s
    mu_p: float = 0.048  # m^2/V/s

    # Intrinsic carrier concentration (Si, 300K)
    ni: float = 1.45e16  # 1/m^3 (≈1.45e10 cm^-3)

    # Geometry
    L: float = 2e-6
    N: int = 801

    # Doping (abrupt)
    NA_left: float = 1e24  # 1/m^3
    ND_right: float = 1e24  # 1/m^3

    # Bias sweep
    V_start: float = -0.5
    V_stop: float = 0.8
    V_step: float = 0.05

    # Gummel iteration controls
    max_iter: int = 200
    tol_phi: float = 1e-6
    tol_carrier: float = 1e-3
    relax_phi: float = 0.3
    relax_carrier: float = 0.5
    min_density: float = 1.0

    # Current reporting (area)
    area: float = 1e-12

    # Optional SRH recombination (not enabled by default)
    enable_srh: bool = False
    tau_n: float = 1e-6
    tau_p: float = 1e-6
    n1: float = 1.45e16
    p1: float = 1.45e16


def bernoulli(psi: np.ndarray) -> np.ndarray:
    """Stable Bernoulli function B(psi) = psi/(exp(psi)-1)."""
    psi = np.asarray(psi)
    out = np.empty_like(psi, dtype=float)
    small = np.abs(psi) < 1e-3
    out[small] = 1.0 - psi[small] / 2.0 + psi[small] ** 2 / 12.0

    large = ~small
    psi_l = psi[large]
    # Avoid overflow/underflow
    psi_l = np.clip(psi_l, -100, 100)
    exp_psi = np.exp(psi_l)
    out[large] = psi_l / (exp_psi - 1.0)
    return out


def build_doping_profile(x: np.ndarray, p: Params) -> Tuple[np.ndarray, np.ndarray]:
    """Abrupt doping: left half p-type, right half n-type."""
    ND = np.zeros_like(x)
    NA = np.zeros_like(x)
    mid = 0.5 * p.L
    NA[x <= mid] = p.NA_left
    ND[x > mid] = p.ND_right
    return ND, NA


def equilibrium_carriers(ND: np.ndarray, NA: np.ndarray, ni: float) -> Tuple[np.ndarray, np.ndarray]:
    """Charge neutrality + mass action law for initial guess."""
    net = ND - NA
    n0 = 0.5 * (net + np.sqrt(net ** 2 + 4 * ni ** 2))
    p0 = ni ** 2 / n0
    return n0, p0


def solve_poisson(phi_bc: Tuple[float, float], n: np.ndarray, p: np.ndarray,
                  ND: np.ndarray, NA: np.ndarray, pms: Params, dx: float) -> np.ndarray:
    """Solve Poisson with Dirichlet boundary conditions."""
    eps = pms.eps_si_rel * pms.eps0
    N = n.size
    main = np.full(N - 2, 2.0 * eps / dx ** 2)
    off = np.full(N - 3, -eps / dx ** 2)
    A = diags([off, main, off], offsets=[-1, 0, 1], format="csc")

    rho = pms.q * (p - n + ND - NA)
    b = -rho[1:-1]
    b[0] += eps / dx ** 2 * phi_bc[0]
    b[-1] += eps / dx ** 2 * phi_bc[1]

    phi_inner = spsolve(A, b)
    phi = np.zeros(N)
    phi[0] = phi_bc[0]
    phi[-1] = phi_bc[1]
    phi[1:-1] = phi_inner
    return phi


def solve_electron_continuity(phi: np.ndarray, n_bc: Tuple[float, float],
                              pms: Params, dx: float) -> np.ndarray:
    """Solve steady-state electron continuity with SG discretization."""
    N = phi.size
    Vt = pms.kB * pms.T / pms.q
    psi = (phi[1:] - phi[:-1]) / Vt

    Bp = bernoulli(psi)
    Bm = bernoulli(-psi)
    C = pms.q * pms.mu_n * Vt / dx

    lower = C * Bm[:-1]  # a_i
    upper = C * Bp[1:]   # c_i
    main = -C * (Bm[1:] + Bp[:-1])  # b_i

    rhs = np.zeros(N - 2)
    rhs[0] -= lower[0] * n_bc[0]
    rhs[-1] -= upper[-1] * n_bc[1]

    A = diags([lower[1:], main, upper[:-1]], offsets=[-1, 0, 1], format="csc")
    n_inner = spsolve(A, rhs)

    n = np.zeros(N)
    n[0] = n_bc[0]
    n[-1] = n_bc[1]
    n[1:-1] = n_inner
    return n


def solve_hole_continuity(phi: np.ndarray, p_bc: Tuple[float, float],
                          pms: Params, dx: float) -> np.ndarray:
    """Solve steady-state hole continuity with SG discretization."""
    N = phi.size
    Vt = pms.kB * pms.T / pms.q
    psi = (phi[1:] - phi[:-1]) / Vt

    Bp = bernoulli(psi)
    Bm = bernoulli(-psi)
    C = pms.q * pms.mu_p * Vt / dx

    lower = -C * Bp[:-1]
    upper = -C * Bm[1:]
    main = C * (Bp[1:] + Bm[:-1])

    rhs = np.zeros(N - 2)
    rhs[0] -= lower[0] * p_bc[0]
    rhs[-1] -= upper[-1] * p_bc[1]

    A = diags([lower[1:], main, upper[:-1]], offsets=[-1, 0, 1], format="csc")
    p_inner = spsolve(A, rhs)

    p = np.zeros(N)
    p[0] = p_bc[0]
    p[-1] = p_bc[1]
    p[1:-1] = p_inner
    return p


def compute_currents(phi: np.ndarray, n: np.ndarray, p: np.ndarray,
                     pms: Params, dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Vt = pms.kB * pms.T / pms.q
    psi = (phi[1:] - phi[:-1]) / Vt
    Bp = bernoulli(psi)
    Bm = bernoulli(-psi)
    Cn = pms.q * pms.mu_n * Vt / dx
    Cp = pms.q * pms.mu_p * Vt / dx

    Jn = Cn * (n[1:] * Bp - n[:-1] * Bm)
    Jp = -Cp * (p[1:] * Bm - p[:-1] * Bp)
    J = Jn + Jp
    x_mid = (np.arange(n.size - 1) + 0.5) * dx
    return x_mid, Jn, Jp, J


def srh_recombination(n: np.ndarray, p: np.ndarray, pms: Params) -> np.ndarray:
    denom = pms.tau_p * (n + pms.n1) + pms.tau_n * (p + pms.p1)
    return (n * p - pms.ni ** 2) / denom


def gummel_solve_for_bias(Vapp: float, state_init: Dict[str, np.ndarray],
                          x: np.ndarray, ND: np.ndarray, NA: np.ndarray,
                          pms: Params) -> Dict[str, np.ndarray]:
    dx = x[1] - x[0]
    phi = state_init["phi"].copy()
    n = state_init["n"].copy()
    p = state_init["p"].copy()

    phi_bc = (0.0, Vapp)

    for it in range(pms.max_iter):
        n_old = n.copy()
        p_old = p.copy()
        phi_old = phi.copy()

        n_bc = (pms.ni ** 2 / max(NA[0], pms.min_density), max(ND[-1], pms.min_density))
        p_bc = (max(NA[0], pms.min_density), pms.ni ** 2 / max(ND[-1], pms.min_density))

        n_new = solve_electron_continuity(phi, n_bc, pms, dx)
        p_new = solve_hole_continuity(phi, p_bc, pms, dx)

        n_new = np.maximum(n_new, pms.min_density)
        p_new = np.maximum(p_new, pms.min_density)

        if pms.enable_srh:
            R = srh_recombination(n_new, p_new, pms)
            # Simple source-term correction (explicit); optional extension
            n_new = np.maximum(n_new - pms.relax_carrier * (R * dx), pms.min_density)
            p_new = np.maximum(p_new - pms.relax_carrier * (R * dx), pms.min_density)

        phi_new = solve_poisson(phi_bc, n_new, p_new, ND, NA, pms, dx)

        n = (1 - pms.relax_carrier) * n_old + pms.relax_carrier * n_new
        p = (1 - pms.relax_carrier) * p_old + pms.relax_carrier * p_new
        phi = (1 - pms.relax_phi) * phi_old + pms.relax_phi * phi_new

        err_phi = np.max(np.abs(phi - phi_old))
        err_n = np.max(np.abs(n - n_old) / np.maximum(n_old, pms.min_density))
        err_p = np.max(np.abs(p - p_old) / np.maximum(p_old, pms.min_density))

        if err_phi < pms.tol_phi and err_n < pms.tol_carrier and err_p < pms.tol_carrier:
            break
    else:
        print(f"Warning: Gummel did not converge at Vapp={Vapp:.3f} V")

    x_mid, Jn, Jp, J = compute_currents(phi, n, p, pms, dx)
    return {
        "phi": phi,
        "n": n,
        "p": p,
        "x_mid": x_mid,
        "Jn": Jn,
        "Jp": Jp,
        "J": J,
        "iterations": it + 1,
    }


def sweep_biases(pms: Params) -> Tuple[List[float], List[float], Dict[float, Dict[str, np.ndarray]]]:
    x = np.linspace(0.0, pms.L, pms.N)
    ND, NA = build_doping_profile(x, pms)
    n0, p0 = equilibrium_carriers(ND, NA, pms.ni)
    phi0 = np.zeros_like(x)

    state = {"phi": phi0, "n": n0, "p": p0}

    V_list = np.arange(pms.V_start, pms.V_stop + 0.5 * pms.V_step, pms.V_step)
    results = {}
    currents = []

    for Vapp in V_list:
        res = gummel_solve_for_bias(Vapp, state, x, ND, NA, pms)
        results[Vapp] = res
        state = {"phi": res["phi"], "n": res["n"], "p": res["p"]}

        J = res["J"]
        J_mean = np.mean(J)
        currents.append(J_mean * pms.area)
        spread = (np.max(J) - np.min(J)) / max(np.mean(np.abs(J)), 1e-30)
        print(f"V={Vapp: .3f} V | I={J_mean * pms.area: .3e} A | J spread={spread:.3e}")

    return list(V_list), currents, results


def plot_results(pms: Params, x: np.ndarray, results: Dict[float, Dict[str, np.ndarray]],
                 V_plot: List[float], V_list: List[float], currents: List[float]) -> None:
    fig1, ax1 = plt.subplots()
    for V in V_plot:
        res = results[V]
        ax1.plot(x * 1e6, res["phi"], label=f"V={V:.2f} V")
    ax1.set_xlabel("x (um)")
    ax1.set_ylabel("phi (V)")
    ax1.legend()
    ax1.set_title("Electrostatic potential")

    fig2, ax2 = plt.subplots()
    for V in V_plot:
        res = results[V]
        ax2.semilogy(x * 1e6, res["n"], label=f"n, V={V:.2f} V")
        ax2.semilogy(x * 1e6, res["p"], linestyle="--", label=f"p, V={V:.2f} V")
    ax2.set_xlabel("x (um)")
    ax2.set_ylabel("Carrier density (1/m^3)")
    ax2.legend()
    ax2.set_title("Carrier densities")

    fig3, ax3 = plt.subplots()
    for V in V_plot:
        res = results[V]
        E = -np.gradient(res["phi"], x)
        ax3.plot(x * 1e6, E, label=f"V={V:.2f} V")
    ax3.set_xlabel("x (um)")
    ax3.set_ylabel("Electric field (V/m)")
    ax3.legend()
    ax3.set_title("Electric field")

    fig4, ax4 = plt.subplots()
    for V in V_plot:
        res = results[V]
        ax4.plot(res["x_mid"] * 1e6, res["J"], label=f"V={V:.2f} V")
    ax4.set_xlabel("x (um)")
    ax4.set_ylabel("Current density (A/m^2)")
    ax4.legend()
    ax4.set_title("Total current density (should be flat)")

    fig5, ax5 = plt.subplots()
    ax5.semilogy(V_list, np.abs(currents), marker="o")
    ax5.set_xlabel("Vapp (V)")
    ax5.set_ylabel("|I| (A)")
    ax5.set_title("Diode I-V (semilogy)")
    ax5.grid(True, which="both")

    plt.tight_layout()
    plt.show()


def save_iv_csv(V_list: List[float], currents: List[float]) -> None:
    data = np.column_stack([V_list, currents])
    np.savetxt("iv.csv", data, delimiter=",", header="Vapp(V),I(A)", comments="")


def main() -> None:
    pms = Params()
    x = np.linspace(0.0, pms.L, pms.N)

    V_list, currents, results = sweep_biases(pms)
    save_iv_csv(V_list, currents)

    V_plot = [V for V in [0.0, 0.2, 0.6] if V in results]
    if not V_plot:
        V_plot = [V_list[0], V_list[len(V_list) // 2], V_list[-1]]

    plot_results(pms, x, results, V_plot, V_list, currents)

    print("\nVapp(V), I(A)")
    for V, I in zip(V_list, currents):
        print(f"{V: .3f}, {I: .3e}")


if __name__ == "__main__":
    main()
