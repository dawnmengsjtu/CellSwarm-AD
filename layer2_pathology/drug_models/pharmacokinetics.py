# -*- coding: utf-8 -*-
"""Pharmacokinetics (PK) module.

Implements one-compartment and two-compartment models for oral and IV administration.
Supports calculation of Cmax, Tmax, AUC, and half-life.

One-compartment oral model:
    dC/dt = (D * ka * exp(-ka*t)) / Vd - ke * C
    Analytical solution:
    C(t) = (D * ka) / (Vd * (ka - ke)) * (exp(-ke*t) - exp(-ka*t))   [oral]
    C(t) = (D / Vd) * exp(-ke*t)                                       [IV bolus]

Two-compartment IV infusion model:
    dCc/dt = R(t)/Vc - ke*Cc - k12*Cc + k21*Cp*(Vp/Vc)
    dCp/dt = k12*Cc*(Vc/Vp) - k21*Cp
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import integrate


class RouteOfAdministration(Enum):
    """Route of drug administration."""
    ORAL = "oral"
    IV_BOLUS = "iv_bolus"
    IV_INFUSION = "iv_infusion"


@dataclass
class PKParameters:
    """Pharmacokinetic parameters for one-compartment model.

    Attributes:
        dose: Administered dose (mg or mg/kg)
        ka: Absorption rate constant (1/h), oral only
        Vd: Volume of distribution (L or L/kg)
        ke: Elimination rate constant (1/h)
        F: Bioavailability fraction (0-1), oral only
        route: Route of administration
        t_infusion: Infusion duration (h), IV infusion only
    """
    dose: float           # mg
    Vd: float             # L
    ke: float             # 1/h
    ka: float = 1.0       # 1/h (oral)
    F: float = 1.0        # bioavailability
    route: RouteOfAdministration = RouteOfAdministration.ORAL
    t_infusion: float = 0.0  # h, for IV infusion

    def __post_init__(self) -> None:
        if self.dose <= 0:
            raise ValueError("dose must be positive")
        if self.Vd <= 0:
            raise ValueError("Vd must be positive")
        if self.ke <= 0:
            raise ValueError("ke must be positive")
        if not (0 < self.F <= 1):
            raise ValueError("Bioavailability F must be in (0, 1]")


@dataclass
class PKResult:
    """Results from PK simulation."""
    times: List[float]
    concentrations: List[float]
    Cmax: float
    Tmax: float
    AUC: float
    half_life: float
    clearance: float      # CL = ke * Vd  (L/h)
    parameters: PKParameters

    def concentration_at(self, t: float) -> float:
        """Interpolate concentration at arbitrary time t."""
        return float(np.interp(t, self.times, self.concentrations))


class PKModel:
    """One-compartment pharmacokinetic model.

    Supports oral, IV bolus, and IV infusion routes.
    Uses both analytical solutions and numerical ODE integration.
    """

    def __init__(self, params: PKParameters) -> None:
        self.params = params

    # ------------------------------------------------------------------
    # Analytical concentration functions
    # ------------------------------------------------------------------

    def _concentration_oral(self, t: float) -> float:
        """Analytical solution for oral one-compartment model."""
        p = self.params
        D_eff = p.dose * p.F
        if abs(p.ka - p.ke) < 1e-9:
            # Flip-flop edge case: use numerical limit
            return (D_eff * p.ka / p.Vd) * t * math.exp(-p.ke * t)
        return (D_eff * p.ka) / (p.Vd * (p.ka - p.ke)) * (
            math.exp(-p.ke * t) - math.exp(-p.ka * t)
        )

    def _concentration_iv_bolus(self, t: float) -> float:
        """Analytical solution for IV bolus."""
        p = self.params
        C0 = p.dose / p.Vd
        return C0 * math.exp(-p.ke * t)

    def _concentration_iv_infusion(self, t: float) -> float:
        """Analytical solution for IV infusion.

        During infusion (0 <= t <= t_inf):
            C(t) = R0/(CL) * (1 - exp(-ke*t))
        After infusion (t > t_inf):
            C(t) = C(t_inf) * exp(-ke*(t - t_inf))
        """
        p = self.params
        CL = p.ke * p.Vd
        R0 = p.dose / p.t_infusion  # infusion rate (mg/h)
        t_inf = p.t_infusion
        if t <= t_inf:
            return (R0 / CL) * (1 - math.exp(-p.ke * t))
        else:
            C_end = (R0 / CL) * (1 - math.exp(-p.ke * t_inf))
            return C_end * math.exp(-p.ke * (t - t_inf))

    def concentration(self, t: float) -> float:
        """Compute plasma concentration at time t (h)."""
        if t < 0:
            return 0.0
        route = self.params.route
        if route == RouteOfAdministration.ORAL:
            return max(0.0, self._concentration_oral(t))
        elif route == RouteOfAdministration.IV_BOLUS:
            return max(0.0, self._concentration_iv_bolus(t))
        elif route == RouteOfAdministration.IV_INFUSION:
            return max(0.0, self._concentration_iv_infusion(t))
        else:
            raise ValueError(f"Unknown route: {route}")

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        t_end: float = 24.0,
        n_points: int = 500,
    ) -> PKResult:
        """Simulate PK profile over [0, t_end] hours.

        Args:
            t_end: End time in hours
            n_points: Number of time points

        Returns:
            PKResult with full concentration-time profile and summary metrics
        """
        times = list(np.linspace(0, t_end, n_points))
        concs = [self.concentration(t) for t in times]

        # Summary metrics
        Cmax = max(concs)
        Tmax = times[concs.index(Cmax)]
        AUC = float(np.trapz(concs, times))
        half_life = math.log(2) / self.params.ke
        clearance = self.params.ke * self.params.Vd

        return PKResult(
            times=times,
            concentrations=concs,
            Cmax=Cmax,
            Tmax=Tmax,
            AUC=AUC,
            half_life=half_life,
            clearance=clearance,
            parameters=self.params,
        )

    def simulate_multiple_doses(
        self,
        n_doses: int,
        dosing_interval: float,
        t_extra: float = 0.0,
        n_points: int = 1000,
    ) -> PKResult:
        """Simulate multiple-dose regimen (superposition principle).

        Args:
            n_doses: Number of doses
            dosing_interval: Interval between doses (h)
            t_extra: Extra time to simulate after last dose (h)
            n_points: Total number of time points

        Returns:
            PKResult for the full multi-dose profile
        """
        t_end = (n_doses - 1) * dosing_interval + t_extra
        if t_end <= 0:
            t_end = dosing_interval
        times = np.linspace(0, t_end, n_points)
        concs = np.zeros(n_points)

        dose_times = [i * dosing_interval for i in range(n_doses)]
        for td in dose_times:
            for i, t in enumerate(times):
                dt = t - td
                if dt >= 0:
                    # Temporarily shift time
                    concs[i] += self.concentration(dt)

        Cmax = float(np.max(concs))
        idx_max = int(np.argmax(concs))
        Tmax = float(times[idx_max])
        AUC = float(np.trapz(concs, times))
        half_life = math.log(2) / self.params.ke
        clearance = self.params.ke * self.params.Vd

        return PKResult(
            times=list(times),
            concentrations=list(concs),
            Cmax=Cmax,
            Tmax=Tmax,
            AUC=AUC,
            half_life=half_life,
            clearance=clearance,
            parameters=self.params,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def half_life_from_ke(ke: float) -> float:
        """Compute half-life from elimination rate constant."""
        if ke <= 0:
            raise ValueError("ke must be positive")
        return math.log(2) / ke

    @staticmethod
    def ke_from_half_life(t_half: float) -> float:
        """Compute elimination rate constant from half-life."""
        if t_half <= 0:
            raise ValueError("half-life must be positive")
        return math.log(2) / t_half

    def steady_state_concentration(self, dosing_interval: float) -> float:
        """Approximate average steady-state concentration (Css,avg).

        Css,avg = F * Dose / (CL * tau)
        """
        p = self.params
        CL = p.ke * p.Vd
        return (p.F * p.dose) / (CL * dosing_interval)


# ======================================================================
# Two-Compartment Model
# ======================================================================

@dataclass
class TwoCompartmentPKParameters:
    """Pharmacokinetic parameters for two-compartment model.

    Attributes:
        dose: Administered dose (mg)
        Vc: Central compartment volume (L)
        Vp: Peripheral compartment volume (L)
        ke: Elimination rate constant from central compartment (1/h)
        k12: Central → peripheral transfer rate constant (1/h)
        k21: Peripheral → central transfer rate constant (1/h)
        ka: Absorption rate constant (1/h), oral only
        F: Bioavailability fraction (0-1)
        route: Route of administration
        t_infusion: Infusion duration (h), IV infusion only
    """
    dose: float           # mg
    Vc: float             # Central volume (L)
    Vp: float             # Peripheral volume (L)
    ke: float             # Elimination rate from central (1/h)
    k12: float            # Central→peripheral transfer rate (1/h)
    k21: float            # Peripheral→central transfer rate (1/h)
    ka: float = 0.0       # Absorption rate (oral only)
    F: float = 1.0        # Bioavailability
    route: RouteOfAdministration = RouteOfAdministration.IV_INFUSION
    t_infusion: float = 1.0  # h

    def __post_init__(self) -> None:
        if self.dose <= 0:
            raise ValueError("dose must be positive")
        if self.Vc <= 0:
            raise ValueError("Vc must be positive")
        if self.Vp <= 0:
            raise ValueError("Vp must be positive")
        if self.ke <= 0:
            raise ValueError("ke must be positive")
        if self.k12 < 0:
            raise ValueError("k12 must be non-negative")
        if self.k21 < 0:
            raise ValueError("k21 must be non-negative")
        if not (0 < self.F <= 1):
            raise ValueError("Bioavailability F must be in (0, 1]")

    @property
    def beta(self) -> float:
        """Terminal (β) elimination rate constant.

        β is the smaller eigenvalue of the two-compartment system:
        α, β = 0.5 * ((ke + k12 + k21) ± sqrt((ke + k12 + k21)^2 - 4*ke*k21))
        """
        s = self.ke + self.k12 + self.k21
        p = self.ke * self.k21
        discriminant = s * s - 4 * p
        return 0.5 * (s - math.sqrt(max(discriminant, 0.0)))

    @property
    def alpha(self) -> float:
        """Distribution (α) rate constant (faster phase)."""
        s = self.ke + self.k12 + self.k21
        p = self.ke * self.k21
        discriminant = s * s - 4 * p
        return 0.5 * (s + math.sqrt(max(discriminant, 0.0)))

    @property
    def terminal_half_life(self) -> float:
        """Terminal (β-phase) half-life in hours."""
        b = self.beta
        if b <= 0:
            return float('inf')
        return math.log(2) / b


class TwoCompartmentPKModel:
    """Two-compartment pharmacokinetic model.

    Solves the two-compartment ODE system using scipy.integrate.solve_ivp.
    Provides the same interface as PKModel: simulate(), concentration(),
    steady_state_concentration().
    """

    def __init__(self, params: TwoCompartmentPKParameters) -> None:
        self.params = params
        # Cache for the last simulation result
        self._cached_result: Optional[PKResult] = None
        self._cached_t_end: Optional[float] = None

    def _ode_system(self, t: float, y: np.ndarray) -> List[float]:
        """ODE right-hand side for two-compartment model.

        State vector y = [Cc, Cp] (concentrations in central and peripheral).
        """
        Cc, Cp = y
        p = self.params

        # Infusion rate
        if p.route == RouteOfAdministration.IV_INFUSION:
            R = (p.dose / p.t_infusion) if t <= p.t_infusion else 0.0
        elif p.route == RouteOfAdministration.IV_BOLUS:
            R = 0.0  # bolus handled via initial condition
        elif p.route == RouteOfAdministration.ORAL:
            R = p.dose * p.F * p.ka * math.exp(-p.ka * t)
        else:
            R = 0.0

        dCc_dt = R / p.Vc - p.ke * Cc - p.k12 * Cc + p.k21 * Cp * (p.Vp / p.Vc)
        dCp_dt = p.k12 * Cc * (p.Vc / p.Vp) - p.k21 * Cp

        return [dCc_dt, dCp_dt]

    def _solve(self, t_end: float, n_points: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve the ODE system and return (times, Cc, Cp)."""
        p = self.params
        t_eval = np.linspace(0, t_end, n_points)

        # Initial conditions
        if p.route == RouteOfAdministration.IV_BOLUS:
            Cc0 = p.dose / p.Vc
        else:
            Cc0 = 0.0
        Cp0 = 0.0

        sol = integrate.solve_ivp(
            self._ode_system,
            [0, t_end],
            [Cc0, Cp0],
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-10,
            max_step=min(0.5, t_end / 100),
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        Cc = np.maximum(sol.y[0], 0.0)
        Cp = np.maximum(sol.y[1], 0.0)
        return sol.t, Cc, Cp

    def concentration(self, t: float) -> float:
        """Compute central compartment concentration at time t (h).

        Uses cached simulation if available, otherwise solves for a
        suitable time range.
        """
        if t < 0:
            return 0.0
        if t == 0:
            if self.params.route == RouteOfAdministration.IV_BOLUS:
                return self.params.dose / self.params.Vc
            return 0.0

        # Use cached result if it covers the requested time
        if (self._cached_result is not None
                and self._cached_t_end is not None
                and t <= self._cached_t_end):
            return float(np.interp(
                t, self._cached_result.times, self._cached_result.concentrations
            ))

        # Solve fresh for this time point
        t_end = max(t * 1.1, 10.0)
        times, Cc, _ = self._solve(t_end, n_points=max(500, int(t_end * 10)))
        return float(np.interp(t, times, Cc))

    def simulate(
        self,
        t_end: float = 24.0,
        n_points: int = 500,
    ) -> PKResult:
        """Simulate PK profile over [0, t_end] hours.

        Args:
            t_end: End time in hours
            n_points: Number of time points

        Returns:
            PKResult with full concentration-time profile and summary metrics
        """
        times, Cc, _ = self._solve(t_end, n_points)

        times_list = list(times)
        concs_list = list(Cc)

        Cmax = float(np.max(Cc))
        Tmax = float(times[np.argmax(Cc)])
        AUC = float(np.trapz(Cc, times))
        half_life = self.params.terminal_half_life
        clearance = self.params.ke * self.params.Vc

        result = PKResult(
            times=times_list,
            concentrations=concs_list,
            Cmax=Cmax,
            Tmax=Tmax,
            AUC=AUC,
            half_life=half_life,
            clearance=clearance,
            parameters=self.params,  # type: ignore[arg-type]
        )

        # Cache for subsequent concentration() calls
        self._cached_result = result
        self._cached_t_end = t_end

        return result

    def simulate_multiple_doses(
        self,
        n_doses: int,
        dosing_interval: float,
        t_extra: float = 0.0,
        n_points: int = 1000,
    ) -> PKResult:
        """Simulate multiple-dose regimen via ODE with pulsed dosing.

        Args:
            n_doses: Number of doses
            dosing_interval: Interval between doses (h)
            t_extra: Extra time to simulate after last dose (h)
            n_points: Total number of time points

        Returns:
            PKResult for the full multi-dose profile
        """
        p = self.params
        t_end = (n_doses - 1) * dosing_interval + t_extra
        if t_end <= 0:
            t_end = dosing_interval
        t_eval = np.linspace(0, t_end, n_points)

        dose_times = [i * dosing_interval for i in range(n_doses)]

        def ode_multi(t: float, y: np.ndarray) -> List[float]:
            Cc, Cp = y
            R = 0.0
            for td in dose_times:
                t_rel = t - td
                if t_rel < 0:
                    continue
                if p.route == RouteOfAdministration.IV_INFUSION:
                    if t_rel <= p.t_infusion:
                        R += p.dose / p.t_infusion
                elif p.route == RouteOfAdministration.ORAL:
                    R += p.dose * p.F * p.ka * math.exp(-p.ka * t_rel)

            dCc = R / p.Vc - p.ke * Cc - p.k12 * Cc + p.k21 * Cp * (p.Vp / p.Vc)
            dCp = p.k12 * Cc * (p.Vc / p.Vp) - p.k21 * Cp
            return [dCc, dCp]

        # IV bolus: first dose as initial condition, rest as impulses
        # For simplicity with solve_ivp, we handle bolus by adding to Cc at dose times
        if p.route == RouteOfAdministration.IV_BOLUS:
            # Solve segment by segment
            Cc_all = np.zeros(n_points)
            Cp_all = np.zeros(n_points)
            Cc_state, Cp_state = 0.0, 0.0

            for dose_idx in range(n_doses):
                t_start = dose_times[dose_idx]
                t_stop = dose_times[dose_idx + 1] if dose_idx + 1 < n_doses else t_end
                Cc_state += p.dose / p.Vc  # bolus

                mask = (t_eval >= t_start) & (t_eval <= t_stop + 1e-12)
                t_seg = t_eval[mask]
                if len(t_seg) == 0:
                    continue

                sol = integrate.solve_ivp(
                    ode_multi, [t_start, t_stop], [Cc_state, Cp_state],
                    t_eval=t_seg, method='RK45', rtol=1e-8, atol=1e-10,
                )
                Cc_all[mask] = np.maximum(sol.y[0], 0.0)
                Cp_all[mask] = np.maximum(sol.y[1], 0.0)
                Cc_state = float(sol.y[0, -1])
                Cp_state = float(sol.y[1, -1])

            Cc = Cc_all
        else:
            sol = integrate.solve_ivp(
                ode_multi, [0, t_end], [0.0, 0.0],
                t_eval=t_eval, method='RK45', rtol=1e-8, atol=1e-10,
                max_step=min(0.5, t_end / 100),
            )
            if not sol.success:
                raise RuntimeError(f"ODE solver failed: {sol.message}")
            Cc = np.maximum(sol.y[0], 0.0)

        Cmax = float(np.max(Cc))
        Tmax = float(t_eval[np.argmax(Cc)])
        AUC = float(np.trapz(Cc, t_eval))
        half_life = p.terminal_half_life
        clearance = p.ke * p.Vc

        return PKResult(
            times=list(t_eval),
            concentrations=list(Cc),
            Cmax=Cmax,
            Tmax=Tmax,
            AUC=AUC,
            half_life=half_life,
            clearance=clearance,
            parameters=p,  # type: ignore[arg-type]
        )

    def steady_state_concentration(self, dosing_interval: float) -> float:
        """Approximate average steady-state concentration (Css,avg).

        For two-compartment model:
        Css,avg = F * Dose / (CL * tau)
        where CL = ke * Vc (clearance from central compartment).
        """
        p = self.params
        CL = p.ke * p.Vc
        return (p.F * p.dose) / (CL * dosing_interval)
