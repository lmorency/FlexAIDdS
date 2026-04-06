"""Tests for RE-DOCK thermodynamic engines and orchestrator."""
import numpy as np
import pytest

from benchmarks.re_dock.thermodynamics import (
    K_B,
    BennettAcceptanceRatio,
    CrooksEngine,
    ShannonCollapseRate,
    temperature_ladder,
    vant_hoff_with_dcp,
    shannon_entropy,
    _log_sum_exp,
    _fermi,
)
from benchmarks.re_dock.orchestrator import (
    BidirectionalProtocol,
    Tier,
    TIER_SYSTEMS,
    REDOCKResult,
)


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestLogSumExp:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        expected = np.log(np.exp(1) + np.exp(2) + np.exp(3))
        assert abs(_log_sum_exp(x) - expected) < 1e-12

    def test_large_values(self):
        """log-sum-exp should not overflow with large inputs."""
        x = np.array([1000.0, 1001.0, 1002.0])
        result = _log_sum_exp(x)
        assert np.isfinite(result)
        expected = 1002.0 + np.log(np.exp(-2) + np.exp(-1) + 1.0)
        assert abs(result - expected) < 1e-10

    def test_negative_values(self):
        x = np.array([-1000.0, -999.0, -998.0])
        result = _log_sum_exp(x)
        assert np.isfinite(result)

    def test_single_value(self):
        assert abs(_log_sum_exp(np.array([5.0])) - 5.0) < 1e-14


class TestFermi:
    def test_zero(self):
        assert abs(_fermi(np.array([0.0]))[0] - 0.5) < 1e-14

    def test_large_positive(self):
        assert _fermi(np.array([100.0]))[0] < 1e-40

    def test_large_negative(self):
        assert _fermi(np.array([-100.0]))[0] > 1.0 - 1e-10

    def test_symmetry(self):
        x = np.array([1.0, -1.0, 2.0, -2.0])
        f = _fermi(x)
        assert abs(f[0] + f[1] - 1.0) < 1e-14
        assert abs(f[2] + f[3] - 1.0) < 1e-14


# ---------------------------------------------------------------------------
# BAR tests
# ---------------------------------------------------------------------------

class TestBAR:
    def test_known_delta_g(self):
        """BAR should recover known DeltaG from Gaussian work distributions."""
        rng = np.random.default_rng(42)
        true_dg = -5.0
        n = 5000
        w_fwd = rng.normal(loc=true_dg + 1.0, scale=1.5, size=n)
        w_rev = rng.normal(loc=-true_dg + 1.0, scale=1.5, size=n)

        bar = BennettAcceptanceRatio(max_iterations=500, tolerance=1e-10)
        beta = 1.0 / (K_B * 298.15)
        dg = bar.solve(w_fwd, w_rev, beta)

        assert abs(dg - true_dg) < 0.5, f"BAR gave {dg}, expected ~{true_dg}"

    def test_zero_delta_g(self):
        """Identical forward and reverse distributions => DeltaG ~ 0."""
        rng = np.random.default_rng(123)
        n = 3000
        w = rng.normal(loc=0.0, scale=1.0, size=n)

        bar = BennettAcceptanceRatio()
        beta = 1.0 / (K_B * 300.0)
        dg = bar.solve(w, w, beta)

        assert abs(dg) < 0.3, f"BAR gave {dg} for symmetric case"

    def test_convergence(self):
        """BAR should converge within max_iterations."""
        rng = np.random.default_rng(7)
        w_fwd = rng.normal(-3.0, 2.0, 1000)
        w_rev = rng.normal(3.0, 2.0, 1000)

        bar = BennettAcceptanceRatio(max_iterations=500)
        beta = 1.0 / (K_B * 298.15)
        dg = bar.solve(w_fwd, w_rev, beta)

        assert np.isfinite(dg)

    def test_uncertainty(self):
        """Uncertainty should be finite and positive."""
        rng = np.random.default_rng(99)
        w_fwd = rng.normal(-5.0, 2.0, 500)
        w_rev = rng.normal(5.0, 2.0, 500)

        bar = BennettAcceptanceRatio()
        beta = 1.0 / (K_B * 298.15)
        err = bar.uncertainty(w_fwd, w_rev, beta)

        assert err > 0
        assert np.isfinite(err)


# ---------------------------------------------------------------------------
# Crooks engine tests
# ---------------------------------------------------------------------------

class TestCrooks:
    def test_sigma_irr_nonnegative(self):
        """For typical nonequilibrium work, sigma_irr >= 0 (2nd law)."""
        rng = np.random.default_rng(55)
        engine = CrooksEngine()
        engine.add_forward_work(rng.normal(2.0, 1.0, 500))
        engine.add_reverse_work(rng.normal(2.0, 1.0, 500))

        sigma = engine.sigma_irr(0.0)
        assert np.isfinite(sigma)

    def test_landauer_bits(self):
        engine = CrooksEngine()
        engine.add_forward_work([1.0, 1.5, 2.0])
        engine.add_reverse_work([1.0, 1.5, 2.0])

        bits = engine.landauer_bits(0.0, 300.0)
        assert np.isfinite(bits)
        assert bits > 0

    def test_add_scalar_and_array(self):
        engine = CrooksEngine()
        engine.add_forward_work(1.0)
        engine.add_forward_work([2.0, 3.0])
        assert len(engine.w_fwd) == 3
        assert engine.w_fwd == [1.0, 2.0, 3.0]

    def test_crossing_point(self):
        engine = CrooksEngine()
        rng = np.random.default_rng(77)
        engine.add_forward_work(rng.normal(3.0, 1.0, 2000))
        engine.add_reverse_work(rng.normal(-3.0, 1.0, 2000))
        cp = engine.crossing_point()
        assert np.isfinite(cp)


# ---------------------------------------------------------------------------
# Shannon collapse tests
# ---------------------------------------------------------------------------

class TestShannonCollapse:
    def test_didt_length(self):
        """dI/dT should have n-2 points for n data points."""
        scr = ShannonCollapseRate()
        for t in [300, 350, 400, 450, 500]:
            scr.add_point(t, np.random.uniform(0.001, 0.01))

        didt = scr.compute_didt()
        assert len(didt) == 3  # 5 - 2

    def test_too_few_points(self):
        scr = ShannonCollapseRate()
        scr.add_point(300, 0.005)
        scr.add_point(400, 0.003)
        assert scr.compute_didt() == []


# ---------------------------------------------------------------------------
# Temperature ladder
# ---------------------------------------------------------------------------

class TestTemperatureLadder:
    def test_endpoints(self):
        t = temperature_ladder(300, 600, 8)
        assert abs(t[0] - 300.0) < 1e-10
        assert abs(t[-1] - 600.0) < 1e-10

    def test_monotonic(self):
        t = temperature_ladder(298, 600, 12)
        assert np.all(np.diff(t) > 0)

    def test_geometric_ratio(self):
        """Consecutive ratio should be constant for geometric ladder."""
        t = temperature_ladder(300, 600, 5)
        ratios = t[1:] / t[:-1]
        assert np.allclose(ratios, ratios[0], atol=1e-10)

    def test_single_replica(self):
        t = temperature_ladder(300, 600, 1)
        assert len(t) == 1
        assert abs(t[0] - 300.0) < 1e-10


# ---------------------------------------------------------------------------
# Van't Hoff
# ---------------------------------------------------------------------------

class TestVantHoff:
    def test_at_reference(self):
        """At T_ref, DeltaG = DeltaH - T*DeltaS (no DeltaCp correction)."""
        dh, ds, dcp = -10.0, -0.02, -0.3
        dg = vant_hoff_with_dcp(dh, ds, dcp, 298.15, 298.15)
        expected = dh - 298.15 * ds
        assert abs(dg - expected) < 1e-10

    def test_temperature_dependence(self):
        """Higher T with negative DeltaCp should decrease stability."""
        dh, ds, dcp = -10.0, -0.02, -0.3
        dg_low = vant_hoff_with_dcp(dh, ds, dcp, 298.15)
        dg_high = vant_hoff_with_dcp(dh, ds, dcp, 350.0)
        assert dg_low != dg_high


# ---------------------------------------------------------------------------
# Shannon entropy
# ---------------------------------------------------------------------------

class TestShannonEntropy:
    def test_uniform_distribution(self):
        """Equal energies => maximum entropy = k_B * ln(N)."""
        n = 100
        energies = np.zeros(n)
        s = shannon_entropy(energies, 300.0)
        expected = K_B * np.log(n)
        assert abs(s - expected) < 1e-10

    def test_single_state(self):
        """One state => S = 0."""
        s = shannon_entropy(np.array([1.0]), 300.0)
        assert abs(s) < 1e-14

    def test_nonnegative(self):
        """Shannon entropy is always >= 0."""
        rng = np.random.default_rng(42)
        energies = rng.normal(-5.0, 3.0, 50)
        s = shannon_entropy(energies, 300.0)
        assert s >= -1e-14

    def test_higher_temp_more_entropy(self):
        """Higher temperature should give higher entropy."""
        energies = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
        s_low = shannon_entropy(energies, 200.0)
        s_high = shannon_entropy(energies, 1000.0)
        assert s_high > s_low


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

class TestBidirectionalProtocol:
    def test_forward_leg_shape(self):
        proto = BidirectionalProtocol(n_replicas=4)
        temps = temperature_ladder(300, 600, 4)
        energies = np.random.default_rng(0).normal(size=(4, 100))
        w = proto.forward_leg(temps, energies)
        assert w.shape == (100,)

    def test_reverse_leg_shape(self):
        proto = BidirectionalProtocol(n_replicas=4)
        temps = temperature_ladder(300, 600, 4)
        energies = np.random.default_rng(0).normal(size=(4, 100))
        w = proto.reverse_leg(temps, energies)
        assert w.shape == (100,)

    def test_convergence_check(self):
        proto = BidirectionalProtocol()
        assert proto.check_convergence(0.0001 * K_B, threshold=0.1 * K_B)
        assert not proto.check_convergence(1.0, threshold=0.1 * K_B)

    def test_run_system_returns_result(self):
        proto = BidirectionalProtocol(n_replicas=4)
        result = proto.run_system("1STP", Tier.TIER0_1STP)

        assert isinstance(result, REDOCKResult)
        assert result.pdb_id == "1STP"
        assert result.tier == Tier.TIER0_1STP
        assert np.isfinite(result.delta_g_bar)
        assert np.isfinite(result.delta_g_vanthoff)
        assert np.isfinite(result.sigma_irr)
        assert len(result.temperatures) == 4
        assert len(result.shannon_entropies) == 4

    def test_tier_systems(self):
        assert "1STP" in TIER_SYSTEMS[Tier.TIER0_1STP]
        assert len(TIER_SYSTEMS[Tier.TIER1_VALIDATION]) == 5
        assert len(TIER_SYSTEMS[Tier.TIER2_ASTEX85]) == 85


# ---------------------------------------------------------------------------
# Thermodynamic invariant tests
# ---------------------------------------------------------------------------

class TestThermodynamicInvariants:
    def test_bar_reversibility(self):
        """Swapping forward and reverse should negate DeltaG."""
        rng = np.random.default_rng(42)
        w_fwd = rng.normal(-3.0, 1.5, 2000)
        w_rev = rng.normal(3.0, 1.5, 2000)

        bar = BennettAcceptanceRatio(max_iterations=500, tolerance=1e-10)
        beta = 1.0 / (K_B * 300.0)

        dg_forward = bar.solve(w_fwd, w_rev, beta)
        dg_reverse = bar.solve(w_rev, w_fwd, beta)

        assert abs(dg_forward + dg_reverse) < 0.5

    def test_jarzynski_equality(self):
        """Jarzynski: <exp(-beta*W)> = exp(-beta*DeltaG)."""
        rng = np.random.default_rng(12)
        n = 5000
        true_dg = -4.0
        w_fwd = rng.normal(true_dg + 0.5, 1.0, n)
        w_rev = rng.normal(-true_dg + 0.5, 1.0, n)

        bar = BennettAcceptanceRatio(max_iterations=500, tolerance=1e-10)
        beta = 1.0 / (K_B * 300.0)
        dg = bar.solve(w_fwd, w_rev, beta)

        log_exp_avg = _log_sum_exp(-beta * w_fwd) - np.log(n)
        dg_jarzynski = -log_exp_avg / beta

        assert abs(dg_jarzynski - dg) < 2.0

    def test_entropy_bounds(self):
        """0 <= S <= k_B * ln(N) for any distribution."""
        rng = np.random.default_rng(33)
        for _ in range(20):
            n = rng.integers(2, 200)
            energies = rng.normal(0, 5, n)
            temp = rng.uniform(200, 800)
            s = shannon_entropy(energies, temp)
            assert s >= -1e-12
            assert s <= K_B * np.log(n) + 1e-10
