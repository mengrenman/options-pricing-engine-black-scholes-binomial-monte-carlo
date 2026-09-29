"""Tests for the surface pane's local-vol views: the repriced smile, delta and dynamics,
forward consistency, and the view group that holds them.

Skipped when the optional `web` extra is not installed, like tests/test_webapp.py.
"""
import inspect
import json
import math
import pathlib
import re
import threading
from html.parser import HTMLParser

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("webapp")   # the sdist ships this file but not webapp/

from fastapi.testclient import TestClient

import optpricer as op
from optpricer.calibration import SVIParams, VolSurface
from webapp import engine, localvol
from webapp.main import DEFAULTS, LV_VIEW_LIST, SurfaceRequest, app

S0, R, Q = 100.0, 0.05, 0.0


def _fit(skew=-0.10, curvature=0.10, r=R, q=Q, spot=S0, atm_vol=0.2):
    return engine.fit_surface(spot, r, q, atm_vol, skew, curvature, 0.02)


def flat_surface(vol, spot=S0, r=R, q=Q, expiries=engine.SURFACE_EXPIRIES) -> VolSurface:
    """Total variance vol^2 T at every strike: constant implied and local vol."""
    slices = {T: SVIParams(a=vol * vol * T, b=0.0, rho=0.0, m=0.0, sigma=0.1, expiry=T)
              for T in expiries}
    curve = {T: spot * math.exp((r - q) * T) for T in expiries}
    return VolSurface(slices, forward_curve=curve, label="flat")


def term_surface(vol_at, spot=S0, r=R, q=Q, expiries=engine.SURFACE_EXPIRIES) -> VolSurface:
    """No smile (b = 0), only a term structure: total variance vol_at(T)^2 T at every strike,
    so local vol depends on time alone and the local-vol round trip is exact."""
    slices = {T: SVIParams(a=vol_at(T) ** 2 * T, b=0.0, rho=0.0, m=0.0, sigma=0.1, expiry=T)
              for T in expiries}
    curve = {T: spot * math.exp((r - q) * T) for T in expiries}
    return VolSurface(slices, forward_curve=curve, label="term")


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def fitted():
    """The app's default surface, as the page fits it."""
    return _fit()


@pytest.fixture(scope="module")
def steep():
    """The surface the old defaults fitted: butterfly arbitrage in the left wing."""
    return _fit(skew=-0.30, curvature=0.60)


def _form(surface_json, **over):
    return {"surface": surface_json, "S0": "100", "t": "1.0", "r": str(R), "q": str(Q), **over}


def _all_finite_or_none(obj):
    """True if every number nested in ``obj`` is finite (None is how a gap is sent)."""
    if isinstance(obj, dict):
        return all(_all_finite_or_none(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_all_finite_or_none(v) for v in obj)
    return not isinstance(obj, float) or math.isfinite(obj)


class _Views(HTMLParser):
    """The local-vol column's tabs and views, read out of a /ui/surface fragment."""

    def __init__(self):
        super().__init__()
        self.tabs, self.views, self.inputs = {}, {}, {}
        self._body_depth = None
        self._depth = 0
        self.body_children = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self._depth += 1
        if tag == "button" and a.get("role") == "tab" and a.get("data-view") in self._lv_keys():
            self.tabs[a["data-view"]] = a["aria-selected"] == "true"
        if tag == "div" and "view" in a.get("class", "").split() and a.get("data-group") == "lv":
            self.views[a["data-view"]] = {"hidden": "hidden" in a, "attrs": a}
        if tag == "input" and a.get("id"):
            self.inputs[a["id"]] = a
        if tag == "div" and "view-body" in a.get("class", "").split():
            self._body_depth = self._depth
        elif self._body_depth is not None and self._depth > self._body_depth:
            self.body_children += 1

    def handle_endtag(self, tag):
        if self._body_depth is not None and self._depth == self._body_depth:
            self._body_depth = None
        self._depth -= 1

    @staticmethod
    def _lv_keys():
        return {v["key"] for v in LV_VIEW_LIST}


def _parse(frag) -> _Views:
    p = _Views()
    p.feed(frag)
    return p


# ---------------------------------------------------------------------------
# Item 1: the default smile
# ---------------------------------------------------------------------------
class TestDefaultSmile:
    def test_defaults_are_the_mild_smile_everywhere(self, client):
        assert (DEFAULTS["skew"], DEFAULTS["curvature"]) == (-0.10, 0.10)
        req = SurfaceRequest()
        assert (req.skew, req.curvature) == (-0.10, 0.10)
        page = client.get("/").text
        assert re.search(r'name="skew" value="-0\.1"', page)
        assert re.search(r'name="curvature" value="0\.1"', page)

    def test_api_fit_with_no_body_uses_the_new_defaults(self, client):
        surf = VolSurface.from_dict(client.post("/api/surface/fit", json={}).json()["surface"])
        want = engine.fit_surface(100, 0, 0, 0.2, -0.10, 0.10, 0.02)["surface"]
        assert float(surf.iv_from_logm(-0.3, 1.0)) == pytest.approx(
            float(want.iv_from_logm(-0.3, 1.0)), abs=1e-12)

    @pytest.mark.parametrize("T,limit", [(1.0, 4.0), (2.0, 16.0)])
    def test_default_surface_is_repriced_within_a_few_vol_bps(self, fitted, T, limit):
        """Measured 2.5 and 13.7 vol bps at T = 1 and 2, in the left wing (4.1 and 16.4
        before the local vol was read at the middle of each time step)."""
        res = localvol.lv_smile(fitted["surface_json"], S0, T, R, Q)
        assert max(abs(d) for d in res["diff_bps"]) < limit
        assert res["clipped_share"] < 0.01

    @pytest.mark.parametrize("T,at_least,share", [(1.0, 900.0, 0.01), (2.0, 1200.0, 0.10)])
    def test_old_defaults_are_the_reason_they_changed(self, steep, T, at_least, share):
        """The steep surface has butterfly arbitrage in its extrapolated wing, so the
        local-vol PDE reprices it up to 1,046 vol bps off at T = 1 and 1,415 at T = 2, and
        at T = 2 a fifth of the nodes within four ATM standard deviations of the forward
        hold a clamped local vol."""
        res = localvol.lv_smile(steep["surface_json"], S0, T, R, Q)
        assert max(abs(d) for d in res["diff_bps"] if d is not None) > at_least
        assert res["clipped_share"] > share


# ---------------------------------------------------------------------------
# Item 4: forward consistency
# ---------------------------------------------------------------------------
LV_ROUTES = [
    ("/ui/localvol", "/api/surface/localvol"),
    ("/ui/lvsmile", "/api/surface/lvsmile"),
    ("/ui/lvdynamics", "/api/surface/lvdynamics"),
]


def _tampered(surface_dict, factor):
    out = json.loads(json.dumps(surface_dict))
    out["forward_curve"] = [[T, F * factor] for T, F in out["forward_curve"]]
    return out


def _tampered_at(surface_dict, expiry, factor):
    """The surface with its forward at one quoted expiry multiplied by ``factor``."""
    out = json.loads(json.dumps(surface_dict))
    out["forward_curve"] = [[T, F * factor if T == expiry else F]
                            for T, F in out["forward_curve"]]
    return out


class TestForwardConsistency:
    @pytest.mark.parametrize("ui,api", LV_ROUTES)
    def test_a_consistent_surface_is_accepted(self, client, fitted, ui, api):
        assert 'role="alert"' not in client.post(ui, data=_form(fitted["surface_json"],
                                                                t="0.5")).text
        body = {"surface": fitted["surface"].to_dict(), "S0": 100, "t": 0.5, "r": R}
        assert client.post(api, json=body).status_code == 200

    @pytest.mark.parametrize("ui,api", LV_ROUTES)
    def test_a_forward_a_million_times_too_large_is_refused(self, client, fitted, ui, api):
        """A feasibility study priced a call at 54.39 at S0 = 100 on such a surface, with no
        sign that anything was off."""
        surf = _tampered(fitted["surface"].to_dict(), 1e6)
        frag = client.post(ui, data=_form(json.dumps(surf), t="0.5")).text
        assert 'role="alert"' in frag and "forward" in frag
        resp = client.post(api, json={"surface": surf, "S0": 100, "t": 0.5, "r": R})
        assert resp.status_code == 400 and "forward" in resp.json()["error"]

    @pytest.mark.parametrize("ui,api", LV_ROUTES)
    def test_a_surface_fitted_at_another_spot_is_refused(self, client, fitted, ui, api):
        frag = client.post(ui, data=_form(fitted["surface_json"], S0="120", t="0.5")).text
        assert 'role="alert"' in frag and "different spot, rate or dividend yield" in frag
        body = {"surface": fitted["surface"].to_dict(), "S0": 120, "t": 0.5, "r": R}
        assert client.post(api, json=body).status_code == 400

    @pytest.mark.parametrize("field,value", [("r", "0.10"), ("q", "0.03")])
    def test_a_surface_fitted_at_another_rate_or_yield_is_refused(self, client, fitted,
                                                                  field, value):
        frag = client.post("/ui/lvsmile", data=_form(fitted["surface_json"],
                                                     **{field: value})).text
        assert 'role="alert"' in frag and "forward" in frag

    def test_the_tolerance_is_half_a_percent(self, fitted):
        surf = fitted["surface"].to_dict()
        for factor, ok in [(1.004, True), (0.996, True), (1.006, False), (0.994, False)]:
            s = VolSurface.from_dict(_tampered(surf, factor))
            if ok:
                engine.check_forward(s, S0, 1.0, R, Q)
            else:
                with pytest.raises(engine.InputError, match="forward"):
                    engine.check_forward(s, S0, 1.0, R, Q)

    @pytest.mark.parametrize("r", [0.30, 0.50])
    def test_a_correct_surface_is_not_refused_at_a_high_rate(self, r):
        """Between quoted expiries the forward is interpolated linearly, which sits above
        the exponential: 1.1% at r = 30% and 3.2% at r = 50%, at t = 1.5. A plain 0.5% rule
        would refuse a surface fitted at exactly these inputs."""
        s = _fit(r=r)["surface"]
        plain = s.forward_at(1.5, r, 0.0) / (S0 * math.exp(r * 1.5)) - 1.0
        assert plain > engine.FORWARD_TOLERANCE
        for t in (0.3, 0.75, 1.46, 1.5, 1.9):
            engine.check_forward(s, S0, t, r, 0.0)

    def test_the_slack_does_not_hide_a_tampered_quote(self):
        s = _fit(r=0.5)["surface"]
        bad = VolSurface.from_dict(_tampered(s.to_dict(), 1.02))
        with pytest.raises(engine.InputError, match="forward"):
            engine.check_forward(bad, S0, 1.5, 0.5, 0.0)

    @pytest.mark.parametrize("expiry,factor", [(0.5, 1e6), (0.5, 1.5), (0.25, 0.5), (0.1, 3.0)])
    @pytest.mark.parametrize("ui,api", LV_ROUTES[1:])
    def test_a_curve_edited_at_an_earlier_expiry_is_refused_by_the_pde_views(
            self, client, fitted, ui, api, expiry, factor):
        """The PDE reads the forward at every time step from 0 to T, so a forward edited at an
        expiry before T matters even when the forward at T itself is untouched. Checking T
        alone let this through: with the forward at 0.5 multiplied by 1e6 and T = 1, the
        Dynamics view priced the put at 36.22 against Black-Scholes 8.61."""
        surf = _tampered_at(fitted["surface"].to_dict(), expiry, factor)
        assert VolSurface.from_dict(surf).forward_at(1.0, R, Q) == pytest.approx(
            S0 * math.exp(R), rel=1e-9)                        # T = 1 is untouched
        frag = client.post(ui, data=_form(json.dumps(surf), t="1.0")).text
        assert 'role="alert"' in frag and f"forward at t={expiry:g}" in frag
        resp = client.post(api, json={"surface": surf, "S0": 100, "t": 1.0, "r": R})
        assert resp.status_code == 400 and "forward" in resp.json()["error"]

    def test_only_the_expiries_the_pde_reads_are_checked(self, fitted):
        """A time before the quoted expiry at or after T reads only the forward points up to
        that expiry, so an edit beyond it changes nothing the view uses."""
        surf = fitted["surface"].to_dict()
        late = VolSurface.from_dict(_tampered_at(surf, 2.0, 5.0))
        engine.check_forward(late, S0, 1.0, R, Q, through_expiry=True)
        with pytest.raises(engine.InputError, match="forward at t=1.5"):
            engine.check_forward(late, S0, 1.5, R, Q, through_expiry=True)   # 1.5 leans on 2.0
        early = VolSurface.from_dict(_tampered_at(surf, 0.1, 2.0))
        with pytest.raises(engine.InputError, match="forward at t=0.1 "):
            engine.check_forward(early, S0, 0.25, R, Q, through_expiry=True)
        engine.check_forward(early, S0, 0.25, R, Q)          # t alone is exact there: passes

    def test_the_slice_view_reads_one_time_so_it_checks_one(self, client, fitted):
        """Dupire's formula at time t reads only forward_at(t), so /ui/localvol keeps checking
        that alone (and keeps extrapolating past the last expiry)."""
        surf = _tampered_at(fitted["surface"].to_dict(), 0.5, 1e6)
        frag = client.post("/ui/localvol", data=_form(json.dumps(surf), t="1.0")).text
        assert 'role="alert"' not in frag
        frag = client.post("/ui/localvol", data=_form(json.dumps(surf), t="0.5")).text
        assert 'role="alert"' in frag

    @pytest.mark.parametrize("r", [0.05, 0.30])
    def test_the_extra_check_accepts_a_correct_surface_at_every_time(self, r):
        s = _fit(r=r)["surface"]
        for t in (0.05, 0.1, 0.3, 0.75, 1.0, 1.46, 1.9, 2.0):
            engine.check_forward(s, S0, t, r, 0.0, through_expiry=True)

    def test_the_tolerance_constant_sits_beside_the_check_not_in_the_limits_block(self):
        """Phase 1 adds its own constants after MC_SEED; keeping this one away from them
        keeps the two branches from conflicting when they are merged."""
        src = pathlib.Path(engine.__file__).read_text()
        assert src.index("FORWARD_TOLERANCE = 0.005") > src.index("def load_surface(")
        assert src.index("FORWARD_TOLERANCE = 0.005") < src.index("def check_forward(")

    @pytest.mark.parametrize("ui,api", LV_ROUTES)
    def test_a_surface_with_no_forward_curve_is_an_input_error_not_a_crash(
            self, client, fitted, ui, api):
        """forward_at raises a bare ValueError there, which would have been a 500."""
        surf = fitted["surface"].to_dict()
        surf["forward_curve"] = []
        with pytest.raises(engine.InputError):
            engine.check_forward(VolSurface.from_dict(surf), S0, 0.5, R, Q)
        frag = client.post(ui, data=_form(json.dumps(surf), t="0.5"))
        assert frag.status_code == 200 and 'role="alert"' in frag.text
        resp = client.post(api, json={"surface": surf, "S0": 100, "t": 0.5, "r": R})
        assert resp.status_code == 400 and "error" in resp.json()

    def test_the_existing_view_still_extrapolates_beyond_the_quoted_expiries(self, client,
                                                                             fitted):
        frag = client.post("/ui/localvol", data=_form(fitted["surface_json"], t="3.0")).text
        assert 'role="alert"' not in frag and "extrapolated" in frag
        body = {"surface": fitted["surface"].to_dict(), "S0": 100, "t": 5.0, "r": R}
        res = client.post("/api/surface/localvol", json=body).json()
        assert res["extrapolated"] is True

    def test_only_the_new_views_restrict_the_time(self, client, fitted):
        for t, bad in [("0.04", True), ("0.05", False), ("2.0", False), ("2.5", True)]:
            for route in ("/ui/lvsmile", "/ui/lvdynamics"):
                frag = client.post(route, data=_form(fitted["surface_json"], t=t)).text
                assert ('role="alert"' in frag) is bad, (route, t)
                if bad:
                    assert "Time must be in [0.05, 2]" in frag


# ---------------------------------------------------------------------------
# The memo and the grid
# ---------------------------------------------------------------------------
class TestMemo:
    def test_memoized_prices_equal_unmemoized_ones_exactly(self, fitted):
        """Against the same function read at the same times: the memo changes nothing."""
        surf = fitted["surface"]
        table = localvol.LocalVolTable(surf, R, Q, 1.0)
        plain = localvol._mid_step(op.dupire_local_vol_func(surf, R, Q), 1.0)
        for K, kind in [(90.0, "put"), (100.0, "call"), (115.0, "call"), (100.0, "call")]:
            kw = {"N_S": 300, "N_t": 100, "ref_vol": 0.25, "S_max_mult": 4.0}
            assert (op.fd_price_local_vol(S0, K, 1.0, R, Q, table, kind, **kw)
                    == op.fd_price_local_vol(S0, K, 1.0, R, Q, plain, kind, **kw))

    def test_the_memo_lives_in_the_request_never_in_module_state(self, fitted):
        surf = fitted["surface"]
        a = localvol.LocalVolTable(surf, R, Q, 1.0)
        b = localvol.LocalVolTable(surf, R, Q, 1.0)
        S = np.linspace(50.0, 200.0, 301)
        a(S, 0.5)
        assert len(a._memo) == 1 and len(b._memo) == 0
        assert not hasattr(localvol.LocalVolTable, "_memo")
        state = [k for k, v in vars(localvol).items()
                 if not k.startswith("__") and isinstance(v, (dict, list, set)) and v]
        assert state == []                               # no cache, registry or counter

    def test_cached_arrays_are_read_only(self, fitted):
        table = localvol.LocalVolTable(fitted["surface"], R, Q, 1.0)
        values = table(np.linspace(50.0, 200.0, 301), 0.5)
        with pytest.raises(ValueError):
            values[0] = 9.0

    def test_a_different_grid_is_never_handed_another_grids_values(self, fitted):
        table = localvol.LocalVolTable(fitted["surface"], R, Q, 1.0)
        a = table(np.linspace(50.0, 200.0, 301), 0.5)
        b = table(np.linspace(60.0, 210.0, 301), 0.5)
        c = table(np.linspace(50.0, 200.0, 151), 0.5)
        assert not np.array_equal(a, b) and a.shape != c.shape

    def test_local_vol_is_read_half_a_step_after_the_time_the_solver_asks_for(self, fitted):
        """The solver asks for the step from t + dt back to t at t, the earlier end. Dupire's
        formula is evaluated at t + dt / 2, and the memo is still keyed on the solver's t."""
        surf, T = fitted["surface"], 1.0
        table = localvol.LocalVolTable(surf, R, Q, T)
        plain = op.dupire_local_vol_func(surf, R, Q)
        S = np.linspace(60.0, 160.0, 201)
        half = 0.5 * T / localvol.LV_N_T
        assert half == 0.005
        assert np.array_equal(table(S, 0.30), plain(S, 0.30 + half))
        assert not np.array_equal(table(S, 0.30), plain(S, 0.30))
        assert next(iter(table._memo))[0] == 0.30
        shifted = localvol._ShiftedTable(surf, R, Q, T, S0, 0.8, 300, 3)
        assert np.array_equal(shifted._at(0.30), plain(shifted.spots, 0.30 + half))

    def test_eleven_strikes_read_one_dupire_evaluation_per_time_step(self, fitted,
                                                                     monkeypatch):
        calls = []
        real = localvol.dupire_local_vol_func

        def counting(surface, r, q, **kw):
            f = real(surface, r, q, **kw)

            def g(S, t):
                calls.append(t)
                return f(S, t)
            return g

        monkeypatch.setattr(localvol, "dupire_local_vol_func", counting)
        res = localvol.lv_smile(fitted["surface_json"], S0, 1.0, R, Q)
        assert len(res["strike"]) == 11
        assert len(calls) == localvol.LV_N_T          # 1,100 without the memo

    def test_clipped_nodes_are_counted_from_the_table(self, fitted, steep):
        """The library's warnings are de-duplicated per process, so they cannot be counted."""
        ok = localvol.LocalVolTable(fitted["surface"], R, Q, 1.0)
        S = np.exp(np.linspace(math.log(30.0), math.log(300.0), 301))
        ok(S, 0.5)
        clipped, total = ok.clipped()
        assert (clipped, total) == (0, 301)
        bad = localvol.LocalVolTable(steep["surface"], R, Q, 1.0)
        bad(S, 1.0)
        clipped, total = bad.clipped()
        assert total == 301 and clipped > 0
        vals = bad._memo[next(iter(bad._memo))]
        at_bound = np.count_nonzero((vals >= localvol.LOCAL_VOL_CAP * (1 - 1e-9))
                                    | (vals <= localvol.LOCAL_VOL_FLOOR * (1 + 1e-9)))
        assert clipped == at_bound

    def test_only_the_nodes_in_the_window_are_counted(self, steep):
        """The share is taken over a window of spots, so it does not depend on how far out a
        particular view sizes its grid."""
        table = localvol.LocalVolTable(steep["surface"], R, Q, 2.0)
        S = np.exp(np.linspace(math.log(30.0), math.log(300.0), 301))
        table(S, 1.0)
        everything = table.clipped()
        vals = table._memo[next(iter(table._memo))]
        at_bound = ((vals >= localvol.LOCAL_VOL_CAP * (1 - 1e-9))
                    | (vals <= localvol.LOCAL_VOL_FLOOR * (1 + 1e-9)))
        clamped_spots = S[at_bound]
        assert everything[0] == clamped_spots.size > 0
        lo = float(clamped_spots.max()) * 1.01
        assert table.clipped(lo, 300.0)[0] == 0            # a window above every clamped node
        clipped, total = table.clipped(0.0, lo)
        assert (clipped, total) == (everything[0], int(np.count_nonzero(S <= lo)))
        assert table.clipped(1e5, 1e6) == (0, 0)            # a window with no nodes at all

    def test_strikes_snap_to_grid_nodes(self):
        dx = 0.01
        snapped = localvol._snap(S0, dx, [95.3, 100.0, 104.77, 131.9])
        nodes = np.log(snapped / S0) / dx
        assert np.allclose(nodes, np.round(nodes), atol=1e-9)
        asked = np.array([95.3, 100.0, 104.77, 131.9])
        assert np.all(np.abs(np.log(snapped / asked)) <= dx / 2 + 1e-12)

    def test_the_grid_widens_only_when_carry_pushes_a_strike_to_the_edge(self):
        vol = 0.2
        grid_vol, _, _ = localvol._grid(1.0, 0.05, 0.0, vol, reach=0.6)
        assert grid_vol == vol                                # 0.6 is 3 std, well inside
        grid_vol, half, _ = localvol._grid(2.0, 0.05, 0.0, 0.1, reach=0.524)
        assert grid_vol > 0.1                                 # 3.7 std of a 4 std grid
        assert half - 0.524 >= localvol.LV_EDGE_MARGIN * grid_vol * math.sqrt(2.0) - 1e-12


# ---------------------------------------------------------------------------
# Item 2: the repriced smile
# ---------------------------------------------------------------------------
class TestSmile:
    @pytest.mark.parametrize("vol", [0.1, 0.2, 0.4])
    @pytest.mark.parametrize("T", [0.05, 0.25, 0.5, 1.0, 2.0])
    def test_a_flat_vol_surface_is_repriced_within_one_vol_bp(self, vol, T):
        """The self-test: with constant vol, Dupire local vol is that vol and the PDE
        must give back the Black-Scholes smile, which is flat."""
        s = flat_surface(vol)
        res = localvol.lv_smile(s.to_json(), S0, T, R, Q)
        assert res["unpriced"] == 0
        assert all(abs(d) < 1.0 for d in res["diff_bps"]), res["diff_bps"]
        assert all(v == pytest.approx(vol, abs=2e-4) for v in res["iv_pde"])

    def test_flat_surface_with_carry_that_reaches_the_grid_edge(self):
        """10% vol for two years at r = 5% put the far strike 0.3 standard deviations from
        the edge of the 4-standard-deviation grid, and it repriced 2.9 vol bps low."""
        res = localvol.lv_smile(flat_surface(0.1).to_json(), S0, 2.0, 0.05, 0.0)
        assert max(abs(d) for d in res["diff_bps"]) < 1.0

    @pytest.mark.parametrize("vol,T,r,q", [(0.8, 2.0, 0.05, 0.0), (0.4, 2.0, 0.0, 0.3),
                                           (0.4, 0.1, 0.0, 0.3), (0.2, 1.0, 0.2, 0.02)])
    def test_a_flat_surface_is_within_five_vol_bps_at_large_carry_or_vol(self, vol, T, r, q):
        """The one-basis-point self-test holds at a 5% rate with no dividend and vol up to 40%.
        Outside that, a 300 x 100 grid is off by 1 to 4 bps (measured up to 3.8), which is
        discretization, not a bug: this is the README's scope for the claim."""
        res = localvol.lv_smile(flat_surface(vol, r=r, q=q).to_json(), S0, T, r, q)
        assert res["unpriced"] == 0
        assert max(abs(d) for d in res["diff_bps"]) < 5.0

    @pytest.mark.parametrize("slope,T", [(0.10, 2.0), (-0.10, 2.0), (0.10, 1.0), (0.05, 0.5)])
    def test_a_surface_that_only_has_a_term_structure_is_repriced_within_a_vol_bp(self, slope,
                                                                                  T):
        """With no smile, local vol depends on time alone and the round trip is exact, so the
        gap must vanish. The library's solver reads local vol at the earlier end of each time
        step, which alone gave -7.1 and +9.6 vol bps at T = 2 for slopes of +0.10 and -0.10
        (32 bps at +0.50); read at the middle of the step it is under half a bp."""
        s = term_surface(lambda T_: 0.3 + slope * math.sqrt(T_))
        res = localvol.lv_smile(s.to_json(), S0, T, R, Q)
        assert res["unpriced"] == 0
        assert max(abs(d) for d in res["diff_bps"]) < 1.0, res["diff_bps"]

    def test_reading_local_vol_at_the_step_start_would_fail_the_term_structure_test(self):
        """The check above has teeth: the same solve with the solver's own times leaves
        several vol bps, so the mid-step shift is what removes them."""
        s = term_surface(lambda T_: 0.3 + 0.10 * math.sqrt(T_))
        f = op.dupire_local_vol_func(s, R, Q)
        res = localvol.lv_smile(s.to_json(), S0, 2.0, R, Q)
        K = res["strike"][5]
        kind = res["kind"][5]
        p = op.fd_price_local_vol(S0, K, 2.0, R, Q, f, kind, N_S=localvol.LV_N_S,
                                  N_t=localvol.LV_N_T, ref_vol=res["grid_vol"],
                                  S_max_mult=localvol.LV_S_MAX_MULT)
        iv = localvol.implied_vol(S0, K, 2.0, R, Q, p, kind)
        want = float(s.iv_from_logm(math.log(K / s.forward_at(2.0, R, Q)), 2.0))
        assert abs(iv - want) * 1e4 > 5.0
        assert abs(res["atm_gap_bps"]) < 1.0

    def test_eleven_out_of_the_money_strikes_around_the_forward(self, fitted):
        res = localvol.lv_smile(fitted["surface_json"], S0, 1.0, R, Q)
        F = res["forward"]
        K = np.array(res["strike"])
        assert len(K) == 11 and np.all(np.diff(K) > 0)
        assert F == pytest.approx(S0 * math.exp(R), rel=1e-6)
        assert res["kind"] == ["put" if k < F else "call" for k in K]
        logm = np.log(K / F)
        limit = min(0.5, 3 * float(fitted["surface"].iv_from_logm(0.0, 1.0)))
        assert np.max(np.abs(logm)) <= limit + 0.01
        assert np.max(np.abs(logm)) > 0.9 * limit
        assert np.abs(logm[5]) == np.min(np.abs(logm))          # the middle one is ATM
        # log-spaced: equal steps in ln K, up to whole-node snapping
        steps = np.diff(logm)
        assert steps.max() - steps.min() < 0.01

    def test_strikes_are_priced_out_of_the_money_and_gaps_are_prices_not_vols(self, fitted):
        res = localvol.lv_smile(fitted["surface_json"], S0, 0.5, R, Q)
        surf = fitted["surface"]
        for K, iv_svi, iv_pde, gap in zip(res["strike"], res["iv_svi"], res["iv_pde"],
                                          res["diff_bps"]):
            assert iv_svi == pytest.approx(
                float(surf.iv_from_logm(math.log(K / res["forward"]), 0.5)), abs=1e-5)
            assert gap == pytest.approx((iv_pde - iv_svi) * 1e4, abs=0.2)

    def test_compares_against_the_surfaces_own_iv_at_the_forward_not_volsurface_iv(
            self, fitted):
        """VolSurface.iv(K, T) takes its forward from the curve at the surface's own
        interpolation; the comparison uses iv_from_logm(ln(K / forward_at(T, r, q)), T)."""
        surf = fitted["surface"]
        res = localvol.lv_smile(fitted["surface_json"], S0, 0.75, R, Q)
        F = surf.forward_at(0.75, R, Q)
        want = [round(float(surf.iv_from_logm(math.log(K / F), 0.75)), 5)
                for K in res["strike"]]
        assert res["iv_svi"] == pytest.approx(want, abs=2e-5)

    def test_ref_vol_is_the_largest_implied_vol_over_the_strikes_with_a_floor(self, fitted):
        res = localvol.lv_smile(fitted["surface_json"], S0, 1.0, R, Q)
        surf = fitted["surface"]
        atm = float(surf.iv_from_logm(0.0, 1.0))
        limit = min(0.5, 3 * atm)
        edge = surf.iv_from_logm(np.linspace(-limit, limit, 11), 1.0)
        assert res["ref_vol"] == pytest.approx(float(edge.max()), abs=1e-5)
        assert res["ref_vol"] > atm                      # the wing, not the ATM vol
        low = localvol.lv_smile(flat_surface(0.03).to_json(), S0, 1.0, R, Q)
        assert low["ref_vol"] == localvol.LV_MIN_REF_VOL

    def test_the_status_reports_the_atm_gap_and_the_share_of_clamped_nodes(self, client,
                                                                          fitted, steep):
        frag = client.post("/ui/lvsmile", data=_form(fitted["surface_json"], t="0.5")).text
        assert re.search(r'ATM gap [-+]\d+\.\d vol bps', frag)
        assert re.search(r"0\.00% of nodes at floor or cap", frag)
        assert 'class="warn"' not in frag
        few = client.post("/ui/lvsmile", data=_form(fitted["surface_json"], t="2.0")).text
        assert "0.03% of nodes at floor or cap" in few and 'class="warn"' in few
        bad = client.post("/ui/lvsmile", data=_form(steep["surface_json"], t="2.0")).text
        share = float(re.search(r"([\d.]+)% of nodes at floor or cap", bad).group(1))
        assert share > 10 and 'class="warn"' in bad
        assert bad.count('role="status"') == 1            # one status line above the chart

    @pytest.mark.parametrize("T", [0.5, 1.0, 2.0])
    def test_both_tabs_report_the_same_share_of_clamped_nodes_for_one_surface(
            self, fitted, steep, T):
        """The count is taken over the same window of spots in both views, whatever size grid
        each builds. Over the whole grid the old defaults read 31% on the smile tab and 8% on
        the dynamics tab at T = 1."""
        for s in (fitted, steep):
            smile = localvol.lv_smile(s["surface_json"], S0, T, R, Q)
            dyn = localvol.lv_dynamics(s["surface_json"], S0, None, T, R, Q)
            assert smile["clip_range"] == dyn["clip_range"]
            assert dyn["clipped_share"] == pytest.approx(smile["clipped_share"], abs=0.01)
            assert smile["grid_nodes"] > 0 and dyn["grid_nodes"] > 0

    def test_the_window_is_four_atm_standard_deviations_around_the_forward(self, fitted):
        res = localvol.lv_smile(fitted["surface_json"], S0, 1.0, R, Q)
        atm = float(fitted["surface"].iv_from_logm(0.0, 1.0))
        half = 4.0 * atm
        lo, hi = res["clip_range"]
        assert lo == pytest.approx(res["forward"] * math.exp(-half), rel=1e-6)
        assert hi == pytest.approx(res["forward"] * math.exp(half), rel=1e-6)

    def test_the_status_tooltip_says_the_gap_includes_forward_interpolation_at_high_carry(
            self, client):
        """Between quoted expiries the surface's forward is interpolated linearly, above the
        exponential the PDE grows spot at; the gap picks that mismatch up."""
        s = _fit(r=0.3, atm_vol=0.4)
        frag = client.post("/ui/lvsmile", data=_form(s["surface_json"], t="1.25",
                                                     r="0.3")).text
        res = localvol.lv_smile(s["surface_json"], S0, 1.25, 0.3, Q)
        assert res["forward_gap"] == pytest.approx(0.0089, abs=0.0005)
        assert "fit interpolates forwards linearly between quoted expiries" in frag
        quiet = client.post("/ui/lvsmile", data=_form(_fit()["surface_json"], t="1.0")).text
        assert "interpolates forwards" not in quiet
        assert localvol.lv_smile(_fit()["surface_json"], S0, 1.0, R, Q)["forward_gap"] == 0.0

    def test_the_chart_has_the_line_the_dots_and_a_gap_strip_on_a_shared_axis(self, client,
                                                                              fitted):
        frag = client.post("/ui/lvsmile", data=_form(fitted["surface_json"])).text
        assert 'id="lv-smile-chart"' in frag
        assert "setTimeout(function" in frag and "}, 0);" in frag     # drawn after the swap
        assert 'mode: "lines"' in frag and 'mode: "markers"' in frag
        assert 'xaxis: "x2"' in frag and 'matches: "x"' in frag
        assert "vol bps" in frag and "plotLayout(" in frag
        assert frag.count("Plotly.newPlot") == 1

    def test_the_json_twin_carries_the_same_numbers(self, client, fitted):
        body = {"surface": fitted["surface"].to_dict(), "S0": 100, "t": 1.0, "r": R}
        api = client.post("/api/surface/lvsmile", json=body).json()
        direct = localvol.lv_smile(fitted["surface_json"], S0, 1.0, R, Q)
        assert api == json.loads(json.dumps(direct))

    def test_every_number_is_finite_or_null(self, client, fitted, steep):
        for s in (fitted, steep):
            for T in (0.05, 0.5, 2.0):
                res = localvol.lv_smile(s["surface_json"], S0, T, R, Q)
                assert _all_finite_or_none(res)
                json.dumps(res, allow_nan=False)

    def test_extreme_but_legal_surfaces_never_500(self, client):
        """Every surface input at its limits; each is a result or an inline refusal."""
        for atm, skew, curv in [(0.01, -2.0, 5.0), (2.0, 2.0, 0.0), (0.05, -2.0, 0.0),
                                (1.0, 0.0, 5.0), (0.01, 0.0, 0.0)]:
            try:
                fit = _fit(skew=skew, curvature=curv, atm_vol=atm)
            except engine.InputError:
                continue
            for T in (0.05, 1.0, 2.0):
                for route in ("/ui/lvsmile", "/ui/lvdynamics"):
                    resp = client.post(route, data=_form(fit["surface_json"], t=str(T)))
                    assert resp.status_code == 200, (atm, skew, curv, T, route)

    def test_total_volatility_and_carry_are_capped(self):
        wild = flat_surface(2.0)
        with pytest.raises(engine.InputError, match="capped at 2"):
            localvol.lv_smile(wild.to_json(), S0, 2.0, R, Q)      # 2.0 x sqrt(2) = 2.83
        s = _fit(r=0.5)
        with pytest.raises(engine.InputError, match="Carry"):
            localvol.lv_smile(s["surface_json"], S0, 1.0, 0.5, Q)   # 0.5 > 0.3 x 0.8

    @pytest.mark.parametrize("price", [0.0, -1.0, float("nan"), 1e9])
    def test_prices_with_no_implied_vol_are_a_gap_not_a_number(self, price):
        assert localvol.implied_vol(S0, 110.0, 1.0, R, Q, price, "call") is None

    def test_price_below_intrinsic_has_no_implied_vol(self):
        intrinsic = S0 - 90.0 * math.exp(-R)
        assert localvol.implied_vol(S0, 90.0, 1.0, R, Q, intrinsic - 0.5, "call") is None

    @pytest.mark.parametrize("K,T,kind,vol", [
        (150.0, 0.25, "call", 0.2),
        (50.0, 0.25, "put", 0.3),
        (100.0, 1.0, "call", 0.05),
        (110.0, 2.0, "call", 1.5),
    ])
    def test_scalar_brent_recovers_the_vol(self, K, T, kind, vol):
        opt = op.OptionSpec(S0=S0, K=K, T=T, r=R, sigma=vol, q=Q)
        price = op.bs_price(opt, kind)
        got = localvol.implied_vol(S0, K, T, R, Q, price, kind)
        assert got == pytest.approx(vol, abs=1e-6)

    def test_a_price_newton_from_one_start_cannot_invert_is_inverted(self):
        """bs_implied_vol_vec starts every entry at one vol and returns NaN on many
        out-of-the-money prices; the guarded scalar Brent finds the root whenever there is one."""
        got = localvol.implied_vol(S0, 200.0, 0.25, R, Q, 0.01, "call")
        assert got is not None
        opt = op.OptionSpec(S0=S0, K=200.0, T=0.25, r=R, sigma=got, q=Q)
        assert op.bs_price(opt, "call") == pytest.approx(0.01, abs=1e-9)

    def test_bad_input_is_an_inline_error_and_a_400(self, client, fitted):
        frag = client.post("/ui/lvsmile", data=_form(fitted["surface_json"], t="abc"))
        assert frag.status_code == 200 and "Time must be a number" in frag.text
        resp = client.post("/api/surface/lvsmile",
                           json={"surface": fitted["surface"].to_dict(), "S0": 100, "t": 9.0})
        assert resp.status_code == 400 and "Time must be in" in resp.json()["error"]
        garbage = client.post("/ui/lvsmile", data=_form("{}"))
        assert 'role="alert"' in garbage.text


# ---------------------------------------------------------------------------
# Item 3: delta and dynamics
# ---------------------------------------------------------------------------
class TestDynamics:
    @pytest.mark.parametrize("vol,T", [(0.2, 1.0), (0.2, 0.25), (0.4, 0.5), (0.1, 2.0)])
    def test_on_a_flat_surface_local_vol_delta_and_gamma_are_black_scholes(self, vol, T):
        """With constant vol the two deltas coincide, so any gap on a real surface is the
        model, not the numerics."""
        s = flat_surface(vol)
        for K in (None, 92.0, 108.0):
            res = localvol.lv_dynamics(s.to_json(), S0, K, T, R, Q)
            price, delta, gamma = res["rows"]
            assert price["lv"] == pytest.approx(price["bs"], abs=2e-3)
            assert delta["lv"] == pytest.approx(delta["bs"], abs=5e-4)
            assert gamma["lv"] == pytest.approx(gamma["bs"], rel=5e-3)

    def test_local_vol_delta_differs_from_black_scholes_by_design(self, fitted):
        res = localvol.lv_dynamics(fitted["surface_json"], S0, None, 1.0, R, Q)
        delta = res["rows"][1]
        assert abs(delta["lv"] - delta["bs"]) > 0.03
        assert delta["lv"] < delta["bs"] < 0.0            # a put: the LV delta is more negative

    def test_local_vol_delta_agrees_with_milstein_monte_carlo(self, fitted):
        """The PDE's delta, from three node-shifted solves, against a bump-and-reprice of
        Milstein paths under the same local-vol function (common random numbers)."""
        res = localvol.lv_dynamics(fitted["surface_json"], S0, None, 1.0, R, Q)
        K, kind = res["strike"], res["kind"]
        f = op.dupire_local_vol_func(fitted["surface"], R, Q)

        def price(spot):
            paths = op.milstein_local_vol_paths(spot, R, Q, 1.0, 32, 20_000, f, seed=11)[-1]
            pay = np.maximum(K - paths, 0.0) if kind == "put" else np.maximum(paths - K, 0.0)
            return math.exp(-R) * float(pay.mean())

        mc_delta = (price(S0 + 1.0) - price(S0 - 1.0)) / 2.0
        assert mc_delta == pytest.approx(res["rows"][1]["lv"], abs=0.006)
        assert abs(mc_delta - res["rows"][1]["bs"]) > 0.03      # and not the sticky-strike one

    def test_node_shifted_windows_reproduce_solves_on_their_own_grids(self, fitted):
        """A solve at spot S0 e^{j dx} builds a grid that is the base grid moved j nodes, so
        it must read the same local vol as a window of one wide table."""
        surf = fitted["surface"]
        T, ref = 1.0, 0.25
        half = 4.0 * ref * math.sqrt(T)
        dx = 2 * half / localvol.LV_N_S
        table = localvol._ShiftedTable(surf, R, Q, T, S0, half, localvol.LV_N_S, 7)
        plain = localvol._mid_step(op.dupire_local_vol_func(surf, R, Q), T)
        K = 104.0                            # between nodes: the strike need not sit on one
        for j in (-7, -1, 0, 1, 4, 7):
            kw = {"N_S": localvol.LV_N_S, "N_t": localvol.LV_N_T, "ref_vol": ref,
                  "S_max_mult": 4.0}
            a = op.fd_price_local_vol(S0 * math.exp(j * dx), K, T, R, Q, table.window(j),
                                      "call", **kw)
            b = op.fd_price_local_vol(S0 * math.exp(j * dx), K, T, R, Q, plain, "call", **kw)
            assert a == pytest.approx(b, abs=1e-8)

    def test_a_window_refuses_a_grid_that_does_not_line_up(self, fitted):
        table = localvol._ShiftedTable(fitted["surface"], R, Q, 1.0, S0, 1.0, 300, 3)
        window = table.window(0)
        with pytest.raises(ValueError, match="line up"):
            window(np.linspace(50.0, 200.0, 301), 0.5)
        with pytest.raises(ValueError, match="line up"):
            window(table.spots[:200], 0.5)

    def test_all_solves_read_one_table_one_dupire_evaluation_per_time_step(self, fitted,
                                                                           monkeypatch):
        calls = []
        real = localvol.dupire_local_vol_func

        def counting(surface, r, q, **kw):
            f = real(surface, r, q, **kw)

            def g(S, t):
                calls.append(t)
                return f(S, t)
            return g

        monkeypatch.setattr(localvol, "dupire_local_vol_func", counting)
        localvol.lv_dynamics(fitted["surface_json"], S0, None, 0.5, R, Q)
        assert len(calls) == localvol.LV_N_T             # 11 to 13 solves, one table

    def test_the_default_strike_is_the_atm_forward_rounded(self, fitted):
        assert localvol.default_strike(100.0, 0.5, 0.05, 0.0) == 103.0
        assert localvol.default_strike(100.0, 1.0, 0.05, 0.0) == 105.0
        assert localvol.default_strike(1000.0, 1.0, 0.05, 0.0) == 1050.0
        assert localvol.default_strike(0.5, 1.0, 0.05, 0.0) == 0.526
        res = localvol.lv_dynamics(fitted["surface_json"], S0, None, 0.5, R, Q)
        assert res["strike"] == 103.0
        assert res["strike"] == localvol.lv_dynamics(
            fitted["surface_json"], S0, "  ", 0.5, R, Q)["strike"]

    def test_the_default_strike_never_leaves_the_strike_limits(self, fitted):
        """A spot near its cap carried forward is above the 1e6 cap on strikes; the default
        used to be refused for it, though the user never typed a strike."""
        assert localvol.default_strike(1e6, 1.0, 0.05, 0.0) == localvol.MAX_STRIKE
        assert localvol.default_strike(1e-4, 30.0, -0.5, 0.0) == engine.MIN_PRICE
        assert localvol.default_strike(2e5, 1.0, 0.05, 0.0) == 210000.0    # untouched inside
        big = _fit(spot=1e6)
        for K in (None, ""):
            res = localvol.lv_dynamics(big["surface_json"], 1e6, K, 1.0, R, Q)
            assert res["strike"] == localvol.MAX_STRIKE
            assert res["kind"] == "put" and _all_finite_or_none(res)

    def test_the_strike_is_priced_as_typed_not_moved_to_a_node(self, fitted):
        """The spots move by whole nodes, which keeps the strike's place among the nodes
        whatever it is, so nothing is gained by moving it (and 105 read as 104.81)."""
        for K in ("103", "105", "97.3"):
            res = localvol.lv_dynamics(fitted["surface_json"], S0, K, 0.5, R, Q)
            assert res["strike"] == float(K)
            assert "strike_asked" not in res
        res = localvol.lv_dynamics(fitted["surface_json"], S0, "105", 1.0, R, Q)
        iv = float(fitted["surface"].iv_from_logm(math.log(105.0 / res["forward"]), 1.0))
        assert res["iv_strike"] == pytest.approx(iv, abs=1e-5)       # the strike as typed
        opt = op.OptionSpec(S0=S0, K=105.0, T=1.0, r=R, sigma=iv, q=Q)
        assert res["rows"][0]["bs"] == pytest.approx(float(op.bs_price(opt, "put")), rel=1e-5)

    @pytest.mark.parametrize("vol,T", [(0.2, 1.0), (0.4, 0.5), (0.1, 2.0)])
    def test_an_unsnapped_strike_is_as_accurate_as_a_snapped_one_on_a_flat_surface(self, vol,
                                                                                    T):
        s = flat_surface(vol)
        F = S0 * math.exp(R * T)
        limit = min(0.5, 3 * vol * math.sqrt(T))
        for frac in (-0.9, -0.37, 0.0, 0.41, 0.9):
            res = localvol.lv_dynamics(s.to_json(), S0, F * math.exp(frac * limit), T, R, Q)
            price, delta, gamma = res["rows"]
            assert price["lv"] == pytest.approx(price["bs"], rel=1e-2, abs=1e-3)
            assert delta["lv"] == pytest.approx(delta["bs"], abs=1e-4)
            assert gamma["lv"] == pytest.approx(gamma["bs"], rel=5e-3)

    def test_strikes_outside_the_range_are_refused_with_the_range(self, fitted):
        with pytest.raises(engine.InputError, match=r"too far from the forward .* from "):
            localvol.lv_dynamics(fitted["surface_json"], S0, 500.0, 0.5, R, Q)
        with pytest.raises(engine.InputError, match="Strike"):
            localvol.lv_dynamics(fitted["surface_json"], S0, "abc", 0.5, R, Q)
        with pytest.raises(engine.InputError, match="Strike must be in"):
            localvol.lv_dynamics(fitted["surface_json"], S0, -3.0, 0.5, R, Q)

    def test_the_table_has_three_rows_and_the_chart_eleven_spots(self, fitted):
        res = localvol.lv_dynamics(fitted["surface_json"], S0, None, 0.5, R, Q)
        assert [r["name"] for r in res["rows"]] == ["Price", "Delta", "Gamma"]
        assert len(res["spot"]) == len(res["iv_lv"]) == len(res["spot_pct"]) == 11
        assert np.all(np.diff(res["spot"]) > 0)
        assert res["spot_pct"][5] == 0.0
        assert -8.0 < res["spot_pct"][0] < -3.0 and 3.0 < res["spot_pct"][-1] < 8.0
        assert all(v is not None for v in res["iv_lv"])

    def test_implied_vol_at_a_fixed_strike_falls_as_spot_rises_under_local_vol(self, fitted):
        """The sticky-local-vol answer: the smile does not move with spot, so the vol at a
        fixed strike is not fixed. A sticky-strike model would draw the flat line."""
        res = localvol.lv_dynamics(fitted["surface_json"], S0, None, 1.0, R, Q)
        iv = np.array(res["iv_lv"])
        assert np.all(np.diff(iv) < 0)
        assert iv[0] - iv[-1] > 0.005
        assert res["iv_lv"][5] == pytest.approx(res["iv_strike"], abs=2e-4)

    def test_spots_are_whole_nodes_apart(self, fitted):
        res = localvol.lv_dynamics(fitted["surface_json"], S0, None, 0.5, R, Q)
        steps = np.diff(np.log(res["spot"]))
        assert steps.max() - steps.min() < 1e-4       # equal steps: whole nodes each

    def test_the_fragment_labels_the_delta_plainly(self, client, fitted):
        frag = client.post("/ui/lvdynamics", data=_form(fitted["surface_json"], t="0.5")).text
        assert "sticky-local-vol" in frag and "sticky-strike" in frag
        assert "by design" in frag and "fixed in absolute spot" in frag
        table = re.search(r"<table.*?</table>", frag, re.DOTALL).group(0)
        assert re.findall(r'<th scope="row">([^<]*)</th>', table) == ["Price", "Delta", "Gamma"]
        assert "Local vol" in table and "Black-Scholes" in table
        assert 'id="lv-dynamics-chart"' in frag
        assert "setTimeout(function" in frag and "}, 0);" in frag
        assert 'name: "Sticky strike (flat)"' in frag
        assert 'name: "Local vol (sticky local vol)"' in frag     # the implied vol of its price
        assert "Local-vol price" not in frag

    def test_the_note_and_warning_are_one_short_line_each_so_the_page_does_not_grow(
            self, client, steep):
        """The 422 px column holds about 70 characters of 12 px text per line. The note was
        two sentences that wrapped to three lines, and the clamped-node warning added more;
        the page bottom is meant to stay under about 920 px. The long text is in tooltips."""
        frag = client.post("/ui/lvdynamics", data=_form(steep["surface_json"], t="2.0")).text
        notes = re.findall(r'<p class="note lv-note"[^>]*>([^<]*)</p>', frag)
        warns = re.findall(r'<p class="warn"[^>]*>([^<]*)</p>', frag)
        assert len(notes) == 1 and len(notes[0]) <= 62
        assert len(warns) == 1 and len(warns[0]) <= 62 and "floor or cap" in warns[0]
        assert frag.count("<p ") == 2

    def test_the_header_says_european_and_shows_the_strike_as_typed(self, client, fitted):
        frag = client.post("/ui/lvdynamics", data=_form(fitted["surface_json"], t="1.0",
                                                        K="105")).text
        head = re.search(r"<thead>.*?</thead>", frag, re.DOTALL).group(0)
        assert ">European put, K 105</th>" in head
        assert "OTM" not in re.sub(r'title="[^"]*"', "", head)      # only in the tooltip
        assert "out of the money against the forward 105.13" in head
        assert "(spot 100)" in head
        call = client.post("/ui/lvdynamics", data=_form(fitted["surface_json"], t="1.0",
                                                        K="112.5")).text
        assert ">European call, K 112.5</th>" in call

    def test_small_numbers_keep_their_significant_figures_in_the_table(self, client):
        """Fixed four decimals printed gamma as 0.0000 at a spot of 50,000 (it is 1.9e-5), so
        both models seemed to have none; the strike showed as 0.05 at a spot of 0.05."""
        big = _fit(spot=50_000.0)
        frag = client.post("/ui/lvdynamics",
                           data=_form(big["surface_json"], S0="50000", t="1.0")).text
        row = re.search(r'<th scope="row">Gamma</th><td class="num">([^<]*)</td>'
                        r'<td class="num">([^<]*)</td>', frag)
        assert row and all(float(v) > 0 for v in row.groups())
        assert all(len(v.replace(".", "").lstrip("0")) == 4 for v in row.groups())
        tiny = _fit(spot=0.05)
        res = localvol.lv_dynamics(tiny["surface_json"], 0.05, None, 1.0, R, Q)
        frag = client.post("/ui/lvdynamics",
                           data=_form(tiny["surface_json"], S0="0.05", t="1.0")).text
        assert f"K {localvol.sig(res['strike'], 5, True)}</th>" in frag
        assert res["strike"] == 0.0526                     # 0.05 e^0.05, three figures

    @pytest.mark.parametrize("x,digits,want", [
        (8.60093, 4, "8.601"), (-0.492, 4, "-0.4920"), (1.9e-5, 4, "0.00001900"),
        (2500.34, 4, "2500"), (104.81, 5, "104.81"), (0.0, 4, "0"), (None, 4, "n/a"),
        (float("nan"), 4, "n/a")])
    def test_sig_formats_to_significant_figures(self, x, digits, want):
        assert localvol.sig(x, digits) == want

    def test_sig_can_drop_trailing_zeros(self):
        assert localvol.sig(105.0, 5, True) == "105"
        assert localvol.sig(112.5, 5, True) == "112.5"
        assert localvol.sig(1050000.0, 5, True) == "1050000"
        assert localvol.sig(0.0525, 5, True) == "0.0525"

    def test_the_json_twin_takes_an_optional_strike(self, client, fitted):
        body = {"surface": fitted["surface"].to_dict(), "S0": 100, "t": 0.5, "r": R}
        default = client.post("/api/surface/lvdynamics", json=body).json()
        assert default["strike"] == 103.0
        given = client.post("/api/surface/lvdynamics", json={**body, "K": 95.0}).json()
        assert given["strike"] == 95.0 and given["kind"] == "put"
        assert _all_finite_or_none(default) and _all_finite_or_none(given)
        bad = client.post("/api/surface/lvdynamics", json={**body, "K": 900.0})
        assert bad.status_code == 400 and "too far" in bad.json()["error"]

    def test_a_surface_with_clamped_local_vol_says_so(self, client, steep):
        frag = client.post("/ui/lvdynamics",
                           data=_form(steep["surface_json"], t="1.0", K="80")).text
        assert "bounds, not computed values" in frag

    def test_every_number_is_finite_or_null(self, fitted, steep):
        for s in (fitted, steep):
            for T in (0.05, 0.5, 2.0):
                res = localvol.lv_dynamics(s["surface_json"], S0, None, T, R, Q)
                assert _all_finite_or_none(res)
                json.dumps(res, allow_nan=False)


# ---------------------------------------------------------------------------
# Item 5: the view group
# ---------------------------------------------------------------------------
SURFACE_FORM = {"S0": "100", "r": "0.05", "q": "0", "atm_vol": "0.2", "skew": "-0.1",
                "curvature": "0.1", "term_slope": "0.02"}


class TestViewGroup:
    def test_three_views_with_short_labels(self):
        assert [v["key"] for v in LV_VIEW_LIST] == ["at-t", "smile", "dynamics"]
        assert [v["label"] for v in LV_VIEW_LIST] == ["Local vol", "Repriced smile", "Dynamics"]
        assert [v["route"] for v in LV_VIEW_LIST] == ["/ui/localvol", "/ui/lvsmile",
                                                      "/ui/lvdynamics"]
        # Beside the 149 px time control in a 421.9 px column (font metrics, SF at weight 600,
        # uppercase): 31 characters of tab label are 246 px at 11 px with 0.02em spacing and
        # 12 px gaps, a 404 px row; the earlier 12 px / 0.03em was 427 px and wrapped.
        assert sum(len(v["label"]) for v in LV_VIEW_LIST) <= 31

    def test_the_tab_row_is_set_small_enough_to_share_a_line_with_the_time_input(self):
        css = pathlib.Path("webapp/templates/base.html").read_text()
        rule = re.search(r"\.lv-col \.tabs \[role=\"tab\"\] \{([^}]*)\}", css).group(1)
        assert "font-size: 11px" in rule and "letter-spacing: 0.02em" in rule
        assert "white-space: nowrap" in rule
        assert ".lv-col .bar { column-gap: 8px; }" in css
        assert ".lv-col form.inline input { width: 5em; }" in css

    @pytest.mark.parametrize("key", [v["key"] for v in LV_VIEW_LIST])
    def test_every_key_renders_that_tab_selected_and_only_that_view_visible(self, client, key):
        parsed = _parse(client.post("/ui/surface", data={**SURFACE_FORM, "lv_view": key}).text)
        assert parsed.tabs == {k: k == key for k in parsed.tabs}
        assert set(parsed.tabs) == {v["key"] for v in LV_VIEW_LIST}
        assert {k for k, v in parsed.views.items() if not v["hidden"]} == {key}
        assert set(parsed.views) == set(parsed.tabs)
        hidden = [v for k, v in parsed.views.items() if k != key]
        assert len(hidden) == 2 and all(v["hidden"] for v in hidden)
        assert (client.post("/ui/surface", data={**SURFACE_FORM, "lv_view": key}).text
                .count(f'name="lv_view" value="{key}"') == 1)

    @pytest.mark.parametrize("junk", ["", "nope", "<script>"])
    def test_an_unknown_view_falls_back_to_the_first(self, client, junk):
        parsed = _parse(client.post("/ui/surface", data={**SURFACE_FORM, "lv_view": junk}).text)
        assert [k for k, sel in parsed.tabs.items() if sel] == ["at-t"]

    def test_each_view_loads_from_its_route_with_the_form(self, client):
        parsed = _parse(client.post("/ui/surface", data=SURFACE_FORM).text)
        for v in LV_VIEW_LIST:
            a = parsed.views[v["key"]]["attrs"]
            assert a["hx-post"] == v["route"] and a["hx-include"] == "#lv-form"
            assert a["hx-trigger"] == "shown, refresh" and a["data-follows"] == "lv-form"

    def test_the_dynamics_view_keeps_its_input_outside_the_swapped_content(self, client):
        frag = client.post("/ui/surface", data={**SURFACE_FORM, "lv_view": "dynamics"}).text
        parsed = _parse(frag)
        dyn = parsed.views["dynamics"]["attrs"]
        assert dyn["hx-target"] == "find .view-body"
        for key in ("at-t", "smile"):
            assert "hx-target" not in parsed.views[key]["attrs"]
        strike = parsed.inputs["lv-strike"]
        assert strike["form"] == "lv-form" and strike["name"] == "K"
        assert strike["value"] == "" and strike["placeholder"] == "ATM forward"
        assert "lv-k-auto" not in parsed.inputs         # nothing to keep in step with time
        assert parsed.body_children == 0               # the swap target starts empty
        view = re.search(r'data-view="dynamics".*?<div class="view-body"></div>\s*</div>',
                         frag, re.DOTALL).group(0)
        assert view.index('id="lv-strike"') < view.index('class="view-body"')

    def test_the_strike_input_is_included_in_the_surface_form_so_it_survives_a_refit(
            self, client):
        page = client.get("/").text
        assert re.search(r'hx-include="#lv-t, #lv-view, #lv-strike"', page)

    def test_a_typed_strike_survives_a_refit_and_an_empty_one_stays_empty(self, client):
        """The field used to be prefilled with the ATM forward at the time in place when the
        surface was fitted, so moving the Time input left it stale (103 after T went from 0.5
        to 2, where the forward is 110.5) while reading as the default. Empty is the default
        now, and the server works it out for each request, so it cannot go stale."""
        edited = client.post("/ui/surface", data={**SURFACE_FORM, "K": " 95 "})
        assert _parse(edited.text).inputs["lv-strike"]["value"] == "95"
        for t in ("0.5", "2", ""):
            blank = client.post("/ui/surface", data={**SURFACE_FORM, "t": t, "K": ""})
            assert _parse(blank.text).inputs["lv-strike"]["value"] == ""
        assert 'name="K_auto"' not in client.post("/ui/surface", data=SURFACE_FORM).text

    def test_a_junk_strike_is_echoed_safely_and_refused_by_the_view(self, client, fitted):
        frag = client.post("/ui/surface", data={**SURFACE_FORM, "K": '"><script>x</script>'}).text
        assert "<script>x</script>" not in frag
        assert "&lt;script&gt;x" in frag
        bad = client.post("/ui/lvdynamics", data=_form(fitted["surface_json"], K="abc")).text
        assert 'role="alert"' in bad and "Strike must be a number" in bad

    def test_an_empty_strike_follows_the_time_input(self, client, fitted):
        """The default strike is the ATM forward at the request's time, whatever the field held
        when the surface was fitted."""
        for t, K in (("0.5", 103.0), ("1.0", 105.0), ("2.0", 111.0)):
            res = client.post("/api/surface/lvdynamics", json={
                "surface": fitted["surface"].to_dict(), "S0": 100, "t": float(t), "r": R}).json()
            assert res["strike"] == K

    def test_the_time_input_and_the_view_choice_still_survive_a_refit(self, client):
        frag = client.post("/ui/surface", data={**SURFACE_FORM, "t": "1.25",
                                                "lv_view": "smile"}).text
        assert 'value="1.25"' in frag and 'name="lv_view" value="smile"' in frag

    def test_chart_views_are_drawn_after_the_swap_settles(self, client, fitted):
        for route in ("/ui/localvol", "/ui/lvsmile", "/ui/lvdynamics"):
            frag = client.post(route, data=_form(fitted["surface_json"], t="0.5")).text
            assert re.search(r"setTimeout\(function \(\) \{.*Plotly\.newPlot", frag, re.DOTALL)
            assert "}, 0);" in frag, route

    def test_css_fills_the_column_and_places_the_new_rules_after_bar_tabs(self):
        css = pathlib.Path("webapp/templates/base.html").read_text()
        assert re.search(r"\.lv-col \.view \.chart \{ flex: 1 1 0; min-height: 260px", css)
        assert re.search(r"\.lv-col \.view-body \{ flex: 1 1 auto; display: flex; "
                         r"flex-direction: column; \}", css)
        assert ".lv-col .view-body > :not(.chart) { flex: 0 0 auto; }" in css
        lines = css.splitlines()
        i = next(n for n, line in enumerate(lines) if line.strip().startswith(".bar .tabs {"))
        block = "\n".join(lines[i + 1:i + 30])
        assert ".lv-col .tabs" in block and ".lv-col .view-body" in block
        assert "table.compact" in block and "label.lv-strike" in block

    def test_a_refit_after_the_toggle_keeps_the_3d_charts_mode(self, client):
        frag = client.post("/ui/surface", data=SURFACE_FORM).text
        assert 'onclick="surfaceMode(\'iv\')"' in frag and 'onclick="surfaceMode(\'tv\')"' in frag
        assert "window.surfaceChartMode" in frag


# ---------------------------------------------------------------------------
# Item 6: total variance on the 3D chart
# ---------------------------------------------------------------------------
class TestTotalVariance:
    def test_toggle_sits_in_the_heading_row_not_in_a_new_row(self, client):
        frag = client.post("/ui/surface", data=SURFACE_FORM).text
        tabs = re.search(r'<div class="tabs" role="tablist" data-group="s3d">(.*?)</div>',
                         frag, re.DOTALL).group(1)
        assert re.findall(r"<button[^>]*>([^<]*)</button>", tabs) == [
            "Implied vol surface", "Total variance"]
        assert "<h3>Implied volatility surface</h3>" not in frag
        assert frag.index('data-group="s3d"') < frag.index('id="surface-chart"')

    def test_total_variance_is_computed_in_the_browser_from_the_grid_it_has(self, client):
        frag = client.post("/ui/surface", data=SURFACE_FORM).text
        assert "G.iv.map((row, i) => row.map((v) => v * v * G.T[i]))" in frag
        assert frag.count("Plotly.newPlot(\"surface-chart\"") == 1

    def test_the_browser_side_formula_matches_the_surfaces_total_variance(self, fitted):
        """w = IV^2 T from the rounded grid the page ships, against the surface's own w."""
        surf = fitted["surface"]
        grid = fitted["grid"]
        for i in (0, 10, 29):
            T = grid["T"][i]
            mine = np.array(grid["iv"][i]) ** 2 * T
            ref = surf.total_var_from_logm(np.array(grid["k"]), T)
            assert np.allclose(mine, ref, atol=3e-5)


# ---------------------------------------------------------------------------
# Server hazards for the new routes
# ---------------------------------------------------------------------------
class TestHazards:
    def test_no_global_warning_state_is_swapped(self):
        assert "catch_warnings(" not in inspect.getsource(localvol)

    @pytest.mark.parametrize("path", ["webapp/localvol.py", "webapp/engine.py", "webapp/main.py",
                                      "tests/test_webapp_surface.py"])
    def test_lines_keep_to_the_projects_99_column_limit(self, path):
        """pyproject sets line-length = 99; ruff's default rules do not enforce it."""
        too_long = [n for n, line in enumerate(pathlib.Path(path).read_text().splitlines(), 1)
                    if len(line) > 99]
        assert too_long == []

    def test_the_new_routes_are_sync(self):
        paths = {"/ui/lvsmile", "/ui/lvdynamics", "/api/surface/lvsmile",
                 "/api/surface/lvdynamics"}
        routes = [r for r in app.routes if getattr(r, "path", "") in paths]
        assert {r.path for r in routes} == paths
        for r in routes:
            assert not inspect.iscoroutinefunction(r.endpoint), r.path

    def test_concurrent_requests_on_different_surfaces_do_not_share_a_memo(self, fitted,
                                                                           steep):
        """Threads share the process: a memo kept in module state would hand one request
        another's local vol. Each thread's answer must equal the sequential one."""
        cases = [(fitted, 0.5), (steep, 1.0), (fitted, 2.0), (steep, 0.5)]
        want = [(localvol.lv_smile(s["surface_json"], S0, T, R, Q),
                 localvol.lv_dynamics(s["surface_json"], S0, None, T, R, Q)) for s, T in cases]
        got, errs = [None] * len(cases) * 2, []

        def run(i):
            s, T = cases[i % len(cases)]
            try:
                got[i] = (localvol.lv_smile(s["surface_json"], S0, T, R, Q),
                          localvol.lv_dynamics(s["surface_json"], S0, None, T, R, Q))
            except Exception as exc:  # noqa: BLE001 - a thread's exception is otherwise lost
                errs.append(exc)

        threads = [threading.Thread(target=run, args=(i,)) for i in range(len(got))]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert not errs
        for i, pair in enumerate(got):
            assert pair == want[i % len(cases)]

    def test_hostile_surface_fields_are_clean_errors(self, client, fitted):
        for route in ("/ui/lvsmile", "/ui/lvdynamics"):
            for payload in ("[" * 1000 + "]" * 1000, "", "null", '{"schema": 1}'):
                resp = client.post(route, data=_form(payload))
                assert resp.status_code == 200 and 'role="alert"' in resp.text
        for field, value in [("S0", "0"), ("S0", "nan"), ("r", "9"), ("q", "abc"), ("t", "")]:
            resp = client.post("/ui/lvdynamics", data=_form(fitted["surface_json"],
                                                             **{field: value}))
            assert resp.status_code == 200 and 'role="alert"' in resp.text, (field, value)
