"""Tests for the FastAPI + HTMX prototype in webapp/.

Skipped when the optional `web` extra is not installed, so the core suite
still runs with numpy + scipy only.
"""
import html
import inspect
import json
import re

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("webapp")   # the sdist ships this file but not webapp/

from fastapi.testclient import TestClient

import optpricer as op
from optpricer.calibration import VolSurface
from webapp import engine
from webapp.main import app

BASE = {"S0": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "q": 0.0, "sigma": 0.2}


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _engine(res, name):
    return next(e for e in res["engines"] if e["name"] == name)


# ---------------------------------------------------------------------------
# JSON API: numbers must match the library
# ---------------------------------------------------------------------------
class TestPriceAPI:
    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_black_scholes_matches_the_library_exactly(self, client, kind):
        res = client.post("/api/price", json={**BASE, "kind": kind}).json()
        ref = op.bs_price(op.OptionSpec(S0=100, K=100, T=1, r=0.05, sigma=0.2), kind)
        assert _engine(res, "Black-Scholes")["price"] == ref

    def test_engines_agree_with_black_scholes(self, client):
        res = client.post("/api/price", json={**BASE, "kind": "call"}).json()
        bs = _engine(res, "Black-Scholes")["price"]
        assert abs(_engine(res, "Finite difference")["price"] - bs) < 2e-3
        assert abs(_engine(res, "Binomial (European)")["price"] - bs) < 1e-2
        mc = _engine(res, "Monte Carlo")
        assert abs(mc["price"] - bs) < 4 * mc["stderr"]

    def test_american_put_carries_an_early_exercise_premium(self, client):
        res = client.post("/api/price", json={**BASE, "kind": "put"}).json()
        amer = _engine(res, "Binomial (American)")["price"]
        euro = _engine(res, "Binomial (European)")["price"]
        assert amer > euro

    def test_greeks_match_the_library(self, client):
        res = client.post("/api/price", json={**BASE, "kind": "call"}).json()
        ref = op.bs_greeks(op.OptionSpec(S0=100, K=100, T=1, r=0.05, sigma=0.2), "call")
        for k, v in ref.items():
            assert res["greeks"][k] == pytest.approx(v, rel=1e-12)

    @pytest.mark.parametrize("override,match", [
        ({"T": 0.0}, "Expiry"),            # bs_price_vec returns NaN here instead of raising
        ({"T": -1.0}, "Expiry"),
        ({"sigma": 0.0}, "Volatility"),
        ({"S0": -5.0}, "Spot"),
        ({"kind": "straddle"}, "Option type"),
        ({"mc_paths": 10**9}, "Monte Carlo paths"),   # the one-line denial of service
        ({"r": 3.0}, "Rate"),
    ])
    def test_bad_input_is_a_400_with_a_readable_message(self, client, override, match):
        resp = client.post("/api/price", json={**BASE, "kind": "call", **override})
        assert resp.status_code == 400
        assert match in resp.json()["error"]


# ---------------------------------------------------------------------------
# Surface: fit, serialize, rehydrate
# ---------------------------------------------------------------------------
class TestSurfaceAPI:
    def test_fitted_surface_is_loadable_and_accurate(self, client):
        res = client.post("/api/surface/fit", json={"S0": 100, "r": 0.03}).json()
        surf = VolSurface.from_dict(res["surface"])
        assert len(surf.slices) == len(engine.SURFACE_EXPIRIES)
        assert max(f["rmse_bps"] for f in res["fit"]) < 25.0

    def test_local_vol_from_a_client_held_surface(self, client):
        surf = client.post("/api/surface/fit", json={"S0": 100, "r": 0.03}).json()["surface"]
        lv = client.post("/api/surface/localvol",
                         json={"surface": surf, "S0": 100, "t": 0.5, "r": 0.03}).json()
        assert np.all(np.isfinite(lv["local_vol"])) and min(lv["local_vol"]) > 0.0
        assert lv["extrapolated"] is False

    def test_tampered_surface_is_refused(self, client):
        surf = client.post("/api/surface/fit", json={}).json()["surface"]
        surf["slices"][0]["extra"] = 1
        resp = client.post("/api/surface/localvol", json={"surface": surf, "S0": 100, "t": 0.5})
        assert resp.status_code == 400 and "unrecognized" in resp.json()["error"]

    def test_oversized_surface_is_refused_before_parsing(self):
        huge = json.dumps({"schema": "optpricer.volsurface", "version": 1,
                           "slices": [{"a": 0.04}] * 20_000})
        with pytest.raises(engine.InputError, match="exceeds"):
            engine.load_surface(huge)


# ---------------------------------------------------------------------------
# HTMX fragments
# ---------------------------------------------------------------------------
class TestHTMX:
    def test_index_loads_htmx_and_plotly(self, client):
        page = client.get("/").text
        assert "htmx.org@2.0.4" in page and "plotly-2.35.2" in page
        assert 'hx-post="/ui/price"' in page and 'hx-post="/ui/surface"' in page

    def test_price_fragment(self, client):
        frag = client.post("/ui/price", data={**{k: str(v) for k, v in BASE.items()},
                                              "kind": "call"}).text
        assert "Black-Scholes" in frag and "ladder-chart" in frag

    def test_bad_input_renders_an_inline_error_with_status_200(self, client):
        """HTMX does not swap 4xx responses, so a 400 would look like nothing happened."""
        resp = client.post("/ui/price", data={**{k: str(v) for k, v in BASE.items()},
                                              "T": "0", "kind": "call"})
        assert resp.status_code == 200
        assert 'role="alert"' in resp.text and "Expiry" in resp.text

    def test_blank_field_is_an_error_not_a_crash(self, client):
        resp = client.post("/ui/price", data={**{k: str(v) for k, v in BASE.items()},
                                              "sigma": "", "kind": "call"})
        assert resp.status_code == 200 and "Volatility is required" in resp.text

    def test_stateless_round_trip_through_the_page(self, client):
        """The surface travels to the browser in a hidden field and back again."""
        frag = client.post("/ui/surface", data={"S0": "100", "r": "0.03", "q": "0",
                                                "atm_vol": "0.2", "skew": "-0.3",
                                                "curvature": "0.6", "term_slope": "0.02"}).text
        hidden = re.search(r'name="surface" value="([^"]*)"', frag).group(1)
        surface_json = html.unescape(hidden)
        VolSurface.from_json(surface_json)                      # parses cleanly
        lv = client.post("/ui/localvol", data={"surface": surface_json, "S0": "100",
                                               "t": "0.5", "r": "0.03", "q": "0"}).text
        assert "lv-chart" in lv and 'role="alert"' not in lv


# ---------------------------------------------------------------------------
# Server hazards identified before building
# ---------------------------------------------------------------------------
class TestServerHazards:
    def test_every_route_is_sync(self):
        """CPU-bound NumPy inside `async def` would block the event loop for every
        request; plain `def` routes run in FastAPI's thread pool instead."""
        ours = [r for r in app.routes
                if getattr(getattr(r, "endpoint", None), "__module__", "") == "webapp.main"]
        assert len(ours) >= 7, "expected this app's page, fragment and API routes"
        for route in ours:          # FastAPI's own /openapi.json and /docs are exempt
            assert not inspect.iscoroutinefunction(route.endpoint), route.path

    def test_monte_carlo_is_serial(self, monkeypatch):
        """A process pool per request costs ~0.8 s and needs a __main__ guard."""
        seen = []
        real = op.euro_price_mc

        def spy(*a, **kw):
            seen.append(kw.get("n_workers"))
            return real(*a, **kw)

        monkeypatch.setattr(engine.op, "euro_price_mc", spy)
        engine.price_all(100, 100, 1, 0.05, 0, 0.2, "call")
        assert seen == [1]


class TestHonestDisplay:
    def test_clipped_local_vol_is_flagged(self):
        """dupire_local_vol clips to [0.01, 5.0] without warning. At an extrapolated
        t=3 on a steep smile, 4 of 81 points sat at the 5.0 cap and the page drew
        them as if computed."""
        s = engine.fit_surface(100, 0.05, 0, 0.2, -0.3, 0.6, 0.02)
        res = engine.local_vol_slice(s["surface_json"], 100, 3.0, 0.05, 0)
        assert res["clipped"] > 0
        assert any("clamped, not computed" in w for w in res["warnings"])

    def test_unclipped_slice_raises_no_flag(self):
        s = engine.fit_surface(100, 0.05, 0, 0.2, -0.3, 0.6, 0.02)
        res = engine.local_vol_slice(s["surface_json"], 100, 0.5, 0.05, 0)
        assert res["clipped"] == 0 and res["warnings"] == []

    def test_clip_bounds_come_from_the_library(self):
        """The page counts clipped points against the library's own bounds,
        imported rather than copied, so they cannot drift apart."""
        from optpricer import calibration

        assert engine.LOCAL_VOL_FLOOR is calibration.LOCAL_VOL_FLOOR
        assert engine.LOCAL_VOL_CAP is calibration.LOCAL_VOL_CAP

    def test_clip_note_is_shown_once(self):
        """The library now warns too. Its message has no count, so the page keeps
        its own counted note and drops the library's rather than saying it twice."""
        s = engine.fit_surface(100, 0.05, 0, 0.2, -0.3, 0.6, 0.02)
        res = engine.local_vol_slice(s["surface_json"], 100, 3.0, 0.05, 0)
        clip_notes = [w for w in res["warnings"] if "clamped, not computed" in w]
        assert len(clip_notes) == 1
        assert f"of {len(res['local_vol'])} points" in clip_notes[0]


# ---------------------------------------------------------------------------
# Regressions from the pre-PR adversarial review
# ---------------------------------------------------------------------------
class TestReviewRegressions:
    # --- the one finding rated important -----------------------------------
    def test_engine_never_swaps_global_warning_state(self):
        """catch_warnings swaps process-wide state; routes run on a thread pool,
        so concurrent requests could corrupt warning handling server-wide."""
        src = inspect.getsource(engine)
        assert "catch_warnings(" not in src

    def test_concurrent_local_vol_leaves_warning_filters_intact(self, client):
        import threading
        import warnings as _w

        before = list(_w.filters)
        surf = client.post("/api/surface/fit", json={"S0": 100, "r": 0.05}).json()["surface"]
        errs = []

        def hit():
            try:
                for t in (0.5, 3.0):
                    r = client.post("/api/surface/localvol",
                                    json={"surface": surf, "S0": 100, "t": t, "r": 0.05})
                    assert r.status_code == 200
            except Exception as e:  # noqa: BLE001 - a thread's exception is otherwise lost
                errs.append(e)

        threads = [threading.Thread(target=hit) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert not errs
        assert list(_w.filters) == before

    # --- nothing user-controllable may 500 ---------------------------------
    @pytest.mark.parametrize("override", [
        {"T": 1e-30}, {"sigma": 1e-15}, {"S0": 5e-324, "K": 5e-324},
        {"T": 1e-5}, {"sigma": 1e-4},
    ])
    def test_tiny_inputs_are_refused_not_crashed(self, client, override):
        body = {**BASE, "kind": "call", **override}
        assert client.post("/api/price", json=body).status_code == 400
        ui = client.post("/ui/price", data={k: str(v) for k, v in body.items()})
        assert ui.status_code == 200 and 'role="alert"' in ui.text

    def test_huge_mc_paths_is_a_400(self, client):
        r = client.post("/api/price", json={**BASE, "kind": "call", "mc_paths": 10**400})
        assert r.status_code == 400

    def test_deeply_nested_surface_is_a_clean_error(self, client):
        payload = "[" * 1000 + "]" * 1000                    # 2 KB, RecursionError
        r = client.post("/ui/localvol", data={"surface": payload, "S0": "100", "t": "0.5",
                                              "r": "0", "q": "0"})
        assert r.status_code == 200 and 'role="alert"' in r.text

    def test_oversized_integer_in_surface_is_a_clean_error(self, client):
        surf = client.post("/api/surface/fit", json={}).json()["surface"]
        text = json.dumps(surf).replace('"a": ', '"a": ' + "1" + "0" * 310 + ", \"_x\": ", 1)
        r = client.post("/ui/localvol", data={"surface": text, "S0": "100", "t": "0.5",
                                              "r": "0", "q": "0"})
        assert r.status_code == 200 and 'role="alert"' in r.text

    def test_negative_forward_surface_is_a_400_not_an_all_nan_chart(self, client):
        surf = client.post("/api/surface/fit", json={}).json()["surface"]
        surf["forward_curve"] = [[T, -100.0] for T, _ in surf["forward_curve"]]
        r = client.post("/api/surface/localvol", json={"surface": surf, "S0": 100, "t": 0.5})
        assert r.status_code == 400 and "forward" in r.json()["error"]

    # --- the body limit runs before FastAPI parses anything -----------------
    def test_oversized_body_is_refused_before_parsing(self, client):
        r = client.post("/api/surface/localvol", content=b"{" + b" " * 2_000_000 + b"}",
                        headers={"content-type": "application/json"})
        assert r.status_code == 413 and "exceeds" in r.json()["error"]

    def test_oversized_chunked_body_is_refused(self, client):
        def chunks():
            for _ in range(40):
                yield b" " * 50_000                          # 2 MB, no Content-Length
        r = client.post("/api/price", content=chunks(),
                        headers={"content-type": "application/json"})
        assert r.status_code == 413

    def test_normal_bodies_pass_the_limit(self, client):
        assert client.post("/api/price", json={**BASE, "kind": "call"}).status_code == 200

    # --- contract and presentation -----------------------------------------
    def test_schema_errors_are_400_with_the_documented_shape(self, client):
        r = client.post("/api/price", json={"S0": "not a number"})
        assert r.status_code == 400 and "error" in r.json()

    def test_total_volatility_is_capped(self, client):
        """Past sigma*sqrt(T) of 4 the FD grid drifts by tens of bps and MC can be
        dozens of standard errors wrong while reporting a tiny standard error."""
        r = client.post("/api/price", json={**BASE, "kind": "call", "sigma": 5.0, "T": 4.0})
        assert r.status_code == 400 and "sqrt(expiry)" in r.json()["error"]
        ok = client.post("/api/price", json={**BASE, "kind": "call", "sigma": 2.0, "T": 4.0})
        assert ok.status_code == 200

    def test_binomial_failure_no_longer_hides_the_other_engines(self, client):
        res = client.post("/api/price", json={**BASE, "kind": "call", "T": 10.0,
                                              "r": 0.5, "sigma": 0.01}).json()
        assert _engine(res, "Black-Scholes")["price"] is not None
        amer = _engine(res, "Binomial (American)")
        assert amer["price"] is None and "needs volatility above" in amer["unavailable"]

    def test_range_message_does_not_round_an_out_of_range_value_into_range(self):
        with pytest.raises(engine.InputError, match=r"got 30\.0000001"):
            engine.validate_option(100, 100, 30.0000001, 0.05, 0, 0.2, "call")

    def test_local_vol_time_survives_a_surface_refit(self, client):
        form = {"S0": "100", "r": "0.03", "q": "0", "atm_vol": "0.2", "skew": "-0.3",
                "curvature": "0.6", "term_slope": "0.02", "t": "1.75"}
        assert 'value="1.75"' in client.post("/ui/surface", data=form).text

    def test_spinner_rule_matches_the_indicator_itself(self):
        import pathlib
        css = pathlib.Path("webapp/templates/base.html").read_text()
        assert ".htmx-request.spin" in css

    def test_early_exercise_text_matches_the_numbers(self):
        """The page says an American call only beats the European with a dividend
        or a negative rate. Check both directions against the tree."""
        def premium(r, q):
            o = op.OptionSpec(S0=100, K=90, T=1.0, r=r, sigma=0.2, q=q)
            return op.crr(o, op.CALL, N=1000, american=True) - op.crr(o, op.CALL, N=1000)
        assert abs(premium(0.05, 0.0)) < 1e-9          # no dividend, positive rate: equal
        assert premium(-0.02, 0.0) > 0.1                # negative rate: early exercise pays
        assert premium(0.05, 0.03) > 0.0                # dividend below the rate: still pays
        page = pathlib_text("webapp/templates/index.html")
        assert "negative" in page and "never for a call on a stock with no dividend" not in page


def pathlib_text(path):
    import pathlib
    return pathlib.Path(path).read_text()
