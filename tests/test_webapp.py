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

    def test_clip_bounds_match_the_library(self):
        """If the library's clip changes, this flags the drift."""
        from optpricer.calibration import dupire_local_vol

        src = inspect.getsource(dupire_local_vol)
        assert f"np.clip(sigma_loc, {engine.LOCAL_VOL_FLOOR}, {engine.LOCAL_VOL_CAP})" in src
