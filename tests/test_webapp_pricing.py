"""Tests for the pricing pane of the web app: the engines and Greeks tables, the grid-range
rule, implied-vol mode and the scenarios view.

Skipped when the optional `web` extra is not installed, like tests/test_webapp.py.
"""
import html
import json
import math
import pathlib
import re

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("webapp")

from fastapi.testclient import TestClient

import optpricer as op
from optpricer.fem import fem_price
from optpricer.pde import fd_greeks, fd_price
from webapp import engine
from webapp.main import app

BASE = {"S0": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "q": 0.0, "sigma": 0.2}
FORM = {k: str(v) for k, v in BASE.items()}
ARGS = (100.0, 100.0, 1.0, 0.05, 0.0, 0.2)
WEBAPP = pathlib.Path(engine.__file__).parent


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _entry(res, name):
    return next(e for e in res["engines"] if e["name"] == name)


def _strip(cell):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", cell)).split())


def _table(frag, cls):
    """The body rows of <table class=cls> as lists of cell text."""
    table = re.search(rf'<table class="{cls}">(.*?)</table>', frag, re.DOTALL).group(1)
    body = re.search(r"<tbody>(.*?)</tbody>", table, re.DOTALL).group(1)
    head = re.search(r"<thead>(.*?)</thead>", table, re.DOTALL).group(1)
    rows = [re.findall(r"<td[^>]*>(.*?)</td>", r, re.DOTALL) for r in re.findall(r"<tr>(.*?)</tr>", body, re.DOTALL)]
    return [_strip(c) for c in re.findall(r"<th[^>]*>(.*?)</th>", head, re.DOTALL)], \
           [[_strip(c) for c in r] for r in rows], rows


def _price_fragment(client, **override):
    return client.post("/ui/price", data={**FORM, "kind": "call", **override}).text


# ---------------------------------------------------------------------------
# The grid-range rule
# ---------------------------------------------------------------------------
class TestGridRule:
    def test_typical_contract_keeps_the_default_grid(self):
        g = engine.fd_grid(*ARGS)
        assert g == {"S_max_mult": 4.0, "N_S": 400, "N_t": 400, "widened": False, "capped": False}

    def test_far_strike_widens_to_reach_it_with_margin(self):
        S0, K, T, r, q, s = 100.0, 160.0, 1.0, 0.05, 0.0, 0.2
        g = engine.fd_grid(S0, K, T, r, q, s)
        total_vol = s * math.sqrt(T)
        assert g["widened"] and not g["capped"]
        # the range reaches the strike (and the forward) plus FD_MARGIN of total vol
        assert g["S_max_mult"] * total_vol >= math.log(K / S0) + engine.FD_MARGIN * total_vol - 1e-12
        assert g["N_S"] > 400 and g["N_S"] % 2 == 0     # scaled with the range, node on S0
        # ... at the spacing the default grid has
        assert 2 * g["S_max_mult"] * total_vol / g["N_S"] == pytest.approx(
            2 * 4.0 * total_vol / 400, rel=0.02)

    def test_drift_widens_it_too(self):
        g = engine.fd_grid(100, 100, 5.0, 0.30, 0.0, 0.2)        # (r - q) T = 1.5, 3.4 total vols
        assert g["widened"] and g["S_max_mult"] == pytest.approx(1.5 / (0.2 * math.sqrt(5)) + 3.0)

    def test_widening_is_only_when_needed(self):
        # strike less than one total vol from spot: reach + 3 stays under the base of 4
        assert not engine.fd_grid(100, 110, 1.0, 0.02, 0.0, 0.2)["widened"]
        assert engine.fd_grid(100, 130, 1.0, 0.02, 0.0, 0.2)["widened"]

    def test_scaling_is_capped(self):
        g = engine.fd_grid(100, 10_000, 0.1, 0.05, 0.0, 0.01)
        assert g["N_S"] == engine.FD_MAX_N_S and g["capped"]

    @pytest.mark.parametrize("args,kind", [
        ((100, 300, 1.0, 0.25, 0.0, 0.12), "put"),
        ((100, 200, 1.0, 0.25, 0.0, 0.12), "put"),
        ((100, 300, 3.0, 0.25, 0.0, 0.12), "put"),
        ((100, 160, 3.0, 0.25, 0.04, 0.12), "call"),
    ])
    def test_rule_fixes_contracts_the_fixed_grid_gets_wrong(self, args, kind):
        """Prices off by 1.5 to 17 percent with the library's fixed 4-sigma grid."""
        opt = op.OptionSpec(S0=args[0], K=args[1], T=args[2], r=args[3], q=args[4], sigma=args[5])
        bs = op.bs_price(opt, kind)
        g = engine.fd_grid(*args)
        kw = {"N_S": g["N_S"], "N_t": g["N_t"], "S_max_mult": g["S_max_mult"]}
        for price in (fd_price(opt, kind, **kw), fem_price(opt, kind, **kw)):
            assert abs(price - bs) / bs * 1e4 < 3.0                 # basis points
        fixed = fd_price(opt, kind, N_S=400, N_t=400)
        assert abs(fixed - bs) / bs * 1e4 > 20.0

    def test_every_finite_difference_and_element_call_uses_the_helper(self, monkeypatch):
        calls = []
        for name in ("fd_greeks", "fd_price", "fem_price"):
            real = getattr(engine, name)

            def spy(*a, _real=real, _name=name, **kw):
                calls.append((_name, a, kw))
                return _real(*a, **kw)

            monkeypatch.setattr(engine, name, spy)
        args = (100.0, 300.0, 1.0, 0.25, 0.0, 0.12)
        engine.price_all(*args, "put")
        g = engine.fd_grid(*args)
        assert sorted(n for n, _, _ in calls) == ["fd_greeks"] * 2 + ["fd_price"] * 4 + ["fem_price"]
        for name, a, kw in calls:
            assert kw["N_S"] == g["N_S"] and kw["N_t"] == g["N_t"]
            # the range in log spot is the helper's, also for the volatility-bumped solves
            assert a[0].sigma * kw["S_max_mult"] == pytest.approx(0.12 * g["S_max_mult"], rel=1e-12)

    def test_method_cells_say_when_the_range_was_widened(self, client):
        wide = _price_fragment(client, K="300", sigma="0.12", r="0.25", kind="put")
        _, rows, _ = _table(wide, "engines")
        assert "range widened" in rows[3][4] and "range widened" in rows[4][4]
        _, rows, _ = _table(_price_fragment(client), "engines")
        assert "widened" not in rows[3][4] and "widened" not in rows[4][4]

    def test_fd_and_fem_share_one_grid_and_say_so(self):
        res = engine.price_all(*ARGS, "call")
        assert "not independent evidence" in _entry(res, "Finite difference")["detail"]
        assert _entry(res, "Finite element")["detail"] == _entry(res, "Finite difference")["detail"]


# ---------------------------------------------------------------------------
# Engines table
# ---------------------------------------------------------------------------
class TestEnginesTable:
    def test_five_rows_with_american_as_a_column(self, client):
        head, rows, _ = _table(_price_fragment(client, kind="put"), "engines")
        assert head == ["Engine", "Price", "vs BS (bps)", "American", "Method"]
        assert [r[0] for r in rows] == ["Black-Scholes", "Binomial", "Monte Carlo",
                                        "Finite difference", "Finite element"]
        assert "Binomial (American)" not in _price_fragment(client, kind="put")

    def test_american_column_values(self, client):
        res = engine.price_all(*ARGS, "put")
        _, rows, _ = _table(_price_fragment(client, kind="put"), "engines")
        american = {r[0]: r[3] for r in rows}
        assert american["Binomial"] == f"{_entry(res, 'Binomial (American)')['price']:.6f}"
        assert american["Finite difference"] == f"{_entry(res, 'Finite difference (American)')['price']:.6f}"
        for name in ("Black-Scholes", "Monte Carlo", "Finite element"):
            assert american[name] == "–"
        # the European price sits in the Price column of the same row
        assert rows[1][1] == f"{_entry(res, 'Binomial (European)')['price']:.6f}"

    def test_caption_gives_both_premiums_on_one_line(self, client):
        frag = _price_fragment(client, kind="put")
        res = engine.price_all(*ARGS, "put")
        line = re.search(r'<p class="caption">(.*?)</p>', frag, re.DOTALL).group(1)
        assert "\n" not in line
        p = res["early_exercise_premium"]
        assert line == f"Early-exercise premium: tree {p['tree']:.4f}, grid {p['grid']:.4f}"
        assert p["tree"] == pytest.approx(0.5193, abs=1e-3) and p["grid"] == pytest.approx(0.5154, abs=1e-3)

    def test_rounding_level_premium_does_not_render_as_negative_zero(self, client, monkeypatch):
        real = engine.fd_greeks

        def nudged(opt, kind, **kw):
            g = real(opt, kind, **kw)
            if kw["american"]:
                g["price"] -= 1e-9
            return g

        monkeypatch.setattr(engine, "fd_greeks", nudged)
        frag = _price_fragment(client)
        assert "grid 0.0000" in frag and "-0.0000" not in frag

    def test_no_dividend_call_has_no_premium(self):
        res = engine.price_all(*ARGS, "call")
        assert abs(res["early_exercise_premium"]["tree"]) < 1e-9
        assert abs(res["early_exercise_premium"]["grid"]) < 1e-5

    def test_finite_element_row_is_the_library_at_the_fd_grid(self):
        res = engine.price_all(*ARGS, "call")
        assert _entry(res, "Finite element")["price"] == fem_price(
            op.OptionSpec(S0=100, K=100, T=1, r=0.05, sigma=0.2), "call", N_S=400, N_t=400)

    def test_fd_row_is_the_one_solve_that_also_gives_the_greeks(self):
        res = engine.price_all(*ARGS, "put")
        opt = op.OptionSpec(S0=100, K=100, T=1, r=0.05, sigma=0.2)
        assert _entry(res, "Finite difference")["price"] == fd_price(opt, "put", N_S=400, N_t=400)
        assert _entry(res, "Finite difference (American)")["price"] == fd_price(
            opt, "put", N_S=400, N_t=400, american=True)

    def test_each_solve_is_done_once(self, monkeypatch):
        """Price and Greeks come from the same two solves (European, American); the American
        vega and rho add four more, all American. Nothing is solved twice."""
        calls = []
        real_g, real_p = engine.fd_greeks, engine.fd_price
        monkeypatch.setattr(engine, "fd_greeks", lambda *a, **k: calls.append(("g", k["american"])) or real_g(*a, **k))
        monkeypatch.setattr(engine, "fd_price", lambda *a, **k: calls.append(("p", k["american"])) or real_p(*a, **k))
        engine.price_all(*ARGS, "put")
        assert calls.count(("g", False)) == 1 and calls.count(("g", True)) == 1
        assert calls.count(("p", True)) == 4 and ("p", False) not in calls

    def test_monte_carlo_is_labeled_as_pairs(self):
        """euro_price_mc counts antithetic pairs: 200,000 of them are 400,000 terminal prices."""
        mc = _entry(engine.price_all(*ARGS, "call"), "Monte Carlo")
        assert mc["note"] == "200,000 pairs, antithetic + control variate"
        assert mc["pairs"] == 200_000 and mc["terminal_prices"] == 400_000
        assert "400,000" in mc["detail"]                    # the tooltip on the Method cell
        assert "paths" not in mc["note"]

    def test_monte_carlo_z_is_shown_under_the_standard_error(self, client):
        res = engine.price_all(*ARGS, "call")
        mc = _entry(res, "Monte Carlo")
        assert mc["z"] == pytest.approx((mc["price"] - _entry(res, "Black-Scholes")["price"]) / mc["stderr"])
        frag = _price_fragment(client)
        cell = re.search(r'<td class="num">[0-9.]+<span class="stack">± [0-9.]+</span>'
                         r'<span class="stack" title="[^"]*">z = ([-+0-9.]+)</span></td>', frag)
        assert cell and float(cell.group(1)) == pytest.approx(mc["z"], abs=0.05)
        assert abs(mc["z"]) < 3          # the fixed seed gives a fair draw here

    def test_large_z_gets_the_warn_style(self, client, monkeypatch):
        monkeypatch.setattr(engine.op, "euro_price_mc", lambda *a, **k: (10.0, 0.01))
        frag = _price_fragment(client)
        assert re.search(r'<span class="stack warn" title="[^"]*">z = -45\.1</span>', frag)
        monkeypatch.setattr(engine.op, "euro_price_mc", lambda *a, **k: (10.4506, 0.01))
        assert 'class="stack warn"' not in _price_fragment(client)

    def test_zero_standard_error_has_no_z(self, monkeypatch):
        monkeypatch.setattr(engine.op, "euro_price_mc", lambda *a, **k: (0.0, 0.0))
        assert _entry(engine.price_all(*ARGS, "call"), "Monte Carlo")["z"] is None

    def test_a_failing_engine_does_not_hide_the_others(self, client, monkeypatch):
        def boom(*a, **k):
            raise ValueError("solver blew up")

        monkeypatch.setattr(engine, "fem_price", boom)
        _, rows, _ = _table(_price_fragment(client), "engines")
        assert rows[4][1] == "–" and "Unavailable: solver blew up" in rows[4][4]
        assert rows[3][1] != "–"


# ---------------------------------------------------------------------------
# JSON API compatibility
# ---------------------------------------------------------------------------
class TestApiCompatibility:
    OLD_ENGINES = ("Black-Scholes", "Binomial (European)", "Binomial (American)", "Monte Carlo",
                   "Finite difference")

    def test_every_old_entry_and_field_is_still_there(self, client):
        res = client.post("/api/price", json={**BASE, "kind": "call"}).json()
        assert {"inputs", "engines", "greeks"} <= set(res)
        names = [e["name"] for e in res["engines"]]
        assert tuple(names[:5]) == self.OLD_ENGINES
        for e in res["engines"][:5]:
            assert {"name", "price", "note", "diff_bps"} <= set(e)
        assert "stderr" in _entry(res, "Monte Carlo")
        assert set(res["greeks"]) == {"delta", "gamma", "vega", "theta", "rho"}
        assert set(res["inputs"]) == {"S0", "K", "T", "r", "q", "sigma", "kind"}

    def test_only_entries_and_fields_are_added(self, client):
        res = client.post("/api/price", json={**BASE, "kind": "put"}).json()
        names = [e["name"] for e in res["engines"]]
        assert names[5:] == ["Finite difference (American)", "Finite element"]
        assert _entry(res, "Monte Carlo")["z"] is not None
        assert {"greeks_fd", "early_exercise_premium", "grid"} <= set(res)
        assert "ladder" not in res and "rows" not in res

    def test_black_scholes_and_old_engines_are_unchanged_where_the_grid_is(self, client):
        res = client.post("/api/price", json={**BASE, "kind": "call"}).json()
        opt = op.OptionSpec(S0=100, K=100, T=1, r=0.05, sigma=0.2)
        assert _entry(res, "Black-Scholes")["price"] == op.bs_price(opt, "call")
        assert _entry(res, "Finite difference")["price"] == fd_price(opt, "call", N_S=400, N_t=400)
        assert _entry(res, "Binomial (American)")["price"] == op.crr(opt, "call", N=500, american=True)

    def test_response_is_strict_json(self, client):
        r = client.post("/api/price", json={**BASE, "kind": "put"})
        json.loads(r.text, parse_constant=lambda c: pytest.fail(f"non-finite {c} in the response"))

    @pytest.mark.parametrize("body", [
        {"S0": 100, "K": 100, "T": 1, "r": 0.05, "sigma": 0.2, "market_price": 10.0},        # both
        {"S0": 100, "K": 100, "T": 1, "r": 0.05},                                       # neither
    ])
    def test_sigma_and_market_price_are_exclusive(self, client, body):
        r = client.post("/api/price", json={**body, "kind": "call"})
        assert r.status_code == 400 and "error" in r.json()


# ---------------------------------------------------------------------------
# Greeks table
# ---------------------------------------------------------------------------
class TestGreeksTable:
    def test_columns_and_rows(self, client):
        head, rows, _ = _table(_price_fragment(client, kind="put"), "greeks")
        assert head == ["Greek", "Black-Scholes", "FD grid", "FD American"]
        assert [r[0] for r in rows] == ["Delta", "Gamma", "Vega", "Theta", "Rho"]

    def test_units_moved_into_tooltips(self, client):
        frag = _price_fragment(client)
        for greek, unit in [("Delta", "per unit of spot"), ("Gamma", "delta per unit of spot"),
                            ("Vega", "per unit of vol"), ("Theta", "per year"),
                            ("Rho", "per unit of rate")]:
            assert f'<td title="{unit}">{greek}</td>' in frag
        assert "<th>Units</th>" not in frag

    def test_grid_columns_come_from_fd_greeks(self):
        res = engine.price_all(*ARGS, "put")
        opt = op.OptionSpec(S0=100, K=100, T=1, r=0.05, sigma=0.2)
        for key, american in (("european", False), ("american", True)):
            ref = fd_greeks(opt, "put", N_S=400, N_t=400, american=american)
            for greek in ("delta", "gamma", "theta"):
                assert res["greeks_fd"][key][greek] == ref[greek]

    def test_european_grid_greeks_agree_with_black_scholes(self):
        res = engine.price_all(*ARGS, "call")
        bs, fd = res["greeks"], res["greeks_fd"]["european"]
        assert fd["delta"] == pytest.approx(bs["delta"], abs=2e-4)
        assert fd["gamma"] == pytest.approx(bs["gamma"], abs=2e-5)
        assert fd["theta"] == pytest.approx(bs["theta"], rel=1e-3)
        assert fd["vega"] is None and fd["rho"] is None

    @pytest.mark.parametrize("args,kind", [
        ((100.0, 100.0, 1.0, 0.05, 0.0, 0.2), "put"),
        ((100.0, 110.0, 1.0, 0.04, 0.0, 0.25), "put"),
        ((100.0, 95.0, 0.75, 0.02, 0.05, 0.3), "call"),
    ])
    def test_american_vega_and_rho_match_a_fine_tree(self, args, kind):
        """Central differences of the American grid price, grid held fixed, against central
        differences of a 4000-step tree: within 1 percent (measured within 0.2)."""
        S0, K, T, r, q, s = args
        res = engine.price_all(*args, kind)

        def tree(**bump):
            p = {"S0": S0, "K": K, "T": T, "r": r, "sigma": s, "q": q} | bump
            return op.crr(op.OptionSpec(**p), kind, N=4000, american=True)

        hs, hr = 0.01, 0.001
        vega = (tree(sigma=s + hs) - tree(sigma=s - hs)) / (2 * hs)
        rho = (tree(r=r + hr) - tree(r=r - hr)) / (2 * hr)
        got = res["greeks_fd"]["american"]
        assert got["vega"] == pytest.approx(vega, rel=0.01)
        assert got["rho"] == pytest.approx(rho, rel=0.01)

    def test_american_table_cells_show_numbers_not_dashes(self, client):
        _, rows, _ = _table(_price_fragment(client, kind="put"), "greeks")
        by = {r[0]: r for r in rows}
        for greek in ("Delta", "Gamma", "Theta", "Vega", "Rho"):
            assert re.fullmatch(r"-?\d+\.\d{5}", by[greek][3]), greek
        assert by["Vega"][2] == "–" and by["Rho"][2] == "–"     # one solve does not give these

    def test_unavailable_american_vega_shows_a_dash_with_the_reason(self, client, monkeypatch):
        def boom(*a, **k):
            raise ValueError("bump failed")

        monkeypatch.setattr(engine, "fd_price", boom)
        frag = _price_fragment(client, kind="put")
        assert 'title="Not computed: bump failed">–</span>' in frag
        _, rows, _ = _table(frag, "greeks")
        assert re.fullmatch(r"-?\d+\.\d{5}", rows[0][3])        # delta still comes from fd_greeks

    def test_no_greek_comes_from_a_tree_bump(self, monkeypatch):
        """A 1 percent bump of the tree measured +78.8 percent error in gamma; the tree
        is used for its two prices only."""
        calls = []
        real = op.crr
        monkeypatch.setattr(engine.op, "crr", lambda *a, **k: calls.append(k) or real(*a, **k))
        engine.price_all(*ARGS, "put")
        assert len(calls) == 2 and {c["N"] for c in calls} == {engine.BINOMIAL_STEPS}


# ---------------------------------------------------------------------------
# Implied-vol mode
# ---------------------------------------------------------------------------
class TestImpliedVol:
    @pytest.mark.parametrize("kind", ["call", "put"])
    @pytest.mark.parametrize("args,sigma", [
        ((100, 100, 1.0, 0.05, 0.0), 0.2703),
        ((100, 130, 0.5, 0.03, 0.02), 0.45),
        ((100, 70, 2.0, 0.05, 0.0), 0.12),
        ((100, 105, 0.02, 0.0, 0.0), 0.8),
        ((50, 55, 4.0, -0.01, 0.03), 1.5),
    ])
    def test_round_trip_recovers_sigma(self, kind, args, sigma):
        S0, K, T, r, q = args
        px = op.bs_price(op.OptionSpec(S0=S0, K=K, T=T, r=r, sigma=sigma, q=q), kind)
        res = engine.price_all(S0, K, T, r, q, None, kind, market_price=px)
        assert abs(res["implied_vol"]["sigma"] - sigma) < 1e-6
        assert res["inputs"]["sigma"] == res["implied_vol"]["sigma"]
        assert _entry(res, "Black-Scholes")["price"] == pytest.approx(px, abs=1e-8)
        assert res["implied_vol"]["vega"] == pytest.approx(res["greeks"]["vega"])

    def test_every_engine_and_the_greeks_use_the_solved_sigma(self):
        px = op.bs_price(op.OptionSpec(S0=100, K=100, T=1, r=0.05, sigma=0.31), "put")
        res = engine.price_all(*ARGS[:5], None, "put", market_price=px)
        ref = engine.price_all(*ARGS[:5], 0.31, "put")
        for name in ("Binomial (American)", "Finite difference", "Finite element"):
            assert _entry(res, name)["price"] == pytest.approx(_entry(ref, name)["price"], abs=1e-6)
        assert res["greeks_fd"]["american"]["vega"] == pytest.approx(
            ref["greeks_fd"]["american"]["vega"], rel=1e-6)

    def test_the_fragment_shows_one_result_line_above_the_table(self, client):
        frag = _price_fragment(client, sigma="10.4506", solve="iv")
        line = re.search(r'<p class="iv-line"[^>]*>(.*?)</p>', frag, re.DOTALL).group(1)
        assert re.fullmatch(r"Implied vol <strong>20\.00%</strong> \(vega 37\.5\)", line)
        assert frag.index("iv-line") < frag.index('<table class="engines">')
        assert 'class="iv-line"' not in _price_fragment(client)

    def test_ladder_and_scenarios_use_the_same_solved_sigma(self):
        px = op.bs_price(op.OptionSpec(S0=100, K=110, T=1, r=0.05, sigma=0.33), "call")
        lad = engine.spot_ladder(100, 110, 1, 0.05, 0, None, "call", market_price=px)
        assert lad["ladder"]["spot"][60] == 100.0
        assert lad["ladder"]["price"][60] == pytest.approx(px, abs=1e-6)
        sc = engine.scenario_grid(100, 110, 1, 0.05, 0, None, "call", market_price=px)
        assert sc["sigma"] == pytest.approx(0.33, abs=1e-6) and sc["base"] == pytest.approx(px, abs=1e-8)

    def test_one_helper_settles_the_volatility_for_every_view(self, monkeypatch):
        seen = []
        real = engine.resolve_option
        monkeypatch.setattr(engine, "resolve_option", lambda *a, **k: seen.append(k) or real(*a, **k))
        engine.price_all(*ARGS, "call")
        engine.spot_ladder(*ARGS, "call")
        engine.scenario_grid(*ARGS, "call")
        assert len(seen) == 3

    def test_brent_solver_is_the_libraries_scalar_one(self, monkeypatch):
        calls = []
        real = op.implied_vol
        monkeypatch.setattr(engine.op, "implied_vol", lambda *a, **k: calls.append(k) or real(*a, **k))
        engine.price_all(*ARGS[:5], None, "call", market_price=10.4506)
        assert len(calls) == 1 and calls[0]["bracket"] == (engine.MIN_VOL, 4.0)     # T = 1

    @pytest.mark.parametrize("price,match", [
        (-0.5, "below intrinsic"),
        (0.0, "no time value"),
    ])
    def test_out_of_bounds_prices_get_specific_messages(self, price, match):
        with pytest.raises(engine.InputError, match=match):
            engine.solve_implied_vol(100, 120, 1.0, 0.05, 0.0, price, "call")

    def test_below_intrinsic_puts_and_calls(self):
        # discounted intrinsic of the call: 100 - 80 e^{-0.05} = 23.90
        with pytest.raises(engine.InputError, match="below intrinsic.*23\\.9"):
            engine.solve_implied_vol(100, 80, 1.0, 0.05, 0.0, 20.0, "call")
        # put: 100 e^{-0.05} - 80 = 15.12
        with pytest.raises(engine.InputError, match="below intrinsic.*15\\.1"):
            engine.solve_implied_vol(80, 100, 1.0, 0.05, 0.0, 10.0, "put")

    def test_price_at_the_lower_bound_has_no_time_value(self):
        lower = 100 - 80 * math.exp(-0.05)
        with pytest.raises(engine.InputError, match="no time value"):
            engine.solve_implied_vol(100, 80, 1.0, 0.05, 0.0, lower, "call")

    def test_price_too_high_for_the_maximum_volatility(self):
        # T = 4: the cap on sigma*sqrt(T) makes sigma_max 2.0, not 5
        above = op.bs_price(op.OptionSpec(S0=100, K=100, T=4.0, r=0.05, sigma=2.2), "call")
        with pytest.raises(engine.InputError, match="above the Black-Scholes price"):
            engine.solve_implied_vol(100, 100, 4.0, 0.05, 0.0, above, "call")
        below = op.bs_price(op.OptionSpec(S0=100, K=100, T=4.0, r=0.05, sigma=1.9), "call")
        assert engine.solve_implied_vol(100, 100, 4.0, 0.05, 0.0, below, "call")["sigma"] == pytest.approx(1.9, abs=1e-6)

    def test_guard_evaluates_black_scholes_at_both_bracket_ends(self, monkeypatch):
        seen = []
        real = op.bs_price

        def spy(opt, kind="call"):
            seen.append(opt.sigma)
            return real(opt, kind)

        monkeypatch.setattr(engine.op, "bs_price", spy)
        engine.solve_implied_vol(100, 100, 1.0, 0.05, 0.0, 10.4506, "call")
        assert seen[:2] == [engine.MIN_VOL, 4.0]

    def test_sigma_max_never_breaks_the_total_vol_cap(self):
        for T in np.geomspace(engine.MIN_EXPIRY, 30.0, 400):
            lo, hi = engine._vol_bracket(float(T))
            assert lo < hi <= 5.0 and hi * math.sqrt(T) <= engine.MAX_TOTAL_VOL

    @pytest.mark.parametrize("bad,match", [
        ("nan", "finite"), ("inf", "finite"), ("-inf", "finite"),
        ("", "Market price is required"), ("abc", "Market price must be a number"),
    ])
    def test_non_finite_and_malformed_prices(self, bad, match):
        with pytest.raises(engine.InputError, match=match):
            engine.price_all(*ARGS[:5], None, "call", market_price=bad)

    def test_other_inputs_are_still_validated(self):
        with pytest.raises(engine.InputError, match="Expiry"):
            engine.price_all(100, 100, 0.0, 0.05, 0.0, None, "call", market_price=10.0)

    def test_both_or_neither_is_refused(self):
        with pytest.raises(engine.InputError, match="not both"):
            engine.resolve_option(*ARGS[:5], 0.2, "call", market_price=10.0)
        with pytest.raises(engine.InputError, match="Give a volatility"):
            engine.resolve_option(*ARGS[:5], None, "call")

    def test_unknown_solve_mode_is_refused(self, client):
        with pytest.raises(engine.InputError, match="Solve for"):
            engine.vol_source("delta", "0.2")
        frag = client.post("/ui/price", data={**FORM, "kind": "call", "solve": "delta"})
        assert frag.status_code == 200 and 'role="alert"' in frag.text and "Solve for" in frag.text

    def test_weak_vega_is_flagged(self):
        """Where vega is tiny the solved vol moves a lot for a tiny price change; the page says so."""
        far = op.bs_price(op.OptionSpec(S0=100, K=130, T=0.1, r=0.05, sigma=0.2), "call")   # 3.8e-5
        iv = engine.solve_implied_vol(100, 130, 0.1, 0.05, 0.0, far, "call")
        assert iv["weak"] is True and iv["vega"] < engine.LOW_VEGA
        assert engine.solve_implied_vol(100, 100, 1.0, 0.05, 0.0, 10.4506, "call")["weak"] is False

    def test_the_fragment_warns_about_weak_vega(self, client):
        far = op.bs_price(op.OptionSpec(S0=100, K=130, T=0.1, r=0.05, sigma=0.2), "call")
        frag = _price_fragment(client, K="130", T="0.1", sigma=repr(far), solve="iv")
        assert "Low vega" in frag
        assert "Low vega" not in _price_fragment(client, sigma="10.4506", solve="iv")

    # --- the API ------------------------------------------------------------
    def test_api_solves_from_a_market_price(self, client):
        px = op.bs_price(op.OptionSpec(S0=100, K=100, T=1, r=0.05, sigma=0.27), "call")
        r = client.post("/api/price", json={"S0": 100, "K": 100, "T": 1, "r": 0.05,
                                            "market_price": px, "kind": "call"})
        assert r.status_code == 200
        res = r.json()
        assert res["implied_vol"]["sigma"] == pytest.approx(0.27, abs=1e-6)
        assert res["inputs"]["sigma"] == pytest.approx(0.27, abs=1e-6)
        assert "implied_vol" not in client.post("/api/price", json={**BASE, "kind": "call"}).json()

    @pytest.mark.parametrize("price,match", [(-1.0, "below intrinsic"), (0.0, "no time value"),
                                             (1e6, "above the Black-Scholes price")])
    def test_api_errors_are_400_with_the_message(self, client, price, match):
        r = client.post("/api/price", json={"S0": 100, "K": 100, "T": 1, "market_price": price})
        assert r.status_code == 400 and match in r.json()["error"]

    def test_api_refuses_a_non_finite_market_price(self, client):
        r = client.post("/api/price", content='{"S0": 100, "K": 100, "T": 1, "market_price": NaN}',
                        headers={"content-type": "application/json"})
        assert r.status_code == 400

    def test_ui_errors_are_inline_with_status_200(self, client):
        for route in ("/ui/price", "/ui/ladder", "/ui/scenarios"):
            r = client.post(route, data={**FORM, "K": "120", "sigma": "0", "kind": "call", "solve": "iv"})
            assert r.status_code == 200 and 'role="alert"' in r.text and "no time value" in r.text, route

    def test_the_form_has_a_solve_for_select_and_a_relabelable_field(self, client):
        page = client.get("/").text
        assert re.search(r'<select name="solve" id="solve-for"[^>]*>\s*<option value="price" selected>Price</option>'
                         r'\s*<option value="iv">Implied vol</option>', page)
        assert '<span id="vol-label">Volatility</span>' in page and 'name="sigma" id="vol-input"' in page
        assert 'label.textContent = iv ? "Market price" : "Volatility"' in page


# ---------------------------------------------------------------------------
# Scenarios view
# ---------------------------------------------------------------------------
class TestScenarios:
    def test_grid_shape_and_base(self):
        res = engine.scenario_grid(*ARGS, "call")
        assert len(res["spot"]) == 13 and len(res["vol"]) == 9 and len(res["rate_shock_pts"]) == 5
        assert res["spot_shock_pct"][0] == -30.0 and res["spot_shock_pct"][-1] == 30.0
        assert res["vol_shock_pts"][0] == -10.0 and res["vol_shock_pts"][-1] == 10.0
        assert res["rate_shock_pts"] == [-2.0, -1.0, 0.0, 1.0, 2.0]
        pnl = np.array(res["pnl"], dtype=float)
        assert pnl.shape == (5, 9, 13)                       # [rate][vol][spot]
        assert pnl[2, 4, 6] == 0.0                           # rate 0, vol 0, spot 100%
        assert res["base"] == op.bs_price(op.OptionSpec(S0=100, K=100, T=1, r=0.05, sigma=0.2), "call")

    def test_pnl_is_stress_test_minus_the_base_price(self):
        opt = op.OptionSpec(S0=100, K=105, T=0.5, r=0.03, sigma=0.25, q=0.01)
        res = engine.scenario_grid(100, 105, 0.5, 0.03, 0.01, 0.25, "put")
        cube = op.stress_test(opt, "put", np.linspace(0.7, 1.3, 13), np.linspace(-0.1, 0.1, 9),
                              np.array([-0.02, -0.01, 0, 0.01, 0.02]), pricer="bs")
        base = op.bs_price(opt, "put")
        pnl = np.array(res["pnl"], dtype=float)
        assert np.allclose(np.transpose(pnl, (2, 1, 0)), cube - base, atol=1e-6)
        assert res["spot"] == pytest.approx(list(100 * np.linspace(0.7, 1.3, 13)))

    def test_cells_below_the_apps_minimum_vol_are_masked(self):
        # sigma 0.05: shocks -0.10, -0.075 and -0.05 give vols <= 0
        res = engine.scenario_grid(100, 100, 1, 0.05, 0.0, 0.05, "call")
        masked = [all(v is None for rate in res["pnl"] for v in rate[j]) for j in range(9)]
        assert masked == [True, True, True] + [False] * 6
        assert res["masked_vols"] == 3 and res["vol"][:3] == [None] * 3 and res["vol"][3] == 0.025

    def test_masking_uses_the_apps_minimum_not_the_libraries_floor(self):
        """stress_test floors a vol at 1e-6 and prices happily at 0.0005; the app's minimum
        is 0.001, so that row must be blank."""
        below = engine.scenario_grid(100, 100, 1, 0.05, 0.0, 0.1005, "call")
        assert below["masked_vols"] == 1 and below["vol"][0] is None
        at = engine.scenario_grid(100, 100, 1, 0.05, 0.0, 0.101, "call")     # exactly MIN_VOL
        assert at["masked_vols"] == 0 and at["vol"][0] == pytest.approx(engine.MIN_VOL)

    def test_masked_cells_are_null_never_nan(self, client):
        r = client.post("/ui/scenarios", data={**FORM, "sigma": "0.05", "kind": "call"})
        assert "NaN" not in r.text and "Infinity" not in r.text
        res = engine.scenario_grid(100, 100, 1, 0.05, 0.0, 0.05, "put")
        json.dumps(res, allow_nan=False)

    def test_all_zero_pnl_shows_a_note_instead_of_the_map(self, client):
        deep = {**FORM, "K": "100000", "T": "0.05", "sigma": "0.05", "kind": "call"}
        res = engine.scenario_grid(100, 100000, 0.05, 0.05, 0.0, 0.05, "call")
        assert res["all_zero"]
        frag = client.post("/ui/scenarios", data=deep).text
        assert "No P&amp;L to map" in frag and 'id="scen-chart"' not in frag
        assert "window.scenDraw = null" in frag                   # the selector goes quiet
        assert not engine.scenario_grid(*ARGS, "call")["all_zero"]

    def test_fragment_draws_a_diverging_map_centered_on_zero(self, client):
        frag = client.post("/ui/scenarios", data={**FORM, "kind": "call"}).text
        assert 'id="scen-chart"' in frag and "too far out of the money" not in frag
        assert 'type: "heatmap"' in frag and "zmid: 0" in frag
        assert '[0, color("--accent")]' in frag and '[1, color("--accent-2")]' in frag
        assert 'mix(color("--panel"), color("--line")' in frag       # neutral just off --panel
        assert "colorbar:" in frag and 'symbol: "circle-open"' in frag       # colorbar, base marker
        assert "hovertemplate" in frag and "plotLayout(" in frag
        assert "getComputedStyle" in frag                # theme colors, so dark mode works
        assert not re.search(r"#[0-9a-fA-F]{6}", frag)                       # no hard-coded colors

    def test_fragment_is_drawn_after_the_old_one_is_removed(self, client):
        frag = client.post("/ui/scenarios", data={**FORM, "kind": "call"}).text
        assert "setTimeout(function" in frag and "}, 0);" in frag

    def test_rate_slices_travel_in_the_fragment(self, client):
        frag = client.post("/ui/scenarios", data={**FORM, "kind": "call"}).text
        assert "const R = " in frag and '"rate_shock_pts": [-2.0, -1.0, 0.0, 1.0, 2.0]' in frag
        assert 'window.scenDraw = function' in frag

    def test_view_is_a_second_tab_in_the_price_group(self, client):
        page = client.get("/").text
        assert re.search(r'<div class="tabs" role="tablist" data-group="price">\s*'
                         r'<button type="button" role="tab" data-view="ladder" aria-selected="true">Price and delta across spot</button>\s*'
                         r'<button type="button" role="tab" data-view="scenarios" aria-selected="false">Scenarios</button>', page)
        view = re.search(r'<div class="view scen-view"([^>]*)>', page).group(1)
        assert 'data-group="price"' in view and 'data-view="scenarios"' in view
        assert 'data-follows="price-form"' in view and " hidden" in view
        assert 'hx-post="/ui/scenarios"' in view and 'hx-include="#price-form"' in view
        assert 'hx-trigger="shown, refresh"' in view and 'hx-target="find .view-body"' in view

    def test_rate_selector_is_outside_the_swapped_content(self, client):
        page = client.get("/").text
        view = re.search(r'<div class="view scen-view".*?</section>', page, re.DOTALL).group(0)
        assert view.index('id="scen-rate"') < view.index('class="view-body"')
        assert re.search(r'<div class="view-body"></div>', view)        # what htmx swaps starts empty

    def test_chart_and_selector_fit_the_ladder_slot(self):
        css = (WEBAPP / "templates" / "base.html").read_text()
        px = lambda pattern: int(re.search(pattern, css, re.DOTALL).group(1))
        slot = px(r"\.chart \{ width: 100%; height: (\d+)px; \}")
        bar = px(r"\.scen-bar \{ height: (\d+)px;")
        body = px(r"\.scen-view \.view-body, \.scen-chart, \.scen-note \{ height: (\d+)px;")
        assert bar + body == slot == 320

    def test_new_css_sits_right_after_the_tall_chart_rule(self):
        css = (WEBAPP / "templates" / "base.html").read_text()
        assert css.index(".chart.tall { height: 420px; }") + len(".chart.tall { height: 420px; }\n") \
               <= css.index("/* Pricing pane.") < css.index(".error {")

    def test_other_views_are_untouched(self, client):
        page = client.get("/").text
        assert 'hx-post="/ui/ladder"' in page and 'data-view="ladder"' in page
        assert '.chart { width: 100%; height: 320px; }' in (WEBAPP / "templates" / "base.html").read_text()


# ---------------------------------------------------------------------------
# Nothing user-controllable may 500
# ---------------------------------------------------------------------------
class TestExtremes:
    CASES = (
        {"S0": 100, "K": 1e6, "T": 0.01, "r": 0.5, "q": -0.5, "sigma": 0.001},
        {"S0": 1e6, "K": 1e-4, "T": 30, "r": -0.5, "q": 0.5, "sigma": 0.5},
        {"S0": 100, "K": 100, "T": 30, "r": 0.5, "q": -0.5, "sigma": 0.7},
        {"S0": 1e-4, "K": 1e6, "T": 1e-4, "r": 0.0, "q": 0.0, "sigma": 5.0},
        {"S0": 100, "K": 100, "T": 4.0, "r": 0.05, "q": 0.0, "sigma": 2.0},
        {"S0": 100, "K": 100, "T": 1e-4, "r": 0.5, "q": 0.5, "sigma": 0.001},
        {"S0": 100, "K": 0.01, "T": 5, "r": 0.1, "q": 0.0, "sigma": 0.05},
    )

    @pytest.mark.parametrize("case", CASES)
    @pytest.mark.parametrize("kind", ["call", "put"])
    def test_every_route_answers_and_json_stays_finite(self, client, case, kind):
        strict = lambda c: pytest.fail(f"{c} in a response")
        api = client.post("/api/price", json={**case, "kind": kind})
        assert api.status_code in (200, 400)
        if api.status_code == 200:
            json.loads(api.text, parse_constant=strict)
        form = {k: str(v) for k, v in case.items()} | {"kind": kind}
        for route in ("/ui/price", "/ui/ladder", "/ui/scenarios"):
            r = client.post(route, data=form)
            assert r.status_code == 200
            assert "NaN" not in r.text and "Infinity" not in r.text, route
