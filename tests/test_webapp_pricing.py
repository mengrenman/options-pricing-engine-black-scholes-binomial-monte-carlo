"""Tests for the pricing pane of the web app: the engines and Greeks tables, the grid-range
rule, implied-vol mode and the scenarios view.

Skipped when the optional `web` extra is not installed, like tests/test_webapp.py.
"""
import html
import json
import math
import pathlib
import re
import shutil
import subprocess

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


def _tree_price(args, kind):
    """American CRR price, 4000 steps, averaged over N and N + 1: the lattice oscillates with the
    parity of the step count."""
    S0, K, T, r, q, s = args
    opt = op.OptionSpec(S0=S0, K=K, T=T, r=r, q=q, sigma=s)
    return 0.5 * (op.crr(opt, kind, N=4000, american=True) + op.crr(opt, kind, N=4001, american=True))


def _tree_vega_rho(args, kind):
    """Vega and rho of the fine tree by central differences, Richardson-extrapolated in the bump so
    the h^2 term cancels and the reference does not share the app's bump error."""
    def slope(index, h):
        up, dn = list(args), list(args)
        up[index] += h
        dn[index] -= h
        return (_tree_price(up, kind) - _tree_price(dn, kind)) / (2 * h)

    hs, hr = 0.02 * args[5], 0.001
    return (4 * slope(5, hs / 2) - slope(5, hs)) / 3, (4 * slope(3, hr / 2) - slope(3, hr)) / 3


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
            # the four bumped solves behind the American vega and rho take half the time steps
            assert kw["N_S"] == g["N_S"]
            assert kw["N_t"] == (engine.FD_BUMP_N_T if name == "fd_price" else g["N_t"])
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

    @pytest.mark.parametrize("kind,K", [("call", 73.0), ("put", 138.0)])
    def test_deep_in_the_money_monte_carlo_is_not_flagged_for_being_too_exact(self, client, kind, K):
        """Five total volatilities in the money, every path ends in the money and the control
        variate absorbs the payoff: the standard error is 3e-9 while the price agrees with
        Black-Scholes to 1e-8 relative. z came out at -63 and -89 there, and the warn style
        marked an accurate price as a failure."""
        res = engine.price_all(100.0, K, 0.1, 0.05, 0.0, 0.2, kind)
        mc, bs = _entry(res, "Monte Carlo"), _entry(res, "Black-Scholes")
        assert abs(mc["price"] / bs["price"] - 1) < 1e-7           # the price is accurate ...
        assert mc["stderr"] < engine.MC_SE_FLOOR * bs["price"]      # ... and the error bar is not
        assert mc["z"] is None and "control variate" in mc["z_note"]
        frag = _price_fragment(client, K=str(K), T="0.1", kind=kind)
        assert "z =" not in frag and 'class="stack warn"' not in frag
        assert re.search(r'<span class="stack" title="No z: the standard error is below 1e-07 of the '
                         r'price[^"]*">± 0\.000000</span>', frag)

    def test_z_is_kept_while_the_standard_error_is_resolved(self, monkeypatch):
        bs = float(op.bs_price(op.OptionSpec(S0=100, K=100, T=1, r=0.05, sigma=0.2), "call"))
        floor = engine.MC_SE_FLOOR
        monkeypatch.setattr(engine.op, "euro_price_mc", lambda *a, **k: (bs * (1 + 1e-3), bs * floor * 10))
        mc = _entry(engine.price_all(*ARGS, "call"), "Monte Carlo")
        assert mc["z"] == pytest.approx(1e-3 / (floor * 10), rel=1e-6) and mc["z_note"] is None
        monkeypatch.setattr(engine.op, "euro_price_mc", lambda *a, **k: (bs * (1 + 1e-3), bs * floor / 10))
        assert _entry(engine.price_all(*ARGS, "call"), "Monte Carlo")["z"] is None

    def test_a_typical_contract_still_shows_its_z_without_a_note(self):
        mc = _entry(engine.price_all(*ARGS, "call"), "Monte Carlo")
        assert mc["z"] is not None and mc["z_note"] is None and abs(mc["z"]) < 3

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

    def test_the_new_notes_are_fields_of_the_json(self, client):
        res = client.post("/api/price", json={**BASE, "kind": "call"}).json()
        assert _entry(res, "Monte Carlo")["z_note"] is None
        assert res["greeks_fd"]["american"]["rho_note"] is None
        deep = client.post("/api/price", json={"S0": 100, "K": 73, "T": 0.1, "r": 0.05, "sigma": 0.2,
                                               "kind": "call"}).json()
        mc = _entry(deep, "Monte Carlo")
        assert mc["z"] is None and "control variate" in mc["z_note"]
        zero = client.post("/api/price", json={**BASE, "r": 0.0, "kind": "put"}).json()
        assert zero["greeks_fd"]["american"]["rho_note"].startswith("One-sided")
        high = client.post("/api/price", json={**BASE, "T": 4.0, "sigma": 1.5, "kind": "call"}).json()
        assert high["greeks_fd"]["american"]["vega"] is None and "2.5" in high["american_vega_rho_unavailable"]

    def test_the_error_text_for_a_missing_volatility_changed_but_stays_a_400(self, client):
        """Clients that matched on the old text 'sigma: Field required' need the new one."""
        r = client.post("/api/price", json={"S0": 100, "K": 100, "T": 1, "kind": "call"})
        assert r.status_code == 400
        assert r.json()["error"] == "Give a volatility (sigma) or a market price (market_price)."

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
        ((100.0, 90.0, 1.0, 0.05, 0.0, 0.05), "put"),       # a fixed 0.01 bump was 38% off here
    ])
    def test_american_vega_and_rho_match_a_fine_tree(self, args, kind):
        """The American grid price, bumped and differenced, against a 4000-step tree whose bumps
        are Richardson-extrapolated, so the reference does not share the app's own bump error:
        within 1 percent (measured within 0.3)."""
        vega, rho = _tree_vega_rho(args, kind)
        got = engine.price_all(*args, kind)["greeks_fd"]["american"]
        assert got["vega"] == pytest.approx(vega, rel=0.01)
        assert got["rho"] == pytest.approx(rho, rel=0.01)

    @pytest.mark.parametrize("args", [
        (100, 100, 1.0, 0.03, 0.0, 0.02),       # the fixed +/-0.01 bump was 359 to 6 percent
        (100, 100, 5.0, 0.03, 0.0, 0.02),       # off on these
        (100, 100, 0.25, 0.03, 0.0, 0.02),
        (100, 120, 1.0, 0.03, 0.0, 0.05),
        (100, 80, 5.0, 0.03, 0.0, 0.05),
        (100, 150, 5.0, 0.03, 0.0, 0.05),
        (100, 100, 1.0, 0.03, 0.0, 0.10),
        (100, 103.045, 1.0, 0.03, 0.0, 0.005),  # at the forward, volatility half a percent
    ])
    def test_american_vega_is_exact_where_there_is_no_early_exercise(self, args):
        """A call with no dividend and r >= 0 is never exercised early, so the American price is
        the European one and Black-Scholes vega and rho are the exact derivatives to compare with."""
        res = engine.price_all(*args, "call")
        assert res["greeks_fd"]["american"]["vega"] == pytest.approx(res["greeks"]["vega"], rel=0.01)
        assert res["greeks_fd"]["american"]["rho"] == pytest.approx(res["greeks"]["rho"], rel=0.01)

    def test_american_put_without_early_exercise_matches_black_scholes(self):
        res = engine.price_all(100, 110, 1.0, -0.02, 0.0, 0.05, "put")          # r < 0, q = 0
        fd, bs = res["greeks_fd"]["american"], res["greeks"]
        assert fd["vega"] == pytest.approx(bs["vega"], rel=0.01)
        assert fd["rho"] == pytest.approx(bs["rho"], rel=0.01)

    def test_vega_bump_is_proportional_to_sigma_and_capped(self, monkeypatch):
        seen = []
        real = engine.fd_price
        monkeypatch.setattr(engine, "fd_price", lambda opt, *a, **k: seen.append(opt.sigma) or real(opt, *a, **k))
        for sigma, half in ((0.05, 0.001), (0.2, 0.004), (0.8, 0.01)):
            seen.clear()
            engine.price_all(100, 100, 1.0, 0.05, 0.0, sigma, "put")
            bumped = sorted(x for x in seen if x != sigma)
            assert bumped[0] == pytest.approx(sigma - half) and bumped[-1] == pytest.approx(sigma + half)

    def test_the_four_bump_solves_take_half_the_time_steps(self, monkeypatch):
        steps = []
        real = engine.fd_price
        monkeypatch.setattr(engine, "fd_price", lambda *a, **k: steps.append(k["N_t"]) or real(*a, **k))
        engine.price_all(*ARGS, "put")
        assert steps == [engine.FD_BUMP_N_T] * 4 and engine.FD_BUMP_N_T == 200

    # --- rho at a zero rate -------------------------------------------------
    def test_rho_of_a_no_dividend_call_at_zero_rate_is_the_black_scholes_slope(self):
        """At r = 0 a call gains early exercise only for r < 0, so the American price has a kink in
        r there. A central difference averaged the two slopes and was 3.7 percent off."""
        res = engine.price_all(100, 100, 1.0, 0.0, 0.0, 0.2, "call")
        fd = res["greeks_fd"]["american"]
        assert fd["rho"] == pytest.approx(res["greeks"]["rho"], rel=0.005)
        assert fd["rho_note"].startswith("One-sided slope, for r ≥ 0")

    def test_rho_of_a_put_at_zero_rate_is_the_slope_on_the_early_exercise_side(self):
        """A put gains early exercise for r > 0. Its up-slope is -49.8 by a 4000-step tree
        (second-order, same step), against -54.1 for the down-slope and Black-Scholes."""
        res = engine.price_all(100, 100, 1.0, 0.0, 0.0, 0.2, "put")
        hr = 0.001
        v = [_tree_price((100.0, 100.0, 1.0, r, 0.0, 0.2), "put") for r in (0.0, hr, 2 * hr)]
        up = (-3 * v[0] + 4 * v[1] - v[2]) / (2 * hr)
        got = res["greeks_fd"]["american"]["rho"]
        assert got == pytest.approx(up, rel=0.01)
        assert abs(got / res["greeks"]["rho"] - 1) > 0.05                       # not the BS slope

    def test_a_small_negative_rate_uses_the_down_slope(self):
        res = engine.price_all(100, 100, 1.0, -0.0004, 0.0, 0.2, "put")         # no early exercise below 0
        fd = res["greeks_fd"]["american"]
        assert fd["rho"] == pytest.approx(res["greeks"]["rho"], rel=0.005)
        assert fd["rho_note"].startswith("One-sided slope, for r ≤ 0")

    def test_away_from_zero_rho_is_central_with_no_note(self, monkeypatch):
        n = []
        real = engine.fd_price
        monkeypatch.setattr(engine, "fd_price", lambda *a, **k: n.append(1) or real(*a, **k))
        assert engine.price_all(*ARGS, "put")["greeks_fd"]["american"]["rho_note"] is None
        assert len(n) == 4                                  # two for vega, two for rho
        n.clear()
        engine.price_all(100, 100, 1.0, 0.0, 0.0, 0.2, "put")
        assert len(n) == 5                                  # the one-sided slope needs one more

    def test_the_rho_cell_says_when_it_is_one_sided(self, client):
        frag = _price_fragment(client, r="0", kind="put")
        assert re.search(r'<span title="One-sided slope, for r ≥ 0[^"]*">-?\d+\.\d{5}</span>', frag)
        assert "One-sided" not in _price_fragment(client, kind="put")

    # --- where the grid's own error is past 1 percent ------------------------
    def test_high_total_volatility_shows_dashes_with_the_reason(self, client):
        # volatility x sqrt(T) = 3: the grid takes a bumped price 1 to 3 percent off
        res = engine.price_all(100, 100, 4.0, 0.03, 0.0, 1.5, "call")
        am = res["greeks_fd"]["american"]
        assert am["vega"] is None and am["rho"] is None and am["delta"] is not None
        assert "volatility x sqrt(expiry) of 2.5" in res["american_vega_rho_unavailable"]
        frag = _price_fragment(client, T="4", sigma="1.5", r="0.03")
        assert 'title="Not computed: past a volatility x sqrt(expiry) of 2.5' in frag
        _, rows, _ = _table(frag, "greeks")
        assert rows[2][3] == "–" and rows[4][3] == "–" and rows[0][3] != "–"

    def test_just_inside_the_limit_they_are_shown_and_within_one_percent(self):
        res = engine.price_all(100, 100, 6.25, 0.03, 0.0, 1.0, "call")         # total volatility 2.5
        am, bs = res["greeks_fd"]["american"], res["greeks"]
        assert am["vega"] == pytest.approx(bs["vega"], rel=0.01)
        assert am["rho"] == pytest.approx(bs["rho"], rel=0.01)

    def test_rho_alone_is_refused_at_a_near_zero_rate_with_a_high_volatility(self, client):
        # volatility 1.5, r = 0: the grid's spurious early-exercise premium moves rho by 4 percent
        res = engine.price_all(100, 100, 0.25, 0.0, 0.0, 1.5, "call")
        am = res["greeks_fd"]["american"]
        assert am["vega"] == pytest.approx(res["greeks"]["vega"], rel=0.01) and am["rho"] is None
        assert am["rho_note"].startswith("Not computed: with the rate near zero")
        assert res["american_vega_rho_unavailable"] is None
        frag = _price_fragment(client, T="0.25", sigma="1.5", r="0")
        assert 'title="Not computed: with the rate near zero' in frag
        # at a rate of 20 percent (r T = 0.05) there is no such premium: rho is shown, within 1 percent
        ok = engine.price_all(100, 100, 0.25, 0.2, 0.0, 1.5, "call")
        assert ok["greeks_fd"]["american"]["rho"] == pytest.approx(ok["greeks"]["rho"], rel=0.01)

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
        (-0.5, "below the no-arbitrage lower bound"),
        (0.0, "no time value"),
    ])
    def test_out_of_bounds_prices_get_specific_messages(self, price, match):
        with pytest.raises(engine.InputError, match=match):
            engine.solve_implied_vol(100, 120, 1.0, 0.05, 0.0, price, "call")

    def test_below_the_lower_bound_puts_and_calls(self):
        # the bound is the discounted intrinsic value, not S - K: call 100 - 80 e^{-0.05} = 23.90
        with pytest.raises(engine.InputError, match="below the no-arbitrage lower bound.*= 23\\.9"):
            engine.solve_implied_vol(100, 80, 1.0, 0.05, 0.0, 20.0, "call")
        # put: 100 e^{-0.05} - 80 = 15.12
        with pytest.raises(engine.InputError, match="below the no-arbitrage lower bound.*= 15\\.1"):
            engine.solve_implied_vol(80, 100, 1.0, 0.05, 0.0, 10.0, "put")

    def test_the_bound_is_not_called_intrinsic_value_when_it_is_not(self):
        """S = 100, K = 50, T = 2, r = 0, q = 6%: the bound is 38.69 but intrinsic is 50, and a
        European call may trade below intrinsic. A price of 38 is refused for the bound it breaks."""
        with pytest.raises(engine.InputError) as info:
            engine.solve_implied_vol(100, 50, 2.0, 0.0, 0.06, 38.0, "call")
        assert "S e^(-qT) - K e^(-rT) = 38.69" in str(info.value) and "discounted" in str(info.value)
        below = op.bs_price(op.OptionSpec(S0=100, K=50, T=2.0, r=0.0, q=0.06, sigma=0.3), "call")
        assert 38.7 < below < 50            # under intrinsic value, yet inside the bounds: solvable
        assert engine.solve_implied_vol(100, 50, 2.0, 0.0, 0.06, below, "call")["sigma"] == pytest.approx(0.3, abs=1e-6)

    def test_a_price_a_hair_below_the_bound_prints_two_different_numbers(self):
        lower = 100 - 80 * math.exp(-0.05)
        with pytest.raises(engine.InputError) as info:
            engine.solve_implied_vol(100, 80, 1.0, 0.05, 0.0, lower - 1e-6, "call")
        price, bound = re.search(r"price (\S+) is below.*= (\S+)\)", str(info.value)).groups()
        assert price != bound and float(bound) - float(price) == pytest.approx(1e-6, rel=0.1)

    def test_a_price_with_time_value_below_the_minimum_volatility_is_not_called_worthless(self):
        """A call at the forward, 0.02 against the bound of 0: Black-Scholes at the page's minimum
        volatility of 0.1% already gives 0.0399, so the price implies about 0.05% volatility. It
        has time value; it is the volatility that is below what the page prices."""
        K = 100 * math.exp(0.05)
        with pytest.raises(engine.InputError) as info:
            engine.solve_implied_vol(100, K, 1.0, 0.05, 0.0, 0.02, "call")
        msg = str(info.value)
        assert "no time value" not in msg
        assert "implies a volatility below this page's minimum of 0.1%" in msg
        assert "Black-Scholes gives 0.0398" in msg

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

    @pytest.mark.parametrize("price,match", [(-1.0, "below the no-arbitrage lower bound"),
                                             (0.0, "no time value"),
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

    # --- the map is European Black-Scholes, and says so ----------------------
    def test_the_bar_says_the_map_is_european_black_scholes(self, client):
        page = client.get("/").text
        bar = re.search(r'<div class="scen-bar">.*?</div>', page, re.DOTALL).group(0)
        assert "Black-Scholes (European) P&amp;L against the base price" in bar
        assert "ignores early exercise" in bar                       # the tooltip

    # --- small underlyings keep their resolution -----------------------------
    def test_a_tiny_underlying_keeps_six_significant_digits(self):
        """S = K = 1.234e-4: the P&L cells are about 1e-5, and six decimals left 40 distinct values."""
        res = engine.scenario_grid(1.234e-4, 1.234e-4, 1.0, 0.05, 0.0, 0.2, "call")
        cells = [v for rate in res["pnl"] for row in rate for v in row]
        assert not res["all_zero"] and len(set(cells)) > 500          # 585 cells, distinct to 6 digits
        opt = op.OptionSpec(S0=1.234e-4, K=1.234e-4, T=1.0, r=0.05, sigma=0.2)
        cube = op.stress_test(opt, "call", np.linspace(0.7, 1.3, 13), np.linspace(-0.1, 0.1, 9),
                              np.array([-0.02, -0.01, 0, 0.01, 0.02]), pricer="bs")
        want = np.transpose(cube - op.bs_price(opt, "call"), (2, 1, 0))
        got = np.array(res["pnl"], dtype=float)
        assert np.allclose(got, want, rtol=0, atol=np.abs(want).max() * 2e-6)
        assert res["spot"][0] == pytest.approx(0.7 * 1.234e-4, rel=1e-9)  # six decimals gave 8.6e-05

    def test_a_large_underlying_is_not_over_precise(self):
        res = engine.scenario_grid(1e5, 1e5, 1.0, 0.05, 0.0, 0.2, "call")
        cells = [v for rate in res["pnl"] for row in rate for v in row]
        assert all(round(v, 6) == v for v in cells)

    def test_pnl_below_a_hundred_millionth_of_the_scale_is_zero_not_a_map(self):
        # K = 1e5, T = 0.05, sigma = 5%: every cell prices under 1e-3, the cutoff for a 1e5 scale
        assert engine.scenario_grid(100, 100000, 0.05, 0.05, 0.0, 0.05, "call")["all_zero"]
        # ... but the same relative depth with a small underlying still maps
        assert not engine.scenario_grid(1e-4, 1e-4, 1.0, 0.05, 0.0, 0.2, "call")["all_zero"]

    def test_chart_labels_are_built_in_significant_digits_not_fixed_decimals(self, client):
        frag = client.post("/ui/scenarios", data={**FORM, "kind": "call"}).text
        assert "const sig = " in frag and "tickvals: ticks" in frag and "ticktext: ticks.map" in frag
        assert 'tickformat: "+.2f"' not in frag and ":.4f}" not in frag and ":.2f}" not in frag
        assert 'hovertemplate: "%{text}<extra></extra>"' in frag

    JSC = "/System/Library/Frameworks/JavaScriptCore.framework/Versions/Current/Helpers/jsc"
    MOCK = """
var els = {};
function makeEl(id) { return { id: id, hidden: false, options: [], value: "", innerHTML: "",
                               add: function (o) { this.options.push(o); } }; }
var document = { getElementById: function (id) { return els[id] || (els[id] = makeEl(id)); }, documentElement: {} };
function getComputedStyle() { return { getPropertyValue: function (n) {
  return ({"--accent": " #2f6fdb", "--accent-2": " #d9822b", "--ink": " #1d1f23", "--panel": " #ffffff",
           "--line": " #e3e3de"})[n] || ""; } }; }
function Option(t, v) { this.text = t; this.value = v; }
function plotLayout(x) { return x; }
var PLOT_CONFIG = {}, __calls = [], window = {};
var Plotly = { react: function (el, data, layout) { __calls.push(data); }, purge: function () {} };
function setTimeout(f) { f(); }
"""

    def _draw(self, client, tmp_path, **form):
        """Run the scenarios fragment's script in a JavaScript engine with a mocked page and
        Plotly, and return the traces it hands to Plotly.react. Skips where none is installed."""
        exe = shutil.which("node") or (self.JSC if pathlib.Path(self.JSC).exists() else None)
        if exe is None:
            pytest.skip("no JavaScript engine (node or macOS jsc) to run the fragment's script")
        frag = client.post("/ui/scenarios", data={**FORM, "kind": "call", **form}).text
        script = re.search(r"<script>(.*?)</script>", frag, re.DOTALL).group(1)
        prelude = "var print = typeof print === 'undefined' ? console.log : print;\n"
        (tmp_path / "scen.js").write_text(
            prelude + self.MOCK + script + "\nprint(JSON.stringify(__calls[__calls.length - 1]));\n")
        out = subprocess.run([exe, str(tmp_path / "scen.js")], capture_output=True, text=True,
                             timeout=60, check=False)
        assert out.returncode == 0, out.stderr + out.stdout
        return json.loads(out.stdout.strip().splitlines()[-1])

    def test_script_labels_a_tiny_underlyings_pnl_in_significant_digits(self, client, tmp_path):
        heat, base = self._draw(client, tmp_path, S0="0.05", K="0.05")
        cb = heat["colorbar"]
        assert len(set(cb["ticktext"])) == 5 and cb["ticktext"][2] == "0"       # not +0.01 +0.00 -0.00
        assert cb["ticktext"][0] == "-" + cb["ticktext"][4].lstrip("+")
        res = engine.scenario_grid(0.05, 0.05, 1.0, 0.05, 0.0, 0.2, "call")
        top = max(abs(v) for row in res["pnl"][2] for v in row if v is not None)
        assert float(cb["ticktext"][4]) == pytest.approx(top, rel=2e-3)
        assert cb["tickvals"][4] == pytest.approx(top)
        cell = heat["text"][8][12]                                              # vol +10 pts, spot +30%
        assert cell.startswith("Spot 0.065 (+30%)<br>Vol 30.00% (+10 pts)<br>P&L +")
        assert float(cell.rsplit("P&L ", 1)[1]) == pytest.approx(res["pnl"][2][8][12], rel=1e-3)
        assert base["text"][0].startswith("Base case<br>Spot 0.05, vol 20.00%<br>Price 0.0")

    def test_script_hover_text_for_the_default_contract(self, client, tmp_path):
        heat, base = self._draw(client, tmp_path)
        assert base["text"][0] == "Base case<br>Spot 100, vol 20.00%<br>Price 10.4506"
        assert heat["text"][4][6] == "Spot 100 (0%)<br>Vol 20.00% (0 pts)<br>P&L 0"
        assert heat["text"][0][0].startswith("Spot 70 (-30%)<br>Vol 10.00% (-10 pts)<br>P&L -")
        assert heat["zmid"] == 0

    def test_script_blank_cells_have_no_hover_text(self, client, tmp_path):
        heat, _ = self._draw(client, tmp_path, sigma="0.05")        # vol shocks -10, -7.5, -5 are masked
        assert [heat["text"][j][0] for j in range(4)] == ["", "", "", heat["text"][3][0]]
        assert heat["text"][3][0].startswith("Spot 70 (-30%)<br>Vol 2.50%")


# ---------------------------------------------------------------------------
# Display details
# ---------------------------------------------------------------------------
class TestDisplay:
    def test_a_greek_that_rounds_to_zero_prints_without_a_sign(self, client):
        """theta of a deep in-the-money put is -0.0, and gamma and vega of a deep in-the-money
        call are -4e-9 and -2e-6: all three printed as -0.00000."""
        put = engine.price_all(100, 250, 1, 0.05, 0.0, 0.2, "put")["greeks_fd"]["american"]
        assert put["theta"] == 0.0 and math.copysign(1.0, put["theta"]) < 0
        call = engine.price_all(100, 53.8346, 0.25, 0.05, 0.0, 0.2, "call")["greeks_fd"]
        assert -5e-6 < call["european"]["gamma"] < 0 and -5e-6 < call["american"]["vega"] < 0
        for frag, cells in ((_price_fragment(client, K="250", kind="put"), {"Theta": 3}),
                            (_price_fragment(client, K="53.8346", T="0.25"), {"Gamma": 2, "Vega": 3})):
            assert "-0.00000" not in frag
            _, rows, _ = _table(frag, "greeks")
            by = {r[0]: r for r in rows}
            for greek, column in cells.items():
                assert by[greek][column] == "0.00000", greek

    def test_real_negative_values_keep_their_sign(self, client):
        _, rows, _ = _table(_price_fragment(client, kind="put"), "greeks")
        assert rows[0][1].startswith("-0.") and rows[3][1].startswith("-")       # put delta, theta

    def test_a_tiny_difference_from_black_scholes_prints_without_a_sign(self, client, monkeypatch):
        bs = float(op.bs_price(op.OptionSpec(S0=100, K=100, T=1, r=0.05, sigma=0.2), "call"))
        monkeypatch.setattr(engine, "fem_price", lambda *a, **k: bs * (1 - 4e-6))    # -0.04 bp
        _, rows, _ = _table(_price_fragment(client), "engines")
        assert rows[4][2] == "0.0" and rows[0][2] == "0.0"          # finite element, Black-Scholes
        _, other, _ = _table(_price_fragment(client, K="120"), "engines")
        assert all(row[2][0] in "+-" or row[2] == "0.0" for row in other)   # a real gap keeps its sign

    def test_a_rounding_level_premium_of_either_sign_prints_as_zero(self, client, monkeypatch):
        real = engine.fd_greeks

        def nudged(opt, kind, **kw):
            g = real(opt, kind, **kw)
            if kw["american"]:
                g["price"] -= 4e-5              # the premium is -0.00004, which %.4f prints as -0.0000
            return g

        monkeypatch.setattr(engine, "fd_greeks", nudged)
        assert "grid 0.0000" in _price_fragment(client) and "-0.0000" not in _price_fragment(client)

    def test_switching_to_implied_vol_prefills_significant_digits(self, client):
        """px.toFixed(4) turned a price under 5e-5 into 0 and the first solve failed."""
        page = client.get("/").text
        assert "px.toPrecision(6)" in page and "px.toFixed(4)" not in page


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
