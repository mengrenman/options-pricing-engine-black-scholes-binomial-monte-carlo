# optpricer web prototype

A FastAPI + HTMX front end for `optpricer`, with Plotly for charts. It prices one
contract with all five of the library's engines, calibrates an SVI surface, and
checks what Dupire local volatility built from that surface does. Every input
change recomputes on the server.

This is a **prototype**. The volatility surface is calibrated to *synthetic*
quotes from a parametric smile, standing in for market data.

## Run it

From the repository root:

```bash
python -m pip install -e ".[web]"
uvicorn webapp.main:app --reload
```

Then open <http://localhost:8000>. The page loads HTMX and Plotly.js from public
CDNs, so the browser needs internet access; the server does not.

Interactive API docs are generated at <http://localhost:8000/docs>.

## Layout

| Path | Role |
|---|---|
| `engine.py` | Validated, bounded wrappers over `optpricer`. All limits live here. |
| `localvol.py` | The surface pane's repriced-smile and delta/dynamics views. |
| `main.py` | Routes: `/` (page), `/ui/*` (HTMX fragments), `/api/*` (JSON). |
| `templates/` | Jinja templates: the page and one fragment per panel. |

The web app sits outside `src/optpricer` on purpose. The library stays at NumPy
and SciPy only, and none of this ships in its wheel.

## Screen layout

At 1,600 px wide and above, the pricing and surface panes sit side by side. The
target is a 1920 × 1080 display, where the page ends 876 px down. That fits a
maximized browser window, which leaves the page about 940–975 px of height,
without scrolling. Narrower windows stack the panes, and below 860 px each pane
becomes a single column. The pricing pane's two tables need about 830 px to sit
side by side, so in a narrower pane (windows under about 1,780 px) they stack
rather than scroll. The synthetic-quotes disclosure is a badge in the surface
heading, so it stays in view in every layout.

Each pane holds tabs. A tab's view loads from its own route when it is shown and,
while it is visible, again after the inputs settle; hidden views do no work.

## What the panes show

**Pricing.**
- One contract priced by Black-Scholes, a 500-step binomial tree, Monte Carlo,
  finite differences and finite elements, each against Black-Scholes in basis
  points. The tree and the grid also price the American option, in their own
  column, and a caption gives the early-exercise premium from each.
- Greeks from Black-Scholes, from the finite-difference grid, and for the American
  option from the grid.
- "Solve for: Implied vol" reads the volatility field as a market price and
  prices everything at the implied volatility it solves for.
- Tabs: price and delta across spot, and a spot × volatility P&L heat map at five
  rate shocks.

**Volatility surface.**
- SVI fitted slice by slice to synthetic quotes, drawn in 3D as implied vol or as
  total variance.
- Local vol: Dupire local volatility across spot at a chosen time.
- Repriced smile: 11 strikes priced by the local-vol PDE and turned back into
  implied vols, against the surface they came from. This checks whether Dupire
  reproduces the surface.
- Dynamics: local-vol delta and gamma at a chosen strike, against Black-Scholes,
  and how the implied vol at that strike moves as spot moves.

## Design decisions

- **Every route is `def`, not `async def`.** Pricing is CPU-bound NumPy. Inside
  `async def` it would run on the event loop and stall every other request;
  FastAPI runs plain `def` routes in a thread pool.
- **No `warnings.catch_warnings`.** It swaps process-wide state, and because the
  routes run on a thread pool, concurrent requests could corrupt warning
  handling for the whole server. The page computes its notes from the data; the
  library's own warnings go to the server log. The app labels the surfaces it
  fits, so that log line appears once per process rather than once per request.
- **Monte Carlo runs serially.** `n_workers > 1` starts a process pool per
  request, which costs roughly 0.7–0.9 s on macOS and needs an
  `if __name__ == "__main__"` guard that web workers do not provide.
- **Request bodies are capped at 1 MB before FastAPI parses them.** Without this,
  a 50 MB JSON body cost about 580 MB of memory and several seconds before the
  surface-size check could refuse it. The limit checks `Content-Length` and, for
  chunked bodies that have none, reads up to the limit and replays the body.
- **Inputs have real lower bounds, not just "positive".** A vanishing expiry or
  volatility sent the binomial tree's `u - d` to zero and crashed the request.
- **Total volatility `sigma * sqrt(T)` is capped at 4.** Past that the
  finite-difference grid drifts by tens of basis points, and at 8 Monte Carlo is
  dozens of standard errors wrong while reporting a tiny standard error. At 10 it
  returns a price of exactly 0 with a standard error of exactly 0.
- **Engines fail independently.** If the binomial tree cannot price a contract,
  its rows say why (it needs `|r - q| * sqrt(T / N) < sigma`) and the other
  engines still show. Only a Black-Scholes failure fails the request.
- **Bad input never produces a 500.** In a fragment it returns status 200 with an
  inline error, because HTMX does not swap 4xx responses; the JSON API returns
  400 with `{"error": ...}`, including for schema errors FastAPI would otherwise
  report as 422. Every number returned is checked for finiteness first.
- **The server keeps no state.** A fitted surface is serialized with
  `VolSurface.to_json` (about 1 KB) into a hidden form field; the local-vol
  request posts it back and the server rehydrates it. Because the client holds
  it, the payload is treated as untrusted: its size is capped before parsing,
  and deeply nested JSON or oversized numbers become a clean error.
- **Charts are sent as data, not images.** The server returns JSON that Plotly
  draws in the browser, rounded to display precision. Rendering the same 3D
  surface with matplotlib cost about 100 ms and 208 KB and could not be rotated;
  as rounded JSON the grid is 11.2 KB (25.9 KB unrounded).
- **Clipped local vol is counted.** `dupire_local_vol` clamps to `[0.01, 5.0]`
  and raises `LocalVolClipWarning` when it does. That warning's text cannot carry
  a count, since it must stay constant to deduplicate, so the page states its own:
  on a steep smile extrapolated to `t = 3`, 4 of 81 points sit at the cap.
- **htmx attribute settling is off.** By default htmx starts a swapped-in element
  with the `class` of the element it replaces and restores its own about 20 ms
  later. That restore removed the `js-plotly-plot` class Plotly had just added,
  and Plotly's positioning CSS depends on it. As a result, every chart broke
  after the first input change: layers stacked down the page and the 3D
  colorbar drifted off the plot. The page sets `attributesToSettle` to `[]`.
- **Finite-difference and finite-element grids cover the strike and the forward.**
  The library centers its grid on spot, 4 σ√T wide, whatever the strike or drift,
  which left 37% of prices at the app's input caps more than 10 bp from
  Black-Scholes. The app widens the grid, with proportionally more nodes, when the
  strike or the forward sits more than one σ√T from spot; that leaves 8.7%, mostly
  tiny out-of-the-money prices. The Method cell says when the grid was widened.
  The two engines share the grid, so their agreement is not independent evidence.
- **American Greeks come from the grid, never from a bumped tree.** Delta, gamma
  and theta come from the same solve as the price. Vega and rho are central
  differences on a grid held fixed, so the nodes do not move with the bump; on 150
  contracts without early exercise they are within 0.38% and 0.25% of the exact
  values. Gamma from a 1% tree bump was 79% wrong at the default contract. Where
  the grid cannot reach 1% accuracy (total volatility above 2.5), the cells show a
  dash.
- **Implied volatility is solved with a bracketed method, after a guard.** The
  price must lie strictly between the no-arbitrage bound and the Black-Scholes price
  at the page's maximum volatility, and the message says which side it misses. The
  library's vectorized Newton solver is not used: from a fixed start it returns
  NaN on many out-of-the-money prices.
- **Monte Carlo counts antithetic pairs.** 200,000 pairs is 400,000 terminal
  prices. The row shows z, the gap to Black-Scholes in standard errors, and flags
  |z| > 3, except when the standard error is at floating-point noise (deep in the
  money, where the control variate absorbs the whole payoff).
- **The default smile is gentle on purpose.** The earlier default (skew −0.30,
  curvature 0.60) had butterfly arbitrage in its extrapolated wing, and the
  local-vol PDE repriced it up to 1,044 vol bps off at T = 1. The default is now
  skew −0.10, curvature 0.10, which reprices within about 4 vol bps at T = 1 and
  16 at T = 2. Steeper smiles can still be entered, and the repriced-smile view
  shows what happens.
- **The repriced smile is one grid, many strikes.** Every strike shares the grid,
  so the Dupire function is evaluated once per time step for all 11 solves instead
  of once per solve: about 35 ms instead of about 100. The memo lives inside the
  request. The library's local-vol solver reads the volatility at the start of each
  backward time step, a first-order error that showed as up to 32 vol bps on
  surfaces with a steep term structure; the app reads it half a step later, which
  brings that to about 3.
- **Local-vol delta is sticky local vol, and says so.** It holds the local-vol
  function fixed in absolute spot, so it differs from the Black-Scholes
  (sticky-strike) delta by design; a Milstein Monte Carlo check agrees with the
  grid to 0.0006.
- **A surface must match the spot and rates it is used with.** The surface carries
  the forwards it was fitted with. Every local-vol route refuses a surface whose
  forward at the chosen time differs from S0·e^((r−q)t) by more than 0.5%, plus the
  slack of the surface's own linear interpolation between quoted forwards. The two
  PDE views read the forward at every time step, so they check every quoted forward
  up to that time. A tampered forward had priced a call at 54.39 with spot at 100.
- **The local-vol chart takes the height its notes leave.** Beside the 3D chart,
  it shrinks when extrapolation or clipping notes appear, so both charts end on
  the same line and the page does not grow. It is drawn after a zero timeout,
  because htmx runs a fragment's script before removing the old fragment, which
  would otherwise still be taking up space when the chart is measured.

## Measured latency

Compute time per request, measured in-process as CPU time, 25 runs each. Another
job held the machine at a load average of 80–100 on 16 cores, which made
wall-clock times unrepeatable and inflates these too: surface calibration, which
this work did not change, measured 238 ms here against 166 ms end to end on an
idle machine. Treat them as upper bounds.

| Request | Median | 90th percentile |
|---|---|---|
| Price fragment: five engines, American prices and Greeks | 58 ms | 63 ms |
| Price fragment in implied-vol mode | 59 ms | 65 ms |
| Spot ladder view | 0.3 ms | 0.3 ms |
| Scenarios view | 1.6 ms | 1.6 ms |
| Surface calibration | 238 ms | 250 ms |
| Local vol at a time | 0.1 ms | 0.1 ms |
| Repriced smile | 33 ms | 37 ms |
| Delta and dynamics | 39 ms | 42 ms |

The price fragment now does about three times the work it did with four engines
and no American Greeks: 20 ms of CPU against 65 ms, timed alternately under the
same load. Calibration is the only step slow enough to notice, so it shows a
spinner and debounces input by 400 ms instead of 250 ms.

## JSON API

| Method and path | Body | Returns |
|---|---|---|
| `GET /api/health` | none | `{"status": "ok"}` |
| `POST /api/price` | `S0, K, T`, and `sigma` or `market_price`; optional `r, q, kind, mc_paths` | prices from every engine (the tree and the grid also American), Black-Scholes and grid Greeks, the early-exercise premium, and the grid used; with `market_price`, the implied vol |
| `POST /api/surface/fit` | smile parameters, all optional | a serialized `VolSurface` and per-slice fit error |
| `POST /api/surface/localvol` | `surface` (as returned above), `S0, t`, optional `r, q` | local and implied vol across spot |
| `POST /api/surface/lvsmile` | the same | the smile repriced by the local-vol PDE against the surface's own, with the gap in vol bps |
| `POST /api/surface/lvdynamics` | the same, plus optional `K` (default: the ATM forward) | local-vol and Black-Scholes delta and gamma, and implied vol at `K` as spot moves |

## Known limitations

- Quotes are synthetic; nothing here reads market data yet.
- HTMX and Plotly.js come from CDNs, pinned to exact versions.
- No authentication, rate limiting or deployment configuration.
- The implied vol solved from a market price is the European Black-Scholes vol,
  even when the price is meant as an American one.
- The library's finite-difference and finite-element solvers leave the dividend
  factor out of their far boundary values. At typical inputs this moves prices
  with a dividend by about 1e-5 to 1e-4; a fix is planned together with the
  notebook numbers it changes.
- Heston and SABR paths, exotics and the delta-hedge backtest stay in the
  notebooks until library bugs they rest on are fixed.
