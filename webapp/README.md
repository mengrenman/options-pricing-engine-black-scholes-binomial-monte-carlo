# optpricer web prototype

A FastAPI + HTMX front end for `optpricer`, with Plotly for charts. It prices one
contract with all five engines, calibrates an SVI surface, and draws Dupire local
volatility from it. Every input change recomputes on the server.

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
| `main.py` | Routes: `/` (page), `/ui/*` (HTMX fragments), `/api/*` (JSON). |
| `templates/` | Jinja templates: the page and one fragment per panel. |

The web app sits outside `src/optpricer` on purpose. The library stays at NumPy
and SciPy only, and none of this ships in its wheel.

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

## Measured latency

End-to-end HTTP round trips to a live server on localhost, 25 requests each, on
an otherwise idle machine. They scale with machine load: under a load average
near 250 the same requests took roughly twice as long.

| Request | Median | 90th percentile |
|---|---|---|
| Page load | 2 ms | 3 ms |
| Price fragment, all five engines | 33 ms | 74 ms |
| Local vol from a client-held surface | 4 ms | 8 ms |
| Surface calibration fragment | 166 ms | 204 ms |
| JSON price API | 36 ms | 49 ms |

Calibration is the only step slow enough to notice, so it shows a spinner and
debounces input by 400 ms instead of 250 ms.

## JSON API

| Method and path | Body | Returns |
|---|---|---|
| `GET /api/health` | none | `{"status": "ok"}` |
| `POST /api/price` | `S0, K, T, sigma`, optional `r, q, kind, mc_paths` | prices from all five engines and Black-Scholes Greeks |
| `POST /api/surface/fit` | smile parameters, all optional | a serialized `VolSurface` and per-slice fit error |
| `POST /api/surface/localvol` | `surface` (as returned above), `S0, t`, optional `r, q` | local and implied vol across spot |

## Known limitations

- Quotes are synthetic; nothing here reads market data yet.
- HTMX and Plotly.js come from CDNs, pinned to exact versions.
- No authentication, rate limiting or deployment configuration.
- American pricing appears only in the binomial row; the other engines are European.
