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
- **Monte Carlo runs serially.** `n_workers > 1` starts a process pool per
  request, which costs roughly 0.7–0.9 s on macOS and needs an
  `if __name__ == "__main__"` guard that web workers do not provide.
- **Every size parameter is capped** (Monte Carlo paths, surface payload bytes,
  slice count), so one request cannot hang a worker.
- **Expiry must be strictly positive.** `bs_price_vec` returns NaN at `T = 0`
  instead of raising, so the check happens before the library is called.
- **Bad input in a fragment returns status 200 with an inline error.** HTMX does
  not swap 4xx responses, so a 400 would look like nothing happened. The JSON
  API returns a proper 400.
- **The server keeps no state.** A fitted surface is serialized with
  `VolSurface.to_json` (about 1 KB) into a hidden form field; the local-vol
  request posts it back and the server rehydrates it. Because the client holds
  it, the payload is treated as untrusted: its size is capped before parsing and
  `VolSurface.from_json` validates it strictly.
- **Charts are sent as data, not images.** The server returns JSON that Plotly
  draws in the browser. Rendering the same 3D surface with matplotlib cost about
  100 ms and 208 KB and could not be rotated; as JSON it is about 0.5 ms and 11 KB.
- **Clipped local vol is flagged.** `dupire_local_vol` clamps its output to
  `[0.01, 5.0]` without warning. The page counts points at either bound and says
  they are clamped, not computed. On a steep smile extrapolated to `t = 3`,
  4 of 81 points hit the cap.

## Measured latency

End-to-end on localhost, 25 requests each:

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
