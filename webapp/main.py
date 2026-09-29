"""FastAPI + HTMX prototype front end for optpricer.

Two kinds of route share one engine module:

* ``/ui/*`` return HTML fragments for HTMX to swap into the page.  Bad input
  comes back as an inline error fragment with status 200, because HTMX does
  not swap 4xx responses and the user would otherwise see nothing happen.
* ``/api/*`` return JSON, with bad input as HTTP 400.

Every route is a plain ``def``, not ``async def``.  The work is CPU-bound
NumPy; inside an ``async def`` it would run on the event loop and block every
other request.  FastAPI runs ``def`` routes in a thread pool instead.

Run from the repository root:  uvicorn webapp.main:app --reload
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from . import engine

app = FastAPI(title="optpricer", version="0.1.0-prototype")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

DEFAULTS = {"S0": 100, "K": 100, "T": 1.0, "r": 0.05, "q": 0.0, "sigma": 0.20,
            "kind": "call", "atm_vol": 0.20, "skew": -0.30, "curvature": 0.60,
            "term_slope": 0.02, "t": 0.5}


@app.exception_handler(engine.InputError)
def _input_error(_request: Request, exc: engine.InputError):
    return JSONResponse(status_code=400, content={"error": str(exc)})


def _error(request: Request, message: str):
    return templates.TemplateResponse(request, "partials/error.html", {"message": message})


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"d": DEFAULTS})


# ---------------------------------------------------------------------------
# HTMX fragments
# ---------------------------------------------------------------------------
@app.post("/ui/price", response_class=HTMLResponse)
def ui_price(request: Request, S0: str = Form(""), K: str = Form(""), T: str = Form(""),
             r: str = Form(""), q: str = Form(""), sigma: str = Form(""),
             kind: str = Form("call")):
    try:
        result = engine.price_all(S0, K, T, r, q, sigma, kind)
    except engine.InputError as exc:
        return _error(request, str(exc))
    return templates.TemplateResponse(request, "partials/pricing.html", {"res": result})


@app.post("/ui/surface", response_class=HTMLResponse)
def ui_surface(request: Request, S0: str = Form(""), r: str = Form(""), q: str = Form(""),
               atm_vol: str = Form(""), skew: str = Form(""), curvature: str = Form(""),
               term_slope: str = Form("")):
    try:
        result = engine.fit_surface(S0, r, q, atm_vol, skew, curvature, term_slope)
    except engine.InputError as exc:
        return _error(request, str(exc))
    # The local-vol form below the chart posts these back with the surface.
    ctx = {"res": result, "S0": S0, "r": r, "q": q, "t": DEFAULTS["t"]}
    return templates.TemplateResponse(request, "partials/surface.html", ctx)


@app.post("/ui/localvol", response_class=HTMLResponse)
def ui_localvol(request: Request, surface: str = Form(""), S0: str = Form(""),
                t: str = Form(""), r: str = Form(""), q: str = Form("")):
    try:
        result = engine.local_vol_slice(surface, S0, t, r, q)
    except engine.InputError as exc:
        return _error(request, str(exc))
    return templates.TemplateResponse(request, "partials/localvol.html", {"res": result})


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------
class PriceRequest(BaseModel):
    S0: float
    K: float
    T: float
    r: float = 0.0
    q: float = 0.0
    sigma: float
    kind: str = "call"
    mc_paths: int = engine.DEFAULT_MC_PATHS


class SurfaceRequest(BaseModel):
    S0: float = 100.0
    r: float = 0.0
    q: float = 0.0
    atm_vol: float = 0.2
    skew: float = -0.3
    curvature: float = 0.6
    term_slope: float = 0.02


class LocalVolRequest(BaseModel):
    surface: dict
    S0: float
    t: float
    r: float = 0.0
    q: float = 0.0


@app.get("/api/health")
def api_health():
    return {"status": "ok"}


@app.post("/api/price")
def api_price(req: PriceRequest):
    res = engine.price_all(req.S0, req.K, req.T, req.r, req.q, req.sigma, req.kind,
                           mc_paths=req.mc_paths)
    res.pop("ladder")
    return res


@app.post("/api/surface/fit")
def api_surface_fit(req: SurfaceRequest):
    res = engine.fit_surface(req.S0, req.r, req.q, req.atm_vol, req.skew,
                             req.curvature, req.term_slope)
    return {"surface": res["surface"].to_dict(), "fit": res["fit"]}


@app.post("/api/surface/localvol")
def api_local_vol(req: LocalVolRequest):
    return engine.local_vol_slice(json.dumps(req.surface), req.S0, req.t, req.r, req.q)
