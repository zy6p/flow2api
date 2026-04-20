"""Standalone remote_browser control-plane service."""

from __future__ import annotations

import asyncio
import hmac
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .core.logger import debug_logger
from .services.browser_captcha import BrowserCaptchaService


def _get_api_key() -> str:
    api_key = (os.environ.get("REMOTE_BROWSER_API_KEY", "") or "").strip()
    if not api_key:
        raise RuntimeError("REMOTE_BROWSER_API_KEY is required")
    return api_key


def _verify_bearer_token(authorization: Optional[str] = Header(default=None)) -> str:
    expected = _get_api_key()
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization[7:].strip()
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    return token


async def _get_browser_service() -> BrowserCaptchaService:
    return await BrowserCaptchaService.get_instance(db=None)


async def _warmup_browser_pool(service: BrowserCaptchaService) -> None:
    try:
        await service.warmup_browser_slots()
    except Exception as exc:
        debug_logger.log_warning(f"[RemoteBrowser] warmup failed: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _get_api_key()
    service = await _get_browser_service()
    app.state.browser_service = service
    if os.environ.get("REMOTE_BROWSER_WARMUP", "true").strip().lower() not in {"0", "false", "no", "off"}:
        asyncio.create_task(_warmup_browser_pool(service))
    try:
        yield
    finally:
        try:
            await service.close()
        except Exception as exc:
            debug_logger.log_warning(f"[RemoteBrowser] shutdown close failed: {exc}")


app = FastAPI(title="Flow2API Remote Browser", lifespan=lifespan)


class SolveRequest(BaseModel):
    project_id: str = Field(..., min_length=1)
    action: str = Field(default="IMAGE_GENERATION")
    token_id: Optional[int] = None


class PrefillRequest(BaseModel):
    project_id: str = Field(..., min_length=1)
    action: str = Field(default="IMAGE_GENERATION")
    token_id: Optional[int] = None


class SessionErrorRequest(BaseModel):
    error_reason: Optional[str] = "upstream_error"


class SessionFinishRequest(BaseModel):
    status: Optional[str] = "success"


class CustomScoreRequest(BaseModel):
    website_url: str
    website_key: str
    verify_url: str
    action: str = "homepage"
    enterprise: bool = False


@app.get("/api/v1/health")
async def health() -> Dict[str, Any]:
    service = await _get_browser_service()
    return {
        "ok": True,
        "service": "remote_browser",
        "stats": service.get_stats(),
    }


@app.post("/api/v1/prefill")
async def prefill(
    request: PrefillRequest,
    _: str = Depends(_verify_bearer_token),
) -> Dict[str, Any]:
    service = await _get_browser_service()
    asyncio.create_task(_warmup_browser_pool(service))
    return {
        "success": True,
        "project_id": request.project_id,
        "action": request.action,
    }


@app.post("/api/v1/solve")
async def solve(
    request: SolveRequest,
    _: str = Depends(_verify_bearer_token),
) -> Dict[str, Any]:
    started_at = time.time()
    service = await _get_browser_service()
    token, session_id = await service.get_token(
        project_id=request.project_id,
        action=request.action,
        token_id=request.token_id,
    )
    if not token or session_id is None:
        raise HTTPException(status_code=503, detail="Failed to obtain reCAPTCHA token")

    fingerprint = await service.get_fingerprint(session_id)
    return {
        "success": True,
        "token": token,
        "session_id": str(session_id),
        "fingerprint": fingerprint or {},
        "token_elapsed_ms": int((time.time() - started_at) * 1000),
    }


@app.post("/api/v1/custom-score")
async def custom_score(
    request: CustomScoreRequest,
    _: str = Depends(_verify_bearer_token),
) -> Dict[str, Any]:
    service = await _get_browser_service()
    payload, browser_id = await service.get_custom_score(
        website_url=request.website_url,
        website_key=request.website_key,
        verify_url=request.verify_url,
        action=request.action,
        enterprise=request.enterprise,
    )
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Invalid custom score payload")

    fingerprint = await service.get_fingerprint(browser_id)
    result = dict(payload)
    verify_result = result.get("verify_result")
    success = bool(result.get("token")) or (
        isinstance(verify_result, dict) and verify_result.get("success") is True
    )
    result.setdefault("verify_mode", "remote_browser_page")
    result["success"] = success
    result["fingerprint"] = fingerprint or {}
    return result


@app.post("/api/v1/sessions/{session_id}/finish")
async def finish_session(
    session_id: str,
    request: SessionFinishRequest,
    _: str = Depends(_verify_bearer_token),
) -> Dict[str, Any]:
    service = await _get_browser_service()
    await service.report_request_finished(session_id)
    return {
        "success": True,
        "session_id": session_id,
        "status": request.status or "success",
    }


@app.post("/api/v1/sessions/{session_id}/error")
async def mark_session_error(
    session_id: str,
    request: SessionErrorRequest,
    _: str = Depends(_verify_bearer_token),
) -> Dict[str, Any]:
    service = await _get_browser_service()
    await service.report_error(
        browser_ref=session_id,
        error_reason=request.error_reason or "upstream_error",
    )
    return {
        "success": True,
        "session_id": session_id,
        "error_reason": request.error_reason or "upstream_error",
    }
