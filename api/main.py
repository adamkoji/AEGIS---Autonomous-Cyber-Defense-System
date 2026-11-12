# api/main.py
from typing import List, Optional
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.services.adversarial_guard import run_adversarial_evasion_simulation
from api.services.model_engine import engine

app = FastAPI(
    title="AEGIS Cyber-Defense & Adversarial Threat Intelligence API",
    description=(
        "Full-Stack AI Cyber-Defense Platform powered by a 111-Feature PyTorch Self-Attention Transformer, "
        "Calibrated LightGBM Ensemble, Live BeautifulSoup4 DOM Scanner, and OWASP/MITRE Adversarial Prompt-Injection Guardrails."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class URLScanRequest(BaseModel):
    url: str = Field(..., description="Target URL to analyze for phishing and adversarial threats")
    html_content: Optional[str] = Field(None, description="Optional raw HTML/DOM content to inspect")
    deep_scan: bool = Field(False, description="Enable live concurrent DNS, SSL, and HTTP probes")


class PayloadScanRequest(BaseModel):
    payload: Optional[str] = Field(None, description="Raw text, prompt, HTML DOM snippet, or API payload")
    text: Optional[str] = Field(None, description="Alias for payload text")
    content: Optional[str] = Field(None, description="Alias for payload content")
    html: Optional[str] = Field(None, description="Alias for HTML snippet")
    context: str = Field("raw_payload", description="Inspection context")


class BatchScanRequest(BaseModel):
    urls: Optional[List[str]] = Field(None, description="List of URLs to scan")
    items: Optional[List[str]] = Field(None, description="Alias for list of URLs/targets")
    deep_scan: bool = Field(False, description="Enable live DNS/SSL/HTTP probes")


class AdversarialSimulateRequest(BaseModel):
    target: Optional[str] = Field("https://paypal.com/signin", description="Base target URL or brand for mutation")
    url: Optional[str] = Field(None, description="Alias for target URL")
    base_url: Optional[str] = Field(None, description="Alias for base_url")


@app.get("/api/health")
def health_check():
    return engine.get_health()


@app.post("/api/scan/url")
@app.post("/analyze-url")
def scan_url_endpoint(req: URLScanRequest):
    if not req.url or not req.url.strip():
        raise HTTPException(status_code=400, detail="URL not provided")
    return engine.scan_url(
        url=req.url.strip(),
        html_content=req.html_content,
        deep_scan=req.deep_scan,
    )


@app.post("/api/scan/payload")
def scan_payload_endpoint(req: PayloadScanRequest):
    raw_input = req.payload or req.text or req.content or req.html or ""
    if not raw_input.strip():
        raise HTTPException(status_code=400, detail="Payload content not provided")
    return engine.scan_payload(content=raw_input, context=req.context)


@app.post("/api/scan/batch")
def scan_batch_endpoint(req: BatchScanRequest):
    targets = req.urls or req.items or []
    if not targets:
        raise HTTPException(status_code=400, detail="No URLs provided in batch request")
    return engine.scan_batch(items=targets[:50], deep_scan=req.deep_scan)


@app.post("/api/adversarial/simulate")
def simulate_adversarial_endpoint(
    req: Optional[AdversarialSimulateRequest] = Body(default=None),
):
    seed = "https://paypal.com/signin"
    if req is not None:
        seed = req.url or req.base_url or req.target or seed
    return run_adversarial_evasion_simulation(seed)


@app.get("/api/model/metrics")
def model_metrics_endpoint():
    return engine.get_metrics()


@app.get("/api/threats/recent")
def recent_threats_endpoint():
    return engine.get_recent_telemetry()
