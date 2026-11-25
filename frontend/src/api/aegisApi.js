const API_BASE = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");

async function requestJson(endpoint, options = {}) {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    let errorDetail = `HTTP ${response.status}`;
    try {
      const errJson = await response.json();
      errorDetail = errJson.detail || errJson.error || errorDetail;
    } catch {
      // ignore json parse failure
    }
    throw new Error(errorDetail);
  }

  return response.json();
}

export async function scanUrl(url, htmlContent = null, deepScan = false, options = {}) {
  return requestJson("/api/scan/url", {
    method: "POST",
    body: JSON.stringify({
      url,
      html_content: htmlContent || null,
      deep_scan: Boolean(deepScan),
    }),
    signal: options.signal,
  });
}

export async function scanPayload(payload, context = "raw_payload", options = {}) {
  return requestJson("/api/scan/payload", {
    method: "POST",
    body: JSON.stringify({
      payload,
      context,
    }),
    signal: options.signal,
  });
}

export async function scanBatch(urls, options = {}) {
  return requestJson("/api/scan/batch", {
    method: "POST",
    body: JSON.stringify({
      urls,
      deep_scan: Boolean(options.deepScan),
    }),
    signal: options.signal,
  });
}

export async function runAdversarialSimulation(target = "https://paypal.com/signin", options = {}) {
  return requestJson("/api/adversarial/simulate", {
    method: "POST",
    body: JSON.stringify({
      target,
    }),
    signal: options.signal,
  });
}

export async function getModelMetrics(options = {}) {
  return requestJson("/api/model/metrics", {
    method: "GET",
    signal: options.signal,
  });
}

export async function getRecentThreats(options = {}) {
  return requestJson("/api/threats/recent", {
    method: "GET",
    signal: options.signal,
  });
}

export async function getHealth(options = {}) {
  return requestJson("/api/health", {
    method: "GET",
    signal: options.signal,
  });
}
