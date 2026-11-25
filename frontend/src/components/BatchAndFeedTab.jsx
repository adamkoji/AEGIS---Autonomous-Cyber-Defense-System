import React, { useState, useEffect } from "react";
import { Sparkles, CheckCircle2, ShieldAlert } from "lucide-react";
import { scanBatch } from "../api/aegisApi.js";

const SAMPLE_LINKS = [
  "https://www.google.com",
  "http://paypa1-secure-login-update.xyz/verify/account?user=admin@evil.com",
  "https://github.com",
  "http://192.168.1.55/apple-id-signin/index.html",
  "https://www.microsoft.com/en-us",
  "http://micros0ft-account-verify.top/login/oauth2?redirect=http://evil.ru",
];

export default function BatchAndFeedTab() {
  const [batchText, setBatchText] = useState(SAMPLE_LINKS.join("\n"));
  const [batchResult, setBatchResult] = useState(null);
  const [scanning, setScanning] = useState(false);

  const handleRunBatch = async (textOverride = null) => {
    const urls = (textOverride || batchText)
      .split(/\r?\n/)
      .map((u) => u.trim())
      .filter(Boolean);

    if (urls.length === 0) return;

    setScanning(true);
    try {
      const res = await scanBatch(urls, { deepScan: false });
      setBatchResult(res);
    } catch (err) {
      console.error("Bulk scan error:", err);
    } finally {
      setScanning(false);
    }
  };

  useEffect(() => {
    handleRunBatch(SAMPLE_LINKS.join("\n"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="w-full max-w-5xl mx-auto pt-2 pb-8 space-y-6">
      {/* Bulk Scan Input Card */}
      <div className="aegis-glow-box rounded-[30px] p-6 sm:p-8">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-5">
          <div>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-[#1E2238] dark:text-white tracking-tight">
              <span className="font-editorial italic font-semibold text-4xl sm:text-5xl pr-2.5 text-[#5B96E5] dark:text-[#8DB9F0]">
                Bulk Scan
              </span>
              Check Multiple Links at Once
            </h2>
            <p className="mt-1.5 text-sm text-[#5A6282] dark:text-[#E9E8F8]/80">
              Paste one website link per line to check them all together.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5 shrink-0">
            <button
              type="button"
              onClick={() => {
                const text = SAMPLE_LINKS.join("\n");
                setBatchText(text);
                handleRunBatch(text);
              }}
              className="px-4 py-2.5 rounded-full bg-white/85 dark:bg-white/10 text-xs font-bold text-[#1E2238] dark:text-white border border-white/80 dark:border-white/15 hover:border-[#8DB9F0] hover:-translate-y-0.5 hover:scale-[1.03] active:scale-95 transition-all duration-200"
            >
              Load Sample Links
            </button>

            <button
              type="button"
              onClick={() => handleRunBatch()}
              disabled={scanning}
              className="aegis-btn-shimmer bg-[#1E2238] hover:bg-[#2A304D] dark:bg-[#8DB9F0] dark:text-[#0D111E] text-white rounded-full px-6 py-2.5 text-xs sm:text-sm font-semibold flex items-center gap-2 hover:scale-105 hover:-translate-y-0.5 active:scale-95 transition-all duration-200"
            >
              <Sparkles className="w-4 h-4" />
              <span>{scanning ? "Scanning..." : "Scan All"}</span>
            </button>
          </div>
        </div>

        <textarea
          rows={4}
          value={batchText}
          onChange={(e) => setBatchText(e.target.value)}
          placeholder="Paste links here (one per line)..."
          className="w-full rounded-2xl bg-white/85 dark:bg-[#0D111E]/85 p-4 text-xs sm:text-sm font-mono text-[#1E2238] dark:text-white border border-white/80 dark:border-white/15 focus:outline-none focus:border-[#8DB9F0] transition-colors"
        />
      </div>

      {/* Clean Results List */}
      {batchResult && (
        <div className="aegis-card rounded-[28px] p-6">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
            <h3 className="text-lg font-bold text-[#1E2238] dark:text-white">
              Scan Results ({batchResult.total_scanned} Links)
            </h3>
            <div className="flex items-center gap-2 text-xs font-bold">
              <span className="px-3 py-1 rounded-full bg-[#8DB9F0]/40 text-[#1E2238] dark:text-[#8DB9F0]">
                {batchResult.legitimate_count} Safe
              </span>
              <span className="px-3 py-1 rounded-full bg-[#E88D6D]/30 text-[#1E2238] dark:text-[#F5B69E]">
                {batchResult.phishing_count} Phishing
              </span>
            </div>
          </div>

          <div className="space-y-2.5">
            {(batchResult.results || []).map((item, idx) => {
              const isPhish = item.prediction === "Phishing" || item.prediction === "Suspicious";
              const riskPct = Math.round((item.ensemble_score || 0) * 100);
              return (
                <div
                  key={`${item.url}-${idx}`}
                  className="p-3.5 rounded-2xl bg-white/75 dark:bg-[#0D111E]/65 border border-white/80 dark:border-white/10 hover:border-[#8DB9F0] hover:-translate-y-0.5 hover:shadow-md transition-all duration-200 flex flex-col sm:flex-row sm:items-center justify-between gap-3"
                >
                  <div className="text-xs sm:text-sm font-mono truncate text-[#1E2238] dark:text-white max-w-xl">
                    {item.url}
                  </div>

                  <div className="flex items-center gap-3 shrink-0">
                    <span
                      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold ${
                        isPhish
                          ? "bg-[#E88D6D] text-white"
                          : "bg-[#8DB9F0] text-[#1E2238]"
                      }`}
                    >
                      {isPhish ? (
                        <ShieldAlert className="w-3.5 h-3.5" />
                      ) : (
                        <CheckCircle2 className="w-3.5 h-3.5" />
                      )}
                      {isPhish ? "Phishing" : "Safe"}
                    </span>

                    <span className="text-xs sm:text-sm font-extrabold text-[#1E2238] dark:text-white w-20 text-right">
                      {riskPct}% Risk
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
