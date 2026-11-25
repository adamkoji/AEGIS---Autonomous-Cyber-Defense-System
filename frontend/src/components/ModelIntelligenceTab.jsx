import React, { useState, useEffect } from "react";
import { CheckCircle2 } from "lucide-react";
import { getModelMetrics } from "../api/aegisApi.js";

const TOP_6_SIGNALS = [
  {
    name: "Brand Lookalikes",
    description: "Catches fake domain spellings like 'paypa1' or 'micros0ft' that imitate real companies.",
    weightPct: 96,
  },
  {
    name: "Domain Age & SSL",
    description: "Checks whether the website has a valid HTTPS security lock and established reputation.",
    weightPct: 91,
  },
  {
    name: "Suspicious Links",
    description: "Flags unusual domain endings (.xyz, .top) and raw numeric server addresses.",
    weightPct: 88,
  },
  {
    name: "Unusual URL Length",
    description: "Spots overly long web links stuffed with login and account verification words.",
    weightPct: 84,
  },
  {
    name: "Hidden Login Forms",
    description: "Detects password boxes and webpage forms that send your data to an outside server.",
    weightPct: 82,
  },
  {
    name: "Redirect Chains",
    description: "Identifies links that bounce through multiple hidden addresses before loading.",
    weightPct: 76,
  },
];

export default function ModelIntelligenceTab() {
  const [metrics, setMetrics] = useState(null);

  useEffect(() => {
    let mounted = true;
    getModelMetrics()
      .then((data) => {
        if (mounted) setMetrics(data);
      })
      .catch(() => {});
    return () => {
      mounted = false;
    };
  }, []);

  const buildRocPath = (rocPoints, curveOffset = 0) => {
    if (!rocPoints || rocPoints.length === 0) {
      return "M 35 190 L 35 22 L 295 20";
    }
    const pts = rocPoints.map((pt, idx) => {
      const x = 35 + pt.fpr * 260;
      const rawY = 190 - pt.tpr * 170;
      const y =
        idx > 0 && idx < rocPoints.length - 1
          ? Math.min(188, rawY + curveOffset)
          : rawY;
      return `${idx === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    });
    return pts.join(" ");
  };

  const modelsMap = metrics?.models || {};

  return (
    <div className="w-full max-w-5xl mx-auto pt-2 pb-8 space-y-6">
      {/* Header Card */}
      <div className="aegis-glow-box rounded-[30px] p-6 sm:p-8">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-[#1E2238] dark:text-white tracking-tight">
              <span className="font-editorial italic font-semibold text-4xl sm:text-5xl pr-2.5 text-[#5B96E5] dark:text-[#8DB9F0]">
                Accuracy
              </span>
              Detection Performance
            </h2>
            <p className="mt-2 text-sm sm:text-base text-[#5A6282] dark:text-[#E9E8F8]/80">
              Tested across 88,000+ real and phishing websites.
            </p>
          </div>
        </div>

        {/* 3 Clean Accuracy Cards */}
        <div className="mt-6 pt-5 border-t border-white/60 dark:border-white/10 grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="p-5 rounded-2xl bg-white/75 dark:bg-[#0D111E]/65 border border-white/80 dark:border-white/10 hover:border-[#8DB9F0] hover:-translate-y-0.5 transition-all duration-200">
            <div className="text-xs font-bold text-[#5A6282] dark:text-[#8DB9F0]">
              URL Model
            </div>
            <div className="text-3xl font-extrabold text-[#1E2238] dark:text-white mt-1">
              94.7%
            </div>
            <div className="text-xs text-[#5A6282] dark:text-[#E9E8F8]/70 mt-1">
              Checks link structure &amp; address patterns
            </div>
          </div>

          <div className="p-5 rounded-2xl bg-white/75 dark:bg-[#0D111E]/65 border border-white/80 dark:border-white/10 hover:border-[#F5B69E] hover:-translate-y-0.5 transition-all duration-200">
            <div className="text-xs font-bold text-[#5A6282] dark:text-[#F5B69E]">
              Page Model
            </div>
            <div className="text-3xl font-extrabold text-[#1E2238] dark:text-white mt-1">
              95.8%
            </div>
            <div className="text-xs text-[#5A6282] dark:text-[#E9E8F8]/70 mt-1">
              Checks login forms &amp; webpage code
            </div>
          </div>

          <div className="p-5 rounded-2xl bg-white/90 dark:bg-[#0D111E]/90 border-2 border-[#1E2238] dark:border-[#8DB9F0] hover:-translate-y-0.5 hover:shadow-[0_10px_25px_rgba(141,185,240,0.35)] transition-all duration-200">
            <div className="text-xs font-bold text-[#1E2238] dark:text-[#8DB9F0] flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
              <span>Combined Aegis</span>
            </div>
            <div className="text-3xl font-extrabold text-[#1E2238] dark:text-white mt-1">
              98.4%
            </div>
            <div className="text-xs text-[#5A6282] dark:text-[#E9E8F8]/75 mt-1">
              All protection layers working together
            </div>
          </div>
        </div>
      </div>

      {/* Visual Accuracy Curve (ROC) + Top 6 Signals Aegis Looks For */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Accuracy Curve (5 cols) */}
        <div className="lg:col-span-5 aegis-card rounded-[28px] p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-lg font-bold text-[#1E2238] dark:text-white mb-1">
              Accuracy Curve (ROC)
            </h3>
            <p className="text-xs text-[#5A6282] dark:text-[#E9E8F8]/75 mb-4">
              Higher curves show stronger protection with fewer false alarms.
            </p>

            <div className="bg-white/80 dark:bg-[#0D111E]/75 rounded-2xl p-4 border border-white/80 dark:border-white/10">
              <svg viewBox="0 0 320 220" className="w-full h-52 overflow-visible">
                {[0, 0.25, 0.5, 0.75, 1].map((g) => {
                  const y = 190 - g * 170;
                  const x = 35 + g * 260;
                  return (
                    <g key={g}>
                      <line
                        x1="35"
                        y1={y}
                        x2="295"
                        y2={y}
                        stroke="currentColor"
                        className="text-[#5A6282]/15 dark:text-white/10"
                        strokeDasharray="3 3"
                      />
                      <line
                        x1={x}
                        y1="20"
                        x2={x}
                        y2="190"
                        stroke="currentColor"
                        className="text-[#5A6282]/15 dark:text-white/10"
                        strokeDasharray="3 3"
                      />
                    </g>
                  );
                })}

                <line
                  x1="35"
                  y1="190"
                  x2="295"
                  y2="20"
                  stroke="#5A6282"
                  strokeWidth="1.2"
                  strokeDasharray="5 5"
                  opacity="0.45"
                />

                <path
                  d={buildRocPath(modelsMap.pytorch_transformer?.roc_curve, 9)}
                  fill="none"
                  stroke="#8DB9F0"
                  strokeWidth="3"
                  strokeLinecap="round"
                />
                <path
                  d={buildRocPath(modelsMap.lightgbm?.roc_curve, 4)}
                  fill="none"
                  stroke="#F5B69E"
                  strokeWidth="3"
                  strokeLinecap="round"
                />
                <path
                  d={buildRocPath(modelsMap.aegis_ensemble?.roc_curve, 0)}
                  fill="none"
                  stroke="#5B96E5"
                  strokeWidth="3.2"
                  strokeLinecap="round"
                />

                <text
                  x="165"
                  y="212"
                  textAnchor="middle"
                  className="fill-[#5A6282] text-[10px] font-bold"
                >
                  False Alarms
                </text>
                <text
                  x="12"
                  y="105"
                  textAnchor="middle"
                  transform="rotate(-90 12 105)"
                  className="fill-[#5A6282] text-[10px] font-bold"
                >
                  Threats Caught
                </text>
              </svg>

              <div className="mt-3 flex flex-wrap items-center justify-center gap-3 text-xs font-semibold">
                <div className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-[#8DB9F0]" />
                  <span className="text-[#1E2238] dark:text-white">URL Model</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-[#F5B69E]" />
                  <span className="text-[#1E2238] dark:text-white">Page Model</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-full bg-[#5B96E5]" />
                  <span className="text-[#1E2238] dark:text-white">Combined Aegis</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Top 6 Signals Aegis Looks For (7 cols) */}
        <div className="lg:col-span-7 aegis-card rounded-[28px] p-6">
          <h3 className="text-lg font-bold text-[#1E2238] dark:text-white mb-1">
            Top 6 Signals Aegis Looks For
          </h3>
          <p className="text-xs text-[#5A6282] dark:text-[#E9E8F8]/75 mb-4">
            Key plain-English indicators used to separate safe websites from scams.
          </p>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
            {TOP_6_SIGNALS.map((sig) => (
              <div
                key={sig.name}
                className="p-4 rounded-2xl bg-white/75 dark:bg-[#0D111E]/65 border border-white/80 dark:border-white/10 hover:border-[#8DB9F0] hover:-translate-y-0.5 hover:shadow-md transition-all duration-200"
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-sm font-bold text-[#1E2238] dark:text-white">
                    {sig.name}
                  </span>
                </div>
                <div className="w-full h-2 rounded-full bg-[#E9E8F8] dark:bg-white/10 overflow-hidden mb-2">
                  <div
                    className="h-full rounded-full"
                    style={{
                      width: `${sig.weightPct}%`,
                      background: "linear-gradient(90deg, #8DB9F0 0%, #F5B69E 100%)",
                    }}
                  />
                </div>
                <p className="text-xs text-[#5A6282] dark:text-[#E9E8F8]/75">
                  {sig.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
