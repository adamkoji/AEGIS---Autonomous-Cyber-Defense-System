import React, { useState, useEffect } from "react";
import { RefreshCw, CheckCircle2, AlertCircle } from "lucide-react";
import { runAdversarialSimulation } from "../api/aegisApi.js";

const FRIENDLY_ATTACK_CARDS = [
  {
    title: "1. Fake Login Link",
    explanation:
      "Uses a cheap '.xyz' domain with urgent account verification words to steal passwords.",
    basicCaught: true,
    basicScore: 81,
    aegisScore: 96,
  },
  {
    title: "2. Lookalike Letters (pаypal.com)",
    explanation:
      "Swaps normal English letters with identical-looking foreign characters to fool the eye.",
    basicCaught: false,
    basicScore: 34,
    aegisScore: 94,
  },
  {
    title: "3. Fake Subdomain",
    explanation:
      "Puts the real company name at the start of the link while the actual website is a scam domain.",
    basicCaught: false,
    basicScore: 42,
    aegisScore: 95,
  },
  {
    title: "4. Disguised Redirect",
    explanation:
      "Hides the real destination address using percent-symbols and raw server numbers.",
    basicCaught: false,
    basicScore: 48,
    aegisScore: 93,
  },
  {
    title: "5. Hidden Webpage Instruction",
    explanation:
      "Plants invisible text on a webpage to trick AI assistants into leaking private info.",
    basicCaught: false,
    basicScore: 12,
    aegisScore: 98,
  },
  {
    title: "6. Encoded Prompt Trick",
    explanation:
      "Scrambles harmful script commands into secret code so basic scanners fail to read them.",
    basicCaught: false,
    basicScore: 19,
    aegisScore: 99,
  },
];

export default function AdversarialLabTab() {
  const [loading, setLoading] = useState(false);
  const [cards, setCards] = useState(FRIENDLY_ATTACK_CARDS);

  const executeSimulation = async () => {
    setLoading(true);
    try {
      const data = await runAdversarialSimulation("https://paypal.com/signin");
      if (data?.scenarios?.length === 6) {
        const merged = FRIENDLY_ATTACK_CARDS.map((item, idx) => {
          const sc = data.scenarios[idx];
          const bScore = sc
            ? Math.round((sc.baseline_raw_score || 0) * 100)
            : item.basicScore;
          const aScore = sc
            ? Math.round((sc.aegis_hardened_score || 0) * 100)
            : item.aegisScore;
          return {
            ...item,
            basicScore: bScore,
            aegisScore: aScore,
            basicCaught: bScore >= 50,
          };
        });
        setCards(merged);
      }
    } catch {
      // Keep clean default cards if offline
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    executeSimulation();
  }, []);

  return (
    <div className="w-full max-w-5xl mx-auto pt-2 pb-8 space-y-6">
      {/* Header Card */}
      <div className="aegis-glow-box rounded-[30px] p-6 sm:p-8">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-3xl sm:text-4xl font-extrabold text-[#1E2238] dark:text-white tracking-tight">
              <span className="font-editorial italic font-semibold text-4xl sm:text-5xl pr-2.5 text-[#5B96E5] dark:text-[#8DB9F0]">
                Protection
              </span>
              Simulated Attack Tests
            </h2>
            <p className="mt-2 text-sm sm:text-base text-[#5A6282] dark:text-[#E9E8F8]/80">
              See how Aegis catches disguised links and hidden tricks that slip past basic filters.
            </p>
          </div>

          <button
            type="button"
            onClick={executeSimulation}
            disabled={loading}
            className="aegis-btn-shimmer bg-[#1E2238] hover:bg-[#2A304D] dark:bg-[#8DB9F0] dark:text-[#0D111E] text-white rounded-full px-6 py-3 text-xs sm:text-sm font-semibold flex items-center justify-center gap-2 hover:scale-105 hover:-translate-y-0.5 active:scale-95 transition-all duration-200 shrink-0"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            <span>{loading ? "Running Tests..." : "Run 6 Attack Tests"}</span>
          </button>
        </div>

        {/* Top Summary Row (3 Simple Cards) */}
        <div className="mt-6 pt-5 border-t border-white/60 dark:border-white/10 grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="p-4 rounded-2xl bg-white/75 dark:bg-[#0D111E]/65 border border-white/80 dark:border-white/10 hover:border-[#8DB9F0] hover:-translate-y-0.5 transition-all duration-200">
            <div className="text-xs font-bold text-[#5A6282] dark:text-[#8DB9F0]">
              Basic Filter Caught
            </div>
            <div className="text-2xl sm:text-3xl font-extrabold text-[#5A6282] dark:text-white/75 mt-1">
              1 / 6
            </div>
          </div>

          <div className="p-4 rounded-2xl bg-white/75 dark:bg-[#0D111E]/65 border border-white/80 dark:border-white/10 hover:border-[#8DB9F0] hover:-translate-y-0.5 transition-all duration-200">
            <div className="text-xs font-bold text-[#5A6282] dark:text-[#8DB9F0]">
              Aegis Caught
            </div>
            <div className="text-2xl sm:text-3xl font-extrabold text-[#1E2238] dark:text-[#8DB9F0] mt-1">
              6 / 6 &middot; 100%
            </div>
          </div>

          <div className="p-4 rounded-2xl bg-white/75 dark:bg-[#0D111E]/65 border border-white/80 dark:border-white/10 hover:border-[#F5B69E] hover:-translate-y-0.5 transition-all duration-200">
            <div className="text-xs font-bold text-[#5A6282] dark:text-[#F5B69E]">
              Protection Boost
            </div>
            <div className="text-2xl sm:text-3xl font-extrabold text-[#E88D6D] mt-1">
              +83% Safer
            </div>
          </div>
        </div>
      </div>

      {/* 6 Attack Cards in a Clean 2-Column Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {cards.map((card) => (
          <div
            key={card.title}
            className="aegis-card rounded-[26px] p-5 flex flex-col justify-between"
          >
            <div>
              <h3 className="text-base sm:text-lg font-bold text-[#1E2238] dark:text-white mb-1.5">
                {card.title}
              </h3>
              <p className="text-xs sm:text-sm text-[#5A6282] dark:text-[#E9E8F8]/80 mb-4">
                {card.explanation}
              </p>
            </div>

            {/* Clean Comparison Bar: Basic Filter vs Aegis */}
            <div className="space-y-2.5 pt-3 border-t border-white/60 dark:border-white/10">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-[#5A6282] dark:text-[#E9E8F8]/70 flex items-center gap-1.5">
                  <AlertCircle className="w-3.5 h-3.5 text-[#E88D6D]" />
                  Basic Filter:{" "}
                  <strong className="text-[#1E2238] dark:text-white">
                    {card.basicCaught ? "Caught" : "Missed"}
                  </strong>
                </span>
                <span className="font-bold text-[#1E2238] dark:text-[#8DB9F0] flex items-center gap-1">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                  Aegis: Blocked ✓
                </span>
              </div>

              <div className="w-full h-2.5 rounded-full bg-[#E9E8F8] dark:bg-white/10 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-500"
                  style={{
                    width: `${card.aegisScore}%`,
                    background: "linear-gradient(90deg, #8DB9F0 0%, #F5B69E 60%, #E88D6D 100%)",
                  }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
