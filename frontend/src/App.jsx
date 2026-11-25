import React, { useState, useEffect } from "react";
import {
  Shield,
  Flame,
  BarChart3,
  Layers,
  Sun,
  Moon,
  Settings,
  Menu,
  X,
  PanelLeftOpen,
  PanelLeftClose,
  CheckCircle2,
  Trash2,
} from "lucide-react";
import ThreatScannerTab from "./components/ThreatScannerTab.jsx";
import AdversarialLabTab from "./components/AdversarialLabTab.jsx";
import ModelIntelligenceTab from "./components/ModelIntelligenceTab.jsx";
import BatchAndFeedTab from "./components/BatchAndFeedTab.jsx";

const NAV_ITEMS = [
  {
    id: "scanner",
    label: "Scanner",
    subtitle: "Check links & messages",
    icon: Shield,
  },
  {
    id: "adversarial",
    label: "Protection Lab",
    subtitle: "Simulated attack tests",
    icon: Flame,
  },
  {
    id: "intelligence",
    label: "Accuracy",
    subtitle: "Detection performance",
    icon: BarChart3,
  },
  {
    id: "batch",
    label: "Bulk Scan",
    subtitle: "Check multiple links",
    icon: Layers,
  },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("scanner");
  const [darkMode, setDarkMode] = useState(false);
  const [dockExpanded, setDockExpanded] = useState(false);
  const [mobileDrawerOpen, setMobileDrawerOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [quickDemoTrigger, setQuickDemoTrigger] = useState(0);
  const [historyClearedNotice, setHistoryClearedNotice] = useState(false);

  useEffect(() => {
    const root = document.documentElement;
    if (darkMode) {
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
    }
  }, [darkMode]);

  const handleTrySample = () => {
    setActiveTab("scanner");
    setQuickDemoTrigger((prev) => prev + 1);
  };

  const handleClearHistory = () => {
    setHistoryClearedNotice(true);
    setTimeout(() => {
      setHistoryClearedNotice(false);
      setSettingsOpen(false);
    }, 900);
  };

  return (
    <div className="lunar-oasis-canvas relative min-h-screen w-full flex flex-col">
      {/* Desktop Left Floating Vertical Capsule Dock */}
      <aside
        className={`hidden md:flex fixed left-5 top-1/2 -translate-y-1/2 z-40 flex-col justify-between bg-[#E9E8F8]/85 dark:bg-[#161B2E]/85 backdrop-blur-2xl border border-white/60 dark:border-white/10 shadow-[0_12px_36px_rgba(30,34,56,0.10)] py-4 px-2.5 rounded-[32px] transition-[width,box-shadow,background-color] duration-300 ease-[cubic-bezier(0.22,1,0.36,1)] ${
          dockExpanded ? "w-60" : "w-16"
        }`}
      >
        <div className="flex flex-col items-stretch gap-2">
          {/* Top Expand / Collapse Button pinned to constant left icon rail */}
          <button
            type="button"
            onClick={() => setDockExpanded(!dockExpanded)}
            title={dockExpanded ? "Collapse menu" : "Expand menu"}
            className="group flex items-center h-11 w-full rounded-full text-[#1E2238] dark:text-[#E9E8F8] hover:bg-white/75 dark:hover:bg-white/10 transition-all duration-200 overflow-hidden"
          >
            <div className="w-11 h-11 flex items-center justify-center shrink-0 transition-transform duration-200 group-hover:scale-110">
              {dockExpanded ? (
                <PanelLeftClose className="w-5 h-5" />
              ) : (
                <PanelLeftOpen className="w-5 h-5" />
              )}
            </div>
            <span
              className={`text-xs font-bold whitespace-nowrap transition-all duration-300 ease-[cubic-bezier(0.22,1,0.36,1)] ${
                dockExpanded
                  ? "opacity-100 translate-x-0 ml-1"
                  : "opacity-0 -translate-x-2 pointer-events-none w-0"
              }`}
            >
              Collapse Menu
            </span>
          </button>

          <div className="w-full h-px bg-black/10 dark:bg-white/10 my-1" />

          {/* 4 Navigation Views with Magnetic Hover & Aura */}
          <nav className="w-full flex flex-col gap-2">
            {NAV_ITEMS.map((item) => {
              const IconComponent = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setActiveTab(item.id)}
                  title={item.label}
                  className={`group relative flex items-center h-12 w-full rounded-full transition-all duration-200 ${
                    isActive
                      ? "bg-[#1E2238] text-white dark:bg-gradient-to-r dark:from-[#8DB9F0] dark:to-[#F5B69E] dark:text-[#0D111E] shadow-[0_8px_22px_rgba(141,185,240,0.48)]"
                      : "text-[#5A6282] dark:text-[#E9E8F8]/75 hover:bg-white/80 dark:hover:bg-white/10 hover:text-[#1E2238] dark:hover:text-white hover:-translate-y-0.5 hover:shadow-[0_6px_16px_rgba(141,185,240,0.28)]"
                  }`}
                >
                  <div className="w-11 h-12 flex items-center justify-center shrink-0 transition-transform duration-200 group-hover:scale-110">
                    <IconComponent className="w-5 h-5" />
                  </div>

                  <div
                    className={`text-left overflow-hidden whitespace-nowrap transition-all duration-300 ease-[cubic-bezier(0.22,1,0.36,1)] ${
                      dockExpanded
                        ? "opacity-100 translate-x-0 max-w-[160px] ml-1 pr-3"
                        : "opacity-0 -translate-x-2 max-w-0 pointer-events-none"
                    }`}
                  >
                    <div className="text-xs font-bold leading-tight truncate">
                      {item.label}
                    </div>
                    <div
                      className={`text-[10px] leading-tight truncate mt-0.5 ${
                        isActive
                          ? "text-white/75 dark:text-[#0D111E]/75"
                          : "text-[#5A6282] dark:text-[#E9E8F8]/55"
                      }`}
                    >
                      {item.subtitle}
                    </div>
                  </div>

                  {!dockExpanded && (
                    <span className="pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity absolute left-16 px-3 py-1.5 rounded-full bg-[#1E2238] text-white text-xs font-semibold whitespace-nowrap shadow-lg z-50">
                      {item.label}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Bottom Dock Controls: Theme Switcher & Settings Popover */}
        <div className="flex flex-col items-stretch gap-2 pt-3 mt-2 border-t border-black/10 dark:border-white/10 relative">
          <button
            type="button"
            onClick={() => setDarkMode(!darkMode)}
            title={darkMode ? "Light Mode" : "Dark Mode"}
            className={`group flex items-center h-11 w-full rounded-full overflow-hidden transition-all duration-200 hover:-translate-y-0.5 ${
              darkMode
                ? "bg-[#F5B69E] text-[#1E2238] shadow-[0_6px_18px_rgba(245,182,158,0.45)]"
                : "bg-white/80 text-[#1E2238] hover:bg-[#F5B69E]/45 hover:shadow-sm"
            }`}
          >
            <div className="w-11 h-11 flex items-center justify-center shrink-0 transition-transform duration-200 group-hover:scale-110">
              {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </div>
            <span
              className={`text-xs font-bold whitespace-nowrap transition-all duration-300 ease-[cubic-bezier(0.22,1,0.36,1)] ${
                dockExpanded
                  ? "opacity-100 translate-x-0 ml-1 pr-3"
                  : "opacity-0 -translate-x-2 max-w-0 pointer-events-none"
              }`}
            >
              {darkMode ? "Light Mode" : "Dark Mode"}
            </span>
          </button>

          <button
            type="button"
            onClick={() => setSettingsOpen(!settingsOpen)}
            title="System Status"
            className="group flex items-center h-11 w-full rounded-full overflow-hidden text-[#5A6282] dark:text-[#E9E8F8]/80 hover:bg-white/75 dark:hover:bg-white/10 hover:-translate-y-0.5 transition-all duration-200"
          >
            <div className="w-11 h-11 flex items-center justify-center shrink-0 transition-transform duration-200 group-hover:scale-110">
              <Settings className="w-5 h-5" />
            </div>
            <span
              className={`text-xs font-bold whitespace-nowrap transition-all duration-300 ease-[cubic-bezier(0.22,1,0.36,1)] ${
                dockExpanded
                  ? "opacity-100 translate-x-0 ml-1 pr-3"
                  : "opacity-0 -translate-x-2 max-w-0 pointer-events-none"
              }`}
            >
              Settings
            </span>
          </button>

          {/* Brief, Human-Friendly Settings Popover */}
          {settingsOpen && (
            <div className="absolute left-full bottom-0 ml-4 w-64 aegis-card rounded-[24px] p-4 shadow-2xl z-50 text-left">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="text-xs font-bold text-[#1E2238] dark:text-white">
                    System Status: Online
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => setSettingsOpen(false)}
                  className="text-xs font-bold text-[#5A6282]"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-2 text-xs mb-3">
                {["URL AI", "Page Scanner", "Brand Guard"].map((layer) => (
                  <div
                    key={layer}
                    className="flex items-center justify-between py-1.5 px-2.5 rounded-xl bg-white/70 dark:bg-white/5"
                  >
                    <span className="font-medium text-[#1E2238] dark:text-[#E9E8F8]">
                      {layer}
                    </span>
                    <span className="font-semibold text-emerald-600 dark:text-emerald-400 flex items-center gap-1">
                      <CheckCircle2 className="w-3.5 h-3.5" /> Ready
                    </span>
                  </div>
                ))}
              </div>

              <button
                type="button"
                onClick={handleClearHistory}
                className="w-full py-2 px-3 rounded-full bg-white/80 dark:bg-white/10 hover:bg-[#F5B69E]/40 text-xs font-semibold text-[#1E2238] dark:text-white flex items-center justify-center gap-1.5 transition-all"
              >
                <Trash2 className="w-3.5 h-3.5" />
                <span>{historyClearedNotice ? "History Cleared ✓" : "Clear History"}</span>
              </button>
            </div>
          )}
        </div>
      </aside>

      {/* Mobile Slide-Over Drawer (< 768px) */}
      {mobileDrawerOpen && (
        <div className="md:hidden fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex">
          <div className="w-72 bg-[#E9E8F8] dark:bg-[#151B30] h-full p-5 flex flex-col justify-between shadow-2xl">
            <div>
              <div className="flex items-center justify-between mb-6">
                <span className="aegis-hero-logo italic font-bold text-3xl">
                  Aegis
                </span>
                <button
                  type="button"
                  onClick={() => setMobileDrawerOpen(false)}
                  className="p-2 rounded-full bg-white/70 dark:bg-white/10"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-2">
                {NAV_ITEMS.map((item) => {
                  const IconComponent = item.icon;
                  const isActive = activeTab === item.id;
                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => {
                        setActiveTab(item.id);
                        setMobileDrawerOpen(false);
                      }}
                      className={`flex items-center gap-3 w-full rounded-2xl px-4 py-3 text-left transition-all ${
                        isActive
                          ? "bg-[#1E2238] text-white dark:bg-[#8DB9F0] dark:text-[#0D111E]"
                          : "text-[#1E2238] dark:text-[#E9E8F8]"
                      }`}
                    >
                      <IconComponent className="w-5 h-5 shrink-0" />
                      <div>
                        <div className="text-sm font-bold">{item.label}</div>
                        <div className="text-[11px] opacity-75">{item.subtitle}</div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            <button
              type="button"
              onClick={() => setDarkMode(!darkMode)}
              className="flex items-center justify-center gap-2 w-full rounded-full py-3 bg-[#F5B69E] text-[#1E2238] font-bold text-xs"
            >
              {darkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
              <span>{darkMode ? "Light Mode" : "Dark Mode"}</span>
            </button>
          </div>
        </div>
      )}

      {/* Top Floating Header Bar: Left Aegis Logo, Center Feature Pills, Right CTA */}
      <header className="w-full pt-5 px-4 sm:px-8 md:pl-24 z-30">
        <div className="max-w-4xl mx-auto rounded-full bg-[#E9E8F8]/80 dark:bg-[#161B2E]/85 backdrop-blur-2xl border border-white/75 dark:border-white/15 px-5 py-2 shadow-[0_14px_40px_-10px_rgba(30,34,56,0.12)] flex items-center justify-between gap-3">
          {/* Left Zone: Mobile Menu Trigger + Stylized Aegis Logo */}
          <div className="flex items-center gap-2.5">
            <button
              type="button"
              onClick={() => setMobileDrawerOpen(true)}
              className="md:hidden p-2 rounded-full bg-white/80 dark:bg-white/10 text-[#1E2238] dark:text-white"
            >
              <Menu className="w-4 h-4" />
            </button>

            <button
              type="button"
              onClick={() => setActiveTab("scanner")}
              className="inline-flex items-center px-2 py-0.5 cursor-pointer focus:outline-none"
            >
              <span className="aegis-hero-logo text-2xl sm:text-3xl font-bold italic tracking-tight select-none leading-none pb-0.5">
                Aegis
              </span>
            </button>
          </div>

          {/* Center Zone: Interactive Feature Micro-Pills */}
          <div className="hidden sm:flex items-center gap-1.5 text-xs font-semibold text-[#5A6282] dark:text-[#E9E8F8]/85">
            <button
              type="button"
              onClick={() => setActiveTab("scanner")}
              className="hover:bg-white/80 dark:hover:bg-white/10 hover:text-[#1E2238] dark:hover:text-white hover:-translate-y-0.5 hover:shadow-sm px-3 py-1 rounded-full transition-all duration-200 cursor-pointer whitespace-nowrap"
            >
              Link Protection
            </button>
            <span className="opacity-35">&middot;</span>
            <button
              type="button"
              onClick={() => setActiveTab("adversarial")}
              className="hover:bg-white/80 dark:hover:bg-white/10 hover:text-[#1E2238] dark:hover:text-white hover:-translate-y-0.5 hover:shadow-sm px-3 py-1 rounded-full transition-all duration-200 cursor-pointer whitespace-nowrap"
            >
              Brand Check
            </button>
            <span className="opacity-35">&middot;</span>
            <button
              type="button"
              onClick={() => setActiveTab("scanner")}
              className="hover:bg-white/80 dark:hover:bg-white/10 hover:text-[#1E2238] dark:hover:text-white hover:-translate-y-0.5 hover:shadow-sm px-3 py-1 rounded-full transition-all duration-200 cursor-pointer whitespace-nowrap"
            >
              AI Prompt Guard
            </button>
          </div>

          {/* Right Zone: Reactive Try Sample Pill Button */}
          <div className="flex items-center justify-end">
            <button
              type="button"
              onClick={handleTrySample}
              className="group bg-white dark:bg-[#232A45] text-[#1E2238] dark:text-white border border-[#8DB9F0]/55 hover:border-[#F5B69E] rounded-full px-4 py-1.5 text-xs font-bold shadow-xs hover:scale-105 hover:-translate-y-0.5 hover:shadow-[0_8px_24px_rgba(245,182,158,0.45)] active:scale-95 transition-all duration-200 flex items-center gap-1.5"
            >
              <span>⚡ Try Sample</span>
              <span className="inline-block transition-transform duration-200 group-hover:translate-x-1">
                &rarr;
              </span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Content Stage */}
      <main
        className={`flex-1 flex flex-col px-4 sm:px-8 pt-6 transition-all duration-300 ${
          dockExpanded ? "md:pl-64" : "md:pl-24"
        }`}
      >
        <div className="flex-1">
          {activeTab === "scanner" && (
            <ThreatScannerTab quickDemoTrigger={quickDemoTrigger} />
          )}
          {activeTab === "adversarial" && <AdversarialLabTab />}
          {activeTab === "intelligence" && <ModelIntelligenceTab />}
          {activeTab === "batch" && <BatchAndFeedTab />}
        </div>
      </main>

      {/* Full-Width Floating Glass Footer (Inset from Page Boundaries) */}
      <footer className="w-full mt-auto pt-8 pb-4 px-4 sm:px-6">
        <div className="w-full rounded-full bg-[#E9E8F8]/65 dark:bg-[#161B2E]/65 backdrop-blur-xl border border-white/60 dark:border-white/10 px-6 py-3 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-[#5A6282] dark:text-slate-400 shadow-xs transition-all duration-300 hover:bg-[#E9E8F8]/90 dark:hover:bg-[#161B2E]/90 hover:shadow-[0_10px_28px_rgba(141,185,240,0.22)]">
          <div className="flex items-center gap-2">
            <span className="font-editorial italic text-lg font-bold text-[#1E2238] dark:text-[#8DB9F0]">
              Aegis
            </span>
            <span>&middot;</span>
            <span className="font-medium text-[#1E2238]/80 dark:text-[#E9E8F8]/80">
              Autonomous AI Cyber Defense
            </span>
          </div>

          <div className="hidden md:block text-xs font-medium text-[#5A6282] dark:text-[#E9E8F8]/70">
            Real-Time Link, Webpage &amp; AI Prompt Protection
          </div>

          <div className="flex items-center gap-2.5 font-semibold text-[#1E2238] dark:text-white">
            <span className="inline-flex items-center gap-1.5 px-3 py-0.5 rounded-full bg-white/75 dark:bg-white/10 border border-white/70 dark:border-white/10 text-[11px]">
              <span className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_#10b981] animate-pulse" />
              <span>All Guards Online</span>
            </span>
            <span className="text-[11px] text-[#5A6282] dark:text-[#8DB9F0]">
              v2.0
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
