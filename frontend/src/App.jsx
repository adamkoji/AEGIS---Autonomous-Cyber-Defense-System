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
        className={`hidden md:flex fixed left-5 top-1/2 -translate-y-1/2 z-40 flex-col justify-between bg-[#E9E8F8]/85 dark:bg-[#161B2E]/85 backdrop-blur-2xl border border-white/60 dark:border-white/10 shadow-[0_12px_36px_rgba(30,34,56,0.10)] py-4 px-2.5 transition-all duration-300 ${
          dockExpanded ? "w-56 rounded-[32px]" : "w-16 rounded-full"
        }`}
        style={{ minHeight: "420px" }}
      >
        <div className="flex flex-col items-center gap-2">
          <button
            type="button"
            onClick={() => setDockExpanded(!dockExpanded)}
            title={dockExpanded ? "Collapse menu" : "Expand menu"}
            className="w-11 h-11 rounded-full flex items-center justify-center text-[#1E2238] dark:text-[#E9E8F8] hover:bg-white/70 dark:hover:bg-white/10 transition-all"
          >
            {dockExpanded ? (
              <PanelLeftClose className="w-5 h-5" />
            ) : (
              <PanelLeftOpen className="w-5 h-5" />
            )}
          </button>

          <div className="w-7 h-px bg-black/10 dark:bg-white/10 my-1" />

          {/* 4 Navigation Views */}
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
                  className={`group relative flex items-center gap-3 w-full rounded-full p-2.5 transition-all ${
                    isActive
                      ? "bg-[#1E2238] text-white dark:bg-gradient-to-r dark:from-[#8DB9F0] dark:to-[#F5B69E] dark:text-[#0D111E] shadow-md"
                      : "text-[#5A6282] dark:text-[#E9E8F8]/75 hover:bg-white/70 dark:hover:bg-white/10 hover:text-[#1E2238] dark:hover:text-white"
                  }`}
                >
                  <div className="w-6 h-6 flex items-center justify-center shrink-0 mx-auto md:mx-0">
                    <IconComponent className="w-5 h-5" />
                  </div>
                  {dockExpanded && (
                    <div className="text-left overflow-hidden">
                      <div className="text-xs font-bold truncate">{item.label}</div>
                      <div
                        className={`text-[10px] truncate ${
                          isActive
                            ? "text-white/75 dark:text-[#0D111E]/75"
                            : "text-[#5A6282] dark:text-[#E9E8F8]/55"
                        }`}
                      >
                        {item.subtitle}
                      </div>
                    </div>
                  )}

                  {!dockExpanded && (
                    <span className="pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity absolute left-16 px-3 py-1.5 rounded-full bg-[#1E2238] text-white text-xs font-semibold whitespace-nowrap shadow-lg">
                      {item.label}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Bottom Dock Controls: Theme Switcher & Settings Popover */}
        <div className="flex flex-col items-center gap-2 pt-3 border-t border-black/10 dark:border-white/10 relative">
          <button
            type="button"
            onClick={() => setDarkMode(!darkMode)}
            title={darkMode ? "Light Mode" : "Dark Mode"}
            className={`flex items-center gap-3 w-full rounded-full p-2.5 transition-all ${
              darkMode
                ? "bg-[#F5B69E] text-[#1E2238] shadow-sm"
                : "bg-white/75 text-[#1E2238] hover:bg-[#F5B69E]/45"
            }`}
          >
            <div className="w-6 h-6 flex items-center justify-center shrink-0 mx-auto md:mx-0">
              {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </div>
            {dockExpanded && (
              <span className="text-xs font-bold">
                {darkMode ? "Light Mode" : "Dark Mode"}
              </span>
            )}
          </button>

          <button
            type="button"
            onClick={() => setSettingsOpen(!settingsOpen)}
            title="System Status"
            className="flex items-center gap-3 w-full rounded-full p-2.5 text-[#5A6282] dark:text-[#E9E8F8]/80 hover:bg-white/70 dark:hover:bg-white/10 transition-all"
          >
            <div className="w-6 h-6 flex items-center justify-center shrink-0 mx-auto md:mx-0">
              <Settings className="w-5 h-5" />
            </div>
            {dockExpanded && <span className="text-xs font-bold">Settings</span>}
          </button>

          {/* Brief, Human-Friendly Settings Popover */}
          {settingsOpen && (
            <div className="absolute left-full bottom-0 ml-4 w-64 aegis-card rounded-[24px] p-4 shadow-2xl z-50 text-left">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500" />
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
                <span className="font-editorial italic font-bold text-3xl text-[#1E2238] dark:text-white">
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

      {/* Simplified Top Floating Pill Bar */}
      <header className="w-full pt-5 px-4 sm:px-8 md:pl-24 z-30">
        <div className="max-w-4xl mx-auto rounded-full bg-[#E9E8F8]/85 dark:bg-[#161B2E]/85 backdrop-blur-2xl px-3 py-2 shadow-sm border border-white/60 dark:border-white/10 flex items-center justify-between gap-2">
          {/* Left Pill: Aegis · Active */}
          <div className="flex items-center gap-2">
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
              className="inline-flex items-center gap-2 bg-white dark:bg-[#232A45] text-[#1E2238] dark:text-white rounded-full px-4 py-1.5 text-xs font-semibold shadow-xs"
            >
              <span className="font-editorial italic text-base font-bold text-[#5B96E5] dark:text-[#8DB9F0]">
                Aegis
              </span>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
              <span className="text-xs text-[#5A6282] dark:text-[#E9E8F8]/90 font-medium">
                Active
              </span>
            </button>
          </div>

          {/* Center Clean Items: Link Protection · Brand Check · AI Prompt Guard */}
          <div className="hidden sm:flex items-center gap-3 text-xs font-semibold text-[#5A6282] dark:text-[#E9E8F8]/85">
            <span>Link Protection</span>
            <span className="opacity-40">&middot;</span>
            <span>Brand Check</span>
            <span className="opacity-40">&middot;</span>
            <span>AI Prompt Guard</span>
          </div>

          {/* Right Button: ⚡ Try Sample → */}
          <button
            type="button"
            onClick={handleTrySample}
            className="bg-white dark:bg-[#232A45] text-[#1E2238] dark:text-white rounded-full px-4 py-1.5 text-xs font-semibold shadow-xs hover:scale-105 transition-all"
          >
            ⚡ Try Sample &rarr;
          </button>
        </div>
      </header>

      {/* Main Content Stage */}
      <main
        className={`flex-1 px-4 sm:px-8 pt-6 transition-all duration-300 ${
          dockExpanded ? "md:pl-64" : "md:pl-24"
        }`}
      >
        {activeTab === "scanner" && (
          <ThreatScannerTab quickDemoTrigger={quickDemoTrigger} />
        )}
        {activeTab === "adversarial" && <AdversarialLabTab />}
        {activeTab === "intelligence" && <ModelIntelligenceTab />}
        {activeTab === "batch" && <BatchAndFeedTab />}
      </main>
    </div>
  );
}
