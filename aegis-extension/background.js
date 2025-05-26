// background.js
const AEGIS_BACKEND_API = "http://127.0.0.1:5000/analyze-url"; // Ensure this matches manifest exactly
const BLOCK_PAGE_URL = chrome.runtime.getURL("block.html");
const CACHE_DURATION_MS = 1 * 60 * 60 * 1000; // 1 hour cache
const BYPASS_CACHE_FOR_DEBUGGING = true; // Set to false to re-enable cache

console.log("AEGIS: Background script loaded and running."); // Initial load check

async function getCachedResult(url) {
    if (BYPASS_CACHE_FOR_DEBUGGING) return null;
    try {
        const result = await chrome.storage.local.get([url]);
        if (result && result[url] && result[url].timestamp) {
            if (Date.now() - result[url].timestamp < CACHE_DURATION_MS) {
                console.log("AEGIS: Cache hit for", url, "- Phishing:", result[url].isPhishing);
                return result[url];
            } else {
                console.log("AEGIS: Cache expired for", url);
                await chrome.storage.local.remove([url]);
            }
        }
    } catch (e) { console.error("AEGIS: Error getting cache", e); }
    return null;
}

async function setCachedResult(url, isPhishing, score) {
    if (BYPASS_CACHE_FOR_DEBUGGING) return;
    try {
        await chrome.storage.local.set({
            [url]: { isPhishing: isPhishing, score: score, timestamp: Date.now() }
        });
        console.log("AEGIS: Cached result for", url);
    } catch (e) { console.error("AEGIS: Error setting cache", e); }
}

chrome.webNavigation.onBeforeNavigate.addListener(
    async (details) => {
        console.log("AEGIS: onBeforeNavigate triggered. URL:", details.url, "Tab ID:", details.tabId, "Frame ID:", details.frameId);

        if (details.frameId !== 0) {
            console.log("AEGIS: Ignoring non-main frame navigation.");
            return;
        }

        const urlToScan = details.url;

        if (!urlToScan || urlToScan.startsWith("chrome://") || urlToScan.startsWith("about:") || urlToScan.startsWith("file://")) {
            console.log("AEGIS: Ignoring internal/filtered URL:", urlToScan);
            return;
        }
        
        // For initial debugging, let's always hit the backend
        if (!BYPASS_CACHE_FOR_DEBUGGING) {
            const cached = await getCachedResult(urlToScan);
            if (cached !== null) {
                if (cached.isPhishing) {
                    console.log("AEGIS: Blocking (from cache)", urlToScan);
                    chrome.tabs.update(details.tabId, { url: `${BLOCK_PAGE_URL}?url=${encodeURIComponent(urlToScan)}&score=${cached.score}&source=cache` });
                } else {
                    console.log("AEGIS: Allowed (from cache, not phishing)", urlToScan);
                }
                return;
            }
            console.log("AEGIS: Cache miss or bypassed for", urlToScan);
        } else {
            console.log("AEGIS: Cache bypassed for debugging for URL:", urlToScan);
        }


        console.log(`AEGIS: Attempting to fetch backend at ${AEGIS_BACKEND_API} for URL:`, urlToScan);
        try {
            const response = await fetch(AEGIS_BACKEND_API, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url: urlToScan }),
            });

            console.log("AEGIS: Fetch call made to backend, response status:", response.status);

            if (!response.ok) {
                console.error("AEGIS: Backend API error!", response.status, response.statusText, "for URL:", urlToScan);
                const errorBody = await response.text();
                console.error("AEGIS: Backend error body:", errorBody);
                return; // Allow navigation on backend error for now
            }

            const data = await response.json();
            if (data.error) {
                console.error("AEGIS: Backend returned an analysis error:", data.error, "for URL:", urlToScan);
                return; // Allow navigation if backend reports an analysis error
            }

            console.log("AEGIS: Backend response for", urlToScan, JSON.stringify(data));
            const isPhishing = data.prediction === "Phishing";
            const score = parseFloat(data.score);

            if (!BYPASS_CACHE_FOR_DEBUGGING) {
                await setCachedResult(urlToScan, isPhishing, score);
            }

            if (isPhishing) {
                console.log("AEGIS: Blocking (from backend)", urlToScan);
                chrome.tabs.update(details.tabId, { url: `${BLOCK_PAGE_URL}?url=${encodeURIComponent(urlToScan)}&score=${score}&source=backend` });
            } else {
                console.log("AEGIS: Allowed (from backend, not phishing)", urlToScan);
            }
        } catch (error) {
            console.error("AEGIS: Error fetching from backend or processing response:", error, "for URL:", urlToScan);
        }
    },
    { url: [{ schemes: ["http", "https"] }] }
);

// Optional: Clear expired cache items periodically
chrome.alarms.create("cacheClearer", { periodInMinutes: 60 }); // Alarm will fire after 60 mins then every 60 mins
chrome.alarms.onAlarm.addListener(async (alarm) => {
    console.log("AEGIS: Alarm triggered:", alarm.name);
    if (alarm.name === "cacheClearer" && !BYPASS_CACHE_FOR_DEBUGGING) {
        try {
            const items = await chrome.storage.local.get(null);
            const urlsToRemove = [];
            for (const url in items) {
                // Check if the item is one of our cache entries (has a timestamp)
                if (items[url] && typeof items[url] === 'object' && items[url].hasOwnProperty('timestamp')) {
                    if (Date.now() - items[url].timestamp >= CACHE_DURATION_MS) {
                        urlsToRemove.push(url);
                    }
                }
            }
            if (urlsToRemove.length > 0) {
                await chrome.storage.local.remove(urlsToRemove);
                console.log("AEGIS: Cleared expired cache items:", urlsToRemove.length);
            } else {
                console.log("AEGIS: No cache items to clear.");
            }
        } catch (e) { console.error("AEGIS: Error clearing cache", e); }
    }
});

console.log("AEGIS: Event listeners for webNavigation and alarms set up.");