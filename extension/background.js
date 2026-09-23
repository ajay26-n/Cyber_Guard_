const API_URL = "https://cyber-guard-r4vionjpd-titans-985d.vercel.app/scan";

const browserApi = (typeof chrome !== "undefined" && chrome.tabs) ? chrome : (typeof browser !== "undefined" ? browser : null);

// Monitor tab navigation
if (browserApi && browserApi.tabs) {
  browserApi.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
    if (changeInfo.status === "complete" && tab.url) {
      inspectTabUrl(tabId, tab.url);
    }
  });

  browserApi.tabs.onActivated.addListener((activeInfo) => {
    browserApi.tabs.get(activeInfo.tabId, (tab) => {
      if (tab && tab.url) {
        inspectTabUrl(tab.id, tab.url);
      }
    });
  });
}

async function inspectTabUrl(tabId, url) {
  if (!url || url.startsWith("chrome://") || url.startsWith("about:") || url.includes("blocked.html") || url.includes("127.0.0.1") || url.includes("localhost") || url.includes("vercel.app")) {
    updateBadge(tabId, "", "");
    return;
  }

  let domain = "";
  try {
    domain = new URL(url).hostname;
  } catch (e) {
    return;
  }

  // 1. Check local blocked storage list first
  if (browserApi.storage && browserApi.storage.local) {
    browserApi.storage.local.get(["blockedDomains"], (res) => {
      const list = res.blockedDomains || [];
      if (list.includes(domain)) {
        blockTab(tabId, url);
        return;
      }
    });
  }

  // 2. Query CyberGuard backend API for threat analysis
  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: url })
    });

    if (res.ok) {
      const data = await res.json();
      if (data.result === "PHISHING" || data.risk_score >= 50) {
        updateBadge(tabId, "RISK", "#ef4444");
        blockTab(tabId, url, data.risk_score, data.ai_explanation);
      } else {
        updateBadge(tabId, "SAFE", "#10b981");
      }
    }
  } catch (e) {
    // Backend offline or unreachable, ignore silent background failures
  }
}

function updateBadge(tabId, text, color) {
  if (browserApi.action && browserApi.action.setBadgeText) {
    browserApi.action.setBadgeText({ tabId: tabId, text: text });
    if (color) {
      browserApi.action.setBadgeBackgroundColor({ tabId: tabId, color: color });
    }
  }
}

function blockTab(tabId, targetUrl, riskScore, reason) {
  const blockPageUrl = browserApi.runtime.getURL("blocked.html") +
    `?target=${encodeURIComponent(targetUrl)}&risk=${riskScore || 85}&reason=${encodeURIComponent(reason || "High risk phishing pattern detected.")}`;

  browserApi.tabs.update(tabId, { url: blockPageUrl });
}
