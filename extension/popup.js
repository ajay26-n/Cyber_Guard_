const API_URL = "https://cyber-guard-rouge.vercel.app/scan";

document.addEventListener("DOMContentLoaded", async () => {
  const activeUrlElem = document.getElementById("active-url");
  const verdictBadge = document.getElementById("verdict-badge");
  const riskMeterBar = document.getElementById("risk-meter-bar");
  const riskScoreVal = document.getElementById("risk-score-value");
  const aiExplanationText = document.getElementById("ai-explanation-text");

  const valDomainAge = document.getElementById("val-domain-age");
  const valSsl = document.getElementById("val-ssl");
  const valRedirects = document.getElementById("val-redirects");
  const valLogin = document.getElementById("val-login");

  const btnManualScan = document.getElementById("btn-manual-scan");
  const manualInput = document.getElementById("manual-url-input");
  const btnToggleBlock = document.getElementById("btn-toggle-block");
  const connStatus = document.getElementById("connection-status");

  let currentScannedUrl = "";
  let currentScannedDomain = "";

  // Helper to query active browser tab (Chrome/Firefox compatible API)
  async function getActiveTabUrl() {
    return new Promise((resolve) => {
      if (typeof chrome !== "undefined" && chrome.tabs && chrome.tabs.query) {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
          if (tabs && tabs[0] && tabs[0].url) {
            resolve(tabs[0].url);
          } else {
            resolve("");
          }
        });
      } else if (typeof browser !== "undefined" && browser.tabs && browser.tabs.query) {
        browser.tabs.query({ active: true, currentWindow: true }).then((tabs) => {
          if (tabs && tabs[0] && tabs[0].url) {
            resolve(tabs[0].url);
          } else {
            resolve("");
          }
        });
      } else {
        resolve("");
      }
    });
  }

  // Check backend server availability
  async function scanUrl(targetUrl) {
    if (!targetUrl || targetUrl.startsWith("chrome://") || targetUrl.startsWith("about:")) {
      activeUrlElem.textContent = targetUrl || "Internal Browser Page";
      verdictBadge.textContent = "SYSTEM PAGE";
      verdictBadge.className = "badge badge-safe";
      aiExplanationText.textContent = "Internal browser pages are safe.";
      riskMeterBar.style.width = "0%";
      riskScoreVal.textContent = "0%";
      return;
    }

    currentScannedUrl = targetUrl;
    activeUrlElem.textContent = targetUrl;
    verdictBadge.textContent = "ANALYZING...";
    verdictBadge.className = "badge badge-scanning";

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: targetUrl })
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();
      connStatus.textContent = "ENGINE ONLINE";
      connStatus.className = "status-pill status-online";

      renderScanResults(data);
    } catch (err) {
      console.error("CyberGuard extension error:", err);
      connStatus.textContent = "OFFLINE";
      connStatus.className = "status-pill status-offline";
      verdictBadge.textContent = "UNREACHABLE";
      verdictBadge.className = "badge badge-scanning";
      aiExplanationText.textContent = `Could not connect to CyberGuard backend at ${API_URL}. Ensure the server is online.`;
    }
  }

  function renderScanResults(data) {
    const isPhishing = data.result === "PHISHING";
    const risk = data.risk_score || 0;

    // Verdict Badge
    verdictBadge.textContent = isPhishing ? "PHISHING DETECTED" : "SAFE WEBSITE";
    verdictBadge.className = isPhishing ? "badge badge-phishing" : "badge badge-safe";

    // Risk Meter
    riskScoreVal.textContent = `${risk}%`;
    riskMeterBar.style.width = `${Math.min(risk, 100)}%`;
    if (risk >= 50) {
      riskMeterBar.style.backgroundColor = "var(--red)";
    } else if (risk >= 25) {
      riskMeterBar.style.backgroundColor = "var(--yellow)";
    } else {
      riskMeterBar.style.backgroundColor = "var(--green)";
    }

    // Matrix
    valDomainAge.textContent = data.domain_age !== null && data.domain_age !== undefined ? `${data.domain_age} days` : "Unknown";
    valSsl.textContent = data.ssl_valid ? "Valid (HTTPS)" : "Invalid / None";
    valSsl.style.color = data.ssl_valid ? "var(--green)" : "var(--red)";

    valRedirects.textContent = data.redirects !== undefined ? `${data.redirects} hops` : "0";
    valLogin.textContent = data.login_page ? "Detected" : "None";
    valLogin.style.color = data.login_page ? "var(--red)" : "var(--text-main)";

    // AI Explanation
    aiExplanationText.textContent = data.ai_explanation || "No suspicious indicators detected.";

    // Update Block Button
    try {
      const urlObj = new URL(data.url || currentScannedUrl);
      currentScannedDomain = urlObj.hostname;
    } catch (e) {
      currentScannedDomain = currentScannedUrl;
    }

    checkBlockedState(currentScannedDomain);
  }

  function checkBlockedState(domain) {
    const storageApi = (typeof chrome !== "undefined" && chrome.storage) ? chrome.storage.local : (typeof browser !== "undefined" ? browser.storage.local : null);
    if (!storageApi) return;

    storageApi.get(["blockedDomains"], (res) => {
      const list = res.blockedDomains || [];
      if (list.includes(domain)) {
        btnToggleBlock.textContent = "🔓 UNBLOCK DOMAIN";
        btnToggleBlock.className = "btn-block btn-unblock";
      } else {
        btnToggleBlock.textContent = "🛡️ BLOCK DOMAIN";
        btnToggleBlock.className = "btn-block";
      }
    });
  }

  // Toggle domain block
  btnToggleBlock.addEventListener("click", () => {
    if (!currentScannedDomain) return;
    const storageApi = (typeof chrome !== "undefined" && chrome.storage) ? chrome.storage.local : (typeof browser !== "undefined" ? browser.storage.local : null);
    if (!storageApi) return;

    storageApi.get(["blockedDomains"], (res) => {
      let list = res.blockedDomains || [];
      if (list.includes(currentScannedDomain)) {
        list = list.filter(d => d !== currentScannedDomain);
      } else {
        list.push(currentScannedDomain);
      }
      storageApi.set({ blockedDomains: list }, () => {
        checkBlockedState(currentScannedDomain);
      });
    });
  });

  // Manual Scan Button
  btnManualScan.addEventListener("click", () => {
    const inputVal = manualInput.value.trim();
    if (inputVal) {
      let target = inputVal;
      if (!target.startsWith("http://") && !target.startsWith("https://")) {
        target = "https://" + target;
      }
      scanUrl(target);
    }
  });

  // Auto-scan active tab URL on popup load
  const activeTabUrl = await getActiveTabUrl();
  if (activeTabUrl) {
    scanUrl(activeTabUrl);
  }
});
