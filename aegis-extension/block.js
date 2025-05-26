// block.js
document.addEventListener('DOMContentLoaded', function() {
    const urlParams = new URLSearchParams(window.location.search);
    const originalUrl = urlParams.get('url');
    const scoreParam = urlParams.get('score'); // Score is a string
    const source = urlParams.get('source');

    const blockedUrlElement = document.getElementById('blockedUrlElement');
    const phishingScoreElement = document.getElementById('phishingScoreElement');
    const threatMimicElement = document.getElementById('threat-mimic');
    const threatNewDomainElement = document.getElementById('threat-new-domain');

    if (blockedUrlElement) {
        blockedUrlElement.textContent = originalUrl ? decodeURIComponent(originalUrl) : 'Unknown URL';
    }

    let numericScore = 0;
    if (scoreParam) {
        numericScore = parseFloat(scoreParam);
        if (phishingScoreElement) {
            phishingScoreElement.textContent = `AEGIS Phishing Score: ${numericScore.toFixed(4)} (Detected via ${source || 'analysis'})`;
        }
    } else if (phishingScoreElement) {
        phishingScoreElement.textContent = 'AEGIS Phishing Score: N/A';
    }

    // Simple logic to show conditional threat details based on score
    if (numericScore > 0.75 && threatMimicElement) { // High score might indicate impersonation
        threatMimicElement.style.display = 'flex';
    }
    if (numericScore > 0.5 && threatNewDomainElement) { // Moderate to high might be new domains
        // In a real scenario, you'd pass specific feature flags from the backend
        // For now, this is just illustrative based on score
        // threatNewDomainElement.style.display = 'flex'; // Example, you might not have this info yet
    }


    document.getElementById('backButton').addEventListener('click', () => {
        window.history.back();
    });

    document.getElementById('proceedButton').addEventListener('click', () => {
        // For a real extension, you would message the background script to
        // temporarily whitelist this URL for the current session/tab.
        // Example:
        // chrome.runtime.sendMessage({action: "allowUrlTemporarily", url: originalUrl}, response => {
        //     if (response && response.success) {
        //         window.location.replace(originalUrl); // Use replace to avoid it being in history if back is pressed
        //     } else {
        //         alert("Could not grant temporary access. Please manage whitelist in extension options.");
        //     }
        // });
        // For now, just an alert:
        const proceed = confirm('WARNING: This site has been identified as potentially dangerous. Proceeding may expose your device and personal information to phishing, malware, or other security threats. Are you absolutely sure you want to continue?');
        if (proceed) {
            // A more robust way to "proceed" would be to navigate away to a neutral page
            // and instruct the user to re-enter the URL, while the background script
            // has temporarily whitelisted it for the next single navigation attempt.
            // Directly navigating here might get re-blocked.
            alert('You have chosen to proceed. Please be extremely cautious. AEGIS cannot protect you further on this specific page if you bypass this warning.');
            // window.location.replace(originalUrl); // This might be blocked again immediately
        }
    });
});