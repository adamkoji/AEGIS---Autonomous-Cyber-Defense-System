# AEGIS (Autonomous Evolving Guardian for Intelligent Security)

AEGIS is a proof-of-concept for a self-learning AI system designed to autonomously detect and mitigate cyberattacks, with an initial focus on phishing websites. This project demonstrates an advanced phishing detection engine utilizing both network/lexical URL analysis and DOM content/structure analysis.

## Key Features Implemented:

*   **Advanced Phishing Detection Engine:**
    *   **Network/Lexical Analysis:** Employs a Transformer model (PyTorch) trained on 100+ features extracted from URLs (lexical characteristics, DNS records, WHOIS data, SSL info, etc.).
    *   **DOM Analysis:** Utilizes a LightGBM/RandomForest model to analyze HTML structure, forms, scripts, and content for suspicious patterns.
    *   **Combined Scoring:** Leverages insights from both analysis methods for improved accuracy.
*   **Flask Web Application:**
    *   A user interface for on-demand URL scanning.
    *   Serves as the backend API for real-time analysis requests.
*   **Browser Extension Prototype (Chrome):**
    *   Monitors web navigation in real-time.
    *   Queries the AEGIS backend for URL risk assessment.
    *   Blocks access to detected phishing sites and displays a user-friendly warning page.

## Technologies Used:

*   **Machine Learning:** Python, PyTorch, Scikit-learn, LightGBM, Pandas, NumPy
*   **Web Backend:** Flask
*   **Feature Extraction:** `requests`, `tldextract`, `dnspython`, `python-whois`, `BeautifulSoup4`
*   **Browser Extension:** JavaScript (Manifest V3), HTML, CSS
*   **Styling:** Tailwind CSS (for the extension's block page, compiled locally)
*   **Development:** Jupyter Notebook, VS Code

## Project Structure:

*   `phishing-detection.ipynb`: Contains the model training and feature engineering logic.
*   `PHISHING_APP/`: Directory for the Flask web application (backend API and UI).
    *   `app.py`: Main Flask application.
    *   `model_def.py`: PyTorch model definition.
    *   `feature_extractor.py`: Functions for extracting features from URLs and DOM.
    *   `model/`: Stores the trained model files (`.pt`, `.pkl`).
    *   `templates/`, `static/`: For the web UI.
*   `aegis-extension/`: Directory for the Chrome browser extension.
    *   `manifest.json`: Extension configuration.
    *   `background.js`: Core logic for navigation interception and API calls.
    *   `block.html`, `block.js`: The warning page displayed for blocked sites.
    *   `styles/`: Contains compiled Tailwind CSS.

## Future AEGIS Vision:

This implementation serves as the "Detector" component of the larger AEGIS vision, which aims to include:
*   Generative AI for dynamic attack simulation.
*   Multi-Agent Reinforcement Learning for adaptive response.
*   Federated learning for privacy-preserving threat intelligence.

## Setup & Running:

(You would add instructions here on how to set up the Python environment, run the Flask app, and load the browser extension).
