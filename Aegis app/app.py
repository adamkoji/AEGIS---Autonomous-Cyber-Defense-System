# app.py
import os
import pickle
import torch
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from urllib.parse import urlparse # Ensure this is at the top
import re
import warnings

# Import model definition and feature extractor
from model_def import PhishingTransformerModel
from feature_extractor import extract_features_for_url, SCALER_FEATURE_ORDER

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch.serialization')
warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn.base')
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.base')

# --- Configuration ---
MODEL_PATH = os.path.join('model', 'phishing_transformer_best.pt')
SCALER_PATH = os.path.join('model', 'tabular_scaler.pkl')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load Model and Scaler ---
model = None
scaler = None
input_dim_from_scaler = None

try:
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully.")

    if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ is not None:
         input_dim_from_scaler = scaler.n_features_in_
    elif hasattr(scaler, 'mean_') and scaler.mean_ is not None:
         input_dim_from_scaler = len(scaler.mean_)
    elif SCALER_FEATURE_ORDER:
        input_dim_from_scaler = len(SCALER_FEATURE_ORDER)
        print(f"Warning: Could not determine input_dim from scaler attributes, using length of SCALER_FEATURE_ORDER: {input_dim_from_scaler}")
    else:
         raise ValueError("Could not determine input dimension for scaler and SCALER_FEATURE_ORDER is empty.")
    print(f"Determined input dimension from scaler (number of features): {input_dim_from_scaler}")

    model = PhishingTransformerModel(
        input_dim=input_dim_from_scaler,
        max_seq_len=5000, # WORKAROUND: To match the .pt file saved with this max_seq_len
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        ff_dim=128,
        dropout=0.1,
        num_classes=1
    )
    print(f"Model structure created with input_dim={input_dim_from_scaler} and max_seq_len=5000 (WORKAROUND).")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
        print("Model state loaded successfully (weights_only=True).")
    except Exception as e_weights:
        print(f"Loading with weights_only=True failed ({e_weights}), falling back to default loading.")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model state loaded successfully (default method).")

    model.to(DEVICE)
    model.eval()
    print(f"Model moved to {DEVICE} and set to evaluation mode.")

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    model, scaler = None, None
except ValueError as e:
     print(f"ValueError during setup: {e}")
     model, scaler = None, None
except RuntimeError as e_runtime:
    print(f"RuntimeError loading model state: {e_runtime}")
    model, scaler = None, None
except Exception as e:
    print(f"An unexpected error occurred during model/scaler loading: {e}")
    import traceback
    traceback.print_exc()
    model, scaler = None, None

# --- Flask App ---
app = Flask(__name__)

def normalize_url_for_analysis(url_string):
    """
    Normalizes a URL, primarily by removing a trailing slash 
    if the path is just '/' or empty and there are no query/fragment.
    Also ensures a scheme is present.
    """
    if not url_string:
        return "" # Return empty if input is empty
    
    url_string = url_string.strip()
    parsed_temp = urlparse(url_string)
    if not parsed_temp.scheme:
        url_string = 'http://' + url_string # Default to http if no scheme
        print(f"Normalizer: Added scheme to '{parsed_temp.geturl()}' -> '{url_string}'")
    
    try:
        parsed = urlparse(url_string)
        # If path is '/' or empty, and no query/fragment, reconstruct without the path
        if (parsed.path == '/' or not parsed.path) and not parsed.query and not parsed.fragment:
            normalized = f"{parsed.scheme}://{parsed.netloc}"
            # print(f"Normalizer: Path was '/' or empty. Normalized '{url_string}' to '{normalized}'")
            return normalized
        
        # General case: rstrip any trailing slash from the full URL if it's not the only char in path
        # This handles domain.com/path/ -> domain.com/path
        # but keeps domain.com/ (which would have been handled above if no further path)
        # However, if the path itself is just '/', the above logic is better.
        # The previous logic handles domain.com/ -> domain.com.
        # This handles domain.com/path1/ -> domain.com/path1
        if parsed.path and parsed.path != '/' and url_string.endswith('/'):
            normalized = url_string.rstrip('/')
            # print(f"Normalizer: Rstripped trailing slash. Normalized '{url_string}' to '{normalized}'")
            return normalized

    except Exception as e:
        print(f"Error during URL normalization for '{url_string}': {e}")
        return url_string # Return original on parsing error
    
    # print(f"Normalizer: URL '{url_string}' returned as is (or with scheme added).")
    return url_string


@app.route('/', methods=['GET', 'POST'])
@app.route('/analyze-url', methods=['POST'])
def index():
    print(f"Flask: Received request for path: {request.path}, method: {request.method}")

    # API Call Handling
    if request.path == '/analyze-url':
        print("Flask: Handling as API call to /analyze-url")
        if model is None or scaler is None:
            print("Flask API Error: Model or Scaler not loaded")
            return jsonify({"error": "Model or Scaler not loaded, cannot perform analysis"}), 500
        
        if not request.is_json:
            print("Flask API Error: Request not JSON")
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        url_to_analyze_raw = data.get('url')
        if not url_to_analyze_raw:
            print("Flask API Error: URL not provided in JSON")
            return jsonify({"error": "URL not provided"}), 400
        
        url_to_analyze = normalize_url_for_analysis(url_to_analyze_raw)
        print(f"API: Raw URL: '{url_to_analyze_raw}', Normalized URL for analysis: '{url_to_analyze}'")
        
        if not url_to_analyze: # Check if normalization resulted in empty string
            print("Flask API Error: URL became empty after normalization.")
            return jsonify({"error": "Invalid URL after normalization"}), 400

        try:
            features_dict = extract_features_for_url(url_to_analyze)
            features_df_for_scaling = pd.DataFrame([features_dict], columns=SCALER_FEATURE_ORDER)
            features_df_for_scaling = features_df_for_scaling.fillna(0)
            scaled_features = scaler.transform(features_df_for_scaling)
            features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(DEVICE)
            
            with torch.no_grad():
                output = model(features_tensor)
                score_val = output.item()

            final_prediction_class = 1 if score_val > 0.5 else 0
            print(f"API: Prediction for '{url_to_analyze}' - Score: {score_val:.4f}, Class: {final_prediction_class}")
            
            return jsonify({
                "url": url_to_analyze_raw, # Return original URL for reference
                "analyzed_url": url_to_analyze, # Show what was actually analyzed
                "prediction": "Phishing" if final_prediction_class == 1 else "Legitimate",
                "score": f"{score_val:.4f}",
            })
        except Exception as e:
            print(f"API Error during analysis for '{url_to_analyze}': {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"Analysis error: {str(e)}"}), 500

    # --- HTML Web UI Handling ---
    results = None
    error = None
    url_input_from_form = request.form.get('url', '') # Keep original form input for display

    if model is None or scaler is None:
        error = "Model or Scaler failed to load. Cannot perform analysis. Please check server logs."
        return render_template('index.html', results=results, error=error, url_input=url_input_from_form)

    if request.method == 'POST':
        url_raw_from_form = url_input_from_form.strip()
        if not url_raw_from_form:
            error = "Please enter a URL."
        else:
            # Scheme check for web UI input
            parsed_init_form = urlparse(url_raw_from_form)
            if not parsed_init_form.scheme:
                 url_to_normalize = 'http://' + url_raw_from_form # Add scheme before normalization
                 # url_input_from_form = url_to_normalize # Update this if you want the form to show the prefixed URL
            elif parsed_init_form.scheme not in ['http', 'https']:
                 error = "Invalid URL scheme. Please use 'http' or 'https'."
                 url_to_normalize = url_raw_from_form # Will likely fail normalization or analysis
            else:
                 url_to_normalize = url_raw_from_form

        if url_raw_from_form and not error: # Analyze if we have a URL and no prior error
            url_analyzed_for_ui = normalize_url_for_analysis(url_to_normalize)
            print(f"WEB UI: Raw URL: '{url_raw_from_form}', Normalized URL for analysis: '{url_analyzed_for_ui}'")

            if not url_analyzed_for_ui: # Check after normalization
                error = "Invalid URL after normalization."
            else:
                try:
                    print(f"WEB UI: Analyzing normalized URL: {url_analyzed_for_ui}")
                    score_val = 0.0

                    features_dict = extract_features_for_url(url_analyzed_for_ui)
                    features_df_for_scaling = pd.DataFrame([features_dict], columns=SCALER_FEATURE_ORDER)
                    features_df_for_scaling = features_df_for_scaling.fillna(0)
                    scaled_features = scaler.transform(features_df_for_scaling)
                    features_tensor = torch.tensor(scaled_features, dtype=torch.float32).to(DEVICE)

                    with torch.no_grad():
                        output = model(features_tensor)
                        score_val = output.item()
                    print(f"WEB UI Model Score (P_phishing): {score_val:.4f}")
                    
                    final_prediction_class = 1 if score_val > 0.5 else 0

                    if score_val > 0.8: risk_level = "Critical"
                    elif score_val > 0.6: risk_level = "High"
                    elif score_val > 0.4: risk_level = "Medium"
                    else: risk_level = "Low"
                    
                    key_metrics = {
                        'URL Length': features_dict.get('length_url', 'N/A'),
                        'Domain Length': features_dict.get('domain_length', 'N/A'),
                        '# Redirects': features_dict.get('qty_redirects', 'N/A'),
                        'SSL Cert Present': 'Yes' if features_dict.get('tls_ssl_certificate', 0) == 1 else 'No'
                    }
                    results = {
                        'url': url_input_from_form, # Show original user input
                        'url_analyzed': url_analyzed_for_ui, # Show what was actually analyzed
                        'prediction': "Phishing" if final_prediction_class == 1 else "Legitimate",
                        'score': f"{score_val:.4f}",
                        'risk_level': risk_level,
                        'status_color': 'red' if final_prediction_class == 1 else ('yellow' if risk_level == 'Medium' else 'green'),
                        'status_icon': 'fa-exclamation-triangle' if final_prediction_class == 1 else ('fa-exclamation-circle' if risk_level == 'Medium' else 'fa-check-circle'),
                        'metrics': key_metrics,
                    }
                except Exception as e:
                    error = f"An error occurred during URL analysis: {str(e)}"
                    print(f"WEB UI Critical error analyzing URL {url_analyzed_for_ui}: {e}")
                    import traceback
                    traceback.print_exc()
    
    return render_template('index.html', results=results, error=error, url_input=url_input_from_form)

if __name__ == '__main__':
    if model is None or scaler is None:
         print("\n--- Cannot start Flask server: Model or Scaler failed to load. Check errors above. ---")
    else:
         print("\n--- Starting AEGIS Phishing Detection Flask server (Network/Lexical Model Only - Workaround max_seq_len) ---")
         print(f"Running on device: {DEVICE}")
         app.run(debug=True, host='127.0.0.1', port=5000)