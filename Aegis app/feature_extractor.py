# feature_extractor.py
import re
from urllib.parse import urlparse, parse_qs # <<< Corrected import
import requests
import tldextract
import time
import numpy as np # Import numpy for handling potential NaNs

# --- Simplified Feature Extractor ---

# Define the exact order of features the scaler expects
# IMPORTANT: This MUST match the order used when the scaler was fit.
# This list is based on the full Mendeley dataset features seen in the notebook.
SCALER_FEATURE_ORDER = [
    'qty_dot_url', 'qty_hyphen_url', 'qty_underline_url', 'qty_slash_url',
    'qty_questionmark_url', 'qty_equal_url', 'qty_at_url', 'qty_and_url',
    'qty_exclamation_url', 'qty_space_url', 'qty_tilde_url', 'qty_comma_url',
    'qty_plus_url', 'qty_asterisk_url', 'qty_hashtag_url', 'qty_dollar_url',
    'qty_percent_url', 'qty_tld_url', 'length_url', 'qty_dot_domain',
    'qty_hyphen_domain', 'qty_underline_domain', 'qty_slash_domain',
    'qty_questionmark_domain', 'qty_equal_domain', 'qty_at_domain',
    'qty_and_domain', 'qty_exclamation_domain', 'qty_space_domain',
    'qty_tilde_domain', 'qty_comma_domain', 'qty_plus_domain',
    'qty_asterisk_domain', 'qty_hashtag_domain', 'qty_dollar_domain',
    'qty_percent_domain', 'qty_vowels_domain', 'domain_length',
    'domain_in_ip', 'server_client_domain', 'qty_dot_directory',
    'qty_hyphen_directory', 'qty_underline_directory', 'qty_slash_directory',
    'qty_questionmark_directory', 'qty_equal_directory', 'qty_at_directory',
    'qty_and_directory', 'qty_exclamation_directory', 'qty_space_directory',
    'qty_tilde_directory', 'qty_comma_directory', 'qty_plus_directory',
    'qty_asterisk_directory', 'qty_hashtag_directory', 'qty_dollar_directory',
    'qty_percent_directory', 'directory_length', 'qty_dot_file',
    'qty_hyphen_file', 'qty_underline_file', 'qty_slash_file',
    'qty_questionmark_file', 'qty_equal_file', 'qty_at_file', 'qty_and_file',
    'qty_exclamation_file', 'qty_space_file', 'qty_tilde_file',
    'qty_comma_file', 'qty_plus_file', 'qty_asterisk_file', 'qty_hashtag_file',
    'qty_dollar_file', 'qty_percent_file', 'file_length', 'qty_dot_params',
    'qty_hyphen_params', 'qty_underline_params', 'qty_slash_params',
    'qty_questionmark_params', 'qty_equal_params', 'qty_at_params',
    'qty_and_params', 'qty_exclamation_params', 'qty_space_params',
    'qty_tilde_params', 'qty_comma_params', 'qty_plus_params',
    'qty_asterisk_params', 'qty_hashtag_params', 'qty_dollar_params',
    'qty_percent_params', 'params_length', 'tld_present_params',
    'qty_params', 'email_in_url', 'time_response', 'domain_spf', 'asn_ip',
    'time_domain_activation', 'time_domain_expiration', 'qty_ip_resolved',
    'qty_nameservers', 'qty_mx_servers', 'ttl_hostname', 'tls_ssl_certificate',
    'qty_redirects', 'url_google_index', 'domain_google_index', 'url_shortened'
]

def _count_chars(text, component):
    """Count occurrences of different characters in the given text component."""
    if not text:
        text = ""
    chars = {
        'dot': '.', 'hyphen': '-', 'underline': '_', 'slash': '/',
        'questionmark': '?', 'equal': '=', 'at': '@', 'and': '&',
        'exclamation': '!', 'space': ' ', 'tilde': '~', 'comma': ',',
        'plus': '+', 'asterisk': '*', 'hashtag': '#', 'dollar': '$',
        'percent': '%'
    }
    result = {}
    for char_name, char in chars.items():
        result[f'qty_{char_name}_{component}'] = text.count(char)
    if component == 'url':
         # Simple TLD counter - adjust list if needed
         result['qty_tld_url'] = sum(1 for tld in ['.com', '.org', '.net', '.edu', '.gov', '.mil', '.int'] if tld in text.lower())
    return result

def _is_ip(domain):
    """Check if domain is an IP address."""
    ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
    return bool(ip_pattern.match(str(domain))) # Ensure domain is string

def _check_server_client(domain):
    """Check for server/client terminology in domain."""
    if not domain: return 0
    server_client_terms = ['server', 'client', 'host', 'proxy']
    return 1 if any(term in str(domain).lower() for term in server_client_terms) else 0

def _get_response_data(url, timeout=5):
    """Get simple response data (redirects, time)."""
    result = {'time_response': 0.0, 'qty_redirects': 0, 'url_shortened': 0}
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        start_time = time.time()
        # Use HEAD request first for efficiency if only checking redirects
        head_response = requests.head(url, headers=headers, timeout=timeout, allow_redirects=False)
        if head_response.status_code in [301, 302, 303, 307, 308]:
            result['qty_redirects'] = 1 # Indicate at least one redirect
            # Optionally follow redirects to count the chain (adds latency)
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            result['time_response'] = time.time() - start_time
            result['qty_redirects'] = len(response.history) if response.history else 1
        else:
             # If no redirect, do a GET to get response time
             response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
             result['time_response'] = time.time() - start_time
             result['qty_redirects'] = 0

        # Check for URL shorteners in the initial URL or the first redirect
        shortener_domains = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly'] # Add more if needed
        initial_domain = urlparse(url).netloc.lower()
        first_redirect_url = response.history[0].url if response.history else ""
        first_redirect_domain = urlparse(first_redirect_url).netloc.lower()

        result['url_shortened'] = 1 if any(sd in initial_domain for sd in shortener_domains) or \
                                        any(sd in first_redirect_domain for sd in shortener_domains) else 0

    except requests.exceptions.RequestException as e:
        print(f"Network error fetching {url}: {e}")
        # Keep defaults on network errors, but maybe flag it?
        pass
    except Exception as e:
        print(f"Error getting response data for {url}: {e}")
        pass # Keep defaults on other errors
    return result

def extract_features_for_url(url):
    """Extract features needed by the model and scaler."""
    features = {key: 0 for key in SCALER_FEATURE_ORDER} # Initialize with zeros

    if not isinstance(url, str) or not url:
        print("Invalid URL input")
        return features # Return zeroed dict if URL is invalid

    # Ensure URL has a scheme for parsing
    original_url_scheme = urlparse(url).scheme
    if not original_url_scheme:
        url = 'http://' + url # Assume http if none provided

    # Basic URL parsing
    try:
        parsed_url = urlparse(url)
        extracted = tldextract.extract(url)
        # Use registered_domain which combines domain and suffix reliably
        domain = extracted.registered_domain if extracted.registered_domain else extracted.domain
        domain_part = parsed_url.netloc
        path = parsed_url.path
        directory_part = '/'.join(path.split('/')[:-1]) if path and '/' in path else ''
        file_part = path.split('/')[-1] if path and '/' in path else ''
        params_part = parsed_url.query
    except Exception as e:
        print(f"Error parsing URL {url}: {e}")
        return features # Return zeroed dict on parsing error

    # Extract character counts
    features.update(_count_chars(url, 'url'))
    features.update(_count_chars(domain_part, 'domain'))
    features.update(_count_chars(directory_part, 'directory'))
    features.update(_count_chars(file_part, 'file'))
    features.update(_count_chars(params_part, 'params'))

    # Length features
    features['length_url'] = len(url)
    features['domain_length'] = len(domain_part) if domain_part else 0
    features['directory_length'] = len(directory_part)
    features['file_length'] = len(file_part)
    features['params_length'] = len(params_part) if params_part else 0

    # Additional domain features
    features['qty_vowels_domain'] = sum(domain_part.lower().count(v) for v in 'aeiou') if domain_part else 0
    features['domain_in_ip'] = 1 if _is_ip(domain_part) else 0
    features['server_client_domain'] = _check_server_client(domain_part)

    # Params features
    features['tld_present_params'] = 1 if params_part and any(tld in params_part.lower() for tld in ['.com', '.org', '.net', '.edu']) else 0
    features['qty_params'] = len(parse_qs(params_part)) if params_part else 0 # Use imported parse_qs

    # Email features
    features['email_in_url'] = 1 if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', url) else 0

    # --- Simplified Network/External Features ---
    response_data = _get_response_data(url)
    features['time_response'] = response_data.get('time_response', 0.0)
    features['qty_redirects'] = response_data.get('qty_redirects', 0)
    features['url_shortened'] = response_data.get('url_shortened', 0)

    # Placeholder for other features expected by the scaler but not extracted here
    # These defaults are based on common scenarios or Mendeley defaults
    # You might need to refine these if they significantly impact predictions
    placeholder_features = {
        'domain_spf': 0, 'asn_ip': 0, 'time_domain_activation': 365, # Default to 1 year old
        'time_domain_expiration': 365, # Default to 1 year expiry
        'qty_ip_resolved': 1, # Assume 1 IP resolved if parsing works
        'qty_nameservers': 2, 'qty_mx_servers': 1, 'ttl_hostname': 3600, # Common default TTL
        'tls_ssl_certificate': 1 if parsed_url.scheme == 'https' else 0, # Check scheme
        'url_google_index': 0, 'domain_google_index': 0 # Assume not indexed by default
    }
    # Only update features that weren't directly calculated
    for key, value in placeholder_features.items():
        if key not in features or features.get(key) == 0: # Update if 0 or missing
             # Check if the key is intended to be updated based on its type
             if key not in ['domain_in_ip', 'server_client_domain', 'tld_present_params',
                           'email_in_url', 'url_shortened', 'tls_ssl_certificate',
                           'domain_spf', 'url_google_index', 'domain_google_index']:
                  features[key] = value
             elif key == 'tls_ssl_certificate': # Ensure tls_ssl is correctly set based on scheme
                  features[key] = 1 if parsed_url.scheme == 'https' else 0


    # Final check to ensure all keys in SCALER_FEATURE_ORDER exist, default to 0 if missing
    final_features = {key: features.get(key, 0) for key in SCALER_FEATURE_ORDER}

    # Convert values to float, handling potential NaNs or infinities just in case
    for key in final_features:
        try:
            val = float(final_features[key])
            if np.isnan(val) or np.isinf(val):
                final_features[key] = 0.0
            else:
                final_features[key] = val
        except (ValueError, TypeError):
             final_features[key] = 0.0 # Default non-numeric to 0


    return final_features # Return the dictionary with all expected keys