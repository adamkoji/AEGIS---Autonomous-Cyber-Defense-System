# api/services/feature_extractor.py
import math
import re
import socket
import ssl
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from urllib.parse import urlparse, parse_qs, urljoin

import dns.resolver
import numpy as np
import requests
import tldextract
from bs4 import BeautifulSoup

from api.services.adversarial_guard import analyze_adversarial_payload, HOMOGLYPH_MAP

# Use offline snapshot for tldextract so it never blocks on network inside sandboxed environments
_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

# 111 Mendeley Phishing Dataset Features in exact order expected by tabular_scaler.pkl
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

# 15 Additional DOM, Brand, Homograph & Adversarial Features used by LightGBM & Ensemble
DOM_ADVERSARIAL_FEATURE_ORDER = [
    'url_entropy',
    'domain_entropy',
    'is_punycode',
    'homograph_spoof_detected',
    'brand_typosquat_score',
    'suspicious_tld',
    'suspicious_keyword_count',
    'is_official_trusted_domain',
    'dom_form_count',
    'external_form_action',
    'password_input_count',
    'hidden_input_count',
    'iframe_trap_count',
    'script_obfuscation_score',
    'prompt_injection_score',
]

LIGHTGBM_FEATURE_ORDER = SCALER_FEATURE_ORDER + DOM_ADVERSARIAL_FEATURE_ORDER

# Official verified brand domains
OFFICIAL_BRAND_DOMAINS = {
    "paypal": {"paypal.com", "paypal.me"},
    "microsoft": {"microsoft.com", "live.com", "office.com", "outlook.com", "azure.com", "windows.com"},
    "google": {"google.com", "googleapis.com", "youtube.com", "gmail.com", "google.co.in", "google.co.uk"},
    "apple": {"apple.com", "icloud.com"},
    "amazon": {"amazon.com", "amazon.in", "amazon.co.uk", "aws.amazon.com", "amazonaws.com"},
    "netflix": {"netflix.com"},
    "github": {"github.com", "githubusercontent.com", "github.io"},
    "binance": {"binance.com", "binance.us"},
    "hdfcbank": {"hdfcbank.com"},
    "sbi": {"onlinesbi.sbi", "sbi.co.in"},
    "chase": {"chase.com"},
    "wellsfargo": {"wellsfargo.com"},
    "bankofamerica": {"bankofamerica.com"},
    "facebook": {"facebook.com", "fb.com", "meta.com"},
    "instagram": {"instagram.com"},
    "linkedin": {"linkedin.com"},
    "coinbase": {"coinbase.com"},
    "metamask": {"metamask.io"},
    "openai": {"openai.com", "chatgpt.com"},
    "dropbox": {"dropbox.com"},
    "adobe": {"adobe.com"},
}

ALL_OFFICIAL_DOMAINS = set().union(*OFFICIAL_BRAND_DOMAINS.values()) | {
    "wikipedia.org", "stackoverflow.com", "python.org", "pytorch.org",
    "huggingface.co", "cloudflare.com", "mozilla.org", "w3.org", "mit.edu",
    "stanford.edu", "harvard.edu", "gov.in", "usa.gov", "nih.gov", "npmjs.com",
}

HIGH_RISK_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "zip", "mov", "buzz",
    "icu", "work", "click", "pw", "cc", "loan", "men", "kim", "rest",
    "fit", "cfd", "sbs", "cyou", " monster", "quest", "support",
}

SUSPICIOUS_KEYWORDS = [
    "login", "verify", "update-account", "secure-signin", "wallet-connect",
    "auth", "password-reset", "signin", "update", "secure", "account",
    "confirm", "billing", "webscr", "unlock", "suspend", "credential",
    "validate", "recover", "security-alert", "apple-id", "banking",
]

LEETSPEAK_TABLE = str.maketrans({
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "@": "a",
    "$": "s",
})


def calculate_shannon_entropy(text: str) -> float:
    """Computes Shannon entropy (bits per character) to identify DGA / random strings."""
    if not text:
        return 0.0
    length = len(text)
    freqs = {}
    for ch in text:
        freqs[ch] = freqs.get(ch, 0) + 1
    entropy = -sum((count / length) * math.log2(count / length) for count in freqs.values())
    return round(entropy, 4)


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Computes exact Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def detect_brand_and_homograph_signals(url: str, domain_part: str, registered_domain: str, path_and_query: str) -> dict:
    """
    Detects IDN Punycode, Unicode Homoglyphs, Leetspeak Brand Typosquatting,
    and Subdomain/Path Brand Spoofing when the registered root domain is NOT the official brand.
    """
    domain_lower = (domain_part or "").lower()
    reg_lower = (registered_domain or "").lower()
    full_lower = (url or "").lower()

    is_official = 1 if reg_lower in ALL_OFFICIAL_DOMAINS else 0
    is_punycode = 1 if "xn--" in domain_lower else 0

    # Check Unicode non-ASCII or confusable characters in domain
    has_non_ascii = any(ord(c) > 127 for c in domain_part)
    has_homoglyph_char = any(c in HOMOGLYPH_MAP for c in domain_part)
    homograph_spoof = 1 if (is_punycode or has_non_ascii or has_homoglyph_char) else 0

    matched_brand = None
    brand_score = 0.0
    spoof_reason = ""

    if not is_official:
        # Normalize leetspeak and homoglyphs for token inspection
        normalized_domain = "".join(HOMOGLYPH_MAP.get(c, c) for c in domain_lower)
        leet_domain = normalized_domain.translate(LEETSPEAK_TABLE)
        # Also try '1' -> 'i' variant (e.g., microsoft / sbi / binance)
        leet_domain_i = normalized_domain.replace("1", "i").replace("0", "o").replace("3", "e").replace("4", "a").replace("5", "s")

        tokens = [t for t in re.split(r"[\.\-_/:?=&]+", domain_lower + "/" + path_and_query.lower()) if len(t) >= 3]
        leet_tokens = [t.translate(LEETSPEAK_TABLE) for t in tokens]

        for brand, official_domains in OFFICIAL_BRAND_DOMAINS.items():
            if reg_lower in official_domains:
                continue

            # 1. Direct brand name in unauthorized domain or subdomain or path
            if brand in domain_lower:
                matched_brand = brand
                brand_score = 0.95
                spoof_reason = f"Brand '{brand}' appears in unauthorized host '{domain_lower}' (official: {sorted(official_domains)[0]})"
                break

            if brand in leet_domain or brand in leet_domain_i:
                matched_brand = brand
                brand_score = 0.96
                homograph_spoof = 1
                spoof_reason = f"Leetspeak/homograph lookalike of '{brand}' in host '{domain_lower}' (official: {sorted(official_domains)[0]})"
                break

            if brand in path_and_query.lower():
                matched_brand = brand
                brand_score = max(brand_score, 0.85)
                spoof_reason = f"Brand '{brand}' targeted in URL path on untrusted host '{domain_lower}'"

            # 2. Levenshtein distance <= 1 on domain tokens (e.g. paypa1, g00gle, micros0ft, amaz0n, netf1ix)
            if len(brand) >= 5:
                for raw_tok, l_tok in zip(tokens, leet_tokens):
                    if abs(len(raw_tok) - len(brand)) <= 1:
                        dist_raw = _levenshtein_distance(raw_tok, brand)
                        dist_leet = _levenshtein_distance(l_tok, brand)
                        if (0 < dist_raw <= 1) or (dist_leet == 0 and raw_tok != brand) or (0 < dist_leet <= 1 and len(brand) >= 6):
                            matched_brand = brand
                            brand_score = max(brand_score, 0.92)
                            homograph_spoof = 1
                            spoof_reason = f"Typosquatted token '{raw_tok}' mimics official brand '{brand}' (edit distance={min(dist_raw, dist_leet)})"
                            break
                if brand_score >= 0.92:
                    break

    return {
        "is_official_trusted_domain": is_official,
        "is_punycode": is_punycode,
        "homograph_spoof_detected": homograph_spoof,
        "brand_typosquat_score": round(float(brand_score), 4),
        "impersonated_brand": matched_brand,
        "brand_spoof_detail": spoof_reason,
    }


def _get_registered_domain(extracted) -> str:
    return getattr(extracted, "top_domain_under_public_suffix", "") or extracted.domain or ""


def extract_dom_features(html_content: str, base_url: str = "") -> dict:
    """
    Live BeautifulSoup4 DOM Structural & Script Scanner.
    Analyzes HTML forms, external form actions, password inputs, hidden inputs,
    iframe traps, external script ratios, JS obfuscation, right-click disabling,
    and favicon domain mismatches.
    """
    default_dom = {
        "dom_analyzed": False,
        "dom_form_count": 0,
        "external_form_action": 0,
        "suspicious_form_action": 0,
        "password_input_count": 0,
        "hidden_input_count": 0,
        "hidden_element_count": 0,
        "iframe_trap_count": 0,
        "total_script_count": 0,
        "external_script_count": 0,
        "external_script_ratio": 0.0,
        "obfuscated_script_count": 0,
        "script_obfuscation_score": 0.0,
        "right_click_disabled": 0,
        "mismatched_favicon": 0,
        "meta_refresh_redirect": 0,
        "external_link_ratio": 0.0,
        "total_links": 0,
        "inline_event_handlers": 0,
        "prompt_injection_score": 0.0,
    }
    if not html_content or not isinstance(html_content, str) or not html_content.strip():
        return default_dom

    try:
        soup = BeautifulSoup(html_content, "html.parser")
        base_domain = _get_registered_domain(_TLD_EXTRACTOR(base_url)).lower() if base_url else ""

        forms = soup.find_all("form")
        dom_form_count = len(forms)
        external_form_action = 0
        suspicious_form_action = 0

        for form in forms:
            action = (form.get("action") or "").strip()
            if not action or action.startswith("#") or action.lower().startswith("javascript:") or action.lower().startswith("mailto:"):
                suspicious_form_action = 1
            elif action.startswith("http://") or action.startswith("https://") or action.startswith("//"):
                action_domain = _get_registered_domain(_TLD_EXTRACTOR(action)).lower()
                if base_domain and action_domain and action_domain != base_domain:
                    external_form_action = 1
                    suspicious_form_action = 1
                elif not base_domain and action_domain:
                    external_form_action = 1

        inputs = soup.find_all("input")
        password_input_count = sum(1 for inp in inputs if (inp.get("type") or "").lower() == "password")
        hidden_input_count = sum(1 for inp in inputs if (inp.get("type") or "").lower() == "hidden")

        # Hidden DOM containers
        hidden_element_count = 0
        inline_event_handlers = 0
        for tag in soup.find_all(True):
            style = (tag.get("style") or "").lower().replace(" ", "")
            if any(h in style for h in ("display:none", "visibility:hidden", "opacity:0", "font-size:0")) or tag.has_attr("hidden"):
                hidden_element_count += 1
            for attr in tag.attrs:
                if isinstance(attr, str) and attr.lower().startswith("on"):
                    inline_event_handlers += 1

        # Iframes
        iframes = soup.find_all("iframe")
        iframe_trap_count = len(iframes)

        # Scripts & Obfuscation
        scripts = soup.find_all("script")
        total_script_count = len(scripts)
        external_script_count = 0
        obfuscated_script_count = 0

        for scr in scripts:
            src = (scr.get("src") or "").strip()
            if src:
                if src.startswith("http://") or src.startswith("https://") or src.startswith("//"):
                    src_domain = _get_registered_domain(_TLD_EXTRACTOR(src)).lower()
                    if not base_domain or (src_domain and src_domain != base_domain):
                        external_script_count += 1
            script_body = scr.string or scr.get_text() or ""
            if re.search(r"(?:eval\s*\(|atob\s*\(|unescape\s*\(|String\.fromCharCode|\\x[0-9a-fA-F]{2}|document\.write\s*\()", script_body):
                obfuscated_script_count += 1

        external_script_ratio = round(external_script_count / max(total_script_count, 1), 4) if total_script_count > 0 else 0.0

        # Disabled right-click check
        raw_lower = html_content.lower()
        right_click_disabled = 1 if (
            "oncontextmenu" in raw_lower and ("return false" in raw_lower or "preventdefault" in raw_lower)
        ) else 0

        # Favicon mismatch check
        mismatched_favicon = 0
        for link in soup.find_all("link"):
            rel_list = [r.lower() for r in (link.get("rel") or [])] if isinstance(link.get("rel"), list) else [str(link.get("rel") or "").lower()]
            if any("icon" in r for r in rel_list):
                href = (link.get("href") or "").strip()
                if href.startswith("http://") or href.startswith("https://"):
                    fav_domain = _get_registered_domain(_TLD_EXTRACTOR(href)).lower()
                    if base_domain and fav_domain and fav_domain != base_domain:
                        mismatched_favicon = 1

        # Meta refresh redirect
        meta_refresh_redirect = 0
        for meta in soup.find_all("meta"):
            if (meta.get("http-equiv") or "").lower() == "refresh":
                meta_refresh_redirect = 1

        # External anchor link ratio
        anchors = soup.find_all("a", href=True)
        total_links = len(anchors)
        ext_links = 0
        for a in anchors:
            href = a["href"].strip()
            if href.startswith("http://") or href.startswith("https://"):
                a_dom = _get_registered_domain(_TLD_EXTRACTOR(href)).lower()
                if base_domain and a_dom and a_dom != base_domain:
                    ext_links += 1
        external_link_ratio = round(ext_links / max(total_links, 1), 4) if total_links > 0 else 0.0

        # Compute script obfuscation composite score (0.0 to 1.0)
        obf_score = 0.0
        if obfuscated_script_count > 0:
            obf_score += min(0.65, 0.35 * obfuscated_script_count)
        if right_click_disabled:
            obf_score += 0.20
        if mismatched_favicon:
            obf_score += 0.15
        if external_script_ratio > 0.6:
            obf_score += 0.15
        if iframe_trap_count > 0:
            obf_score += 0.15
        obf_score = min(1.0, round(obf_score, 4))

        # Check adversarial prompt injection markers inside DOM
        adv_result = analyze_adversarial_payload(html_content)
        prompt_injection_score = float(adv_result.get("risk_score", 0.0))

        return {
            "dom_analyzed": True,
            "dom_form_count": dom_form_count,
            "external_form_action": external_form_action,
            "suspicious_form_action": suspicious_form_action,
            "password_input_count": password_input_count,
            "hidden_input_count": hidden_input_count,
            "hidden_element_count": hidden_element_count,
            "iframe_trap_count": iframe_trap_count,
            "total_script_count": total_script_count,
            "external_script_count": external_script_count,
            "external_script_ratio": external_script_ratio,
            "obfuscated_script_count": obfuscated_script_count,
            "script_obfuscation_score": obf_score,
            "right_click_disabled": right_click_disabled,
            "mismatched_favicon": mismatched_favicon,
            "meta_refresh_redirect": meta_refresh_redirect,
            "external_link_ratio": external_link_ratio,
            "total_links": total_links,
            "inline_event_handlers": inline_event_handlers,
            "prompt_injection_score": prompt_injection_score,
        }
    except Exception:
        return default_dom


def _count_chars(text: str, component: str) -> dict:
    """Counts occurrences of 17 special characters in a URL component."""
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
        result['qty_tld_url'] = sum(
            1 for tld in ['.com', '.org', '.net', '.edu', '.gov', '.mil', '.int', '.xyz', '.top', '.ru', '.cn', '.io', '.co']
            if tld in text.lower()
        )
    return result


def _is_ip(domain: str) -> bool:
    """Checks if domain host is an IPv4 address."""
    clean_host = str(domain).split(":")[0]
    ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
    return bool(ip_pattern.match(clean_host))


def _check_server_client(domain: str) -> int:
    """Checks for server/client terminology in domain."""
    if not domain:
        return 0
    server_client_terms = ['server', 'client', 'host', 'proxy', 'admin']
    return 1 if any(term in str(domain).lower() for term in server_client_terms) else 0


def _probe_dns(hostname: str, timeout: float = 1.2) -> dict:
    """Fast DNS resolution probe for IP count, NS, MX, TTL, and SPF."""
    dns_info = {
        "qty_ip_resolved": 1,
        "qty_nameservers": 2,
        "qty_mx_servers": 1,
        "ttl_hostname": 3600,
        "domain_spf": 0,
        "dns_live_resolved": False,
    }
    if not hostname or _is_ip(hostname):
        return dns_info

    try:
        resolver = dns.resolver.Resolver()
        resolver.timeout = timeout
        resolver.lifetime = timeout

        # A records
        answers = resolver.resolve(hostname, "A")
        dns_info["qty_ip_resolved"] = len(answers)
        dns_info["ttl_hostname"] = int(answers.rrset.ttl) if answers.rrset else 3600
        dns_info["dns_live_resolved"] = True

        # TXT / SPF check
        try:
            txt_ans = resolver.resolve(hostname, "TXT")
            for rdata in txt_ans:
                if "v=spf1" in str(rdata).lower():
                    dns_info["domain_spf"] = 1
                    break
        except Exception:
            pass
    except Exception:
        # Fallback to socket.getaddrinfo if dnspython fails
        try:
            socket.setdefaulttimeout(timeout)
            infos = socket.getaddrinfo(hostname, None)
            ips = {item[4][0] for item in infos if item and len(item) > 4}
            if ips:
                dns_info["qty_ip_resolved"] = len(ips)
                dns_info["dns_live_resolved"] = True
        except Exception:
            pass

    return dns_info


def _probe_ssl(hostname: str, is_https: bool, timeout: float = 1.2) -> dict:
    """Fast SSL/TLS certificate inspection probe."""
    ssl_info = {
        "tls_ssl_certificate": 1 if is_https else 0,
        "ssl_issuer": "Trusted CA" if is_https else "None",
        "ssl_valid_days_remaining": 365 if is_https else 0,
    }
    if not is_https or not hostname or _is_ip(hostname):
        return ssl_info

    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                if cert:
                    ssl_info["tls_ssl_certificate"] = 1
                    issuer = dict(x[0] for x in cert.get("issuer", []))
                    ssl_info["ssl_issuer"] = issuer.get("organizationName") or issuer.get("commonName") or "Verified CA"
    except Exception:
        pass

    return ssl_info


def _probe_http(url: str, timeout: float = 1.5) -> dict:
    """Fast HTTP latency, redirect chain, URL shortener, and live HTML fetcher."""
    shortener_domains = {'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'buff.ly', 'rb.gy', 'tiny.cc'}
    initial_domain = urlparse(url).netloc.lower()
    is_shortened = 1 if any(sd in initial_domain for sd in shortener_domains) else 0

    http_info = {
        "time_response": 0.12,
        "qty_redirects": 0,
        "url_shortened": is_shortened,
        "fetched_html": "",
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AEGIS-ThreatScanner/2.0"}
    try:
        start_time = time.time()
        resp = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        http_info["time_response"] = round(time.time() - start_time, 4)
        http_info["qty_redirects"] = len(resp.history) if resp.history else 0
        if resp.history:
            first_redir = urlparse(resp.history[0].url).netloc.lower()
            if any(sd in first_redir for sd in shortener_domains):
                http_info["url_shortened"] = 1
        if "text/html" in (resp.headers.get("Content-Type") or "").lower():
            http_info["fetched_html"] = resp.text[:100000]
    except Exception:
        pass

    return http_info


def extract_all_features(url: str, html_content: str = None, deep_scan: bool = False) -> dict:
    """
    Extracts all 111 Mendeley URL/Network features + 18 BeautifulSoup DOM features
    + 12 Adversarial, Brand Impersonation, Homograph & Entropy signals.
    Uses concurrent ThreadPoolExecutor probes bounded by a strict 2.5s ceiling.
    """
    mendeley_features = {key: 0.0 for key in SCALER_FEATURE_ORDER}

    if not isinstance(url, str) or not url.strip():
        return {
            "mendeley_features": mendeley_features,
            "lightgbm_features": {key: 0.0 for key in LIGHTGBM_FEATURE_ORDER},
            "dom_features": extract_dom_features(""),
            "brand_signals": {},
            "network_details": {},
            "feature_breakdown": {},
        }

    raw_url = url.strip()
    if not urlparse(raw_url).scheme:
        working_url = "http://" + raw_url
    else:
        working_url = raw_url

    parsed_url = urlparse(working_url)
    extracted = _TLD_EXTRACTOR(working_url)
    registered_domain = getattr(extracted, "top_domain_under_public_suffix", "") or extracted.domain or ""
    tld_suffix = (extracted.suffix or "").lower()
    domain_part = parsed_url.netloc.split(":")[0] if parsed_url.netloc else ""
    path = parsed_url.path or ""

    # Normalize trailing slash on bare root domain so length_url and qty_slash_url match training distribution
    if path == "/" and not parsed_url.query and not parsed_url.fragment:
        working_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        parsed_url = urlparse(working_url)
        path = ""

    directory_part = "/".join(path.split("/")[:-1]) if path and "/" in path else ""
    file_part = path.split("/")[-1] if path and "/" in path else ""
    params_part = parsed_url.query or ""

    # 1. Character count features across 5 URL components (86 features)
    mendeley_features.update(_count_chars(working_url, 'url'))
    mendeley_features.update(_count_chars(domain_part, 'domain'))
    mendeley_features.update(_count_chars(directory_part, 'directory'))
    mendeley_features.update(_count_chars(file_part, 'file'))
    mendeley_features.update(_count_chars(params_part, 'params'))

    # 2. Component lengths & structural flags (11 features)
    mendeley_features['length_url'] = float(len(working_url))
    mendeley_features['domain_length'] = float(len(domain_part))
    mendeley_features['directory_length'] = float(len(directory_part))
    mendeley_features['file_length'] = float(len(file_part))
    mendeley_features['params_length'] = float(len(params_part))

    mendeley_features['qty_vowels_domain'] = float(sum(domain_part.lower().count(v) for v in 'aeiou'))
    mendeley_features['domain_in_ip'] = 1.0 if _is_ip(domain_part) else 0.0
    mendeley_features['server_client_domain'] = float(_check_server_client(domain_part))
    mendeley_features['tld_present_params'] = 1.0 if params_part and any(
        tld in params_part.lower() for tld in ['.com', '.org', '.net', '.edu', '.ru', '.xyz']
    ) else 0.0
    mendeley_features['qty_params'] = float(len(parse_qs(params_part))) if params_part else 0.0
    mendeley_features['email_in_url'] = 1.0 if re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', working_url) else 0.0

    # 3. Brand, Homograph, Entropy & Keyword signals
    brand_signals = detect_brand_and_homograph_signals(
        working_url, domain_part, registered_domain, f"{path}?{params_part}"
    )
    url_entropy = calculate_shannon_entropy(working_url)
    domain_entropy = calculate_shannon_entropy(domain_part)
    suspicious_tld = 1 if tld_suffix in HIGH_RISK_TLDS else 0
    lower_full = working_url.lower()
    suspicious_keyword_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in lower_full)

    # 4. Concurrent Network / DNS / SSL Probes (bounded to 2.5s total)
    is_https = parsed_url.scheme.lower() == "https"
    dns_data = {
        "qty_ip_resolved": 1,
        "qty_nameservers": 2,
        "qty_mx_servers": 1,
        "ttl_hostname": 3600,
        "domain_spf": 0,
        "dns_live_resolved": False,
    }
    ssl_data = {
        "tls_ssl_certificate": 1 if is_https else 0,
        "ssl_issuer": "Trusted CA" if is_https else "None",
        "ssl_valid_days_remaining": 365 if is_https else 0,
    }
    http_data = {
        "time_response": 0.0,
        "qty_redirects": 0,
        "url_shortened": 1 if any(sd in domain_part.lower() for sd in ('bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly')) else 0,
        "fetched_html": "",
    }

    if deep_scan and domain_part and not _is_ip(domain_part):
        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                fut_dns = executor.submit(_probe_dns, domain_part, 1.0)
                fut_ssl = executor.submit(_probe_ssl, domain_part, is_https, 1.0)
                fut_http = executor.submit(_probe_http, working_url, 1.2)
                try:
                    dns_data = fut_dns.result(timeout=1.5)
                except FuturesTimeoutError:
                    pass
                try:
                    ssl_data = fut_ssl.result(timeout=1.5)
                except FuturesTimeoutError:
                    pass
                try:
                    http_data = fut_http.result(timeout=1.8)
                except FuturesTimeoutError:
                    pass
        except Exception:
            pass

    # Populate Mendeley network features (preserving scaler-compatible defaults)
    mendeley_features['time_response'] = float(http_data.get('time_response', 0.0))
    mendeley_features['domain_spf'] = float(dns_data.get('domain_spf', 0))
    mendeley_features['asn_ip'] = 0.0
    mendeley_features['time_domain_activation'] = 365.0
    mendeley_features['time_domain_expiration'] = 365.0
    mendeley_features['qty_ip_resolved'] = float(dns_data.get('qty_ip_resolved', 1))
    mendeley_features['qty_nameservers'] = float(dns_data.get('qty_nameservers', 2))
    mendeley_features['qty_mx_servers'] = float(dns_data.get('qty_mx_servers', 1))
    mendeley_features['ttl_hostname'] = float(dns_data.get('ttl_hostname', 3600))
    mendeley_features['tls_ssl_certificate'] = float(ssl_data.get('tls_ssl_certificate', 1 if is_https else 0))
    mendeley_features['qty_redirects'] = float(http_data.get('qty_redirects', 0))
    mendeley_features['url_google_index'] = 0.0
    mendeley_features['domain_google_index'] = 0.0
    mendeley_features['url_shortened'] = float(http_data.get('url_shortened', 0))

    # Clean NaNs/Infs
    for k, v in mendeley_features.items():
        try:
            fval = float(v)
            mendeley_features[k] = 0.0 if (np.isnan(fval) or np.isinf(fval)) else fval
        except (ValueError, TypeError):
            mendeley_features[k] = 0.0

    # 5. Live BeautifulSoup4 DOM Structural & Script Scanner
    effective_html = html_content if html_content is not None else http_data.get("fetched_html", "")
    dom_features = extract_dom_features(effective_html, working_url)

    #Also check URL itself for adversarial payloads (SQLi, XSS, SSRF, encoded payloads in query)
    url_adv = analyze_adversarial_payload(working_url)
    combined_prompt_adv_score = max(
        float(dom_features.get("prompt_injection_score", 0.0)),
        float(url_adv.get("risk_score", 0.0)),
    )
    dom_features["prompt_injection_score"] = round(combined_prompt_adv_score, 4)

    # 6. Assemble 126-feature dictionary for LightGBM
    lightgbm_features = dict(mendeley_features)
    lightgbm_features.update({
        "url_entropy": float(url_entropy),
        "domain_entropy": float(domain_entropy),
        "is_punycode": float(brand_signals["is_punycode"]),
        "homograph_spoof_detected": float(brand_signals["homograph_spoof_detected"]),
        "brand_typosquat_score": float(brand_signals["brand_typosquat_score"]),
        "suspicious_tld": float(suspicious_tld),
        "suspicious_keyword_count": float(suspicious_keyword_count),
        "is_official_trusted_domain": float(brand_signals["is_official_trusted_domain"]),
        "dom_form_count": float(dom_features["dom_form_count"]),
        "external_form_action": float(dom_features["external_form_action"]),
        "password_input_count": float(dom_features["password_input_count"]),
        "hidden_input_count": float(dom_features["hidden_input_count"]),
        "iframe_trap_count": float(dom_features["iframe_trap_count"]),
        "script_obfuscation_score": float(dom_features["script_obfuscation_score"]),
        "prompt_injection_score": float(dom_features["prompt_injection_score"]),
    })

    # 7. Grouped Feature Breakdown across 4 categories (101+ features)
    lexical_keys = SCALER_FEATURE_ORDER[:97]
    network_keys = SCALER_FEATURE_ORDER[97:]
    adversarial_brand_summary = {
        "url_entropy": url_entropy,
        "domain_entropy": domain_entropy,
        "is_punycode": brand_signals["is_punycode"],
        "homograph_spoof_detected": brand_signals["homograph_spoof_detected"],
        "brand_typosquat_score": brand_signals["brand_typosquat_score"],
        "impersonated_brand": brand_signals["impersonated_brand"],
        "suspicious_tld": suspicious_tld,
        "tld_suffix": tld_suffix,
        "suspicious_keyword_count": suspicious_keyword_count,
        "is_official_trusted_domain": brand_signals["is_official_trusted_domain"],
        "url_adversarial_score": url_adv.get("risk_score", 0.0),
        "dom_prompt_injection_score": dom_features["prompt_injection_score"],
    }

    feature_breakdown = {
        "total_features_analyzed": len(SCALER_FEATURE_ORDER) + len(dom_features) + len(adversarial_brand_summary),
        "categories": {
            "lexical": {
                "label": "Lexical & URL Token Structure",
                "feature_count": 56,
                "full_mendeley_lexical_count": len(lexical_keys),
                "highlights": {
                    "length_url": mendeley_features["length_url"],
                    "domain_length": mendeley_features["domain_length"],
                    "qty_dot_url": mendeley_features["qty_dot_url"],
                    "qty_hyphen_domain": mendeley_features["qty_hyphen_domain"],
                    "qty_slash_url": mendeley_features["qty_slash_url"],
                    "qty_at_url": mendeley_features["qty_at_url"],
                    "domain_in_ip": mendeley_features["domain_in_ip"],
                    "email_in_url": mendeley_features["email_in_url"],
                    "qty_params": mendeley_features["qty_params"],
                },
            },
            "network_dns": {
                "label": "Network, DNS & SSL/TLS Telemetry",
                "feature_count": 15,
                "highlights": {
                    "tls_ssl_certificate": mendeley_features["tls_ssl_certificate"],
                    "ssl_issuer": ssl_data["ssl_issuer"],
                    "qty_ip_resolved": mendeley_features["qty_ip_resolved"],
                    "qty_nameservers": mendeley_features["qty_nameservers"],
                    "qty_mx_servers": mendeley_features["qty_mx_servers"],
                    "ttl_hostname": mendeley_features["ttl_hostname"],
                    "domain_spf": mendeley_features["domain_spf"],
                    "qty_redirects": mendeley_features["qty_redirects"],
                    "time_response": mendeley_features["time_response"],
                    "url_shortened": mendeley_features["url_shortened"],
                },
            },
            "dom_structure": {
                "label": "BeautifulSoup4 Live DOM & Script AST",
                "feature_count": 18,
                "highlights": dom_features,
            },
            "adversarial_brand": {
                "label": "Adversarial AI, Homograph & Brand Guardrails",
                "feature_count": 12,
                "highlights": adversarial_brand_summary,
            },
        },
    }

    return {
        "normalized_url": working_url,
        "domain": domain_part,
        "registered_domain": registered_domain,
        "tld_suffix": tld_suffix,
        "mendeley_features": mendeley_features,
        "lightgbm_features": lightgbm_features,
        "dom_features": dom_features,
        "brand_signals": brand_signals,
        "url_adversarial": url_adv,
        "network_details": {**dns_data, **ssl_data, **{k: v for k, v in http_data.items() if k != "fetched_html"}},
        "feature_breakdown": feature_breakdown,
    }


def extract_features_for_url(url: str) -> dict:
    """Backward-compatible helper returning the 111 Mendeley features dict."""
    res = extract_all_features(url, html_content=None, deep_scan=False)
    return res["mendeley_features"]
