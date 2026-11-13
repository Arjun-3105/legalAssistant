"""
api_manager.py — Dynamic Gemini API Key and Credential Lifecycle Manager.

Handles:
- Dynamic resolution of API keys: User Session State -> Streamlit Secrets -> Environment Variables (.env)
- Live health check and quota exhaustion detection
- Safe Gemini generation wrapper with graceful fallback
- Streamlit sidebar UI component for real-time status and custom key input
"""

import os
import re
from typing import Optional, Tuple, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    st = None
    STREAMLIT_AVAILABLE = False

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False

DEFAULT_MODEL_NAME = "models/gemini-2.0-flash"
FALLBACK_MODEL_NAME = "gemini-1.5-flash"


def get_active_api_key() -> Tuple[Optional[str], str]:
    """
    Resolve the active Gemini API key with strict precedence:
    1. User custom key from Streamlit session state (if provided in UI)
    2. Streamlit Cloud secrets (st.secrets["GEMINI_API_KEY"])
    3. Environment variable (GEMINI_API_KEY)
    
    Returns:
        (api_key, source): source is one of 'user', 'secrets', 'env', or 'none'
    """
    # 1. User custom key in session state
    if STREAMLIT_AVAILABLE and hasattr(st, "session_state"):
        user_key = st.session_state.get("user_gemini_api_key", "").strip()
        if user_key:
            return user_key, "user"

    # 2. Streamlit secrets (used when hosted on Streamlit Cloud)
    if STREAMLIT_AVAILABLE:
        try:
            if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                secret_key = str(st.secrets["GEMINI_API_KEY"]).strip()
                if secret_key:
                    return secret_key, "secrets"
            if hasattr(st, "secrets") and "gemini_api_key" in st.secrets:
                secret_key = str(st.secrets["gemini_api_key"]).strip()
                if secret_key:
                    return secret_key, "secrets"
        except Exception:
            pass

    # 3. Environment variable
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    if env_key:
        return env_key, "env"

    return None, "none"


def is_quota_error(error_str: str) -> bool:
    """Check if an exception string indicates quota exhaustion / credit depletion."""
    err_lower = error_str.lower()
    keywords = [
        "429",
        "resourceexhausted",
        "resource has been exhausted",
        "quota exceeded",
        "rate limit",
        "check your plan and billing",
        "insufficient_quota",
        "daily limit",
        "quota limit",
        "credit",
        "exhausted"
    ]
    return any(k in err_lower for k in keywords)


def is_invalid_key_error(error_str: str) -> bool:
    """Check if an exception string indicates an invalid or unauthorized API key."""
    err_lower = error_str.lower()
    keywords = [
        "api_key_invalid",
        "api key not valid",
        "permission_denied",
        "invalid api key",
        "unauthenticated",
        "403",
        "400"
    ]
    return any(k in err_lower for k in keywords)


def validate_api_key(api_key: str) -> Tuple[bool, str, str]:
    """
    Test an API key with a fast, lightweight call.
    
    Returns:
        (is_valid: bool, message: str, status_code: str)
        status_code: 'SUCCESS' | 'QUOTA_EXHAUSTED' | 'INVALID_KEY' | 'ERROR'
    """
    if not GENAI_AVAILABLE:
        return False, "Google Generative AI package is not installed.", "ERROR"

    clean_key = (api_key or "").strip()
    if not clean_key:
        return False, "API key cannot be empty.", "INVALID_KEY"

    try:
        genai.configure(api_key=clean_key)
        # Try primary model or fallback
        model = None
        for m_name in [DEFAULT_MODEL_NAME, FALLBACK_MODEL_NAME, "models/gemini-1.5-flash"]:
            try:
                model = genai.GenerativeModel(m_name)
                break
            except Exception:
                continue

        if not model:
            model = genai.GenerativeModel(DEFAULT_MODEL_NAME)

        # Send minimal prompt to test creds
        resp = model.generate_content("Ping", generation_config={"max_output_tokens": 5})
        if resp and resp.text:
            return True, "API Key is valid and operational! ✅", "SUCCESS"
        return True, "API Key connected successfully.", "SUCCESS"

    except Exception as e:
        err_msg = str(e)
        if is_quota_error(err_msg):
            return False, "Quota exhausted / out of credits on this API key. 🔴", "QUOTA_EXHAUSTED"
        if is_invalid_key_error(err_msg):
            return False, "Invalid API key or unauthorized access. ❌", "INVALID_KEY"
        return False, f"API test failed: {err_msg}", "ERROR"


def get_gemini_model(model_name: str = DEFAULT_MODEL_NAME) -> Optional[Any]:
    """
    Configure and instantiate the Gemini GenerativeModel using the currently active key.
    """
    if not GENAI_AVAILABLE:
        return None

    api_key, source = get_active_api_key()
    if not api_key:
        return None

    try:
        genai.configure(api_key=api_key)
        try:
            return genai.GenerativeModel(model_name)
        except Exception:
            return genai.GenerativeModel(FALLBACK_MODEL_NAME)
    except Exception as e:
        print(f"[APIManager] Error initializing model: {e}")
        return None


def call_gemini_safely(
    model: Optional[Any],
    prompt: str,
    generation_config: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Execute model.generate_content with full error interception.
    
    Returns:
        (text, status, error_detail)
        status: 'SUCCESS' | 'QUOTA_EXHAUSTED' | 'INVALID_KEY' | 'NO_KEY' | 'ERROR'
    """
    if model is None:
        api_key, source = get_active_api_key()
        if not api_key:
            return None, "NO_KEY", "No Gemini API key is configured."
        model = get_gemini_model()
        if model is None:
            return None, "NO_KEY", "Failed to initialize Gemini model."

    try:
        kwargs = {}
        if generation_config:
            kwargs["generation_config"] = generation_config
        response = model.generate_content(prompt, **kwargs)
        
        # If call was successful and session state exists, clear quota exhausted flag
        if STREAMLIT_AVAILABLE and hasattr(st, "session_state"):
            st.session_state["api_quota_exhausted"] = False
            
        return response.text.strip(), "SUCCESS", None

    except Exception as e:
        err_msg = str(e)
        status = "ERROR"
        if is_quota_error(err_msg):
            status = "QUOTA_EXHAUSTED"
            if STREAMLIT_AVAILABLE and hasattr(st, "session_state"):
                st.session_state["api_quota_exhausted"] = True
                st.session_state["last_api_error"] = err_msg
        elif is_invalid_key_error(err_msg):
            status = "INVALID_KEY"
            if STREAMLIT_AVAILABLE and hasattr(st, "session_state"):
                st.session_state["api_key_invalid"] = True
                st.session_state["last_api_error"] = err_msg
        else:
            if STREAMLIT_AVAILABLE and hasattr(st, "session_state"):
                st.session_state["last_api_error"] = err_msg

        return None, status, err_msg


def get_api_status() -> Dict[str, Any]:
    """Get the current API operational status."""
    api_key, source = get_active_api_key()
    quota_exhausted = False
    key_invalid = False
    last_error = ""

    if STREAMLIT_AVAILABLE and hasattr(st, "session_state"):
        quota_exhausted = st.session_state.get("api_quota_exhausted", False)
        key_invalid = st.session_state.get("api_key_invalid", False)
        last_error = st.session_state.get("last_api_error", "")

    if not api_key:
        return {
            "configured": False,
            "source": "none",
            "is_exhausted": False,
            "is_invalid": False,
            "status_text": "No API Key configured",
            "badge": "⚠️ Not Configured",
            "badge_color": "orange"
        }
    elif quota_exhausted:
        return {
            "configured": True,
            "source": source,
            "is_exhausted": True,
            "is_invalid": False,
            "status_text": f"Quota / Credits Exhausted ({source.capitalize()} Key)",
            "badge": "🔴 Quota Exhausted",
            "badge_color": "red",
            "last_error": last_error
        }
    elif key_invalid:
        return {
            "configured": True,
            "source": source,
            "is_exhausted": False,
            "is_invalid": True,
            "status_text": f"Invalid Key or Permission Denied ({source.capitalize()} Key)",
            "badge": "❌ Invalid Key",
            "badge_color": "red",
            "last_error": last_error
        }
    else:
        label = "Custom Key" if source == "user" else "Backend Key"
        return {
            "configured": True,
            "source": source,
            "is_exhausted": False,
            "is_invalid": False,
            "status_text": f"Operational ({label})",
            "badge": f"🟢 Active ({label})",
            "badge_color": "green"
        }


def render_api_sidebar():
    """
    Render a clean, reactive API Key configuration widget in the Streamlit sidebar.
    Allows users to see real-time backend API status and plug in their personal Gemini key.
    """
    if not STREAMLIT_AVAILABLE:
        return

    with st.sidebar:
        st.markdown("---")
        st.subheader("🔑 Gemini API Credentials")
        
        status = get_api_status()
        
        # Status Alert Banner
        if status["is_exhausted"]:
            st.error(
                "🚨 **API Quota Exhausted!**\n\n"
                "The backend API has run out of credits or hit its rate limit. "
                "Add your personal Gemini API key below to continue without interruptions."
            )
        elif status["is_invalid"]:
            st.error(
                "❌ **API Key Error!**\n\n"
                "The current key is invalid or unauthorized. Please verify your key below."
            )
        elif not status["configured"]:
            st.warning(
                "⚠️ **No API Key Configured!**\n\n"
                "Please add a Gemini API key below to enable agent synthesis and RAG refinement."
            )
        else:
            st.success(f"{status['badge']}")

        # Key Input
        current_user_key = st.session_state.get("user_gemini_api_key", "")
        
        user_input_key = st.text_input(
            "Your Gemini API Key:",
            value=current_user_key,
            type="password",
            placeholder="AIzaSy...",
            help="Your key is only used in this browser session and is never persisted to disk or logged."
        )

        col_save, col_reset = st.columns(2)
        
        with col_save:
            if st.button("💾 Apply & Test", use_container_width=True):
                clean_k = user_input_key.strip()
                if not clean_k:
                    st.error("Please enter a key.")
                else:
                    with st.spinner("Testing API key..."):
                        is_valid, msg, code = validate_api_key(clean_k)
                    if is_valid:
                        st.session_state["user_gemini_api_key"] = clean_k
                        st.session_state["api_quota_exhausted"] = False
                        st.session_state["api_key_invalid"] = False
                        st.success("✅ Valid! Active for this session.")
                        st.rerun()
                    else:
                        st.error(msg)
                        
        with col_reset:
            if st.button("🔄 Use Default", use_container_width=True):
                if "user_gemini_api_key" in st.session_state:
                    del st.session_state["user_gemini_api_key"]
                st.session_state["api_quota_exhausted"] = False
                st.session_state["api_key_invalid"] = False
                st.info("Reset to default backend key.")
                st.rerun()

        # Helpful reference link
        st.caption(
            "👉 Don't have a key? [Get a free Gemini API key](https://aistudio.google.com/app/apikey) "
            "from Google AI Studio in 30 seconds."
        )
