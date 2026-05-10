"""
HTTP wrapper for the Dash app's agent endpoints.
Used by test scripts and future CLI tools; agent callbacks call
the Interviewer/Interpreter classes directly (same process).
"""

import requests

BASE_URL = "http://127.0.0.1:8050"
TIMEOUT  = 10


def configure_design(design_type: str, factors: list[dict], options: dict) -> dict:
    """Push a design configuration to the running Dash app."""
    payload = {"design_type": design_type, "factors": factors, "options": options}
    r = requests.post(f"{BASE_URL}/agent/configure", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get_app_state() -> dict:
    """Read current Dash app state including fitted model results."""
    r = requests.get(f"{BASE_URL}/agent/state", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def wait_for_state_key(key: str, poll_interval: float = 1.0, max_wait: float = 30.0) -> dict:
    """Block until the app state contains a given key. Returns the state dict."""
    import time
    elapsed = 0.0
    while elapsed < max_wait:
        state = get_app_state()
        if state.get(key):
            return state
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise TimeoutError(f"App state key '{key}' did not appear within {max_wait}s")
