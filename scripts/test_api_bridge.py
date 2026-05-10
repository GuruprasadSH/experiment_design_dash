"""
Sprint 1 verification: push a design config via HTTP, confirm app state.

Usage:
    python scripts/test_api_bridge.py

Exit 0 on pass, exit 1 on failure.
"""

import sys
import time
import subprocess
import requests

# Add repo root so agent/ is importable
sys.path.insert(0, ".")

from agent.app_controller import configure_design, wait_for_state_key

APP_URL   = "http://127.0.0.1:8050"
STARTUP_S = 6   # seconds to wait for the Dash server to become ready


def _wait_for_server(timeout: float = 20.0) -> bool:
    """Poll /agent/state until the server responds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{APP_URL}/agent/state", timeout=2)
            if r.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)
    return False


def main() -> int:
    print("=== Sprint 1 API Bridge Verification ===\n")

    # Start Dash server as a subprocess
    print("Starting Dash app …")
    proc = subprocess.Popen(
        [sys.executable, "DOE/app.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        print(f"Waiting up to {STARTUP_S + 14}s for server …")
        if not _wait_for_server(timeout=STARTUP_S + 14):
            print("FAIL: server did not start within timeout")
            return 1
        print("Server is up.\n")

        # Push a 3-factor fractional factorial config
        factors = [
            {"name": "Temperature", "low": 150, "high": 200},
            {"name": "Pressure",    "low": 10,  "high": 30},
            {"name": "Flow",        "low": 5,   "high": 15},
        ]
        options = {
            "resolution":    4,
            "replicates":    1,
            "blocks":        1,
            "randomize":     True,
            "center_points": 2,
        }
        print(f"Pushing design config: fractional, {len(factors)} factors …")
        resp = configure_design("fractional", factors, options)
        print(f"  configure_design response: {resp}")

        print("Polling /agent/state for 'design_type' …")
        state = wait_for_state_key("design_type", poll_interval=1.0, max_wait=15.0)
        print(f"\nApp state:\n{state}\n")

        assert state.get("design_type") == "fractional", (
            f"Expected design_type='fractional', got {state.get('design_type')!r}"
        )
        print("PASS: design_type == 'fractional'")
        return 0

    except AssertionError as e:
        print(f"FAIL: assertion error — {e}")
        return 1
    except TimeoutError as e:
        print(f"FAIL: timeout — {e}")
        return 1
    except Exception as e:
        print(f"FAIL: unexpected error — {e}")
        return 1
    finally:
        print("\nTerminating Dash app …")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
