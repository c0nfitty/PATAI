"""
session.py
----------
Conversation mode session management.

Sessions are stored as JSON files on the IFS under SESSIONS_DIR.
Each file is named {uuid}.json and contains the search history
for one browser session.

Lifecycle:
  - Created when the user enables conversation mode
  - Updated on each search (history appended, last_active refreshed)
  - Expired after SESSION_TTL_SECS of inactivity
  - Deleted explicitly when user clicks "Forget session"
  - Swept by a background thread every SESSION_SWEEP_SECS
"""

import json
import time
import threading
import logging

from config import SESSIONS_DIR, SESSION_MAX_BYTES, SESSION_TTL_SECS, SESSION_SWEEP_SECS

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Internal helpers
# ------------------------------------------------------------------ #

def _path(sid: str):
    """Return the IFS path for a session file."""
    return SESSIONS_DIR / f"{sid}.json"


def _is_expired(data: dict) -> bool:
    """Return True if the session has been inactive too long."""
    return time.time() - data.get("last_active", 0) > SESSION_TTL_SECS


def _trim(data: dict) -> dict:
    """
    Drop the oldest history entries until the session is under
    SESSION_MAX_BYTES. Prevents runaway growth on long sessions.
    """
    raw = json.dumps(data)
    while len(raw.encode()) > SESSION_MAX_BYTES and len(data["history"]) > 1:
        data["history"].pop(0)
        raw = json.dumps(data)
    return data


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #

def load(sid: str) -> dict:
    """
    Load a session from disk.
    Returns an empty session dict if the file doesn't exist or has expired.
    """
    p = _path(sid)
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if not _is_expired(data):
                return data
            # Expired — clean up and return fresh
            p.unlink(missing_ok=True)
            log.debug("Session expired and removed: %s", sid)
    except Exception as e:
        log.warning("Failed to load session %s: %s", sid, e)
    return {"history": [], "last_active": time.time()}


def save(sid: str, data: dict) -> None:
    """
    Persist a session to disk.
    Refreshes last_active and trims if over the byte threshold.
    """
    data["last_active"] = time.time()
    data = _trim(data)
    try:
        _path(sid).write_text(json.dumps(data), encoding="utf-8")
    except Exception as e:
        log.error("Failed to save session %s: %s", sid, e)


def delete(sid: str) -> None:
    """Delete a session file immediately (user clicked Forget)."""
    try:
        _path(sid).unlink(missing_ok=True)
        log.debug("Session deleted: %s", sid)
    except Exception as e:
        log.warning("Failed to delete session %s: %s", sid, e)


def append(sid: str, query: str, rug_ids: list, count: int) -> None:
    """
    Convenience function — load, append one history entry, save.
    Called after each successful search in conversation mode.
    """
    data = load(sid)
    data["history"].append({
        "query":   query,
        "count":   count,
        "rug_ids": rug_ids,
    })
    save(sid, data)


def get_history(sid: str) -> list:
    """Return the history list for a session, or empty list if none."""
    return load(sid).get("history", [])


def get_seen_ids(sid: str) -> list:
    """Return all rug IDs seen across the entire session history."""
    return [rid for h in get_history(sid) for rid in h.get("rug_ids", [])]


# ------------------------------------------------------------------ #
#  Background cleanup thread
# ------------------------------------------------------------------ #

def _sweep():
    """Remove expired session files. Runs in a daemon thread."""
    while True:
        time.sleep(SESSION_SWEEP_SECS)
        try:
            now = time.time()
            removed = 0
            for p in SESSIONS_DIR.glob("*.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if now - data.get("last_active", 0) > SESSION_TTL_SECS:
                        p.unlink(missing_ok=True)
                        removed += 1
                except Exception:
                    # Unreadable file — remove it
                    p.unlink(missing_ok=True)
                    removed += 1
            if removed:
                log.debug("Session sweep removed %d expired file(s)", removed)
        except Exception as e:
            log.warning("Session sweep error: %s", e)


def start_cleanup_thread() -> None:
    """Start the background sweep thread. Call once at app startup."""
    t = threading.Thread(target=_sweep, daemon=True, name="session-sweep")
    t.start()
    log.info("Session cleanup thread started (sweep every %ds, TTL %ds)",
             SESSION_SWEEP_SECS, SESSION_TTL_SECS)
