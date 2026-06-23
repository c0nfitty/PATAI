"""
keywords.py
-----------
Design team keyword taxonomy for AI Pattern Search.

Keywords are stored in MFGDBFA.PATAI_KWDS on IBM i DB2.
The full list is cached in memory and refreshed every CACHE_TTL_SECS.

Used in two places:
  - UI suggestion chips  : random_sample(8)
  - Query expansion prompt: taxonomy_str()

To add/remove/deactivate keywords, update MFGDBFA.PATAI_KWDS directly
(set STSCDE to a non-active value to hide without deleting).
Changes are picked up automatically on the next cache refresh.

Active status code: 1
"""

import random
import time
import logging
import ibm_db_dbi

from config import KEYWORDS_CACHE_TTL_SECS

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Cache
# ------------------------------------------------------------------ #

_cache: list[str] = []
_cache_loaded_at: float = 0.0




def _load_from_db() -> list[str]:
    try:
        conn   = ibm_db_dbi.connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT KEYW FROM MFGDBFA.PATAI_KWDS WHERE STSCDE = 1 ORDER BY KEYW"
        )
        rows = [row[0].strip() for row in cursor.fetchall()]
        conn.close()
        log.info("Keywords loaded from DB2: %d active keywords", len(rows))
        return rows
    except Exception as e:
        log.error("Failed to load keywords from DB2: %s", e)
        return []


def _get_keywords() -> list[str]:
    """Return cached keywords, refreshing if stale or empty."""
    global _cache, _cache_loaded_at
    now = time.time()
    if not _cache or (now - _cache_loaded_at) > KEYWORDS_CACHE_TTL_SECS:
        fresh = _load_from_db()
        if fresh:
            _cache = fresh
            _cache_loaded_at = now
        elif not _cache:
            # DB unavailable and cache is empty — fall back to hardcoded list
            log.warning("DB2 unavailable and cache empty — using fallback keyword list")
            _cache = _FALLBACK
            _cache_loaded_at = now
    return _cache


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #
def get_all() -> list[str]:
    """Return the full list of active keywords (lowercased for matching)."""
    return _get_keywords()


def random_sample(n: int = 8) -> list[str]:
    """Return n unique randomly selected active keywords."""
    kws = _get_keywords()
    return random.sample(kws, min(n, len(kws)))


def taxonomy_str() -> str:
    """Return all active keywords as a comma-separated string for prompt injection."""
    return ", ".join(_get_keywords())


def refresh() -> None:
    """Force a cache refresh — call if keywords have just been updated in DB2."""
    global _cache_loaded_at
    _cache_loaded_at = 0.0
    _get_keywords()
    log.info("Keyword cache manually refreshed")


def get_all_with_status() -> list[dict]:
    """Return all keywords (active and inactive) with status for admin view."""
    try:
        conn   = ibm_db_dbi.connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT KEYW, STSCDE FROM MFGDBFA.PATAI_KWDS ORDER BY KEYW"
        )
        rows = [{"keyword": row[0].strip(), "active": row[1] == 1} for row in cursor.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        log.error("Failed to load keywords with status: %s", e)
        return [{"keyword": kw, "active": True} for kw in _FALLBACK]


def add_keyword(kw: str) -> dict:
    """Insert a new keyword into PATAI_KWDS with active status.
    Returns {"ok": True} or {"ok": False, "error": "..."}
    """
    kw = kw.strip()
    if not kw:
        return {"ok": False, "error": "Keyword cannot be empty"}
    try:
        conn   = ibm_db_dbi.connect()
        cursor = conn.cursor()
        # Check for duplicate (case-insensitive)
        cursor.execute(
            "SELECT COUNT(*) FROM MFGDBFA.PATAI_KWDS WHERE UPPER(KEYW) = UPPER(?)",
            (kw,)
        )
        if cursor.fetchone()[0] > 0:
            conn.close()
            return {"ok": False, "error": f"'{kw}' already exists"}
        cursor.execute(
            "INSERT INTO MFGDBFA.PATAI_KWDS (KWID, KEYW, STSCDE) VALUES (MAPUTIL.guid(), ?, 1) WITH NC",
            (kw,)
        )
        conn.close()
        refresh()
        log.info("Keyword added: %s", kw)
        return {"ok": True}
    except Exception as e:
        log.error("Failed to add keyword '%s': %s", kw, e)
        return {"ok": False, "error": str(e)}


def deactivate_keyword(kw: str) -> dict:
    """Set STSCDE=0 for a keyword (soft delete).
    Returns {"ok": True} or {"ok": False, "error": "..."}
    """
    kw = kw.strip()
    if not kw:
        return {"ok": False, "error": "Keyword cannot be empty"}
    try:
        conn   = ibm_db_dbi.connect()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE MFGDBFA.PATAI_KWDS SET STSCDE = 0 WHERE UPPER(KEYW) = UPPER(?) WITH NC",
            (kw,)
        )
        conn.close()
        refresh()
        log.info("Keyword deactivated: %s", kw)
        return {"ok": True}
    except Exception as e:
        log.error("Failed to deactivate keyword '%s': %s", kw, e)
        return {"ok": False, "error": str(e)}


def reactivate_keyword(kw: str) -> dict:
    """Set STSCDE=1 for a previously deactivated keyword."""
    kw = kw.strip()
    if not kw:
        return {"ok": False, "error": "Keyword cannot be empty"}
    try:
        conn   = ibm_db_dbi.connect()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE MFGDBFA.PATAI_KWDS SET STSCDE = 1 WHERE UPPER(KEYW) = UPPER(?) WITH NC",
            (kw,)
        )
        conn.close()
        refresh()
        log.info("Keyword reactivated: %s", kw)
        return {"ok": True}
    except Exception as e:
        log.error("Failed to reactivate keyword '%s': %s", kw, e)
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------------------ #
#  Fallback — used only if DB2 is unreachable and cache is cold
# ------------------------------------------------------------------ #

_FALLBACK = [
    "Abstract", "Americana", "Animal Skin", "Antique", "Authentic",
    "Basic", "Basket Weave", "Block", "Border", "Botanical",
    "Braid", "Braided", "Casual", "Check", "Checker Board",
    "Chevron", "Circle", "Classical", "Damask", "Diamond",
    "Distressed", "Farmhouse", "Floral", "Fretwork", "Geometric",
    "Gingham", "Global", "Herringbone", "Hooked", "Ikat",
    "Juvenile", "Kilim", "Leaf", "Marble", "Modern",
    "Moroccan", "Novelty", "Ogee", "Ombre", "Oval",
    "Panel", "Persian", "Plaid", "Scroll", "Sisal",
    "Soft Modern", "Southwest", "Stripe", "Textured", "Traditional",
    "Transitional", "Trellis", "Tribal", "Vintage", "Watercolor",
    "Wave", "Weathered",
]
