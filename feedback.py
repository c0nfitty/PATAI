"""
feedback.py
-----------
AI Pattern Search — user feedback logging.

When a design team member flags a result, the rug_id, query, reason,
and any additional context are appended to a JSONL file for periodic review.

One JSON object per line — easy to read with:
    cat /apps/python/patternai/feedback.jsonl

Fields logged per entry:
    rug_id          : the rug that was flagged
    query           : the search query that returned it
    reason          : reason code — 'not_related' | 'should_rank_higher' | 'other'
    notes           : free-text note from the user (optional)
    keyword_adds    : keywords the user wants added to the taxonomy
    keyword_removes : keywords the user wants removed from the taxonomy
    timestamp       : unix epoch

Use the review to identify patterns in mismatched results and improve
the pipeline prompt, combined_text structure, or keyword taxonomy.
"""

import json
import logging
import time
from typing import List, Optional

from config import FEEDBACK_LOG

log = logging.getLogger(__name__)


def record(
    rug_id: str,
    query: str,
    reason: str = "not_related",
    notes: str = "",
    keyword_adds: Optional[List[str]] = None,
    keyword_removes: Optional[List[str]] = None,
    color_corrections: Optional[dict] = None,
) -> bool:
    """
    Append a feedback entry to the JSONL log.

    Args:
        rug_id          : the rug that was flagged
        query           : the search query that returned it
        reason          : reason code (default: 'not_related')
        notes           : optional free-text note from the user
        keyword_adds    : keywords the user requested be added to the taxonomy
        keyword_removes : keywords the user requested be removed from the taxonomy

    Returns True on success, False on failure.
    """
    if not rug_id:
        log.warning("feedback.record called with empty rug_id — skipped")
        return False

    entry = {
        "rug_id":          rug_id,
        "query":           query,
        "reason":          reason,
        "timestamp":       time.time(),
    }
    if notes:
        entry["notes"] = notes
    if keyword_adds:
        entry["keyword_adds"] = keyword_adds
    if keyword_removes:
        entry["keyword_removes"] = keyword_removes
    if color_corrections:
        entry["color_corrections"] = color_corrections

    log.info("Writing feedback to: %s", FEEDBACK_LOG)
    try:
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        log.info(
            "Feedback logged: rug=%s reason=%s kw_adds=%d kw_removes=%d color_fixes=%d",
            rug_id, reason,
            len(keyword_adds or []),
            len(keyword_removes or []),
            len(color_corrections or {}),
        )
        return True
    except Exception as e:
        log.error("Failed to write feedback to %s: %s", FEEDBACK_LOG, e)
        return False
