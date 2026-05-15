"""
feedback.py
-----------
AI Pattern Search — user feedback logging.

When a design team member flags a result as "Not Related",
the rug_id, query, reason, and timestamp are appended to a
JSONL file on the IFS for periodic review.

One JSON object per line — easy to read with:
    cat /apps/python/patternai/feedback.jsonl

Use the review to identify patterns in mismatched results
and improve the pipeline prompt or combined_text structure.
"""

import json
import logging
import time

from config import FEEDBACK_LOG

log = logging.getLogger(__name__)


def record(rug_id: str, query: str, reason: str = "not_related") -> bool:
    """
    Append a feedback entry to the JSONL log.

    Args:
        rug_id : the rug that was flagged
        query  : the search query that returned it
        reason : reason code (default: 'not_related')

    Returns True on success, False on failure.
    """
    if not rug_id:
        log.warning("feedback.record called with empty rug_id — skipped")
        return False

    entry = {
        "rug_id":    rug_id,
        "query":     query,
        "reason":    reason,
        "timestamp": time.time(),
    }

    try:
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        log.info("Feedback logged: rug=%s reason=%s", rug_id, reason)
        return True
    except Exception as e:
        log.error("Failed to write feedback: %s", e)
        return False
