"""
app.py
------
AI Pattern Search — Flask application entry point.

Routes only — no business logic here.
All logic lives in the dedicated modules:
  config.py   : constants and environment variables
  session.py  : conversation mode session management
  keywords.py : DB2-backed keyword taxonomy
  rag.py      : query expansion and RAG summary
  search.py   : Bedrock retrieval, S3 presigning, reranking
  feedback.py : flagged result logging
"""


import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, jsonify, render_template, request

import logging
from pathlib import Path

import config
config.bootstrap()  # Must run before boto3 clients are created
import feedback
import keywords
import rag
import search
import session

# ------------------------------------------------------------------ #
#  Logging
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Bootstrap
# ------------------------------------------------------------------ #

config.bootstrap()
session.start_cleanup_thread()

# ------------------------------------------------------------------ #
#  Flask app
# ------------------------------------------------------------------ #

app = Flask(__name__, template_folder="templates")

# ------------------------------------------------------------------ #
#  Routes
# ------------------------------------------------------------------ #

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/keywords")
def get_keywords():
    """Return a random sample of 8 active keywords from DB2."""
    return jsonify(keywords.random_sample(8))


@app.route("/keywords/refresh", methods=["POST"])
def refresh_keywords():
    """Force a keyword cache refresh — call after updating PATAI_KWDS."""
    keywords.refresh()
    return jsonify({"ok": True})


@app.route("/session/delete", methods=["POST"])
def delete_session():
    """Delete a conversation session file immediately."""
    body = request.get_json() or {}
    sid  = (body.get("session_id") or "").strip()
    if sid:
        session.delete(sid)
    return jsonify({"ok": True})


@app.route("/feedback", methods=["POST"])
def post_feedback():
    """Log a flagged result from the design team."""
    body   = request.get_json() or {}
    rug_id = (body.get("rug_id") or "").strip()
    query  = (body.get("query")  or "").strip()
    reason = (body.get("reason") or "not_related").strip()

    if not rug_id:
        return jsonify({"error": "Missing rug_id"}), 400

    ok = feedback.record(rug_id, query, reason)
    return jsonify({"ok": ok})


@app.route("/search", methods=["POST"])
def do_search():
    """
    Main search endpoint.

    Request JSON:
        query       : str   — user's search query
        max_results : int   — number of results (default 9)
        expand      : bool  — enable query expansion via Claude
        rerank      : bool  — enable Bedrock Cohere reranking
        convo       : bool  — enable conversation mode
        session_id  : str   — session ID (required when convo=true)

    Response JSON:
        rugs           : list of rug dicts
        summary        : RAG-generated summary string
        expanded_query : expanded query string (if expand=true, else null)
        reranked       : bool
        session_id     : session ID (if convo=true, else null)
    """
    body        = request.get_json() or {}
    query       = (body.get("query") or "").strip()
    max_results = int(body.get("max_results", config.DEFAULT_MAX_RESULTS))
    use_expand  = bool(body.get("expand",  False))
    use_rerank  = bool(body.get("rerank",  False))
    use_convo   = bool(body.get("convo",   False))
    sid         = (body.get("session_id") or "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Load session history if conversation mode is active
    history    = session.get_history(sid)  if use_convo and sid else []
    exclude_ids = session.get_seen_ids(sid) if use_convo and sid else []

    try:
        # Optionally expand the query before searching
        search_query = rag.expand_query(query) if use_expand else query

        # Fetch more results when reranking so the reranker has room to work
        fetch_count = max_results * config.RERANK_FETCH_MULT if use_rerank else max_results

        # Run RAG summary and retrieval in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_summary = executor.submit(
                rag.rag_summarise, search_query,
                history if use_convo else None
            )
            fut_rugs = executor.submit(
                search.retrieve_rugs, search_query, fetch_count, exclude_ids
            )
            summary = fut_summary.result()
            rugs    = fut_rugs.result()

        # Optionally rerank and trim to requested count
        if use_rerank and rugs:
            rugs = search.rerank_rugs(search_query, rugs)[:max_results]

        # Persist search to session history
        if use_convo and sid and rugs:
            session.append(
                sid,
                query=query,
                rug_ids=[r["rug_id"] for r in rugs],
                count=len(rugs),
            )

        return jsonify({
            "rugs":           rugs,
            "summary":        summary,
            "expanded_query": search_query if use_expand else None,
            "reranked":       use_rerank,
            "session_id":     sid if use_convo else None,
        })

    except Exception as e:
        log.error("Search error: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------ #
#  Entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    log.info("AI Pattern Search starting on %s:%s", config.FLASK_HOST, config.FLASK_PORT)
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
