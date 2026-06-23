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


import re
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


@app.route("/admin/keywords")
def admin_keywords():
    """Keyword taxonomy admin page."""
    return render_template("admin_keywords.html")


@app.route("/admin/keywords/list")
def admin_keywords_list():
    """Return all keywords with active/inactive status."""
    return jsonify(keywords.get_all_with_status())


@app.route("/admin/keywords/add", methods=["POST"])
def admin_keywords_add():
    """Add a new keyword to PATAI_KWDS."""
    body = request.get_json() or {}
    kw   = (body.get("keyword") or "").strip()
    if not kw:
        return jsonify({"ok": False, "error": "Missing keyword"}), 400
    result = keywords.add_keyword(kw)
    return jsonify(result), (200 if result["ok"] else 400)


@app.route("/admin/keywords/deactivate", methods=["POST"])
def admin_keywords_deactivate():
    """Deactivate (soft-delete) a keyword."""
    body = request.get_json() or {}
    kw   = (body.get("keyword") or "").strip()
    if not kw:
        return jsonify({"ok": False, "error": "Missing keyword"}), 400
    result = keywords.deactivate_keyword(kw)
    return jsonify(result), (200 if result["ok"] else 400)


@app.route("/admin/keywords/reactivate", methods=["POST"])
def admin_keywords_reactivate():
    """Reactivate a previously deactivated keyword."""
    body = request.get_json() or {}
    kw   = (body.get("keyword") or "").strip()
    if not kw:
        return jsonify({"ok": False, "error": "Missing keyword"}), 400
    result = keywords.reactivate_keyword(kw)
    return jsonify(result), (200 if result["ok"] else 400)


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
    """Log a flagged result from the design team.

    All suggested metadata edits are queued in the JSONL for a
    separate batch job — nothing is applied to S3 or the KB here.

    Accepts:
        rug_id             : str        (required)
        query              : str
        reason             : 'not_related' | 'should_rank_higher' | 'other'
        notes              : str        free-text note
        keyword_adds       : list[str]  keywords to add to this rug's metadata
        keyword_removes    : list[str]  keywords to remove from this rug's metadata
        color_corrections  : dict       {old_name: new_name} color fixes
    """
    body   = request.get_json() or {}
    rug_id = (body.get("rug_id") or "").strip()
    query  = (body.get("query")  or "").strip()
    reason = (body.get("reason") or "not_related").strip()
    notes  = (body.get("notes")  or "").strip()

    kw_adds    = [k.strip() for k in (body.get("keyword_adds")    or []) if str(k).strip()]
    kw_removes = [k.strip() for k in (body.get("keyword_removes") or []) if str(k).strip()]
    color_corrections = {
        str(k).strip(): str(v).strip()
        for k, v in (body.get("color_corrections") or {}).items()
        if str(k).strip() and str(v).strip() and str(k).strip() != str(v).strip()
    }

    if not rug_id:
        return jsonify({"error": "Missing rug_id"}), 400

    ok = feedback.record(
        rug_id,
        query,
        reason=reason,
        notes=notes,
        keyword_adds=kw_adds or None,
        keyword_removes=kw_removes or None,
        color_corrections=color_corrections or None,
    )
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
        # Extract year filter from query (e.g. "floral from 2015", "2015 floral")
        year_match  = re.search(r"\b(19|20)\d{2}\b", query)
        year_filter = year_match.group(0) if year_match else None
        clean_query = re.sub(r"\b(19|20)\d{2}\b", "", query).strip(" ,.-") if year_filter else query
        if year_filter:
            log.info("Year filter extracted: %s — searching on: %r", year_filter, clean_query)

        # Optionally expand the query before searching.
        # On load-more (conversation mode with prior history), pass the
        # previous expansion so Claude picks a complementary set of synonyms.
        if use_expand:
            prev_expansion = session.get_last_expansion(sid, query) if use_convo and sid else ""
            search_query = rag.expand_query(clean_query, previous_expansion=prev_expansion)
        else:
            search_query = clean_query

        # Fetch enough candidates to cover dedup losses, excluded seen IDs, and
        # reranker headroom.  Formula: (desired + already-excluded) × multiplier,
        # capped at Bedrock's hard limit of 100.
        fetch_count = min(
            (max_results + len(exclude_ids)) * config.RERANK_FETCH_MULT,
            100,
        )

        # In broad mode, fan out each expanded term as its own Bedrock query
        # for much better recall.  In exact mode, single query as before.
        expanded_terms = search_query.split() if use_expand and search_query != clean_query else None

        # Run RAG summary and retrieval in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_summary = executor.submit(
                rag.rag_summarise, search_query,
                history if use_convo else None
            )
            if expanded_terms:
                fut_rugs = executor.submit(
                    search.retrieve_rugs_multi, expanded_terms, fetch_count, exclude_ids, year_filter
                )
            else:
                fut_rugs = executor.submit(
                    search.retrieve_rugs, search_query, fetch_count, exclude_ids, year_filter
                )
            summary = fut_summary.result()
            rugs    = fut_rugs.result()

        # Rerank if requested, then trim to exactly max_results in all cases
        if use_rerank and rugs:
            rugs = search.rerank_rugs(search_query, rugs)
        rugs = rugs[:max_results]

        # Persist search to session history
        if use_convo and sid and rugs:
            session.append(
                sid,
                query=query,
                rug_ids=[r["rug_id"] for r in rugs],
                count=len(rugs),
                expanded_query=search_query if use_expand else "",
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
