"""
rag.py
------
AI Pattern Search — RAG and query expansion.

Two functions:
  - expand_query()   : uses Claude to enrich a short user query with
                       synonyms, design terms, and taxonomy keywords
                       before it hits the Bedrock knowledge base.

  - rag_summarise()  : calls Bedrock RetrieveAndGenerate to produce
                       a professional natural-language summary of
                       what was found. Optionally incorporates
                       conversation history for context continuity.
"""

import json
import logging
import urllib.request
import urllib.error

import boto3

from config import (
    AWS_REGION,
    KNOWLEDGE_BASE_ID,
    BEDROCK_MODEL_ARN,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    ANTHROPIC_URL,
)
from keywords import taxonomy_str, get_all

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Boto3 client — module-level so it's reused across requests
# ------------------------------------------------------------------ #

_bedrock = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)

# ------------------------------------------------------------------ #
#  RAG system prompt
# ------------------------------------------------------------------ #

_RAG_SYSTEM = (
    "You are an expert rug analyst for a high-end rug retailer. "
    "Help sales representatives and design teams find the right pattern "
    "for each project. Use professional, sales-oriented language referencing "
    "specific patterns, colour palettes, and design characteristics. "
    "Be concise — 2-3 sentences maximum."
)

# ------------------------------------------------------------------ #
#  Query expansion
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
#  Expansion prompt templates
# ------------------------------------------------------------------ #

_EXPAND_SYSTEM = """\
You are a rug taxonomy specialist enriching search queries for a pattern database.

Rules:
1. Keep ALL words from the original query — never remove or rephrase them.
2. Add ONLY 1–2 terms from the TAXONOMY LIST that share a direct, \
obvious design relationship with the query (same motif family, \
same visual concept). If you are not confident a term is directly \
related, do not add it.
3. Do NOT add style/era terms (e.g. Traditional, Vintage, Modern, \
Transitional) unless the query explicitly references a style or era.
4. Return a single line of space-separated terms. No punctuation, \
no explanation, no markdown.

TAXONOMY LIST:
{taxonomy}"""

_EXPAND_SYSTEM_SYNONYM = """\
You are a rug pattern search specialist. Expand a single design term into a \
rich set of synonyms and closely related visual vocabulary to improve recall \
in a vector search over rug pattern descriptions.

Rules:
1. Keep the original term.
2. Add 4–6 synonyms or closely related design/visual terms (e.g. for "floral": \
flower botanical bloom garden petal vine). Focus on words likely to appear in \
written descriptions of rug patterns.
3. Do NOT add style/era terms (Traditional, Vintage, Modern, etc.).
4. Return a single line of space-separated terms. No punctuation, no explanation.
"""

_EXPAND_SYSTEM_SYNONYM_ALT = """\
You are a rug pattern search specialist. A previous search already used the \
terms listed under ALREADY USED. Expand the original term using a DIFFERENT \
angle — complementary synonyms, related motifs, or adjacent visual concepts \
not yet covered — to surface new results in a vector search over rug pattern \
descriptions.

Rules:
1. Keep the original term.
2. Add 4–6 terms NOT in the ALREADY USED list. Focus on words likely to appear \
in written descriptions of rug patterns.
3. Do NOT add style/era terms (Traditional, Vintage, Modern, etc.).
4. Return a single line of space-separated terms. No punctuation, no explanation.

ALREADY USED:
{previous_expansion}
"""

_EXPAND_USER = "Query: {query}"

# Words that signal the user already named a style/era — skip expansion
_STYLE_TERMS = {
    "traditional", "modern", "transitional", "vintage", "antique",
    "casual", "farmhouse", "global", "southwestern", "southwest",
    "soft modern",
}


def _is_taxonomy_match(query: str) -> bool:
    """
    Return True if the entire query (or a close lowercase match) is
    already a single taxonomy term. Expansion adds noise in this case.
    """
    q = query.strip().lower()
    taxonomy_lower = {kw.lower() for kw in get_all()}
    return q in taxonomy_lower


def expand_query(query: str, previous_expansion: str = "") -> str:
    """
    Enrich the user's query with 1–2 directly related taxonomy terms.

    Short-circuits (returns query unchanged) when:
      - The query is already 5+ words (specific enough)
      - The query explicitly names a style/era (expansion would just
        pile on more style noise)

    Note: taxonomy terms are intentionally NOT short-circuited — a bare
    single-word term like "floral" benefits most from synonym expansion
    ("flower botanical bloom garden") to improve vector search recall
    across JSON files that may use different vocabulary.

    Falls back to the original query on any API error.
    """
    query = query.strip()
    if not query:
        return query

    # --- Short-circuit conditions ---
    words = query.lower().split()

    if len(words) >= 5:
        log.info("Query expansion skipped: '%s' is already specific (%d words)", query, len(words))
        return query

    if any(w in _STYLE_TERMS for w in words):
        log.info("Query expansion skipped: '%s' contains style/era term", query)
        return query

    # --- Expand ---
    # Single-word queries: synonym expansion for broader vector recall.
    #   On load-more (previous_expansion provided), use the alt prompt so
    #   Claude picks a different angle and surfaces new results.
    # Multi-word queries: taxonomy cross-linking.
    if len(words) == 1:
        if previous_expansion:
            system_prompt = _EXPAND_SYSTEM_SYNONYM_ALT.format(
                previous_expansion=previous_expansion
            )
            log.info("Using alt synonym expansion (previous: '%s')", previous_expansion)
        else:
            system_prompt = _EXPAND_SYSTEM_SYNONYM
    else:
        taxonomy = taxonomy_str()
        system_prompt = _EXPAND_SYSTEM.format(taxonomy=taxonomy)

    payload = json.dumps({
        "model":      ANTHROPIC_MODEL,
        "max_tokens": 60,
        "system":     system_prompt,
        "messages": [
            {"role": "user", "content": _EXPAND_USER.format(query=query)}
        ],
    }).encode()

    req = urllib.request.Request(
        ANTHROPIC_URL,
        data=payload,
        headers={
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type":      "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data     = json.loads(resp.read().decode())
            expanded = data["content"][0]["text"].strip()

            if not expanded or len(expanded) > 200:
                log.warning("Expansion returned unexpected output, using original")
                return query

            # Ensure original terms survived
            for word in query.split():
                if word.lower() not in expanded.lower():
                    expanded = query + " " + expanded
                    break

            log.info("Query expanded: '%s' → '%s'", query, expanded)
            return expanded

    except Exception as e:
        log.warning("Query expansion failed, using original: %s", e)
        return query

# ------------------------------------------------------------------ #
#  RAG summary
# ------------------------------------------------------------------ #

def rag_summarise(query: str, history = None) -> str:
    """
    Call Bedrock RetrieveAndGenerate to produce a natural-language summary
    of search results.

    If history is provided (conversation mode), the last 3 prior queries
    are prepended as context so the summary acknowledges what was
    previously shown.

    Returns a plain-text summary string, or a fallback message on error.
    """
    # Build context prefix for conversation mode
    context = ""
    if history:
        prior = history[-3:]
        context = "Previous searches in this session:\n"
        context += "".join(f'- "{h["query"]}"\n' for h in prior)
        context += "\nCurrent search: "

    input_text = context + query

    try:
        response = _bedrock.retrieve_and_generate(
            input={"text": input_text},
            retrieveAndGenerateConfiguration={
                "type": "KNOWLEDGE_BASE",
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": KNOWLEDGE_BASE_ID,
                    "modelArn":        BEDROCK_MODEL_ARN,
                    "generationConfiguration": {
                        "promptTemplate": {
                            "textPromptTemplate": (
                                _RAG_SYSTEM
                                + "\n\n$search_results$\n\n$output_format_instructions$"
                            )
                        }
                    },
                    "retrievalConfiguration": {
                        "vectorSearchConfiguration": {"numberOfResults": 10}
                    },
                },
            },
        )
        summary = response["output"]["text"]
        log.debug("RAG summary produced (%d chars)", len(summary))
        return summary

    except Exception as e:
        log.error("RAG summarise error: %s", e)
        return "Results retrieved — summary unavailable."
