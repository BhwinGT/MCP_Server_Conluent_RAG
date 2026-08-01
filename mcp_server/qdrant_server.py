"""
qdrant_server.py
----------------
MCP server exposing two-stage hybrid search over the Confluent support
article knowledge base:

  confluent_search_documents(query, top_k)        -- hybrid retrieve + cross-encoder rerank
  confluent_search_official_articles(...)         -- same, official articles only
  confluent_search_saved_conversations(...)       -- same, saved conversations only
  confluent_collection_info()                     -- returns collection stats
  confluent_save_solved_conversation(...)         -- TOOL: step 1, returns a preview,
                                             writes nothing
  confluent_confirm_save_conversation(token)      -- TOOL: step 2, commits the draft
  confluent_prepare_solved_conversation(...)      -- PROMPT: optional human-curated
                                             entry point to the same flow

Search architecture:
  Stage 1 (retrieval, cheap/broad); dense (semantic) + sparse (BM25 keyword)
    search run in parallel via Qdrant prefetch, fused with Reciprocal Rank Fusion (RRF). This stage's job is RECALL.

  Stage 2 (reranking, precise/narrow); a cross-encoder reads the query
    and EACH candidate chunk together and rescoring the candidate pool.
    This stage's job is PRECISION

    The below command will bring everything neccessary up;
    docker compose up -d
"""

import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Prefetch, FusionQuery, Fusion, SparseVector, PointStruct,
    Filter, FieldCondition, MatchValue,
)
from fastembed import SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from ollama import Client as OllamaClient
from fastmcp import FastMCP
from fastmcp.exceptions import AuthorizationError
from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware import Middleware
from fastmcp.server.auth.providers.azure import AzureProvider
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.chunking import chunk_article

load_dotenv()

# DB + Model
EMBED_MODEL      = os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b")
SPARSE_MODEL     = "Qdrant/bm25"
RERANK_MODEL     = "jinaai/jina-reranker-v2-base-multilingual"
QDRANT_HOST      = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT      = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME  = os.getenv("QDRANT_COLLECTION", "confluent_support")


# MCP server
MCP_HOST = os.getenv("MCP_HOST", "0.0.0.0")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))

# Authentication
AUTH_ENABLED = os.getenv("MCP_AUTH_ENABLED", "false").strip().lower() in ("1", "true")
AZURE_CLIENT_ID     = os.getenv("AZURE_CLIENT_ID", "").strip()
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET", "").strip()
AZURE_TENANT_ID     = os.getenv("AZURE_TENANT_ID", "").strip()
AZURE_API_SCOPE = os.getenv("AZURE_API_SCOPE", "access").strip()
AZURE_TOKEN_ISSUER = os.getenv("AZURE_TOKEN_ISSUER", "").strip()
PUBLIC_MCP_URL = os.getenv("PUBLIC_MCP_URL", "").strip().rstrip("/")

ALLOWED_EMAIL_DOMAINS = [
    d.strip().lower().lstrip("@")
    for d in os.getenv("ALLOWED_EMAIL_DOMAINS", "").split(",")
    if d.strip()
]

ALLOWED_EMAILS = [
    e.strip().lower()
    for e in os.getenv("ALLOWED_EMAILS", "").split(",")
    if e.strip()
]

# Priority order; Entra work accounts often lack "email". Getting this wrong
# fails CLOSED (everyone denied). SECURITY: the Entra app must be registered
# SINGLE-TENANT, or preferred_username can be spoofed from another tenant.
EMAIL_CLAIMS = ("email", "preferred_username", "upn")

JWT_SIGNING_KEY = os.getenv("MCP_JWT_SIGNING_KEY", "").strip()
FASTEMBED_CACHE_DIR = os.getenv("FASTEMBED_CACHE_DIR", "./model_cache")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")


# Retrieval tuning. Rerank cost is linear in pool size (top_k * multiplier),
# and one search can reach ~6GB RSS -- raise MAX_TOP_K with care.
TOP_K            = int(os.getenv("TOP_K_RESULTS", "5"))
MAX_TOP_K                 = 50
CANDIDATE_POOL_MULTIPLIER = 4
MIN_CANDIDATE_POOL        = 15
PREFETCH_MULTIPLIER       = 4
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0"))

# Saving chat history
CHUNK_TARGET_SIZE  = int(os.getenv("CHUNK_SIZE", "300"))
SINGLE_CHUNK_LIMIT = 400
MAX_SUMMARY_WORDS  = 1200
MIN_EXCERPT_WORDS = 20
MAX_EXCERPT_CHARS = 8000
EXCERPT_PREVIEW_CHARS = 1200

GROUNDING_REJECT_BELOW = 0.15
GROUNDING_WARN_BELOW   = 0.40

DUPLICATE_STRONG_ABOVE = 0.85   # measured: 0.922 feedback-loop vs 0.684 genuinely-new
DUPLICATE_WARN_ABOVE   = 0.75
DUPLICATE_PREVIEW_CHARS = 700

DRAFT_TTL_SECONDS  = 1800
MAX_PENDING_DRAFTS = 50
_pending_drafts: dict[str, dict] = {}

# Only words of >=4 chars are considered, so short stopwords need no listing.
_STOPWORDS = {
    "this", "that", "with", "from", "have", "has", "been", "when", "then",
    "they", "them", "there", "their", "which", "would", "could", "should",
    "after", "before", "because", "using", "used", "into", "only", "also",
    "some", "more", "than", "were", "will", "your", "about", "what", "where",
    "while", "these", "those", "such", "each", "both", "does", "done", "make",
    "made", "need", "needs", "want", "like", "just", "very", "over", "under",
    "same", "other", "another", "issue", "problem", "user", "users",
}


def _content_tokens(text: str) -> set[str]:
    """Distinctive lowercase words, the signal for the grounding check."""
    words = re.findall(r"[a-z0-9][a-z0-9._-]{3,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _grounding_overlap(summary: str, excerpt: str) -> float:
    """
    Fraction of the summary's distinctive words that appear in the excerpt.

    1.0 means fully grounded; near 0 means the summary talks about things
    its own cited source never mentions.
    """
    summary_tokens = _content_tokens(summary)
    if not summary_tokens:
        return 1.0
    return len(summary_tokens & _content_tokens(excerpt)) / len(summary_tokens)


def _current_user_email() -> str:
    """
    Authenticated user's address, or "" when auth is off / unavailable.

    Used both by the domain check and to bind a draft to whoever created
    it, so one user cannot confirm another's pending write.
    """
    try:
        token = get_access_token()
    except Exception:
        return ""
    if token is None:
        return ""
    claims = token.claims or {}
    return next(
        (str(claims[c]) for c in EMAIL_CLAIMS if claims.get(c)), ""
    ).strip().lower()


def _find_near_duplicate(text: str) -> dict:
    """
    Closest existing entry to `text`.

    Returns {score, title, source_type, url, text}; score 0.0 with empty
    fields if the collection is empty or the lookup fails. Dense-only: BM25's
    exact-keyword bias would miss a reworded echo, which is the case that matters.
    """
    empty = {"score": 0.0, "title": "", "source_type": "", "url": "", "text": ""}
    try:
        vec = _ollama_client.embed(model=EMBED_MODEL, input=[text])["embeddings"][0]
        hits = _client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec, using="dense", limit=1, with_payload=True,
        ).points
        if not hits:
            return empty
        top = hits[0]
        return {
            "score": float(top.score),
            "title": top.payload.get("title", ""),
            "source_type": top.payload.get("source_type", ""),
            "url": top.payload.get("url", ""),
            "text": top.payload.get("text", ""),
        }
    except Exception:
        # A lookup failure must never interfere with a save.
        return empty


def _prune_drafts() -> None:
    """Drop expired drafts, then enforce the hard cap oldest-first."""
    now = time.time()
    for tok in [t for t, d in _pending_drafts.items()
                if now - d["created_at"] > DRAFT_TTL_SECONDS]:
        _pending_drafts.pop(tok, None)
    while len(_pending_drafts) > MAX_PENDING_DRAFTS:
        oldest = min(_pending_drafts, key=lambda t: _pending_drafts[t]["created_at"])
        _pending_drafts.pop(oldest, None)

_client        = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
_ollama_client = OllamaClient(host=OLLAMA_HOST)
_sparse_model  = SparseTextEmbedding(model_name=SPARSE_MODEL, cache_dir=FASTEMBED_CACHE_DIR)
_reranker      = TextCrossEncoder(model_name=RERANK_MODEL, cache_dir=FASTEMBED_CACHE_DIR)

class EmailDomainMiddleware(Middleware):
    """
    Rejects authenticated users who aren't on the allowlist.

    Two modes, in priority order:
      ALLOWED_EMAILS set        -> only those exact addresses (per-person)
      else ALLOWED_EMAIL_DOMAINS -> anyone on those domains

    Covers GUEST/external users invited into the tenant, whom Entra
    authenticates happily. Middleware, not a per-tool check, so new tools
    are protected by default.
    """

    def _reject_reason(self) -> str | None:
        email = _current_user_email()

        if not email:
            return "no usable email claim on the access token"
        if "@" not in email:
            return f"malformed email claim: {email!r}"

        if ALLOWED_EMAILS:
            if email not in ALLOWED_EMAILS:
                return f"{email} is not on the allowed user list"
            return None

        domain = email.rsplit("@", 1)[1]
        if domain not in ALLOWED_EMAIL_DOMAINS:
            return f"{email} is not on an allowed domain"
        return None

    async def on_message(self, context, call_next):
        if AUTH_ENABLED and (ALLOWED_EMAILS or ALLOWED_EMAIL_DOMAINS):
            reason = self._reject_reason()
            if reason:
                allowed = (
                    f"{len(ALLOWED_EMAILS)} specifically authorised user(s)"
                    if ALLOWED_EMAILS
                    else ", ".join("@" + d for d in ALLOWED_EMAIL_DOMAINS) + " accounts"
                )
                raise AuthorizationError(
                    f"Access denied ({reason}). This server is restricted to {allowed}."
                )
        return await call_next(context)


def _build_auth():
    """
    Returns a configured AzureProvider, or None to run unauthenticated.

    Microsoft Entra is the only provider wired up. To move to a different
    identity provider, swap AzureProvider for another of fastmcp's providers.
    """
    if not AUTH_ENABLED:
        return None

    missing = [
        name for name, val in (
            ("AZURE_CLIENT_ID", AZURE_CLIENT_ID),
            ("AZURE_CLIENT_SECRET", AZURE_CLIENT_SECRET),
            ("AZURE_TENANT_ID", AZURE_TENANT_ID),
            ("PUBLIC_MCP_URL", PUBLIC_MCP_URL),
        ) if not val
    ]
    if missing:
        raise SystemExit(
            f"MCP_AUTH_ENABLED is set but {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} empty. Set them in .env, "
            f"or set MCP_AUTH_ENABLED=false to run without authentication."
        )

    if not PUBLIC_MCP_URL.startswith("https://"):
        raise SystemExit(
            f"PUBLIC_MCP_URL must be the public https:// URL clients reach "
            f"(got {PUBLIC_MCP_URL!r}). Identity providers reject non-HTTPS "
            f"redirect URIs."
        )

    if ALLOWED_EMAILS:
        print(
            f"Auth: restricted to {len(ALLOWED_EMAILS)} specific user(s) "
            f"(ALLOWED_EMAILS is set, so ALLOWED_EMAIL_DOMAINS is ignored).",
            file=sys.stderr,
        )
    elif not ALLOWED_EMAIL_DOMAINS:
        print(
            "WARNING: auth is on but neither ALLOWED_EMAILS nor "
            "ALLOWED_EMAIL_DOMAINS is set -- ANY account that completes login "
            "will be accepted, from any organisation.",
            file=sys.stderr,
        )

    # jwt_signing_key is persisted so a container restart doesn't invalidate every client's session.
    # The OAuth state directory is volume-mounted for the same reason.
    return AzureProvider(
        client_id=AZURE_CLIENT_ID,
        client_secret=AZURE_CLIENT_SECRET,
        tenant_id=AZURE_TENANT_ID,
        base_url=PUBLIC_MCP_URL,
        required_scopes=[AZURE_API_SCOPE],
        additional_authorize_scopes=["openid", "profile", "email"],
        token_issuer=AZURE_TOKEN_ISSUER or None,
        jwt_signing_key=JWT_SIGNING_KEY or None,
    )


mcp = FastMCP("confluent_rag_mcp", auth=_build_auth(), middleware=[EmailDomainMiddleware()],)


def _embed_sparse_query(text: str) -> SparseVector:
    emb = next(_sparse_model.embed([text]))
    return SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())


def _run_search(query: str, top_k: int, source_type: str | None = None) -> str:
    """
    Shared two-stage hybrid search. All three search tools call this.

    source_type=None searches everything; "official_article" or
    "chat_history" restricts the candidate pool to that source.
    """
    try:
        top_k = max(1, min(int(top_k), MAX_TOP_K))

        dense_resp    = _ollama_client.embed(model=EMBED_MODEL, input=[query])
        dense_vector  = dense_resp["embeddings"][0]
        sparse_vector = _embed_sparse_query(query)

        candidate_pool_size = max(top_k * CANDIDATE_POOL_MULTIPLIER, MIN_CANDIDATE_POOL)
        prefetch_limit      = candidate_pool_size * PREFETCH_MULTIPLIER

        source_filter = None
        if source_type:
            source_filter = Filter(must=[FieldCondition(
                key="source_type", match=MatchValue(value=source_type))])

        # --- Stage 1: hybrid retrieval (recall) ---
        fused = _client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(query=dense_vector, using="dense",
                         limit=prefetch_limit, filter=source_filter),
                Prefetch(query=sparse_vector, using="bm25",
                         limit=prefetch_limit, filter=source_filter),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=candidate_pool_size,
            with_payload=True,
        ).points

        scope = {
            "official_article": " in the official Confluent articles",
            "chat_history": " in the saved conversations",
        }.get(source_type or "", "")

        if not fused:
            return f"No relevant documents found{scope}."

        # --- Stage 2: cross-encoder reranking (precision) ---
        candidate_texts = [p.payload.get("text", "") for p in fused]
        rerank_scores = list(_reranker.rerank(query, candidate_texts))

        reranked = sorted(
            zip(fused, rerank_scores), key=lambda pair: pair[1], reverse=True
        )

        # Drop anything below the relevance floor
        strong_enough = [pair for pair in reranked if pair[1] >= RERANK_MIN_SCORE]
        top_results = strong_enough[:top_k]

        if not top_results:
            return f"No relevant documents found{scope}."

        parts = []
        for i, (point, score) in enumerate(top_results):
            title    = point.payload.get("title", "")
            src_type = point.payload.get("source_type", "official_article")
            text     = point.payload.get("text", "")

            if src_type == "chat_history":
                source_label = (
                    f"community-reported fix (AI-assisted, via {point.payload.get('source_ai', 'unknown AI')})"
                )
            else:
                source_label = point.payload.get("url", "unknown")

            parts.append(
                f"[Result {i+1}]  rerank_score={round(score, 4)}  title={title}  "
                f"source_type={src_type}  source={source_label}\n{text}\n"
            )

        return "\n---\n".join(parts)

    except Exception as e:
        return f"Search error: {e}"


_SEARCH_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}


@mcp.tool(annotations={"title": "Search Confluent Knowledge Base (all sources)", **_SEARCH_ANNOTATIONS})
def confluent_search_documents(query: str, top_k: int = TOP_K) -> str:
    """
    Two-stage hybrid search across the WHOLE knowledge base both official
    Confluent support articles and problems saved from past conversations.

    Use this by default. Use confluent_search_official_articles when the user
    wants only vendor-documented answers, or confluent_search_saved_conversations
    when they want only what colleagues have hit and fixed.

    Stage 1: dense (semantic) + sparse (BM25 keyword) search, fused with
    Reciprocal Rank Fusion, to build a broad candidate pool.
    Stage 2: a cross-encoder reranks that pool by reading the query and each
    candidate together, for much higher precision than fused rank alone.

    Each result is labelled with its source_type, so you can tell an official
    article from a community-reported fix.

    Args:
        query: What to search for. Natural language works well, the dense
            half of the search is semantic, so an error message or a
            described symptom is a good query.
        top_k: How many chunks to return. Values above 50 are clamped;
            fewer, stronger results generally produce a better grounded
            answer than a long tail of weak ones.
    """
    return _run_search(query, top_k, source_type=None)


@mcp.tool(annotations={"title": "Search Official Confluent Articles Only", **_SEARCH_ANNOTATIONS})
def confluent_search_official_articles(query: str, top_k: int = TOP_K) -> str:
    """
    Same two-stage hybrid search, restricted to OFFICIAL Confluent support
    articles scraped from the support portal. Saved conversations are excluded.

    Use this when the user wants only vendor-documented, authoritative
    answers. For example when writing something customer-facing, or when
    they explicitly distrust community content.

    Every result carries a real source URL that can be cited.

    Args:
        query: What to search for. Natural language or an exact error message.
        top_k: How many chunks to return (clamped at 50).
    """
    return _run_search(query, top_k, source_type="official_article")


@mcp.tool(annotations={"title": "Search Saved Conversations Only", **_SEARCH_ANNOTATIONS})
def confluent_search_saved_conversations(query: str, top_k: int = TOP_K) -> str:
    """
    Same two-stage hybrid search, restricted to problems SAVED FROM PAST
    CONVERSATIONS by colleagues. Official articles are excluded.

    Use this when the user wants real-world experience rather than official
    documentation, undocumented workarounds, environment-specific gotchas,
    or "has anyone here encounter this problem before?".

    Args:
        query: What to search for. Natural language or an exact error message.
        top_k: How many chunks to return (clamped at 50).
    """
    return _run_search(query, top_k, source_type="chat_history")



@mcp.tool(annotations={
    "title": "Draft a Knowledge Base Entry (step 1 of 2, saves nothing)",
    # readOnlyHint stays True: parks an in-memory draft but writes nothing to
    # the KB, so clients should not prompt. The confirm tool carries the writes.
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": False,
    })

def confluent_save_solved_conversation(
    title: str,
    description: str,
    resolution: str,
    source_excerpt: str,
    applies_to: str = "",
    cause: str = "",
    source_ai: str = "",
    ) -> str:
    """
    STEP 1 of 2. Prepares a solved Confluent problem for the knowledge base
    and returns a preview for the user to check. THIS WRITES NOTHING.
    Nothing is stored until confluent_confirm_save_conversation() is called with the
    token this returns.

    Show the returned preview to the user word-for-word and ask them to confirm.
    Do not call confluent_confirm_save_conversation() until they explicitly say yes.

    ONLY save something that was actually broken, and actually fixed.

    If a similar entry already exists, the preview will show it alongside
    your draft. That is a WARNING, not a rejection: show it to the user, let
    them compare, and save only if they decide it adds something new.

    Good candidates: a fix found by trial and error, an environment-specific gotcha,
    a resolution the user discovered themselves.
    If nothing was actually broken and fixed, do not save.

    Entries are structured like official Confluent support articles
    (Title / Description / Applies To / Cause / Resolution) so they sit
    alongside them in search results.

    Write plain, concise summaries not a raw transcript. Strip greetings,
    dead ends and back-and-forth; keep the actual problem and the actual fix.

    Args:
        title: Short, specific title (e.g. "Consumer group keeps rebalancing
            after increasing session.timeout.ms").
        description: What the problem looked like. Symptoms, error
            messages, relevant context. Plain text, no conversational filler.
        resolution: The steps that resolved it. Plain text, numbered if
            applicable.
        source_excerpt: REQUIRED. The word-for-word part of the conversation this
            summary is based on. Copy the actual messages, do not
            paraphrase or reconstruct them. Include the problem report and
            the messages containing the fix.
            This exists because you may be summarising from a long
            conversation that also covered unrelated topics, or one whose
            earlier parts are no longer fully in your context. Quoting the
            real span is what lets the user see what your summary was
            actually based on. If you cannot locate the relevant messages,
            say so to the user instead of reconstructing them from memory.
        applies_to: Optional -- e.g. "Confluent Cloud", "Confluent Platform
            7.6", "Kafka Streams". Leave blank if unclear.
        cause: Optional -- the underlying cause, if one was clearly
            identified separately from the description.
        source_ai: Optional -- which AI assistant helped (e.g. "Claude",
            "ChatGPT", "Gemini").
    """
    try:
        title = title.strip()
        description = description.strip()
        resolution = resolution.strip()
        applies_to = applies_to.strip()
        cause = cause.strip()
        source_excerpt = source_excerpt.strip()

        if not title or not description or not resolution:
            return "Error: title, description, and resolution are all required."

        if not source_excerpt:
            return (
                "Error: source_excerpt is required. Paste the word-for-word "
                "conversation messages this summary is based on. If you "
                "cannot locate them, tell the user rather than reconstructing."
            )

        excerpt_words = len(source_excerpt.split())
        if excerpt_words < MIN_EXCERPT_WORDS:
            return (
                f"Error: source_excerpt is only {excerpt_words} words, which is "
                f"too short to show what the summary is based on (need at least "
                f"{MIN_EXCERPT_WORDS}). Quote the actual messages describing the "
                f"problem and the fix."
            )

        total_words = len((description + " " + applies_to + " " + cause + " " + resolution).split())
        if total_words > MAX_SUMMARY_WORDS:
            return (
                f"Error: submission is {total_words} words, over the {MAX_SUMMARY_WORDS}-word "
                f"limit. Summarize further -- keep only the actual problem and the actual fix, "
                f"not the full conversation."
            )

        if not _client.collection_exists(COLLECTION_NAME):
            return (
                f"Error: collection '{COLLECTION_NAME}' does not exist yet. "
                f"Run the ingestion pipeline (embed_store.py) at least once before saving conversations."
            )

        overlap = _grounding_overlap(" ".join([description, cause, resolution]), source_excerpt)
        if overlap < GROUNDING_REJECT_BELOW:
            return (
                f"Error: the summary has almost nothing in common with the "
                f"source_excerpt you provided ({overlap:.0%} of its distinctive "
                f"terms appear there). Either the excerpt is from the wrong part "
                f"of the conversation, or the summary describes things the "
                f"excerpt never mentions. Re-check both."
            )

        dup_text = f"{title} {description} {cause}".strip()
        dup = _find_near_duplicate(dup_text)

        _prune_drafts()
        token = uuid.uuid4().hex
        _pending_drafts[token] = {
            "created_at": time.time(),
            "owner": _current_user_email(),
            "fields": {
                "title": title, "description": description,
                "resolution": resolution, "applies_to": applies_to,
                "cause": cause, "source_ai": source_ai.strip(),
                "source_excerpt": source_excerpt[:MAX_EXCERPT_CHARS],
            },
        }

        warnings = []
        if overlap < GROUNDING_WARN_BELOW:
            warnings.append(
                f"!! CHECK CAREFULLY: only {overlap:.0%} of the summary's key terms "
                f"appear in the excerpt. That can be normal for heavy paraphrasing, "
                f"but it can also mean the summary drifted from the source."
            )
        warning = ("\n" + "\n".join(warnings) + "\n") if warnings else ""

        duplicate_block = ""
        if dup["score"] >= DUPLICATE_WARN_ABOVE:
            kind = ("official Confluent article" if dup["source_type"] == "official_article"
                    else "saved conversation")
            headline = ("!! VERY SIMILAR ENTRY ALREADY EXISTS"
                        if dup["score"] >= DUPLICATE_STRONG_ABOVE
                        else "!  A similar entry already exists")
            dup_text_shown = " ".join(dup["text"].split())
            if len(dup_text_shown) > DUPLICATE_PREVIEW_CHARS:
                dup_text_shown = dup_text_shown[:DUPLICATE_PREVIEW_CHARS].rstrip() + " ..."

            lines = [
                "",
                f"{headline}  ({dup['score']:.0%} similar)",
                f"  Existing {kind}: \"{dup['title']}\"",
            ]
            if dup["url"]:
                lines.append(f"  {dup['url']}")
            lines += [
                "  What it already says:",
                "\n".join("    | " + dup_text_shown[i:i + 90]
                          for i in range(0, len(dup_text_shown), 90)),
                "",
                "  Ask the user to compare. Save anyway ONLY if this adds something",
                "  that entry does not -- a different cause, a different fix, or a",
                "  case it does not cover. If your summary came from searching this",
                "  knowledge base, it is already here and saving would duplicate it.",
            ]
            duplicate_block = "\n".join(lines) + "\n"

        excerpt_shown = "\n".join("  | " + l for l in source_excerpt.splitlines() if l.strip())
        if len(excerpt_shown) > EXCERPT_PREVIEW_CHARS:
            excerpt_shown = (excerpt_shown[:EXCERPT_PREVIEW_CHARS].rstrip()
                             + f"\n  | ... [{excerpt_words} words total, truncated for display]")

        preview = "\n".join([
            "NOT SAVED YET -- review this, then confirm.",
            "",
            f"Title       : {title}",
            f"Applies To  : {applies_to or '(none)'}",
            f"Cause       : {cause or '(none)'}",
            "",
            "Description :", description,
            "",
            "Resolution  :", resolution,
            "",
            f"Based on this quoted conversation ({excerpt_words} words):",
            excerpt_shown,
            warning,
            duplicate_block,
            "Show the user EVERYTHING above, including the quoted conversation "
            "and any similarity warning, and ask whether to save it. Only if "
            "they explicitly agree, call:",
            f"    confluent_confirm_save_conversation(draft_token=\"{token}\")",
            f"The draft expires in {DRAFT_TTL_SECONDS // 60} minutes.",
        ])
        return preview

    except Exception as e:
        return f"Draft error: {e}"

@mcp.tool(annotations={
    "title": "Commit Knowledge Base Entry (step 2 of 2, writes)",
    "readOnlyHint": False,
    "destructiveHint": True,
    "idempotentHint": False,
    "openWorldHint": False,
    })
def confluent_confirm_save_conversation(draft_token: str) -> str:
    """
    STEP 2 of 2. Commits a draft created by confluent_save_solved_conversation() to
    the knowledge base.

    Only call this after the user has seen the preview and explicitly
    agreed to save it. If they want changes, call confluent_save_solved_conversation()
    again with the corrected text instead -- that produces a new token.

    Args:
        draft_token: The token from the confluent_save_solved_conversation() preview.
    """
    try:
        _prune_drafts()
        draft = _pending_drafts.get(draft_token.strip())
        if draft is None:
            return (
                "Error: unknown or expired draft token. Drafts expire after "
                f"{DRAFT_TTL_SECONDS // 60} minutes. Call confluent_save_solved_conversation() "
                f"again to produce a fresh preview."
            )

        owner = draft.get("owner", "")
        if owner and owner != _current_user_email():
            return "Error: this draft was created by a different user."

        f = draft["fields"]
        title = f["title"]

        blocks = [
            ("heading", f"# {title}"),
            ("heading", "## Description"),
            ("paragraph", f["description"]),
        ]
        if f["applies_to"]:
            blocks += [("heading", "## Applies To"), ("paragraph", f["applies_to"])]
        if f["cause"]:
            blocks += [("heading", "## Cause"), ("paragraph", f["cause"])]
        blocks += [("heading", "## Resolution"), ("paragraph", f["resolution"])]

        chunks = chunk_article(blocks, CHUNK_TARGET_SIZE, SINGLE_CHUNK_LIMIT)

        conversation_id = str(uuid.uuid4())
        saved_at = datetime.now(timezone.utc).isoformat()

        dense_vecs = _ollama_client.embed(model=EMBED_MODEL, input=chunks)["embeddings"]
        sparse_vecs = []
        for sparse_emb in _sparse_model.embed(chunks):
            sparse_vecs.append(SparseVector(
                indices=sparse_emb.indices.tolist(),
                values=sparse_emb.values.tolist(),
            ))

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": dense_vecs[i], "bm25": sparse_vecs[i]},
                payload={
                    "text": chunks[i],
                    "title": title,
                    "article_id": conversation_id,
                    "chunk_index": i,
                    "source_type": "chat_history",
                    "source_ai": f["source_ai"] or "unknown",
                    "applies_to": f["applies_to"],
                    "updated_at": saved_at,
                    "url": "",
                    "section_id": None,
                    "source_excerpt": f["source_excerpt"] if i == 0 else "",
                    "saved_by": draft.get("owner", ""),
                },
            )
            for i in range(len(chunks))
        ]
        _client.upsert(collection_name=COLLECTION_NAME, points=points)
        _pending_drafts.pop(draft_token.strip(), None)

        return (
            f"Saved. Title: '{title}' -- stored as {len(chunks)} chunk(s) "
            f"in the knowledge base and will now surface in future searches."
        )

    except Exception as e:
        return f"Save error: {e}"


@mcp.prompt()
def confluent_prepare_solved_conversation(chat_excerpt: str, title: str = "") -> str:
    """
    Turns a human-selected excerpt of a solved conversation into a request
    for the AI to summarize it and save it to the knowledge base.

    Use this when YOU (the human) want to choose exactly what gets
    archived, rather than asking the AI to decide what's relevant from its
    own context. Paste in just the relevant part of the conversation --
    the actual problem and how it got solved -- not the whole chat.

    Args:
        chat_excerpt: The relevant portion of the conversation -- paste in
            the problem description and the messages that led to the fix.
            Trim out greetings, dead ends, and anything unrelated yourself;
            only what's pasted here will be considered for summarizing.
        title: Optional. If left blank, the AI will propose one based on
            the excerpt.
    """
    title_instruction = (
        f'Use this title: "{title}"' if title.strip()
        else "Propose a short, specific title for this problem."
    )

    return (
        "The user has selected the following excerpt from a solved "
        "Confluent-related troubleshooting conversation and wants it saved "
        "to the knowledge base, in the same format as official Confluent "
        "support articles.\n\n"
        f"{title_instruction}\n\n"
        "Summarize the excerpt below into these fields, matching the "
        "structure of a real Confluent support article:\n"
        "  - description: what the problem looked like (symptoms, errors)\n"
        "  - applies_to: what this applies to, e.g. 'Confluent Cloud' or "
        "'Confluent Platform 7.6' (omit if unclear)\n"
        "  - cause: the underlying cause, only if one was clearly "
        "identified separately from the description (omit otherwise)\n"
        "  - resolution: the steps that actually fixed it\n"
        "Plain text, no conversational filler, similar in tone to an "
        "official support article.\n\n"
        "Then call confluent_save_solved_conversation, passing the excerpt below "
        "verbatim as source_excerpt. That returns a preview and saves "
        "nothing. Show the preview to the user and only call "
        "confluent_confirm_save_conversation once they explicitly agree.\n\n"
        "--- EXCERPT ---\n"
        f"{chat_excerpt}\n"
        "--- END EXCERPT ---"
    )


@mcp.tool(annotations={
    "title": "Confluent Knowledge Base Stats",
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
})
def confluent_collection_info() -> str:
    """Returns stats about the Qdrant collection (point count, models used, etc.)."""
    try:
        count = _client.count(COLLECTION_NAME).count
        chat_count = _client.count(
            COLLECTION_NAME,
            count_filter={"must": [{"key": "source_type", "match": {"value": "chat_history"}}]},
        ).count
    except Exception as e:
        return f"Could not read collection info: {e}"

    return (
        f"Collection    : {COLLECTION_NAME}\n"
        f"Total chunks  : {count}  ({chat_count} from saved conversations, {count - chat_count} official)\n"
        f"Qdrant        : {QDRANT_HOST}:{QDRANT_PORT}\n"
        f"Dense model   : {EMBED_MODEL}\n"
        f"Sparse model  : {SPARSE_MODEL} (BM25)\n"
        f"Rerank model  : {RERANK_MODEL}\n"
        f"Search mode   : Hybrid retrieval (RRF) -> cross-encoder rerank"
    )


if __name__ == "__main__":
    print(f"Starting MCP server on http://{MCP_HOST}:{MCP_PORT}")
    print(f"MCP endpoint (for Claude Desktop / clients): http://{MCP_HOST}:{MCP_PORT}/mcp")
    print("Leave this running. Press Ctrl+C to stop.")
    mcp.run(transport="http", host=MCP_HOST, port=MCP_PORT)