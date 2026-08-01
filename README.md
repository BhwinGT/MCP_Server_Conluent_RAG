# Confluent Support RAG — MCP Server

An MCP server that gives any AI assistant (Claude, ChatGPT, Gemini) or any local LLM searchable access to a knowledge base of Confluent support articles, and lets it save solved problems back into that knowledge base.

It runs entirely in Docker, is reachable over HTTPS through an ngrok tunnel, and can require Microsoft (Entra) sign-in restriction.

---

## 1. Overview

**What this does.** It exposes a small set of tools over the Model Context Protocol (MCP), the standard interface AI assistants use to call external tools. An assistant can search the knowledge base, and with your explicit approval it can write a solved problem back into it.

### Current state

|                   |                                                             |
| ----------------- | ----------------------------------------------------------- |
| Knowledge base    | 3,283 chunks from official Confluent support articles       |
| Vector store      | Qdrant, collection name `confluent_support`                 |
| Dense embeddings  | `qwen3-embedding:0.6b` via Ollama (1024-d, cosine)          |
| Sparse embeddings | `Qdrant/bm25` via fastembed                                 |
| Reranker          | `jinaai/jina-reranker-v2-base-multilingual` (cross-encoder) |

### The tools

**List of tools and what it does**
1. `confluent_search_documents`: Hybrid search across everything. The default.
2. `confluent_search_official_articles`: Same search, official articles only.
3. `confluent_search_saved_conversations`: Same search, user-saved fixes only.
4. `confluent_save_solved_conversation`: Step 1 of 2. Returns a preview. Writes nothing.
5. `confluent_confirm_save_conversation`: Step 2 of 2. Commits the draft. The only tool that writes.
6. `confluent_collection_info`: Collection stats, chunk counts and models in use.

**MCP prompt** (only supported by some AI assistants/clients.)
1. `confluent_prepare_solved_conversation`, for when a user wants to paste the excerpt themselves.

---

## 2. How it works

```
        MCP client  (Claude · ChatGPT · Gemini · rag/orchestrator.py)
                              │
                              │  HTTPS
                              │
                        ngrok tunnel  ── TLS terminates here
                              │
                              │  plain HTTP, internal Docker network
                              │
                    ┌─────────▼───────────┐
                    │  mcp_server :8000   │   mcp_server/qdrant_server.py
                    │    auth · tools     │
                    └──┬──────────────┬───┘
           embeddings  │              │  hybrid search / save
                ┌──────▼─────┐  ┌─────▼──────┐
                │  Ollama    │  │  Qdrant    │   collection:
                │  :11434    │  │  :6333     │   confluent_support
                └────────────┘  └────────────┘
```

Three flows share one collection.

### Flow A — Ingest (offline, run manually)

```
Confluent Support Portal (Zendesk Help Center API)
   │
   │  fetch_articles_for_category()          (from ingest/scrape_support.py)
   │
   ▼  raw article HTML
   │
   │  extract_blocks()  → [(block_type, text), …]
   │     headings, paragraphs, code, tables (_table_to_markdown)
   │
   ▼
   │
   │  chunk_article()                        (from shared/chunking.py)
   │     ~300-word target, 1-block overlap
   │     each chunk is prepended with its section heading
   │
   ▼  ingest/fetched_docs.json
   │
   │  main()                                 (from ingest/embed_store.py)
   │     embed_dense_batch()  → Ollama       → 1024-d vector
   │     embed_sparse_batch() → BM25         → sparse vector
   │     point_id(article_id, chunk_index)   → deterministic UUID
   │
   ▼
Qdrant.upsert()   source_type="official_article"
```

`(article_id, updated_at)`: Zendesk article IDs are immutable, so an edited article keeps its ID and only its `updated_at` moves. `load_existing_articles()` reads `{article_id: updated_at}` from the collection and each article is classified:

| Case                                    | Action                                                            |
| --------------------------------------- | ----------------------------------------------------------------- |
| ID not stored                           | embed all its chunks (new)                                        |
| stored `updated_at` older than incoming | **delete every point for that article, then re-insert** (updated) |
| otherwise                               | skip (unchanged)                                                  |

Updated articles are deleted before re-inserting rather than upserted in place. That is what clears orphans: if an edit shortens an article from 16 chunks to 15, upserting alone would leave chunk 15 behind, still returnable by search.

Only `source_type="official_article"` points are considered, so saved conversations are never refreshed or deleted by ingestion.

A typical run re-embeds 41 chunks out of 3,281 (1.2%), versus `--reset` which redoes all of them.

Articles missing from a scrape are deliberately left alone, never deleted. An expired `SUPPORT_COOKIE` makes every article look gone, and a sync that trusted absence would wipe the knowledge base. Observed in practice: article `48565010397972` vanished from the scrape but returns HTTP 403, not 404. It still exists, its permissions changed.

### Flow B — Query (live)

All three search tools call one shared `_run_search()`; they differ only by which `source_type` they filter to.

```
query
   │
   │  MAX_TOP_K (50)
   │
   ▼  embed twice: Ollama dense + _embed_sparse_query() BM25
   │
STAGE 1 · recall ──  _client.query_points()
   │
   │  Prefetch(dense, filter) ┐
   │                          ├── FusionQuery(RRF) → candidate pool
   │  Prefetch(bm25,  filter) ┘    size = max(top_k × 4, 15)
   │
   │  The source_type filter is applied to each prefetch branch.
   │
   ▼
STAGE 2 · precision ──  _reranker.rerank()
   │
   │  cross-encoder reads query + each chunk together, one pass per candidate
   │  → sort by score → drop below RERANK_MIN_SCORE → keep top_k
   │
   ▼
formatted results (rerank score, title, source_type, source URL) → the LLM
```

Why two stages?: dense catches meaning, BM25 catches exact strings; RRF fuses them for recall. The cross-encoder then fixes the ordering, because rank-fusion scores tie often. The reranker is expensive (one model pass per candidate), so it only ever sees the small stage-1 pool, never all 3,283 chunks.

### Flow C — Save (live, two steps)

The server never sees the conversation, only what the AI chooses to send. So it can't verify a summary is faithful; the human is the last check. The design makes the AI show its work and makes the human look before anything is written.

```
STEP 1  confluent_save_solved_conversation(title, description, resolution, source_excerpt, applies_to, cause, source_ai)
   │
   │  required fields present?
   │  source_excerpt >= 20 words?                        → else reject
   │  _grounding_overlap(summary, excerpt) < 0.15        → else reject
   │  _find_near_duplicate(text)                         → advisory only, never auto reject
   │  park a draft in memory (owner-bound, 30 min TTL)
   │
   ▼
   PREVIEW + draft_token.  NOTHING IS WRITTEN.
   Shows the quoted excerpt and any similarity warning.
   User checks before moving on to step 2.


STEP 2  confluent_confirm_save_conversation(draft_token)
   │
   │  token valid, unexpired, same owner?
   │  chunk_article() → embed dense + sparse (same path as ingestion)
   │
   ▼
Qdrant.upsert()   source_type="chat_history", + source_excerpt + saved_by
```

Four guards sit in front of a write, each catching what the others can't:

| Guard            | Catches                                                                        |
| ---------------- | ------------------------------------------------------------------------------ |
| `source_excerpt` | summaries from vague recall, or from an unrelated earlier topic in a long chat |
| Grounding check  | a summary describing things its own quoted source never mentions               |
| Duplicate check  | the feedback loop. the system saving its own search answers back in            |
| Draft → confirm  | the AI saving without a human actually looking                                 |

The duplicate check only warns, it does not block. It shows the matching entry's title, URL and existing text so you can compare.

### Data model

Two named vectors per point, so one hybrid query can use both. Both writers emit an identical key set, so code reading the collection never branches on source.

| Payload key      | Official article                                 | Saved conversation           |
| ---------------- | ------------------------------------------------ | ---------------------------- |
| `text`           | the chunk (this is what gets embedded and shown) | same                         |
| `title`          | article title                                    | user-supplied title          |
| `url`            | source URL                                       | `""`                         |
| `article_id`     | Zendesk article id (numeric)                     | random UUID per conversation |
| `chunk_index`    | position in article                              | position in summary          |
| `section_id`     | Zendesk section                                  | `None`                       |
| `updated_at`     | article update time                              | save timestamp (ISO-8601)    |
| `source_type`    | `"official_article"`                             | `"chat_history"`             |
| `source_ai`      | `""`                                             | `"Claude"` / `"ChatGPT"` / …, `"unknown"` if the AI sent none |
| `applies_to`     | `""`                                             | e.g. `"Confluent Platform"`  |
| `source_excerpt` | —                                                | verbatim conversation        |
| `saved_by`       | —                                                | authenticated user's email   |

---

## 3. Setup on a new machine

### Prerequisites

- **Docker Desktop** (or Docker Engine) with Compose v2+.
- **8 GB RAM minimum**, 16 GB recommended.
- ~4 GB free disk for the image, models and vectors.

### Step 1 — Get the code and make a config file

```bash
git clone <your-repo-url> confluent-rag
cd confluent-rag
cp .env.example .env
```

The stack starts with `.env` left exactly as copied. Be aware that the knowledge base is **not** in the repository, so a fresh clone starts with an empty collection and every search returns nothing until Step 3 is done. Fill in what you need:

| Want                                 | Set                                                                                             |
| ------------------------------------ | ----------------------------------------------------------------------------------------------- |
| Just to start the stack              | *(no changes needed)*                                                                           |
| To build the knowledge base          | `SUPPORT_COOKIE`. The other two scrape values ship pre-filled, see Step 3                       |
| Public HTTPS access                  | `NGROK_AUTHTOKEN`, `NGROK_DOMAIN`                                                               |
| Company-only sign-in                 | `MCP_AUTH_ENABLED=true` + all the `AZURE_...` vars + `PUBLIC_MCP_URL` + `ALLOWED_EMAIL_DOMAINS` |
| Cloud inference in `orchestrator.py` | `OLLAMA_API_KEY`                                                                                |

### Step 2 — Start the stack

```bash
docker compose up -d
```

This starts five services: `qdrant`, `ollama`, `ollama_init`, `mcp_server`, `ngrok`.

The first boot will be downloading the embedding model (~640 MB) into the `ollama_data` volume and the BM25 + reranker weights (~1 GB) into `model_cache`. Both persist, so later starts take seconds.

```bash
docker compose ps
docker compose logs -f mcp_server
```

`mcp_server` should reach `(healthy)`. `ollama_init` showing `Exited (0)` is correct.

`ngrok` restarts in a loop until `NGROK_AUTHTOKEN` is set, logging `ERR_NGROK_4018`. That is expected on a fresh clone and affects nothing else. Set the token (see "Remote access"), or run `docker compose stop ngrok` if you only want local access.

### Step 3 — Load the knowledge base

The vectors live in the `qdrant_storage` Docker volume, which is local to one machine and **not** part of the repository. Neither is `ingest/fetched_docs.json`. A fresh clone therefore has to build the knowledge base before anything can be searched. If you're restoring a machine that already has that volume, skip this step.

Three values drive the scrape. `.env.example` ships the first two already filled in:

| Variable               | Value                                    |
| ---------------------- | ---------------------------------------- |
| `SUPPORT_BASE_URL`     | `https://support.confluent.io`           |
| `SUPPORT_CATEGORY_IDS` | `360004404872,360005831891,360005822052` |
| `SUPPORT_COOKIE`       | you must supply this, see below          |

Those three category IDs are Getting Started, Confluent Cloud and Confluent Platform. Blanking `SUPPORT_BASE_URL` does **not** fall back to a default: an empty value in `.env` overrides the fallback inside the script, and every request then fails as a URL with no protocol.

To build it from scratch:

```bash
# 1. Scrape the support portal, **needs** a valid SUPPORT_COOKIE in .env
docker compose run --rm ingestion python ingest/scrape_support.py

# 2. Embed and store
docker compose run --rm ingestion python ingest/embed_store.py
```

Always pass `--rm` these are one-off commands, not services, and without it you accumulate stopped containers.

`SUPPORT_COOKIE` is a browser session cookie copied from a logged-in Confluent support portal session, and it **expires**. Whoever sets this up needs portal access. If the scrape returns 401s or empty results, re-copy it from a logged-in browser. This only affects re-scraping; the server keeps serving whatever is already in Qdrant.

### Step 4 — Verify it worked

```bash
curl http://localhost:6333/collections/confluent_support
```

Inside the `result` object you should see `"status": "green"` and a non-zero `points_count`. A `404` saying the collection doesn't exist means Step 3 hasn't run yet, because the collection is created by `embed_store.py` and not at startup.

Then try a real question end to end:

```bash
docker compose run --rm -it orchestrator
```

This is an interactive terminal client. Ask something like *"why does my consumer group keep rebalancing?"*, you should see retrieved chunks followed by a generated answer. **It sends no auth headers**, so it only works with `MCP_AUTH_ENABLED=false`.

### Step 5 — Connect a real MCP client

Point your client at:

```
https://<your-ngrok-domain>/mcp    (remote)
```

---

## 4. Remote access

The server speaks plain HTTP on port 8000. Remote MCP clients need HTTPS, so something must terminate TLS in front of it. That is the tunnel's job.

A tunnel agent opens an **outbound** connection to the provider and holds it open, which is why this needs no port forwarding or firewall changes. A client hits `https://your-domain/mcp`, the request lands on the provider's edge which owns the certificate, so **TLS terminates there**, and the edge forwards it down the already-open connection to `mcp_server:8000`.

**The agent must run continuously.** That is normal for every tunnel, not an ngrok quirk. It runs as a compose service with `restart: unless-stopped`, so it comes back on its own after a reboot. The host machine must also be powered on with Docker running.

Docker alone cannot replace this. It can publish port 8000 on the host, but it cannot give you a public hostname or a TLS certificate.

**Setup:** put your authtoken and reserved static domain in `.env` (`NGROK_DOMAIN` is the hostname only, no `https://` and no `/mcp`), then `docker compose up -d`. ngrok starts with the stack.

**Verify.** Don't trust the container status, because ngrok logs almost nothing on success:

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://<your-domain>/mcp
```

`406` is the **correct** answer: it's the MCP server saying the client must accept `text/event-stream`. That proves TLS terminated, the tunnel routed, and the server replied. `401` means auth is on and working. `502` means the tunnel is up but can't reach the server. HTML means something else answered.

**Rate limiting.** `ngrok-policy.yml` applies 120 requests/min per IP at the edge. It is the only edge control that costs nothing in client compatibility, because edge *auth* would break MCP clients entirely.

**Cloudflare Tunnel** is wired up as an alternative behind a profile (`docker compose --profile cloudflare up -d`), which reads `CLOUDFLARE_TUNNEL_TOKEN` from `.env`. It needs a domain on Cloudflare but has no request quota, so it is the escape hatch if ngrok's 20k/month becomes a problem. Running both just gives you two public URLs onto the same server, so `docker compose stop ngrok` once you've switched.

---

## 5. Authentication

Optional and **off by default** (`MCP_AUTH_ENABLED=false`), so the stack starts and serves for someone who hasn't been given credentials.

**Auth must live inside the server, not at the tunnel edge.** MCP clients authenticate by exchanging tokens, not by completing a browser login. An edge login page returns a redirect the client can't follow, so it doesn't add auth, it makes the server unreachable.

The server is an OAuth 2.1 **resource server**: it never issues credentials, it only validates tokens Microsoft issued.

```
client → /mcp with no token        → 401 + pointer to /.well-known metadata      → registers, sends user to Microsoft Entra sign-in
Entra  → user signs in             → signed JWT (audience = this server)
client → /mcp + Bearer JWT
             │  
             │
             │
   AzureProvider   verifies signature · issuer · audience
             │
             │  token OK
             │
   EmailDomainMiddleware.on_message()
       _current_user_email() reads email / preferred_username / upn
       ALLOWED_EMAILS set?  → address must be on that exact list
       else                 → domain must be in ALLOWED_EMAIL_DOMAINS
             │
             │  allowed
             │
          tool runs
```

### Entra app registration (one time)

Azure portal → **Microsoft Entra ID** → **App registrations** → **New registration**.

1. **Register it SINGLE-TENANT.** This is a security requirement, not a preference. `preferred_username` is mutable, so with a *multi-tenant* registration a user in another Entra directory could set theirs to end in your domain and pass the email check. Single-tenant means only your own directory can obtain a token for this app.
2. **Redirect URI**, type **Web**, put in exactly `<PUBLIC_MCP_URL>/auth/callback`.
3. **Expose an API** → accept the default Application ID URI (`api://<client-id>`) → **Add a scope** named `access`, consentable by admins and users. This step is mandatory: Entra access tokens carry only *custom* API scopes in the `scp` claim, so the standard OIDC scopes can't be used and the server refuses to start without one.
4. **Manifest** → set `"requestedAccessTokenVersion": 2`. Without this Entra issues v1.0 tokens, whose issuer differs from what the server expects, and every request is rejected with an issuer mismatch. v2 also supplies the `preferred_username` claim the email check wants.
5. **Certificates & secrets** → New client secret. Copy the **value**, not the ID. It is shown only once.

### The four credentials

| Variable| What it's for|
| --- | --- |
| `AZURE_TENANT_ID`     | Which Entra directory may issue tokens. The outer gate: outsiders are rejected by Microsoft before reaching your server.|
| `AZURE_CLIENT_ID`     | Identifies this app. Public. Tokens are checked to have been issued *for this ID*, so one meant for another app can't be replayed.|
| `AZURE_CLIENT_SECRET` | The app's password, proving to Microsoft that requests claiming to be this app really are. Keep secret.|
| `MCP_JWT_SIGNING_KEY` | Not about Microsoft. Signs the server's *own* session tokens. Leave it blank and a random one is generated at every startup, logging everyone out on each restart.|

Generate the signing key once with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

### Who may use the tools

Two modes. Set one in `.env`:

| Mode              | Set                                              | Who gets in                          |
| ----------------- | ------------------------------------------------ | ------------------------------------ |
| **A, domain**     | `ALLOWED_EMAIL_DOMAINS=dev.co.th`               | anyone with an `@dev.co.th` account |
| **B, per person** | `ALLOWED_EMAILS=alice@dev.co.th,bob@dev.co.th` | only those exact addresses           |

When `ALLOWED_EMAILS` is non-empty it **wins** and `ALLOWED_EMAIL_DOMAINS` is ignored entirely, so a colleague on an allowed domain who isn't listed is denied. The two are deliberately **not** combined, which also lets you name an outside contractor (`dave@partner.co.th`) without opening their whole domain. Matching is case-insensitive and exact.

Leaving **both** blank while auth is on accepts any authenticated account. The server logs a warning at startup if you do.

Changing either value needs a restart: `docker compose up -d --force-recreate mcp_server`.

**Two layers, and the second is not redundant.** Single-tenant blocks accounts outside the directory. The middleware additionally covers **guest users invited into** the tenant, contractors and partners whose address may be `@gmail.com` or another company's domain. Entra would authenticate those happily. The check runs as middleware rather than per-tool, so any tool added later is protected by default.

---

## 6. Day-to-day operations

```bash
docker compose up -d                        # start
docker compose down                         # stop (volumes and data survive)
docker compose ps                           # what's running
docker compose logs -f mcp_server           # tail logs
docker compose down -v                      # DESTRUCTIVE: also wipes the knowledge base
```

### After changing code

```bash
docker compose up -d --build mcp_server
```

**`docker compose build` alone is not enough.** It rebuilds the image but leaves the old container running, so your changes silently don't take effect. To confirm the running container matches your source:

```bash
docker exec confluent-rag-mcp-server md5sum /app/mcp_server/qdrant_server.py
md5sum mcp_server/qdrant_server.py
```

### Other tasks

```bash
# Interactive test client (needs MCP_AUTH_ENABLED=false)
docker compose run --rm -it orchestrator

# Re-scrape and re-embed
docker compose run --rm ingestion python ingest/scrape_support.py
docker compose run --rm ingestion python ingest/embed_store.py
docker compose run --rm ingestion python ingest/embed_store.py --reset   # full rebuild

# Collection stats
curl http://localhost:6333/collections/confluent_support
```

### Volumes

| Volume           | Holds                   | Losing it means         |
| ---------------- | ----------------------- | ----------------------- |
| `qdrant_storage` | the knowledge base      | re-scrape and re-embed  |
| `ollama_data`    | the embedding model     | ~640 MB re-download     |
| `model_cache`    | BM25 + reranker weights | ~1 GB re-download       |
| `oauth_state`    | issued OAuth sessions   | everyone signs in again |

---

## 7. Configuration reference

Values marked **(docker)** are overridden by `docker-compose.yml`'s `environment:` block, so they only matter if you run the Python scripts directly on the host.

The Default column is what `.env.example` ships, which is what you get after Step 1. Note that a variable present but blank in `.env` overrides the fallback written into the Python, it does not fall back to it.

### Environment variables

| Variable                                         | Default                  | Notes                                                               |
| ------------------------------------------------ | ------------------------ | ------------------------------------------------------------------- |
| `EMBED_MODEL`                                    | `qwen3-embedding:0.6b`   | Dense embeddings. Always local.                                     |
| `OLLAMA_HOST`                                    | `http://127.0.0.1:11434` | **(docker)** → `http://ollama:11434`                                |
| `OLLAMA_API_KEY`                                 | *(blank)*                | Set, and `orchestrator.py` uses Ollama Cloud for chat.              |
| `INFERENCE_MODEL`                                | `gemma4:31b`             | `orchestrator.py` chat only.                                        |
| `QDRANT_HOST` / `QDRANT_PORT`                    | `localhost` / `6333`     | **(docker)** host → `qdrant`                                        |
| `QDRANT_COLLECTION`                              | `confluent_support`      |                                                                     |
| `MCP_HOST` / `MCP_PORT`                          | `127.0.0.1` / `8000`     | **(docker)** → `0.0.0.0`, which lets other containers reach it      |
| `FASTEMBED_CACHE_DIR`                            | `./model_cache`          | **(docker)** → the mounted volume                                   |
| `TOP_K_RESULTS`                                  | `5`                      | Default results per search                                          |
| `RERANK_MIN_SCORE`                               | `0`                      | **Raw cross-encoder logit, not a 0-1 score.** Negatives are normal, so this can legitimately go below 0 to return more. |
| `CHUNK_SIZE`                                     | `300`                    | Target words per chunk                                              |
| `MCP_AUTH_ENABLED`                               | `false`                  | Turns on Microsoft sign-in                                          |
| `AZURE_CLIENT_ID` / `_SECRET` / `_TENANT_ID`     | *(blank)*                | Required when auth is on                                            |
| `AZURE_API_SCOPE`                                | `access`                 | Must match the scope you exposed in Entra                           |
| `AZURE_TOKEN_ISSUER`                             | *(blank)*                | Only if the app still issues v1.0 tokens                            |
| `PUBLIC_MCP_URL`                                 | *(blank)*                | Public https URL. Must match the Entra redirect URI.                |
| `ALLOWED_EMAIL_DOMAINS`                          | *(blank)*                | Mode A                                                              |
| `ALLOWED_EMAILS`                                 | *(blank)*                | Mode B, wins over Mode A                                            |
| `MCP_JWT_SIGNING_KEY`                            | *(blank)*                | Set it, or restarts log everyone out                                |
| `NGROK_AUTHTOKEN` / `NGROK_DOMAIN`               | *(blank)*                | Domain is hostname only                                             |
| `CLOUDFLARE_TUNNEL_TOKEN`                        | *(blank)*                | Only for `--profile cloudflare`                                     |
| `SUPPORT_BASE_URL`                               | `https://support.confluent.io` | Scraping only. Must not be blank, see Step 3.                 |
| `SUPPORT_CATEGORY_IDS`                           | three IDs, see Step 3    | Scraping only. Comma-separated Zendesk category IDs.                |
| `SUPPORT_COOKIE`                                 | *(blank)*                | Scraping only. You supply it, and it expires.                       |

### Tuning constants

In `mcp_server/qdrant_server.py`, not environment variables.

| Constant                                           | Value           | Controls                                                                                                                                                                            |
| -------------------------------------------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MAX_TOP_K`                                        | `50`            | Ceiling on a caller's `top_k`. Rerank cost is linear in pool size.                                                                                                                  |
| `CANDIDATE_POOL_MULTIPLIER` / `MIN_CANDIDATE_POOL` | `4` / `15`      | Pool handed to the reranker: `max(top_k × 4, 15)`                                                                                                                                   |
| `PREFETCH_MULTIPLIER`                              | `4`             | Candidates each of dense/sparse contributes to fusion                                                                                                                               |
| `GROUNDING_REJECT_BELOW` / `_WARN_BELOW`           | `0.15` / `0.40` | Summary must overlap its cited excerpt. Measured: a faithful summary scored 58%, one from an unrelated topic scored 0%.                                                             |
| `DUPLICATE_STRONG_ABOVE` / `_WARN_ABOVE`           | `0.85` / `0.75` | Similarity at which the preview warns. **Advisory, never blocks.** Measured: 0.922 for a system saving its own answer, 0.684 for a genuinely new fix, 0.525 for an unrelated topic. |
| `MIN_EXCERPT_WORDS`                                | `20`            | Minimum quoted conversation                                                                                                                                                         |
| `MAX_SUMMARY_WORDS`                                | `1200`          | Backstop against a raw transcript being pasted in                                                                                                                                   |
| `MAX_EXCERPT_CHARS`                                | `8000`          | Excerpt is truncated to this before being stored                                                                                                                                    |
| `EXCERPT_PREVIEW_CHARS`                            | `1200`          | How much of the excerpt the step-1 preview shows                                                                                                                                    |
| `DUPLICATE_PREVIEW_CHARS`                          | `700`           | How much of a matching entry the duplicate warning shows                                                                                                                            |
| `DRAFT_TTL_SECONDS` / `MAX_PENDING_DRAFTS`         | `1800` / `50`   | Pending drafts are in memory only, lost on restart                                                                                                                                  |
| `SINGLE_CHUNK_LIMIT`                               | `400`           | Below this word count an entry stays one chunk                                                                                                                                      |

---

## 8. Known limits and risks

**Memory is the real constraint.** Measured on this stack:

| Stage                | RSS     |
| -------------------- | ------- |
| Models loaded, idle  | ~1.5 GB |
| Rerank 20 candidates | ~2.2 GB |
| Rerank 40 candidates | ~2.9 GB |

A single real search at the default `top_k=5` pushed one container to **6.2 GB** on an 8 GB host, after which its network calls began failing, and `OOMKilled=true` has been observed. Growth is roughly linear in candidate count, and `MAX_TOP_K=50` permits a 200-candidate pool which would extrapolate well past 8 GB. Consider a `mem_limit` on `mcp_server` so it fails predictably instead of starving the host, and lowering `MAX_TOP_K` if the full range isn't needed.

Practical consequence: **don't run test scripts via `docker exec` into the running server.** That loads a second copy of the models and gets OOM-killed, which surfaces as misleading "DNS resolution" errors. Use `docker compose run --rm` instead.

**A fabricated excerpt cannot be detected.** The grounding check compares the summary against the excerpt, so an invented pair is internally consistent and passes. The word floor only catches lazy padding. The human reviewing the preview is the last line of defence, which is why the preview shows the excerpt.

**ngrok free tier: 20,000 requests/month, 1 GB transfer.** MCP is chatty, each session costs several HTTP requests. That is roughly 660/day: fine for a couple of people, tight for a team. You get no warning, it just starts failing. The `cloudflare` profile is the way out.

**`orchestrator.py` sends no auth headers**, so it only works with `MCP_AUTH_ENABLED=false`. That is deliberate, since a bypass would be a second way in to secure and audit.

**Ollama Cloud retires models** with little notice, and a retired model fails at answer time with HTTP 410. Only `orchestrator.py` is affected because embeddings are local. List what's available and swap `INFERENCE_MODEL`:

```bash
curl -s -H "Authorization: Bearer $OLLAMA_API_KEY" https://ollama.com/api/tags
```

Being listed doesn't mean you can use it, many return 403 on the free tier.

**There are no automated tests.** Changes are verified by hand.

---

## 9. Troubleshooting

| Symptom                                                 | Cause and fix                                                                                                                            |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `401` on `/mcp`                                         | Auth is on and working. Expected for an unauthenticated request.                                                                         |
| Code changes have no effect                             | You ran `docker compose build` without `up -d`, so the old container is still running.                                                   |
| `AADSTS90072`                                           | The account isn't in your Entra directory. **This is the single-tenant gate working.**                                                   |
| Issuer mismatch in the server log                       | The Entra app still issues v1.0 tokens. Set `"requestedAccessTokenVersion": 2` in the manifest.                                          |
| `redirect_uri_mismatch`                                 | The Entra redirect URI isn't exactly `<PUBLIC_MCP_URL>/auth/callback`, type **Web**.                                                     |
| Server exits at startup naming a variable               | Auth is on but that credential is empty. It fails loudly rather than starting unprotected.                                               |
| `AzureProvider requires at least one non-OIDC scope`    | You skipped "Expose an API". Add a scope matching `AZURE_API_SCOPE`.                                                                     |
| "Access denied … restricted to …"                       | The account authenticated but isn't on the allowlist. Check `ALLOWED_EMAILS`.                                                            |
| Everyone logged out after a restart                     | `MCP_JWT_SIGNING_KEY` is blank, so a new one was generated. Set it.                                                                      |
| Tunnel up but `502`                                     | ngrok can't reach the server. Check `mcp_server` is `(healthy)`.                                                                         |
| ngrok container won't claim the domain                  | The free plan allows **one agent**. A host-side `ngrok.exe` is probably still running.                                                   |
| ngrok restarting, log shows `ERR_NGROK_4018`            | `NGROK_AUTHTOKEN` is blank. Set it, or `docker compose stop ngrok` if you only need local access.                                        |
| `404` from the collections endpoint                     | Step 3 hasn't run. `embed_store.py` creates the collection, startup does not.                                                            |
| Searches return "No relevant documents found"           | The collection is empty (run Step 3), or `RERANK_MIN_SCORE` is set too high.                                                             |
| Search errors mentioning Ollama or name resolution      | Usually memory exhaustion, not networking.                                                                                               |
| Scrape returns 401 or nothing                           | `SUPPORT_COOKIE` expired. Re-copy it from a logged-in browser.                                                                           |
| Scrape fails naming a URL with no protocol              | `SUPPORT_BASE_URL` is blank in `.env`. A blank value overrides the script's fallback rather than deferring to it.                        |
| `warning: The "h0" variable is not set` during a scrape | Compose interpolates `$` in `env_file` values, so `$h0` in the cookie expands to nothing. Escape `$` as `$$` if a scrape starts failing. |
| An edited article still returns old text                | The refresh compares `updated_at`. If the edit didn't move it, use `--reset`.                                                            |
| `mcp_server` never turns healthy on first boot          | Normal for several minutes while model weights download. `start_period` is 300s.                                                         |

---

## 10. Handoff and ownership

### Credentials that must be transferred

These are tied to individual accounts. If the current owner leaves, the service breaks or becomes unmanageable.
| Credential                        | Where it lives         | Risk if not transferred                                        |
| --------------------------------- | ---------------------- | -------------------------------------------------------------- |
| ngrok authtoken + reserved domain | personal ngrok account | Tunnel dies, and a new domain means reconfiguring every client |
| Entra app registration            | Azure tenant           | Nobody can rotate the secret or change the redirect URI        |
| Ollama Cloud API key              | personal ollama.com    | `orchestrator.py` chat stops, search unaffected                |
| `MCP_JWT_SIGNING_KEY`             | `.env` only            | Changing it logs everyone out                                  |

Also note `ALLOWED_EMAILS`: if it's set to one person, **only that person can use the tools.** Add the new owner before you hand over, or switch to `ALLOWED_EMAIL_DOMAINS`.

### Auditing what's been saved

Every saved conversation stores `source_excerpt` (the verbatim conversation it came from, on chunk 0) and `saved_by`. Excerpts are never embedded and never appear in search results, so they exist purely so a bad entry can be traced back.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

q = QdrantClient(host="localhost", port=6333)
pts, _ = q.scroll("confluent_support", limit=100, with_payload=True,
    scroll_filter=Filter(must=[FieldCondition(
        key="source_type", match=MatchValue(value="chat_history"))]))
for p in pts:
    print(p.payload["article_id"], "|", p.payload["saved_by"], "|", p.payload["title"])
```

### Removing a bad entry

A bad entry in a RAG store is self-perpetuating, it surfaces in future searches and shapes future answers. Delete by `article_id`:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, FilterSelector

q = QdrantClient(host="localhost", port=6333)
q.delete("confluent_support", points_selector=FilterSelector(filter=Filter(
    must=[FieldCondition(key="article_id", match=MatchValue(value="<article_id>"))])))
```

### Repository layout

```
mcp_server/qdrant_server.py   the MCP server: tools, search, save, auth
shared/chunking.py            chunk_article(), shared by ingestion and save
ingest/scrape_support.py      step 1, scrape and chunk to fetched_docs.json
ingest/embed_store.py         step 2, embed and upsert into Qdrant
rag/orchestrator.py           interactive terminal test client
docker-compose.yml            the whole stack
Dockerfile                    image shared by server and ingestion
requirements.txt              pinned Python dependencies
ngrok-policy.yml              edge rate limit
.env.example                  config template, copy to .env
.gitignore                    keeps .env, fetched_docs.json and caches out of git
.dockerignore                 keeps the same out of the image
README.md                     this file
```

`ingest/fetched_docs.json` and the Docker volumes are deliberately not in the repository, which is why Step 3 exists.
