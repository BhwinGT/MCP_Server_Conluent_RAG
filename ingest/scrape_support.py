"""
scrape_support.py
------------------
Scrapes Confluent's support portal (a Zendesk Help Center) via its public JSON API

For each configured category, downloads every article's HTML body, then
chunks it in a structure-aware way instead of blind word-count windows,
since support articles are short and heavily step/list structured.

API reference: https://developer.zendesk.com/api-reference/help_center/help-center-api/articles/

Output: ingest/fetched_docs.json
  A list of pre-chunked records:
    {
      "article_id": 35308470797716,
      "title": "How can I track how many ACLs...",
      "url": "https://support.confluent.io/hc/en-us/articles/...",
      "section_id": 203952818,
      "updated_at": "2025-11-03T10:15:00Z",
      "chunk_index": 0,
      "text": "# How can I track how many ACLs...\n\n## Solution\n\n..."
    }
"""

import os
import sys
import json
import time
from pathlib import Path

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.chunking import chunk_article

load_dotenv()

BASE_URL       = os.getenv("SUPPORT_BASE_URL", "https://support.confluent.io")
CATEGORY_IDS   = [c.strip() for c in os.getenv("SUPPORT_CATEGORY_IDS", "").split(",") if c.strip()]
SUPPORT_COOKIE = os.getenv("SUPPORT_COOKIE", "").strip()
OUTPUT_FILE    = Path("./ingest/fetched_docs.json")

# Chunking targets (in words)
CHUNK_TARGET_SIZE   = int(os.getenv("CHUNK_SIZE", "300"))
SINGLE_CHUNK_LIMIT  = 400   # articles shorter than this stay as one chunk

BLOCK_TAGS = {
    "h1": "heading", "h2": "heading", "h3": "heading",
    "h4": "heading", "h5": "heading", "h6": "heading",
    "p": "paragraph",
    "li": "list_item",
    "pre": "code",
    "table": "table",
}

HEADING_MARKDOWN = {
    "h1": "#", "h2": "##", "h3": "###",
    "h4": "####", "h5": "#####", "h6": "######",
}

# Zendesk API pagination
def fetch_articles_for_category(client: httpx.Client, category_id: str) -> list[dict]:
    """
    Follows Zendesk's `next_page` link until exhausted.
    Returns all articles.
    """
    articles = []
    url = f"{BASE_URL}/api/v2/help_center/en-us/categories/{category_id}/articles.json?per_page=100"

    while url:
        resp = client.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        articles.extend(data.get("articles", []))
        url = data.get("next_page")
        time.sleep(0.2)

    return articles

# Structure-aware chunking
def _table_to_markdown(table_el) -> str:
    """
    Converts an HTML table into a Markdown table.
    """
    rows = []
    for tr in table_el.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        rows.append([c.get_text(separator=" ", strip=True) for c in cells])

    rows = [r for r in rows if any(cell for cell in r)]
    if not rows:
        return ""

    col_count = max(len(r) for r in rows)
    header = rows[0] + [""] * (col_count - len(rows[0]))
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * col_count) + " |",
    ]
    for row in rows[1:]:
        row = row + [""] * (col_count - len(row))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)

def extract_blocks(html: str) -> list[tuple[str, str]]:
    """
    Parses article HTML into an ordered list of (block_type, text) tuples,
    using the author's own structure (headings, paragraphs, list items,
    code blocks, tables) as chunk boundaries.
    """
    soup = BeautifulSoup(html, "html.parser")
    blocks = []
    seen = set()

    for el in soup.find_all(list(BLOCK_TAGS.keys())):
        if id(el) in seen:
            continue

        if el.name == "p" and el.find_parent("li"):
            continue

        if el.name != "table" and el.find_parent("table"):
            continue

        block_type = BLOCK_TAGS[el.name]

        if block_type == "table":
            text = _table_to_markdown(el)
            if not text:
                continue

        elif block_type == "heading":
            heading_text = el.get_text(separator=" ", strip=True)
            if not heading_text:
                continue
            text = f"{HEADING_MARKDOWN[el.name]} {heading_text}"

        elif block_type == "code":
            code_text = el.get_text(separator=" ", strip=True)
            if not code_text:
                continue
            text = f"```\n{code_text}\n```"

        elif el.name == "li":
            text = el.get_text(separator=" ", strip=True)
            if not text:
                continue
            parent = el.find_parent(["ol", "ul"])
            if parent and parent.name == "ol":
                siblings = parent.find_all("li", recursive=False)
                try:
                    idx = siblings.index(el) + 1
                    text = f"{idx}. {text}"
                except ValueError:
                    text = f"- {text}"
            else:
                text = f"- {text}"

        else:
            text = el.get_text(separator=" ", strip=True)
            if not text:
                continue

        blocks.append((block_type, text))
        seen.add(id(el))

    if not blocks:
        full_text = soup.get_text(separator=" ", strip=True)
        if full_text:
            blocks = [("paragraph", full_text)]

    return blocks

# Main

def main():
    if not CATEGORY_IDS:
        print("Error: SUPPORT_CATEGORY_IDS is empty in .env")
        return

    print(f"Scraping categories: {CATEGORY_IDS}")

    headers = {"User-Agent": "confluent-rag-scraper/1.0"}
    if SUPPORT_COOKIE:
        headers["Cookie"] = SUPPORT_COOKIE
        print("Loaded browser session cookie -- restricted articles will be included.")
    else:
        print("No SUPPORT_COOKIE found in .env -- scraping anonymously (public articles only).")

    all_records = []

    with httpx.Client(headers=headers) as client:
        for category_id in CATEGORY_IDS:
            print(f"\nFetching articles for category {category_id} ...")
            articles = fetch_articles_for_category(client, category_id)
            print(f"  Found {len(articles)} articles")

            for article in articles:
                blocks = extract_blocks(article.get("body") or "")
                if not blocks:
                    continue

                title = article.get("title", "").strip()
                chunks = chunk_article(blocks, CHUNK_TARGET_SIZE, SINGLE_CHUNK_LIMIT)

                for i, chunk_text_val in enumerate(chunks):
                    full_text = f"# {title}\n\n{chunk_text_val}" if title else chunk_text_val

                    all_records.append({
                        "article_id": article["id"],
                        "title": title,
                        "url": article.get("html_url", ""),
                        "section_id": article.get("section_id"),
                        "updated_at": article.get("updated_at"),
                        "chunk_index": i,
                        "text": full_text,
                    })

    print(f"\nTotal chunks across all articles: {len(all_records)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()