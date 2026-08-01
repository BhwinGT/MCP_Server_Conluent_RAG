"""
shared/chunking.py
-------------------
Structure-aware chunking, shared by:
  - ingest/scrape_support.py
  - mcp_server/qdrant_server.py
"""


def chunk_article(
    blocks: list[tuple[str, str]],
    chunk_target_size: int = 300,
    single_chunk_limit: int = 400,
) -> list[str]:
    all_text = "\n\n".join(b[1] for b in blocks)
    total_words = len(all_text.split())

    if total_words <= single_chunk_limit:
        return [all_text]

    chunks = []
    current_blocks: list[tuple[str, str]] = []
    current_words = 0
    running_heading: str | None = None
    chunk_start_heading: str | None = None

    def flush():
        text = "\n\n".join(b[1] for b in current_blocks)
        if chunk_start_heading and current_blocks[0][1] != chunk_start_heading:
            text = f"{chunk_start_heading}\n\n{text}"
        chunks.append(text)

    for block_type, text in blocks:
        block_words = len(text.split())

        if current_blocks and current_words + block_words > chunk_target_size:
            flush()
            current_blocks = [current_blocks[-1]]
            current_words = len(current_blocks[-1][1].split())
            chunk_start_heading = running_heading

        if block_type == "heading":
            running_heading = text
            if chunk_start_heading is None:
                chunk_start_heading = text

        current_blocks.append((block_type, text))
        current_words += block_words

    if current_blocks:
        flush()

    return chunks
