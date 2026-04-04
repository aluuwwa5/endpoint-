"""Load markdown documents from knowledge_base/ directory."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path(__file__).parent.parent.parent / "knowledge_base"


def load_documents() -> list[dict]:
    """Load all .md files from knowledge_base/ and split into chunks.

    Returns:
        List of dicts with keys: text, source, chunk_id
    """
    documents = []
    chunk_size = 500  # characters per chunk
    chunk_overlap = 100

    if not KNOWLEDGE_DIR.exists():
        logger.warning("Knowledge base directory not found: %s", KNOWLEDGE_DIR)
        return documents

    for md_file in sorted(KNOWLEDGE_DIR.glob("*.md")):
        logger.info("Loading document: %s", md_file.name)
        content = md_file.read_text(encoding="utf-8")

        # Split by paragraphs first, then by chunk size
        paragraphs = content.split("\n\n")
        current_chunk = ""
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                documents.append({
                    "text": current_chunk.strip(),
                    "source": md_file.name,
                    "chunk_id": f"{md_file.stem}_{chunk_idx}",
                })
                # Keep overlap
                current_chunk = current_chunk[-chunk_overlap:] + "\n\n" + para
                chunk_idx += 1
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        # Don't forget the last chunk
        if current_chunk.strip():
            documents.append({
                "text": current_chunk.strip(),
                "source": md_file.name,
                "chunk_id": f"{md_file.stem}_{chunk_idx}",
            })

    logger.info("Loaded %d chunks from %d documents", len(documents), len(list(KNOWLEDGE_DIR.glob("*.md"))))
    return documents
