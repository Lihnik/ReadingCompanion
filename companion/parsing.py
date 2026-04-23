import io
import re

import fitz  # PyMuPDF

from .constants import MAX_CHUNK_CHARS


def split_long_page(page_text: str, page_num: int, base_index: int) -> list:
    if len(page_text) <= MAX_CHUNK_CHARS:
        return [{"index": base_index, "title": f"Page {page_num}", "text": page_text, "page": page_num}]

    paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
    sub_chunks = []
    current = []
    current_len = 0
    sub_num = 0

    for para in paragraphs:
        if current_len + len(para) > MAX_CHUNK_CHARS and current:
            sub_chunks.append({
                "index": base_index + sub_num,
                "title": f"Page {page_num}, Part {sub_num + 1}",
                "text": "\n\n".join(current),
                "page": page_num,
            })
            sub_num += 1
            current = [para]
            current_len = len(para)
        else:
            current.append(para)
            current_len += len(para)

    if current:
        sub_chunks.append({
            "index": base_index + sub_num,
            "title": f"Page {page_num}, Part {sub_num + 1}",
            "text": "\n\n".join(current),
            "page": page_num,
        })
    return sub_chunks


def parse_pdf(file_bytes: bytes) -> list:
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    chunks = []
    chunk_index = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_width = page.rect.width
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0]

        if not text_blocks:
            continue

        # Detect multi-column layout: in a two-column PDF each block is roughly
        # half the page wide, so the median block width falls well below 55% of
        # the page width.  Single-column text typically spans 70-90%.
        median_block_width = sorted(b[2] - b[0] for b in text_blocks)[len(text_blocks) // 2]
        if median_block_width < page_width * 0.55:
            # Multi-column: assign blocks to left/right column by comparing
            # each block's x-centre to the page midpoint, then sort within
            # each column top-to-bottom so the full left column is read before
            # the full right column.
            mid_x = page_width / 2
            text_blocks = sorted(
                text_blocks,
                key=lambda b: (0 if (b[0] + b[2]) / 2 < mid_x else 1, b[1]),
            )
        else:
            # Single column: simple top-to-bottom sort.
            text_blocks = sorted(text_blocks, key=lambda b: b[1])

        page_text = "\n\n".join(b[4].strip() for b in text_blocks if b[4].strip())

        if not page_text.strip():
            continue

        page_chunks = split_long_page(page_text, page_num + 1, chunk_index)
        chunks.extend(page_chunks)
        chunk_index += len(page_chunks)

    doc.close()
    return chunks


def parse_epub(file_bytes: bytes) -> list:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise RuntimeError("EPUB support requires beautifulsoup4: pip install beautifulsoup4")

    import zipfile
    import xml.etree.ElementTree as ET
    import posixpath

    zf = zipfile.ZipFile(io.BytesIO(file_bytes))

    # Locate the OPF package document via META-INF/container.xml
    container = ET.fromstring(zf.read("META-INF/container.xml"))
    opf_path = container.find(
        ".//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile"
    ).get("full-path")
    opf_dir = posixpath.dirname(opf_path)

    opf = ET.fromstring(zf.read(opf_path))
    OPF = "http://www.idpf.org/2007/opf"

    # Build manifest: item id → resolved ZIP path
    manifest = {}
    for item in opf.findall(f"{{{OPF}}}manifest/{{{OPF}}}item"):
        href = item.get("href")
        full = posixpath.normpath(posixpath.join(opf_dir, href)) if opf_dir else href
        manifest[item.get("id")] = full

    # Spine: reading order
    spine_paths = [
        manifest[ref.get("idref")]
        for ref in opf.findall(f"{{{OPF}}}spine/{{{OPF}}}itemref")
        if ref.get("idref") in manifest
    ]

    chunks = []
    chunk_index = 0
    section_num = 0

    for path in spine_paths:
        try:
            content = zf.read(path)
        except KeyError:
            continue

        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "nav"]):
            tag.decompose()
        text = soup.get_text(separator="\n\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)

        if len(text.strip()) < 50:
            continue

        section_num += 1
        heading = soup.find(["h1", "h2", "h3"])
        chapter_title = heading.get_text(strip=True) if heading else f"Section {section_num}"
        if len(chapter_title) > 60:
            chapter_title = chapter_title[:57] + "..."

        sub_chunks = split_long_page(text, section_num, chunk_index)
        for c in sub_chunks:
            c["title"] = (
                chapter_title if len(sub_chunks) == 1
                else f"{chapter_title}, Part {c['index'] - chunk_index + 1}"
            )
        chunks.extend(sub_chunks)
        chunk_index += len(sub_chunks)

    return chunks
