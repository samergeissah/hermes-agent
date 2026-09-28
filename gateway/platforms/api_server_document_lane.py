"""Inbound-document text extraction for the api_server attachment lane.

WHY this seam exists: an uploaded DOCUMENT used to reach the session as a caption plus a filesystem
path (``[Document attached: <name> (<mime>)]`` / ``[Document attached at: <path>]``) and nothing
else. That is fine for a profile with shell/file tools and useless for every other one — the live
failure was a strict intake persona with no file tools answering "I have no way to read it" about a
one-line PDF it had in front of it. Images never had that problem: the gateway pre-analyses them
with the vision backend (``api_server_image_lane``) and puts the DESCRIPTION in the turn, so any
profile can answer from its own context with no tool call. This module is the document-side twin of
that: the text is extracted at ingest and rides in the turn beside the existing path handle.

Design rules (they are the reason this is a separate module and not inline code in the upload
handler):

* **Best-effort, never fatal.** An upload that used to run a turn must keep running one. Every
  extractor failure becomes a plain sentence in the turn, never an exception, never a 5xx.
* **Honest.** A model must never believe it read a document it never saw. Unsupported type,
  extraction failure, an empty text layer (a scan), an oversize file and the truncation cut all say
  so in the turn, in the document's own line.
* **Stdlib only.** The stage venv ships runtime deps only (no pymupdf/pdfminer/pypdf, no
  ``python-docx/openpyxl/olefile``, no ``pdftotext``/``mutool``/``qpdf`` on the host), so the
  extractors are ``zlib``/``zipfile``/``xml``/``re``/``struct`` level: PDF content streams
  (``FlateDecode`` included), the OOXML packages (docx/xlsx/pptx — they are zips with XML inside),
  ODF/ePub, RTF, plain text, ZIP archives (read and walked), and the legacy OLE Office binaries
  (.doc/.xls/.ppt) read straight off the Compound File Binary container with ``struct`` — the
  WordDocument/Workbook/PowerPoint Document streams, no ``olefile``, no converter binary.
  A hand-pinned third-party dependency was deliberately NOT added: this is hand-patched staging
  code that a stage promotion replaces, and a new runtime dependency would be the thing that breaks
  the gateway on the next release, for a capability the stdlib already covers.
* **Bounded.** Text is capped at ``MAX_EXTRACT_CHARS`` (truncation is labelled in the turn), the
  source is skipped above ``MAX_SOURCE_BYTES``, and decompression is capped per stream so a
  compressed bomb cannot wedge the gateway. An uploaded ARCHIVE is bounded separately because it can
  nest: ``MAX_ARCHIVE_ENTRIES`` entries, ``MAX_ARCHIVE_DEPTH`` levels of nested zip,
  ``MAX_ARCHIVE_BYTES`` of member bytes and ``MAX_ARCHIVE_TEXT_CHARS`` of text **per upload**, and
  the limits it hit are stated in the turn (an archive must not be able to make the gateway read
  unbounded data by nesting zips inside zips).
* **Best-effort legacy binaries, honest failures.** A .doc/.xls/.ppt is a real file format with an
  encrypted/complex corner this reader does not cover (Word's piece table for an encrypted or
  Fast-Saved document, BIFF5-era cells, exotic PowerPoint records). Every such file therefore ends
  either with its recovered text or with an explicit "text could not be extracted from this
  application/msword" line — never with silence and never with invented text.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import struct
import zlib
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("gateway.platforms.api_server")


class DocumentUnreadable(ValueError):
    """The upload is not a readable document of its declared type (damaged, or mislabelled).

    Distinct from a generic extractor exception so the turn can say *what* is wrong with the file
    ("the DOCX package could not be opened") instead of a bare "extraction failed".
    """


# Cap on the extracted text that reaches the model's context (~10k tokens).
MAX_EXTRACT_CHARS = 40_000
# Above this the upload is not parsed at all; the turn says so instead (a chat attachment).
MAX_SOURCE_BYTES = 20 * 1024 * 1024
# Decompression caps: one PDF stream / one zip entry / one whole zip. Guards against a bomb.
MAX_STREAM_BYTES = 8 * 1024 * 1024
MAX_ZIP_ENTRY_BYTES = 16 * 1024 * 1024

# Caps for an uploaded ARCHIVE (``.zip``). They are per UPLOAD, not per archive, because an archive
# can hold archives: without a depth limit a zip inside a zip inside a zip recurses until the
# process runs out of something. Every one of these is stated in the turn, so a model reading a
# partly-read archive knows what it did not see.
MAX_ARCHIVE_ENTRIES = 200               # entries actually read, at every nesting level
MAX_ARCHIVE_DEPTH = 2                   # nested zip levels inside the uploaded archive (1 = no nest)
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024    # member bytes read (decompressed), whole upload
MAX_ARCHIVE_TEXT_CHARS = MAX_EXTRACT_CHARS   # text collected from an archive, whole upload

# Plain-text documents, by MIME. Every other ``text/*`` is treated as text too (see
# ``_is_text_mime``); these are the ``application/*`` spellings the accepted-type table carries.
_TEXT_APPLICATION_MIMES = frozenset({
    "application/json", "application/xml", "application/x-yaml", "application/yaml",
    "application/toml", "application/x-toml",
})

_OOXML_WORD = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_OOXML_SHEET = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_OOXML_DECK = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
_ODF_TEXT = "application/vnd.oasis.opendocument.text"
_ODF_SHEET = "application/vnd.oasis.opendocument.spreadsheet"
_ODF_DECK = "application/vnd.oasis.opendocument.presentation"
_EPUB_MIME = "application/epub+zip"

# Legacy OLE (Compound File Binary) Office binaries and generic archives: read here rather than
# refused, see ``_ole_*`` and ``_zip_text`` below. ``application/x-zip-compressed`` is the spelling
# Windows clients send for the same container, so both resolve to the archive reader.
_DOC_MIME = "application/msword"
_XLS_MIME = "application/vnd.ms-excel"
_PPT_MIME = "application/vnd.ms-powerpoint"
_ZIP_MIMES = ("application/zip", "application/x-zip-compressed")

_EXT_MIMES = {
    ".pdf": "application/pdf",
    ".docx": _OOXML_WORD,
    ".xlsx": _OOXML_SHEET,
    ".pptx": _OOXML_DECK,
    ".doc": _DOC_MIME,
    ".xls": _XLS_MIME,
    ".ppt": _PPT_MIME,
    ".zip": "application/zip",
    ".odt": _ODF_TEXT,
    ".ods": _ODF_SHEET,
    ".odp": _ODF_DECK,
    ".epub": _EPUB_MIME,
    ".rtf": "application/rtf",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
    ".xml": "application/xml",
    ".yaml": "application/x-yaml",
    ".yml": "application/x-yaml",
    ".log": "text/plain",
}


# --------------------------------------------------------------------------------------------- #
# Plain text and encoded text
# --------------------------------------------------------------------------------------------- #

def _decode_text(raw: bytes, limit: int) -> str:
    """Decode a text payload as best we can: BOM first, then utf-8, then latin-1 (never fails)."""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or (b"\x00" in raw[:64] and len(raw) > 32):
        for enc in ("utf-16", "utf-16-le", "utf-16-be"):
            try:
                return raw.decode(enc, errors="strict")[:limit]
            except (UnicodeDecodeError, UnicodeError):
                continue
    try:
        return raw.decode("utf-8-sig", errors="strict")[:limit]
    except UnicodeDecodeError:
        return raw.decode("latin-1", errors="replace")[:limit]


def _is_text_mime(mime: str) -> bool:
    return mime.startswith("text/") or mime in _TEXT_APPLICATION_MIMES


def _extract_text(raw: bytes) -> str:
    # Decode the whole declared-text payload (the source is already capped at MAX_SOURCE_BYTES that
    # way) so the note can report an EXACT truncated-at count instead of "some more characters".
    return _decode_text(raw, MAX_SOURCE_BYTES)


# --------------------------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------------------------- #

_PDF_STREAM_RE = re.compile(rb"<<(.*?)>>[ \t\r\n]*stream(\r\n|\n|\r)", re.S)
_PDF_FILTER_RE = re.compile(rb"/Filter\s*(\[[^\]]*\]|/[A-Za-z0-9]+)")
_PDF_LENGTH_RE = re.compile(rb"/Length\s+(\d+)")


def _pdf_apply_filters(dict_bytes: bytes, body: bytes) -> Optional[bytes]:
    """Decode one stream body using the ``/Filter`` chain in its dictionary; ``None`` if unknown.

    Only the lossless byte filters a real PDF puts on a text stream are handled: ``FlateDecode``
    (zlib — what every pdflatex/Word/Chrome PDF uses), ``ASCIIHexDecode`` and ``ASCII85Decode``. An
    image stream or an unsupported filter (DCTDecode/LZWDecode/CCITTFax) returns ``None`` and is
    skipped: this is a text extractor, not a renderer.
    """
    match = _PDF_FILTER_RE.search(dict_bytes)
    if not match:
        return body
    names = [n.lstrip(b"/") for n in re.findall(rb"/([A-Za-z0-9]+)", match.group(1))]
    data = body
    for name in names:
        if name in (b"FlateDecode", b"Fl"):
            try:
                data = zlib.decompressobj().decompress(data, MAX_STREAM_BYTES)
            except zlib.error:
                # Tolerate a truncated/odd stream: keep whatever inflates.
                ok = False
                for wbits in (15, -15, 47):
                    try:
                        data = zlib.decompressobj(wbits).decompress(data, MAX_STREAM_BYTES)
                        ok = True
                        break
                    except zlib.error:
                        continue
                if not ok:
                    return None
        elif name in (b"ASCIIHexDecode", b"AHx"):
            try:
                hexed = re.sub(rb"[^0-9A-Fa-f]", b"", data.split(b">")[0])
                data = bytes.fromhex(hexed.decode("ascii"))[:MAX_STREAM_BYTES]
            except (ValueError, UnicodeDecodeError):
                return None
        elif name in (b"ASCII85Decode", b"A85"):
            chunk = bytes(b for b in data.split(b"~>")[0] if b > 32)
            try:
                data = base64.a85decode(chunk, adobe=False)[:MAX_STREAM_BYTES]
            except (ValueError, TypeError):
                return None
        else:
            return None
    return data


def _pdf_bytes_to_text(data: bytes) -> str:
    """One PDF string as text: UTF-16 when it carries a BOM, else PDFDocEncoding≈latin-1."""
    if data[:2] in (b"\xfe\xff", b"\xff\xfe"):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass
    return data.decode("latin-1", errors="replace")


def _pdf_strings(data: bytes) -> str:
    """The text a PDF content stream draws, in reading order — ``Tj``/``TJ``/``'``/``"`` operands.

    A content stream is a small postfix language, so this is a tiny tokenizer rather than a byte
    grep for ``Tj``: literal strings ``( … )`` (with the escapes and balanced parens the spec
    allows), hex strings ``< … >``, kerning numbers and the positioning operators. ``Td``/``TD``/
    ``T*``/``BT``/``'``/``"`` start a new line, which is what makes a multi-line document readable
    instead of one run-together blob, and a large negative kerning jump is kept as a word gap.
    """
    out: List[str] = []
    i, n = 0, len(data)
    pending_text = False  # text has been emitted on the current line

    def emit(text: str) -> None:
        nonlocal pending_text
        if text:
            out.append(text)
            pending_text = True

    def newline() -> None:
        nonlocal pending_text
        if out and not out[-1].endswith("\n"):
            out.append("\n")
        pending_text = False

    while i < n:
        ch = data[i]
        if ch == 0x28:  # ( literal string
            depth, j, buf = 1, i + 1, bytearray()
            while j < n and depth:
                c = data[j]
                if c == 0x5C and j + 1 < n:  # backslash escape
                    nxt = data[j + 1]
                    mapping = {0x6E: 0x0A, 0x72: 0x0D, 0x74: 0x09, 0x62: 0x08, 0x66: 0x0C}
                    if nxt in mapping:
                        buf.append(mapping[nxt]); j += 2; continue
                    if 0x30 <= nxt <= 0x37:  # up to 3 octal digits
                        k, digits = j + 1, b""
                        while k < n and len(digits) < 3 and 0x30 <= data[k] <= 0x37:
                            digits += bytes([data[k]]); k += 1
                        buf.append(int(digits, 8) & 0xFF); j = k; continue
                    buf.append(nxt); j += 2; continue
                if c == 0x28:
                    depth += 1
                elif c == 0x29:
                    depth -= 1
                    if not depth:
                        break
                buf.append(c); j += 1
            emit(_pdf_bytes_to_text(bytes(buf)))
            i = j + 1
            continue
        if ch == 0x3C:  # < hex string > or << dictionary >>
            if i + 1 < n and data[i + 1] == 0x3C:
                j = data.find(b">>", i + 2)
                i = n if j == -1 else j + 2
                continue
            j = data.find(b">", i + 1)
            if j == -1:
                break
            hexed = re.sub(rb"[^0-9A-Fa-f]", b"", data[i + 1:j])
            if len(hexed) % 2:
                hexed += b"0"
            try:
                emit(_pdf_bytes_to_text(bytes.fromhex(hexed.decode("ascii"))))
            except (ValueError, UnicodeDecodeError):
                pass
            i = j + 1
            continue
        if 0x41 <= ch <= 0x5A or 0x61 <= ch <= 0x7A:  # an operator token
            j = i
            while j < n and (0x41 <= data[j] <= 0x5A or 0x61 <= data[j] <= 0x7A):
                j += 1
            token = data[i:j]
            if token in (b"Td", b"TD", b"T*", b"BJ", b"BT", b"ET"):
                newline()
            i = j
            continue
        if ch == 0x27 or ch == 0x22:  # ' and " also draw a string line
            newline()
            i += 1
            continue
        if 0x30 <= ch <= 0x39 or ch in (0x2B, 0x2D, 0x2E):  # numbers: kerning offsets, coords
            j = i
            while j < n and (data[j:j + 1].isdigit() or data[j:j + 1] in (b".", b"-", b"+")):
                j += 1
            try:
                value = float(data[i:j])
            except ValueError:
                value = 0.0
            if pending_text and value <= -120:
                out.append(" ")
                pending_text = False
            i = j if j > i else i + 1  # never stall: an unparsable token still has to advance
            continue
        i += 1

    text = "".join(out)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _printable_ratio(text: str) -> float:
    """Share of a decoded stream that is readable text (printables + whitespace)."""
    if not text:
        return 0.0
    good = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t ")
    return good / len(text)


def _pdf_text(raw: bytes) -> str:
    """Text of every decodable content stream in the file (file order = page order, best-effort)."""
    chunks: List[str] = []
    stream_count = 0
    for match in _PDF_STREAM_RE.finditer(raw):
        header, eol = match.group(1), match.group(2)
        start = match.end()
        stream_count += 1
        body: Optional[bytes] = None
        length_m = _PDF_LENGTH_RE.search(header)
        if length_m:
            declared = int(length_m.group(1))
            if 0 < declared <= len(raw) - start:
                body = raw[start:start + declared]
        if body is None:
            end = raw.find(b"endstream", start)
            if end == -1:
                continue
            body = raw[start:end]
            if body.endswith(eol):
                body = body[:-len(eol)]
            elif body[-1:] in (b"\r", b"\n"):
                body = body[:-1]
        decoded = _pdf_apply_filters(header, body)
        if decoded is None:
            continue
        text = _pdf_strings(decoded)
        # A stream that "decodes" into mostly binary is an embedded font/program, not page text:
        # keeping it would flood the turn and could push the real content past the cap.
        if text.strip() and _printable_ratio(text) >= 0.75:
            chunks.append(text.strip())
    if not stream_count:
        raise DocumentUnreadable(
            "no readable PDF stream was found — the file is damaged or is not really a PDF")
    return "\n".join(chunks)


# --------------------------------------------------------------------------------------------- #
# Zip-container office documents (docx / xlsx / pptx / odt / ods / odp / epub)
# --------------------------------------------------------------------------------------------- #

_DOCX_PARAGRAPH_RE = re.compile(rb"<w:p[ >].*?</w:p>|<w:p\s*/>", re.S)
_DOCX_TEXT_RE = re.compile(rb"<w:t[^>]*>(.*?)</w:t>", re.S)
_XML_ENTITY = {b"&amp;": b"&", b"&lt;": b"<", b"&gt;": b">", b"&quot;": b'"', b"&apos;": b"'"}


def _unescape_xml(data: bytes) -> bytes:
    for entity, char in _XML_ENTITY.items():
        data = data.replace(entity, char)
    return data


def _read_zip_members(raw: bytes, want: Callable[[str], bool], limit: int,
                      label: str) -> List[Tuple[str, bytes]]:
    """``[(name, bytes)]`` for the zip members ``want`` accepts, capped; raises when unopenable."""
    import io
    import zipfile

    found: List[Tuple[str, bytes]] = []
    total = 0
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except (zipfile.BadZipFile, OSError, ValueError, EOFError) as exc:
        logger.warning("[api_server] %s package is not a readable zip container: %s", label, exc)
        raise DocumentUnreadable(
            f"this file is not a readable {label} package — its zip container is damaged or it is "
            f"not really a {label} file") from exc
    with archive:
        for info in archive.infolist():
            if info.is_dir() or not want(info.filename):
                continue
            if info.file_size > MAX_ZIP_ENTRY_BYTES or total + info.file_size > limit:
                continue
            try:
                data = archive.read(info)
            except (zipfile.BadZipFile, RuntimeError, OSError, ValueError):
                continue
            found.append((info.filename, data))
            total += len(data)
    return found


def _docx_text(raw: bytes) -> str:
    """Paragraph-per-line text of a WordprocessingML package (body, headers/footers, footnotes)."""
    def want(name: str) -> bool:
        return (name == "word/document.xml"
                or name.startswith(("word/header", "word/footer", "word/footnotes",
                                    "word/endnotes")))

    members = _read_zip_members(raw, want, MAX_STREAM_BYTES, "DOCX")
    if not members:
        raise DocumentUnreadable("the DOCX package has no readable document part")
    parts: List[str] = []
    for _, data in members:
        for para in _DOCX_PARAGRAPH_RE.findall(data) or [data]:
            line = "".join(_unescape_xml(t).decode("utf-8", errors="replace")
                           for t in _DOCX_TEXT_RE.findall(para)).strip()
            if line:
                parts.append(line)
    return "\n".join(parts)


def _xlsx_text(raw: bytes) -> str:
    """Sheet-by-sheet rows of a SpreadsheetML package, tab-separated, shared strings resolved."""
    import xml.etree.ElementTree as ET

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    shared: List[str] = []
    for _, data in _read_zip_members(raw, lambda n: n == "xl/sharedStrings.xml",
                                     MAX_STREAM_BYTES, "XLSX"):
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            continue
        for si in root.iter(f"{ns}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{ns}t")))
    parts: List[str] = []
    sheets = _read_zip_members(
        raw, lambda n: n.startswith("xl/worksheets/sheet") and n.endswith(".xml"),
        MAX_STREAM_BYTES, "XLSX")
    if not sheets and not shared:
        raise DocumentUnreadable("the XLSX package has no readable worksheet")
    for name, data in sorted(sheets):
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            continue
        parts.append(f"[{name.rsplit('/', 1)[-1]}]")
        for row in root.iter(f"{ns}row"):
            cells: List[str] = []
            for cell in row.iter(f"{ns}c"):
                kind = cell.get("t")
                value = cell.find(f"{ns}v")
                if kind == "s" and value is not None and (value.text or "").isdigit():
                    idx = int(value.text)
                    cells.append(shared[idx] if 0 <= idx < len(shared) else "")
                elif kind == "inlineStr":
                    cells.append("".join(t.text or "" for t in cell.iter(f"{ns}t")))
                else:
                    cells.append(value.text if value is not None and value.text else "")
            if any(c.strip() for c in cells):
                parts.append("\t".join(cells).rstrip("\t"))
    return "\n".join(parts)


def _pptx_text(raw: bytes) -> str:
    """Slide-by-slide text of a PresentationML package."""
    import xml.etree.ElementTree as ET

    ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
    parts: List[str] = []
    slides = _read_zip_members(
        raw, lambda n: n.startswith("ppt/slides/slide") and n.endswith(".xml"),
        MAX_STREAM_BYTES, "PPTX")
    if not slides:
        raise DocumentUnreadable("the PPTX package has no readable slide")
    for name, data in sorted(slides):
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            continue
        lines = [(t.text or "").strip() for t in root.iter(f"{ns}t")]
        lines = [ln for ln in lines if ln]
        if lines:
            parts.append(f"[{name.rsplit('/', 1)[-1]}] " + " | ".join(lines))
    return "\n".join(parts)


def _markup_text(raw: bytes) -> str:
    """Visible text of an XML/HTML-ish document: tags stripped, block ends kept as newlines."""
    text = re.sub(rb"<[^>]*>", b"\n", raw)
    text = _unescape_xml(text).decode("utf-8", errors="replace")
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def _odf_text(raw: bytes) -> str:
    """ODF documents: the text runs of ``content.xml`` (documents, and cells for sheets)."""
    members = _read_zip_members(raw, lambda n: n == "content.xml", MAX_STREAM_BYTES, "ODF")
    if not members:
        raise DocumentUnreadable("the ODF package has no readable content.xml")
    parts = [_markup_text(data) for _, data in members]
    return "\n".join(p for p in parts if p.strip())


def _epub_text(raw: bytes) -> str:
    """ePub: the readable text of every (x)html member, in package order."""
    members = _read_zip_members(
        raw, lambda n: n.lower().endswith((".xhtml", ".html", ".htm")), MAX_STREAM_BYTES, "ePub")
    if not members:
        raise DocumentUnreadable("the ePub package has no readable (x)html member")
    parts: List[str] = []
    for _, data in members:
        text = _markup_text(data)
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def _rtf_text(raw: bytes) -> str:
    """RTF text: control words/markup dropped, ``\\'xx`` hex escapes and ``\\uNNNN`` decoded.

    ``striprtf`` is not installed, and an RTF is small and regular enough to read directly.
    """
    data = re.sub(rb"\{\\(?:fonttbl|colortbl|stylesheet|info|pict|header|footer|footnote)"
                  rb"[^{}]*\}", b"", raw)
    data = re.sub(rb"\\u(-?\d+)\s*\??", lambda m: chr(int(m.group(1)) % 65536).encode("utf-8"), data)
    data = re.sub(rb"\\'([0-9a-fA-F]{2})", lambda m: bytes([int(m.group(1), 16)]), data)
    data = re.sub(rb"\\par[d]?\b", b"\n", data)
    data = re.sub(rb"\\line\b", b"\n", data)
    data = re.sub(rb"\\tab\b", b"\t", data)
    data = re.sub(rb"\{\\\*[^{}]*\}", b"", data)
    data = re.sub(rb"\\[a-zA-Z]+-?\d*[ ]?", b"", data)
    data = data.replace(b"{", b"").replace(b"}", b"")
    text = _decode_text(data, MAX_SOURCE_BYTES)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return re.sub(r"\n{2,}", "\n\n", text).strip()


# --------------------------------------------------------------------------------------------- #
# Zip archives (``.zip`` — one zip container, walked member by member, bounded)
# --------------------------------------------------------------------------------------------- #

_ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


class _ArchiveBudget:
    """What ONE uploaded archive has cost so far, at every nesting level.

    A single budget object is threaded through the recursive walk instead of per-archive counters,
    because a per-level limit is not a limit: 200 entries at each of N levels is 200*N entries. This
    is the zip-bomb guard for ``.zip`` the way ``MAX_STREAM_BYTES`` is for a single PDF stream.
    """

    __slots__ = ("entries", "bytes_read", "chars", "skipped_count", "skipped_size", "exhausted")

    def __init__(self) -> None:
        self.entries = 0          # non-directory entries actually read
        self.bytes_read = 0       # decompressed member bytes read
        self.chars = 0            # text collected from those members
        self.skipped_count = 0    # entries dropped: unreadable, unsupported type, or a cap was hit
        self.skipped_size = 0     # entries dropped for exceeding the per-entry size cap
        self.exhausted = False    # a cap has been hit; nothing further is read


def _archive_limits_line() -> str:
    """The caps an archive upload runs under, IN THE TURN — a partly-read archive must say so."""
    return ("[Archive limits for this upload: at most {entries} entries, {depth} nested zip "
            "level(s), {bytes} of member content and {chars} characters of text are read.]".format(
                entries=MAX_ARCHIVE_ENTRIES, depth=MAX_ARCHIVE_DEPTH,
                bytes=_human_size(MAX_ARCHIVE_BYTES), chars=MAX_ARCHIVE_TEXT_CHARS))


def _looks_like_zip(data: bytes, name: str) -> bool:
    return name.lower().endswith(".zip") or data[:4] in _ZIP_MAGICS


def _looks_like_text(data: bytes) -> bool:
    """Share of the first 512 bytes that is printable — used ONLY to guess an extensionless member."""
    sample = data[:512]
    if not sample:
        return False
    good = sum(1 for byte in sample if 32 <= byte < 0x7F or byte in (9, 10, 13))
    return good / len(sample) >= 0.9


def _archive_member_text(name: str, data: bytes) -> Optional[str]:
    """Text of one archive member, or ``None`` when no reader applies to its type.

    The member's type comes from its own filename extension (an archive entry never carries a MIME
    type of its own), with a PDF-signature and a plain-text sniff for the extensionless case.
    """
    member_mime = _resolve_mime("", name)
    if member_mime in _ZIP_MIMES:
        return None                                   # the caller recurses, with the shared budget
    reader = _EXTRACTORS.get(member_mime)
    if reader is not None:
        return reader(data)
    if _is_text_mime(member_mime) or (not member_mime and _looks_like_text(data)):
        return _extract_text(data)
    if data[:5] == b"%PDF-":
        return _pdf_text(data)
    return None


def _archive_member(name: str, data: bytes, depth: int, budget: _ArchiveBudget) -> Optional[str]:
    """One member's rendered contribution (``None`` when its type is unsupported — the caller counts).

    Order of evidence, and it matters: a member whose own name says it is an office package (a
    ``.docx`` inside a ``.zip``) is read by ITS reader — the paragraph structure is the useful part
    and its reader is the one that knows where the text lives — and only a member that is a zip
    without a package type (``.zip``, or a zip with an unknown extension) is walked recursively.
    """
    indent = "  " * (depth + 1)
    member_mime = _resolve_mime("", name)
    package_reader = _EXTRACTORS.get(member_mime) if member_mime not in _ZIP_MIMES else None
    if package_reader is None and _looks_like_zip(data, name):
        if depth + 1 > MAX_ARCHIVE_DEPTH:
            marker = (f"{indent}--- {name} (nested zip) ---\n{indent}[not opened: the "
                      f"{MAX_ARCHIVE_DEPTH}-level nested-zip limit for this upload was reached.]")
            budget.chars += len(marker)      # it is text in the turn: it makes the note non-empty
            return marker
        try:
            nested = _zip_text(data, depth=depth + 1, budget=budget)
        except DocumentUnreadable as exc:
            budget.chars += 64
            return f"{indent}--- {name} (nested zip) ---\n{indent}[not extracted: {exc}]"
        except Exception as exc:  # noqa: BLE001 — one odd member must not lose the whole archive
            logger.warning("[api_server] nested zip member %s failed: %s", name, exc)
            budget.chars += 64
            return (f"{indent}--- {name} (nested zip) ---\n"
                    f"{indent}[not extracted: {type(exc).__name__}]")
        return f"{indent}--- {name} (nested zip) ---\n{nested}"
    heading = f"{indent}--- {name} ({member_mime or 'no declared type'}) ---"
    try:
        text = package_reader(data) if package_reader is not None \
            else _archive_member_text(name, data)
    except DocumentUnreadable as exc:
        budget.skipped_count += 1
        budget.chars += 64
        return f"{heading}\n{indent}[not extracted: {exc}]"
    except Exception as exc:  # noqa: BLE001 — one odd member must not lose the whole archive
        budget.skipped_count += 1
        budget.chars += 64
        logger.warning("[api_server] archive member %s failed: %s", name, exc)
        return f"{heading}\n{indent}[not extracted: {type(exc).__name__}: {exc}]"[:2000]
    if text is None:
        budget.skipped_count += 1
        return None
    # A member that "reads" as a wall of NULs or binary noise is not text: the compressor turned
    # padding into a tiny entry, and putting 40 000 control characters in the turn helps no model.
    if not text.strip() or _printable_ratio(text) < 0.6:
        marker = f"{heading}\n{indent}[no extractable text in this entry]"
        budget.chars += len(marker)
        return marker
    room = MAX_ARCHIVE_TEXT_CHARS - budget.chars
    if len(text) > room:
        text = text[:max(room, 0)]
        budget.exhausted = True
    budget.chars += len(text)
    body = "\n".join(indent + line for line in text.splitlines())
    if room <= 0:
        return heading
    return f"{heading}\n{body}"


def _zip_text(raw: bytes, depth: int = 0, budget: Optional[_ArchiveBudget] = None) -> str:
    """Text of a zip archive's readable members — capped at every level (see ``MAX_ARCHIVE_*``).

    The caps in one sentence: entries/bytes/characters are counted for the WHOLE upload and the
    nesting depth is finite, so a zip bomb (many entries, a huge deflate ratio, or archives nested
    inside archives) can only ever make this read ``MAX_ARCHIVE_BYTES`` and stop. Every cap that was
    actually hit is stated in the text, so the model is never told it read an archive it did not.
    """
    import io
    import zipfile

    budget = budget if budget is not None else _ArchiveBudget()
    top_level = not depth
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except (zipfile.BadZipFile, OSError, ValueError, EOFError) as exc:
        logger.warning("[api_server] zip archive is not readable: %s", exc)
        raise DocumentUnreadable(
            "this file is not a readable zip archive — its container is damaged or it is not "
            "really a zip") from exc
    parts: List[str] = [_archive_limits_line()] if top_level else []
    with archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if not infos:
            raise DocumentUnreadable("the zip archive contains no entries")
        total_entries = len(infos)
        for info in infos:
            if budget.exhausted or budget.entries >= MAX_ARCHIVE_ENTRIES \
                    or budget.chars >= MAX_ARCHIVE_TEXT_CHARS:
                budget.exhausted = True
                budget.skipped_count += 1
                continue
            if info.file_size > MAX_ZIP_ENTRY_BYTES:
                budget.skipped_size += 1
                continue
            room = MAX_ARCHIVE_BYTES - budget.bytes_read
            try:
                with archive.open(info) as handle:
                    # A BOUNDED read: a member's declared size is attacker-controlled, so the
                    # decompressor is never asked for more than the remaining budget.
                    data = handle.read(min(MAX_ZIP_ENTRY_BYTES, room))
            except (zipfile.BadZipFile, RuntimeError, OSError, ValueError, EOFError,
                    NotImplementedError):
                budget.skipped_count += 1
                continue
            budget.entries += 1
            budget.bytes_read += len(data)
            member = _archive_member(info.filename, data, depth, budget)
            if member is not None:
                parts.append(member)
            if budget.exhausted or budget.chars >= MAX_ARCHIVE_TEXT_CHARS:
                budget.exhausted = True
    if top_level:
        if budget.skipped_size:
            parts.append("[%d archive entr%s skipped: larger than the %s per-entry limit.]"
                         % (budget.skipped_size, "y was" if budget.skipped_size == 1 else "ies were",
                            _human_size(MAX_ZIP_ENTRY_BYTES)))
        if budget.exhausted:
            parts.append("[The caps above were reached before the archive ended: %d of its %d "
                         "entr%s were read and the rest were not.]"
                         % (budget.entries, total_entries,
                            "y" if total_entries == 1 else "ies"))
        elif budget.skipped_count:
            parts.append("[%d archive entr%s skipped (unsupported type or unreadable member).]"
                         % (budget.skipped_count,
                            "y was" if budget.skipped_count == 1 else "ies were"))
    body = "\n\n".join(part for part in parts if part.strip())
    if top_level and not budget.chars:
        raise DocumentUnreadable(
            "the zip archive holds no extractable text: its %d entr%s carry no readable text "
            "document (or were skipped by the size limits)"
            % (len(infos), "y" if len(infos) == 1 else "ies"))
    return body


# --------------------------------------------------------------------------------------------- #
# Legacy OLE (Compound File Binary) Office binaries: .doc / .xls / .ppt
#
# The container is the same for all three (and for MSG/VSD/etc.): a 512-byte header naming a FAT,
# 128-byte directory entries, and a "mini" stream for anything below the 4096-byte cutoff. Only the
# STREAM inside differs, so the reader is written once and the three format readers sit on top.
# ``olefile`` is not installed on this host and is not going to be pinned into hand-patched staging
# code (see the module docstring); ~120 lines of ``struct`` replaces it.
# --------------------------------------------------------------------------------------------- #

_CFB_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_CFB_FREE = 0xFFFFFFFF
_CFB_END = 0xFFFFFFFE
_CFB_MAX_CHAIN = 1 << 20        # an entry chain longer than this is cyclic: stop, do not loop


class _CompoundFile:
    """Read-only CFB (OLE2) container reader. Every structural surprise raises ``DocumentUnreadable``."""

    def __init__(self, raw: bytes):
        self.raw = raw
        if len(raw) < 512 or raw[:8] != _CFB_MAGIC:
            raise DocumentUnreadable(
                "this is not a compound (OLE) file — the CFB header signature is missing, so it "
                "cannot be a legacy .doc/.xls/.ppt")
        _minor, self.major, byte_order, sector_shift = struct.unpack_from("<HHHH", raw, 24)
        mini_shift = struct.unpack_from("<H", raw, 32)[0]
        if byte_order != 0xFFFE or not 7 <= sector_shift <= 14 or not 4 <= mini_shift <= 12:
            raise DocumentUnreadable(
                "the compound file header is damaged (bad byte order or sector size)")
        self.sector_size = 1 << sector_shift
        self.mini_sector_size = 1 << mini_shift
        self.base = self.sector_size          # the header is padded to one whole sector
        fat_count, = struct.unpack_from("<I", raw, 44)
        first_directory, = struct.unpack_from("<I", raw, 48)
        self.mini_cutoff, = struct.unpack_from("<I", raw, 56)
        first_mini_fat, = struct.unpack_from("<I", raw, 60)
        mini_fat_count, = struct.unpack_from("<I", raw, 64)
        first_difat, = struct.unpack_from("<I", raw, 68)
        difat_count, = struct.unpack_from("<I", raw, 72)
        if self.base + self.sector_size > len(raw):
            raise DocumentUnreadable("the compound file is truncated (no first sector)")
        # DIFAT: 109 sector numbers in the header, then a chain of sectors holding 127 more each.
        fat_sectors = list(struct.unpack_from("<109I", raw, 76))
        sector = first_difat
        for _ in range(min(difat_count, _CFB_MAX_CHAIN)):
            if sector in (_CFB_END, _CFB_FREE) or not self._in_file(sector):
                break
            block = self._sector(sector)
            per_sector = self.sector_size // 4 - 1
            fat_sectors.extend(struct.unpack_from("<%dI" % per_sector, block, 0))
            sector, = struct.unpack_from("<I", block, self.sector_size - 4)
        fat_sectors = [s for s in fat_sectors if s not in (_CFB_END, _CFB_FREE)]
        if fat_count:
            fat_sectors = fat_sectors[:fat_count]
        self.fat: List[int] = []
        for fat_sector in fat_sectors:
            if not self._in_file(fat_sector):
                continue
            self.fat.extend(struct.unpack_from(
                "<%dI" % (self.sector_size // 4), self._sector(fat_sector), 0))
        if not self.fat:
            raise DocumentUnreadable("the compound file has no readable FAT (truncated or damaged)")
        self.entries = self._read_directory(first_directory)
        root = next((e for e in self.entries if e["type"] == 5), None)
        self._mini_fat: List[int] = []
        self._mini_stream: Optional[bytes] = None
        if root is not None and mini_fat_count:
            # The mini FAT is itself a sector chain: each sector names the next in its last slot.
            blocks: List[int] = []
            sector, seen = first_mini_fat, set()
            while sector not in (_CFB_END, _CFB_FREE) and self._in_file(sector) \
                    and sector not in seen and len(blocks) < mini_fat_count:
                seen.add(sector)
                blocks.append(sector)
                nxt, = struct.unpack_from("<I", self._sector(sector), self.sector_size - 4)
                sector = nxt
            for block in blocks:
                self._mini_fat.extend(struct.unpack_from(
                    "<%dI" % (self.sector_size // 4), self._sector(block), 0))
            self._root_entry = root

    # -- primitives ------------------------------------------------------------------------ #
    def _in_file(self, sector: int) -> bool:
        if sector < 0 or sector == _CFB_FREE or sector == _CFB_END:
            return False
        return self.base + (sector + 1) * self.sector_size <= len(self.raw)

    def _sector(self, sector: int) -> bytes:
        start = self.base + sector * self.sector_size
        return self.raw[start:start + self.sector_size]

    def _chain(self, start: int, table: Optional[List[int]] = None) -> List[int]:
        """The sectors of one stream, following the FAT (or the mini FAT); stops on a cycle."""
        entries = self.fat if table is None else table
        chain: List[int] = []
        seen = set()
        sector = start
        while sector not in (_CFB_END, _CFB_FREE) and sector < len(entries) \
                and sector not in seen and len(chain) < _CFB_MAX_CHAIN:
            seen.add(sector)
            chain.append(sector)
            sector = entries[sector]
        return chain

    def _read_directory(self, first_directory: int) -> List[Dict[str, object]]:
        data = b"".join(self._sector(s) for s in self._chain(first_directory))
        entries: List[Dict[str, object]] = []
        for offset in range(0, len(data) - 127, 128):
            entry = self._parse_entry(data[offset:offset + 128])
            if entry is not None:
                entries.append(entry)
        if not entries:
            raise DocumentUnreadable(
                "the compound file directory could not be read (truncated or damaged)")
        return entries

    def _parse_entry(self, chunk: bytes) -> Optional[Dict[str, object]]:
        name_length, obj_type = struct.unpack_from("<HB", chunk, 64)
        if obj_type not in (1, 2, 5):
            return None
        name = ""
        if 2 <= name_length <= 64:
            name = chunk[:name_length - 2].decode("utf-16-le", errors="replace")
        start, = struct.unpack_from("<I", chunk, 116)
        size, = struct.unpack_from("<Q", chunk, 120)
        if self.major == 3:
            size &= 0xFFFFFFFF          # v3 writes only the low 32 bits of the size
        return {"name": name, "type": obj_type, "start": start, "size": size}

    def _mini_stream_data(self) -> bytes:
        if self._mini_stream is None:
            root = getattr(self, "_root_entry", None)
            out = bytearray()
            if root is not None:
                for sector in self._chain(int(root["start"])):
                    out.extend(self._sector(sector))
            self._mini_stream = bytes(out[:int(root["size"])]) if root is not None else b""
        return self._mini_stream

    # -- the one public operation ----------------------------------------------------------- #
    def stream(self, name: str) -> Optional[bytes]:
        """The named stream's bytes, or ``None`` when this file has no such stream."""
        entry = next((e for e in self.entries
                      if e["type"] == 2 and e["name"] == name), None)
        if entry is None:
            return None
        size = int(entry["size"])
        start = int(entry["start"])
        if size < self.mini_cutoff and self._chain(start, self._mini_fat):
            mini = self._mini_stream_data()
            out = bytearray()
            for sector in self._chain(start, self._mini_fat):
                begin = sector * self.mini_sector_size
                out.extend(mini[begin:begin + self.mini_sector_size])
                if len(out) >= size:
                    break
            return bytes(out[:size])
        out = bytearray()
        for sector in self._chain(start):
            if not self._in_file(sector):
                break
            out.extend(self._sector(sector))
            if len(out) >= size:
                break
        return bytes(out[:size])


# -- text helpers shared by the three legacy readers ------------------------------------------- #

def _printable_code(value: int) -> bool:
    try:
        return chr(value).isprintable()
    except (ValueError, OverflowError):
        return False


def _clean_text(text: str) -> str:
    """Format control characters as readable text (CR/FF/VT/bell → line break, NUL → nothing)."""
    text = (text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0b", "\n")
                .replace("\x0c", "\n").replace("\x07", "\t").replace("\x00", ""))
    text = "".join(ch for ch in text if ch in "\n\t" or ch.isprintable())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _ole_plain_runs(data: bytes, encoding: str, minimum: int = 24) -> str:
    """Last-resort text of a binary stream: its longest printable runs in ONE encoding.

    A best-effort FALLBACK, used only when the format reader found nothing although the container
    does carry the stream it expects. The encoding comes from the file's own header (Word's
    ``fExtChar``), so this cannot invent 16-bit text out of 8-bit bytes, and a run shorter than
    ``minimum`` characters is binary noise and is dropped. The caller still refuses the file when
    this comes back empty.
    """
    width = 2 if encoding == "utf-16-le" else 1
    chunks: List[str] = []
    index, length = 0, len(data)
    while index < length:
        end = index
        while end + width <= length:
            value = data[end] if width == 1 else data[end] | (data[end + 1] << 8)
            if value in (9, 10, 13) or (32 <= value < 0xFFFD and _printable_code(value)):
                end += width
                continue
            break
        if end - index >= minimum * width:
            chunks.append(data[index:end].decode(encoding, errors="replace"))
        index = end + width if end > index else index + 1
    return _clean_text("\n".join(chunks))


# -- Word 6/95/97-2003 (.doc) ------------------------------------------------------------------ #

_DOC_FIB_SIGNATURE = 0xA5EC
_DOC_FLAG_COMPLEX = 0x0004           # the text is described by a piece table, not fcMin..fcMac
_DOC_FLAG_EXTENDED_CHARS = 0x1000    # 16-bit characters (fExtChar)
_DOC_FLAG_ONE_TABLE = 0x0200         # the piece table lives in 1Table, not 0Table
_DOC_PIECE_COMPRESSED = 0x40000000   # a piece stored as 8-bit cp1252 at half the offset


def _doc_simple_text(document: bytes, flags: int) -> str:
    """The FIB's ``fcMin..fcMac`` run — the whole text of a non-complex Word document."""
    fc_min, fc_mac = struct.unpack_from("<II", document, 0x18)
    if not 0x20 <= fc_min < fc_mac <= len(document):
        return ""
    body = document[fc_min:fc_mac]
    if flags & _DOC_FLAG_EXTENDED_CHARS:
        return _clean_text(body.decode("utf-16-le", errors="replace"))
    return _clean_text(body.decode("cp1252", errors="replace"))


def _doc_piece_table_text(document: bytes, compound: "_CompoundFile", flags: int) -> str:
    """Text via Word 97+'s piece table (``Clx`` at ``fcClx`` in the 0Table/1Table stream).

    This is the path a real edited document takes: each piece names a(``fc``, compressed) spot in
    the WordDocument stream, and the CP array says how many characters it holds. Both encodings are
    handled (``fCompressed`` pieces are cp1252 at ``fc/2``, the rest UTF-16LE at ``fc``).
    """
    fc_clx, lcb_clx = struct.unpack_from("<II", document, 0x01A2)
    table = compound.stream("1Table" if flags & _DOC_FLAG_ONE_TABLE else "0Table")
    if table is None:
        table = compound.stream("1Table") or compound.stream("0Table")
    if table is None or not lcb_clx or fc_clx + lcb_clx > len(table):
        return ""
    clx = table[fc_clx:fc_clx + lcb_clx]
    offset = 0
    while offset + 3 <= len(clx) and clx[offset] == 0x01:      # Prc: grpprl the text does not need
        size, = struct.unpack_from("<h", clx, offset + 1)
        if size < 0:
            return ""
        offset += 3 + size
    if offset + 5 > len(clx) or clx[offset] != 0x02:           # Pcdt
        return ""
    length, = struct.unpack_from("<I", clx, offset + 1)
    plc = clx[offset + 5:offset + 5 + length]
    pieces = (len(plc) - 4) // 12
    if pieces <= 0:
        return ""
    positions = [struct.unpack_from("<I", plc, 4 * i)[0] for i in range(pieces + 1)]
    chunks: List[str] = []
    base = 4 * (pieces + 1)
    for piece in range(pieces):
        pcd = plc[base + 8 * piece:base + 8 * piece + 8]
        if len(pcd) < 8:
            break
        fc, = struct.unpack_from("<I", pcd, 2)
        count = positions[piece + 1] - positions[piece]
        if fc & _DOC_PIECE_COMPRESSED:
            start = (fc & 0x3FFFFFFF) // 2
            chunks.append(document[start:start + count].decode("cp1252", errors="replace"))
        else:
            start = fc & 0x3FFFFFFF
            chunks.append(document[start:start + count * 2].decode("utf-16-le", errors="replace"))
    return _clean_text("".join(chunks))


def _doc_text(raw: bytes) -> str:
    """Text of a Word 6/95/97-2003 binary ``.doc``: FIB, then the piece table, then a raw scan.

    Best-effort by design, and it says so: an encrypted or badly damaged .doc ends with an explicit
    refusal by name rather than with binary noise presented as the document's contents.
    """
    compound = _CompoundFile(raw)
    document = compound.stream("WordDocument")
    if document is None:
        raise DocumentUnreadable(
            "the compound file has no WordDocument stream — it is an OLE file, but not a Word "
            "binary .doc")
    if len(document) < 0x20:
        raise DocumentUnreadable("the WordDocument stream is truncated")
    if struct.unpack_from("<H", document, 0)[0] != _DOC_FIB_SIGNATURE:
        raise DocumentUnreadable("the WordDocument stream is damaged (no Word FIB signature)")
    flags, = struct.unpack_from("<H", document, 0x0A)
    nfib, = struct.unpack_from("<H", document, 0x02)
    text = ""
    if flags & _DOC_FLAG_COMPLEX and nfib >= 193:
        text = _doc_piece_table_text(document, compound, flags)
    if not text.strip():
        text = _doc_simple_text(document, flags)
    if not text.strip():
        text = _ole_plain_runs(
            document, "utf-16-le" if flags & _DOC_FLAG_EXTENDED_CHARS else "cp1252")
    if not text.strip():
        raise DocumentUnreadable(
            "no recoverable text in this Word binary .doc — its FIB text run, its piece table and "
            "a raw printable-run scan all came back empty (the document is encrypted, or damaged)")
    return text


# -- Excel 97-2003 (.xls) ---------------------------------------------------------------------- #

_BIFF_EOF = 0x000A
_BIFF_BOF = 0x0809
_BIFF_CONTINUE = 0x003C
_BIFF_BOUNDSHEET = 0x0085
_BIFF_SST = 0x00FC
_BIFF_LABELSST = 0x00FD
_BIFF_LABEL = 0x0204
_BIFF_RSTRING = 0x00D6
_BIFF_NUMBER = 0x0203
_BIFF_RK = 0x027E
_BIFF_MULRK = 0x00BD
_BIFF_BOOLERR = 0x0205
_BIFF_FORMULA = 0x0006
_BIFF_STRING = 0x0207
_BIFF_WORKSHEET = 0x0010
_XLS_MAX_ROWS = 2000
_XLS_MAX_COLS = 128


class _BiffCursor:
    """Byte cursor over one record's payload that follows into its ``CONTINUE`` records.

    BIFF8 splits one logical record across ``CONTINUE`` records at 8224-byte boundaries, and the
    shared-string table is the record this bites: a string's character data can begin in one segment
    and end in the next, where it re-states its own compression flag as the first byte. ``read``
    spans the seam for header fields; ``segment_tail``/``advance_segment`` exist for the character
    arrays, which must honour that repeated flag byte.
    """

    def __init__(self, segments: List[bytes]) -> None:
        self.segments = [segment for segment in segments if segment]
        self.index = 0
        self.offset = 0

    def read(self, count: int) -> Optional[bytes]:
        out = bytearray()
        want = count
        while want > 0:
            segment = self.segment_tail()
            if not segment:
                if not self.advance_segment():
                    return None
                continue
            take = min(want, len(segment))
            out += segment[:take]
            self.offset += take
            want -= take
        return bytes(out)

    def segment_tail(self) -> bytes:
        if self.index >= len(self.segments):
            return b""
        return self.segments[self.index][self.offset:]

    def advance_segment(self) -> bool:
        if self.index + 1 >= len(self.segments):
            return False
        self.index += 1
        self.offset = 0
        return True

    def skip(self, count: int) -> bool:
        return self.read(count) is not None


def _biff_records(data: bytes):
    """Yield ``(rectype, payload)`` for every BIFF record in a stream; stops at the last whole one."""
    offset, length = 0, len(data)
    while offset + 4 <= length:
        rectype, size = struct.unpack_from("<HH", data, offset)
        if offset + 4 + size > length:
            return
        yield rectype, data[offset + 4:offset + 4 + size]
        offset += 4 + size


def _biff_cursor_strings(segments: List[bytes]) -> List[str]:
    """The shared-string table's strings, from its payload plus its ``CONTINUE`` segments."""
    cursor = _BiffCursor(segments)
    header = cursor.read(8)
    if header is None:
        return []
    _total, unique = struct.unpack("<II", header)
    strings: List[str] = []
    for _ in range(min(unique, 8192)):
        raw_count = cursor.read(2)
        if raw_count is None:
            break
        count, = struct.unpack("<H", raw_count)
        raw_flags = cursor.read(1)
        if raw_flags is None:
            break
        flags = raw_flags[0]
        wide = bool(flags & 0x01)
        runs = 0
        if flags & 0x08:
            raw_runs = cursor.read(2)
            if raw_runs is None:
                break
            runs, = struct.unpack("<H", raw_runs)
        extra = 0
        if flags & 0x04:
            raw_extra = cursor.read(4)
            if raw_extra is None:
                break
            extra, = struct.unpack("<I", raw_extra)
        chars = _biff_chars(cursor, count, wide)
        if chars is None:
            break
        strings.append(chars)
        if (runs * 4 + extra) and not cursor.skip(runs * 4 + extra):
            break
    return strings


def _biff_chars(cursor: _BiffCursor, count: int, wide: bool) -> Optional[str]:
    """``count`` characters across a ``CONTINUE`` seam, where each new segment re-states its width."""
    out: List[str] = []
    remaining = count
    while remaining > 0:
        tail = cursor.segment_tail()
        width = 2 if wide else 1
        take = min(remaining, len(tail) // width)
        if take <= 0:
            if not cursor.advance_segment():
                return None if remaining else "".join(out)
            flag = cursor.read(1)
            if flag is None:
                return None
            wide = bool(flag[0] & 0x01)
            continue
        chunk = cursor.read(take * width)
        if chunk is None:
            return None
        out.append(chunk.decode("utf-16-le" if wide else "cp1252", errors="replace"))
        remaining -= take
    return "".join(out)


def _biff_unicode_string(payload: bytes, offset: int = 0) -> Tuple[str, int]:
    """``(text, next offset)`` for one XLUnicodeString inside a record payload (best-effort)."""
    if offset + 3 > len(payload):
        return "", len(payload)
    count, = struct.unpack_from("<H", payload, offset)
    flags = payload[offset + 2]
    offset += 3
    wide = bool(flags & 0x01)
    width = 2 if wide else 1
    raw = payload[offset:offset + count * width]
    text = raw.decode("utf-16-le" if wide else "cp1252", errors="replace")
    offset += count * width
    if flags & 0x08 and offset + 2 <= len(payload):
        runs, = struct.unpack_from("<H", payload, offset)
        offset += 2 + runs * 4
    if flags & 0x04 and offset + 4 <= len(payload):
        extra, = struct.unpack_from("<I", payload, offset)
        offset += 4 + extra
    return text, offset


def _biff_rk_value(raw: bytes) -> Optional[float]:
    """A BIFF ``RK`` number: the low two bits say integer vs IEEE double, x100 in bit 1."""
    if len(raw) < 4:
        return None
    encoded, = struct.unpack_from("<i", raw, 0)
    if encoded & 0x02:
        value = float(encoded >> 2)
    else:
        double, = struct.unpack("<d", b"\x00\x00\x00\x00" + struct.pack("<i", encoded & 0xFFFFFFFC))
        value = double
    return value / 100.0 if encoded & 0x01 else value


def _fmt_number(value: float) -> str:
    return f"{value:g}"


def _xls_text(raw: bytes) -> str:
    """Cells of an Excel 97-2003 binary ``.xls``: shared strings, labels, numbers, formulas.

    Best-effort and stated as such: the workbook-globals shared-string table is parsed with its
    ``CONTINUE`` seams honoured, then each worksheet's label/number/formula records are laid back
    into rows. A BIFF5-era or partly damaged file yields whatever its records still describe, and a
    file whose records describe nothing is refused by name.
    """
    compound = _CompoundFile(raw)
    workbook = compound.stream("Workbook") or compound.stream("Book")
    if workbook is None:
        raise DocumentUnreadable(
            "the compound file has no Workbook stream — it is an OLE file, but not an Excel "
            "binary .xls")
    records = list(_biff_records(workbook))
    if not records:
        raise DocumentUnreadable("the Workbook stream holds no BIFF records (truncated)")

    shared: List[str] = []
    for index, (rectype, payload) in enumerate(records):
        if rectype != _BIFF_SST:
            continue
        segments = [payload]
        for next_type, next_payload in records[index + 1:]:
            if next_type != _BIFF_CONTINUE:
                break
            segments.append(next_payload)
        shared = _biff_cursor_strings(segments)
        break

    boundsheet_names: List[str] = []
    sheets: List[Tuple[str, Dict[int, Dict[int, str]]]] = []
    current: Optional[Dict[int, Dict[int, str]]] = None
    pending_string_at: Optional[Tuple[int, int]] = None
    for rectype, payload in records:
        if rectype == _BIFF_BOF:
            # The BIFF8 BOF payload is vers(2) then dt(2): dt names the substream (0x0005 workbook
            # globals, 0x0010 worksheet, 0x0020 chart, 0x0040 macro sheet).
            if len(payload) < 4:
                current = None
                continue
            substream, = struct.unpack_from("<H", payload, 2)
            if substream == _BIFF_WORKSHEET:
                name = boundsheet_names[len(sheets)] if len(sheets) < len(boundsheet_names) \
                    else f"sheet {len(sheets) + 1}"
                current = {}
                sheets.append((name, current))
            else:
                current = None
            continue
        if rectype == _BIFF_BOUNDSHEET and len(payload) >= 8:
            # BoundSheet8: lbPlyPos(4) hsState(1) dt(1) then a ShortXLUnicodeString — a 1-BYTE
            # character count at offset 6, its flag byte at 7, the name from 8.
            count = payload[6]
            wide = bool(payload[7] & 0x01)
            name = payload[8:8 + count * (2 if wide else 1)].decode(
                "utf-16-le" if wide else "cp1252", errors="replace")
            boundsheet_names.append(name.strip() or f"sheet {len(boundsheet_names) + 1}")
            continue
        if current is None:
            continue
        if rectype == _BIFF_LABELSST and len(payload) >= 10:
            row, col, _xf, index = struct.unpack_from("<HHHI", payload, 0)
            if 0 <= index < len(shared) and shared[index]:
                current.setdefault(row, {})[col] = shared[index]
        elif rectype in (_BIFF_LABEL, _BIFF_RSTRING) and len(payload) >= 9:
            row, col, _xf = struct.unpack_from("<HHH", payload, 0)
            value, _ = _biff_unicode_string(payload, 6)
            if value:
                current.setdefault(row, {})[col] = value
        elif rectype == _BIFF_NUMBER and len(payload) >= 14:
            row, col, _xf = struct.unpack_from("<HHH", payload, 0)
            number, = struct.unpack_from("<d", payload, 6)
            current.setdefault(row, {})[col] = _fmt_number(number)
        elif rectype == _BIFF_RK and len(payload) >= 10:
            row, col, _xf = struct.unpack_from("<HHH", payload, 0)
            value = _biff_rk_value(payload[6:10])
            if value is not None:
                current.setdefault(row, {})[col] = _fmt_number(value)
        elif rectype == _BIFF_MULRK and len(payload) >= 12:
            row, first_col = struct.unpack_from("<HH", payload, 0)
            last_col, = struct.unpack_from("<H", payload, len(payload) - 2)
            values = payload[4:len(payload) - 2]
            for step, col in enumerate(range(first_col, last_col + 1)):
                value = _biff_rk_value(values[step * 6 + 2:step * 6 + 6])
                if value is not None:
                    current.setdefault(row, {})[col] = _fmt_number(value)
        elif rectype == _BIFF_BOOLERR and len(payload) >= 8:
            row, col, _xf, value, is_error = struct.unpack_from("<HHHBB", payload, 0)
            current.setdefault(row, {})[col] = ("#" if is_error else "") + ("TRUE" if value else "FALSE")
        elif rectype == _BIFF_FORMULA and len(payload) >= 14:
            row, col, _xf = struct.unpack_from("<HHH", payload, 0)
            result = payload[6:14]
            if result[6:8] == b"\xff\xff":
                if result[0] == 0 and result[2] in (0, 1):
                    current.setdefault(row, {})[col] = "TRUE" if result[2] else "FALSE"
                else:
                    pending_string_at = (row, col)     # the text follows in a STRING record
            else:
                number, = struct.unpack("<d", result)
                current.setdefault(row, {})[col] = _fmt_number(number)
        elif rectype == _BIFF_STRING and len(payload) >= 3:
            value, _ = _biff_unicode_string(payload, 0)
            if value and pending_string_at is not None:
                row, col = pending_string_at
                current.setdefault(row, {})[col] = value
            pending_string_at = None

    parts: List[str] = []
    for name, grid in sheets:
        if not grid:
            continue
        parts.append(f"[{name}]")
        for row in sorted(grid)[:_XLS_MAX_ROWS]:
            cells = [grid[row].get(col, "") for col in range(0, min(len(grid[row]), _XLS_MAX_COLS))]
            line = "\t".join(cells).rstrip("\t")
            if line.strip():
                parts.append(line)
    if not parts and shared:
        # No cell records to place, but the shared table carries text: show it rather than nothing.
        parts = ["[shared strings]"] + [text for text in shared if text.strip()]
    if not parts:
        raise DocumentUnreadable(
            "no cell or string records in this Excel binary .xls (it may be a BIFF5 file, empty, "
            "or damaged)")
    return "\n".join(parts)


# -- PowerPoint 97-2003 (.ppt) ----------------------------------------------------------------- #

_PPT_TEXT_ATOMS = {
    0x0FA0: "utf-16-le",   # TextCharsAtom
    0x0FA8: "cp1252",      # TextBytesAtom
    0x0FBA: "cp1252",      # CString
}


def _ppt_records_text(data: bytes, depth: int) -> List[str]:
    """Text atoms of a PowerPoint record tree (containers hold their children in their own body)."""
    out: List[str] = []
    offset, length, collected = 0, len(data), 0
    while offset + 8 <= length and len(out) < 4096:
        header, rectype, size = struct.unpack_from("<HHI", data, offset)
        body = data[offset + 8:offset + 8 + size]
        if len(body) < size:               # a truncated record ends the walk, it is not guessed at
            break
        if header & 0x000F == 0x000F:       # recVer 0xF: a container, children follow inside
            if depth < 12:
                out.extend(_ppt_records_text(body, depth + 1))
        else:
            encoding = _PPT_TEXT_ATOMS.get(rectype)
            if encoding is not None:
                text = body.decode(encoding, errors="replace").strip("\x00 \t\r\n")
                if text:
                    out.append(text)
                    collected += len(text)
                    if collected >= MAX_EXTRACT_CHARS:
                        break
        offset += 8 + size
    return out


def _ppt_text(raw: bytes) -> str:
    """Text of a PowerPoint 97-2003 binary ``.ppt``: every text atom in the record tree, in order."""
    compound = _CompoundFile(raw)
    stream = compound.stream("PowerPoint Document")
    if stream is None:
        raise DocumentUnreadable(
            "the compound file has no 'PowerPoint Document' stream — it is an OLE file, but not a "
            "PowerPoint binary .ppt")
    text = _clean_text("\n".join(_ppt_records_text(stream, 0)))
    if not text.strip():
        raise DocumentUnreadable(
            "no text records in this PowerPoint binary .ppt — its TextCharsAtom/TextBytesAtom "
            "records are absent (the file is empty, encrypted or damaged)")
    return text


_EXTRACTORS: Dict[str, Callable[[bytes], str]] = {
    "application/pdf": _pdf_text,
    _DOC_MIME: _doc_text,
    _XLS_MIME: _xls_text,
    _PPT_MIME: _ppt_text,
    "application/zip": _zip_text,
    "application/x-zip-compressed": _zip_text,
    _OOXML_WORD: _docx_text,
    _OOXML_SHEET: _xlsx_text,
    _OOXML_DECK: _pptx_text,
    _ODF_TEXT: _odf_text,
    _ODF_SHEET: _odf_text,
    _ODF_DECK: _odf_text,
    _EPUB_MIME: _epub_text,
    "application/rtf": _rtf_text,
    "text/rtf": _rtf_text,
}


def _resolve_mime(mime: str, filename: str) -> str:
    """The MIME to extract with: the declared one, else the filename extension."""
    mime = (mime or "").split(";", 1)[0].strip().lower()
    if _is_text_mime(mime) or mime in _EXTRACTORS:
        return mime
    return _EXT_MIMES.get(os.path.splitext(filename or "")[1].lower(), mime)


def _human_size(count: int) -> str:
    """Bytes as a human phrase for the skip note ("21.0 MB", "900 bytes")."""
    if count >= 1024 * 1024:
        return f"{count / (1024 * 1024):.1f} MB"
    if count >= 1024:
        return f"{count / 1024:.1f} KB"
    return f"{count} bytes"


def _note(text: str) -> str:
    """The extracted text as the line the turn carries, with the truncation cut named."""
    included = text[:MAX_EXTRACT_CHARS]
    if not included.strip():
        return ("[Document contents (extracted, 0 chars): the document was read but contains no "
                "extractable text — it may be a scan or an image-only page. The file is at the "
                "path below if a tool on this profile can read it.]")
    line = f"[Document contents (extracted, {len(included)} chars):\n{included}"
    if len(text) > MAX_EXTRACT_CHARS:
        line += (f"\n\n[Document contents truncated at {MAX_EXTRACT_CHARS} chars — the document is "
                 f"longer; {len(text) - MAX_EXTRACT_CHARS} more characters were not included.]")
    return line + "]"


def _skip_note(reason: str, mime: str) -> str:
    return (f"[Document contents not extracted: {reason}. text could not be extracted from this "
            f"{mime or 'unknown'} in this turn.]")


def extract_note(raw: bytes, mime: str, filename: str = "") -> str:
    """One ``[Document contents …]`` line for an uploaded document — the turn's inline reading.

    Never raises and never returns ``""``: a document turn always carries an explicit statement
    about whether its text was read, so no model can quietly answer about a document it never saw.
    """
    resolved = _resolve_mime(mime, filename)
    display = resolved or mime or "application/octet-stream"
    if not raw:
        return _skip_note("the upload was empty", display)
    if len(raw) > MAX_SOURCE_BYTES:
        return _skip_note(
            f"the file is {_human_size(len(raw))}, above the "
            f"{MAX_SOURCE_BYTES // (1024 * 1024)} MB extraction limit", display)
    extractor: Optional[Callable[[bytes], str]] = _EXTRACTORS.get(resolved)
    if extractor is None and _is_text_mime(resolved):
        extractor = _extract_text
    if extractor is None:
        return _skip_note("no extractor is available for this type", display)
    try:
        text = extractor(raw)
    except DocumentUnreadable as exc:
        logger.warning("[api_server] document unreadable as %s: %s", display, exc)
        return _skip_note(str(exc), display)
    except Exception as exc:  # noqa: BLE001 — an extractor failure must never fail the upload
        logger.warning("[api_server] document text extraction failed for %s: %s", display, exc)
        return _skip_note(f"extraction failed ({type(exc).__name__}: {exc})"[:200], display)
    logger.info("[api_server] extracted %d chars of text from a %s document (%d bytes)",
                len(text), display, len(raw))
    return _note(text)
