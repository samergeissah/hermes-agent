"""Focused tests for document TEXT extraction on ``POST /api/sessions/{id}/chat/attachments``.

The behaviour under test is the document-side twin of what the vision backend already does for
images on the same route: the document's text is read at ingest and put INTO the turn, so a profile
with no file tools (a strict intake persona — the live failure) can answer from its own context.

Covered:

* the extractors, by type: PDF with an uncompressed content stream, PDF with a ``FlateDecode``
  stream (the shape every real-world PDF has), docx, xlsx, and the plain-text family
  (txt/md/csv/json/xml), plus UTF-16 text;
* the archive reader: a real ``.zip`` with text members, a nested zip (walked), an archive whose
  members are office packages (read by their own reader), the depth/entry/byte/character caps of a
  zip-bomb-shaped upload (each cap stated in the turn), and an archive with nothing readable in it;
* the legacy OLE binaries: a generated ``.doc`` (FIB text run AND the complex/piece-table shape),
  ``.xls`` (BIFF8 shared strings incl. a ``CONTINUE``-split table, labels, numbers), ``.ppt``
  (record-tree text atoms), and the honest failures — a mislabelled OLE file, a truncated one, and
  a payload that is not a compound file at all each say what is wrong instead of guessing;
* the caps and the honest failure modes: truncation is labelled at the cap, an oversize source is
  skipped with a stated limit, an unreadable/mislabelled package says so, an unsupported type says
  so — the turn NEVER silently omits the document;
* end to end through the route: the secret word inside an uploaded PDF, ``.zip`` and ``.doc``
  reaches the turn's user content, and an IMAGE upload is untouched (no document note, one inline
  image part).
"""

from __future__ import annotations

import base64
import io
import json
import struct
import time
import zlib
import zipfile
from unittest.mock import patch

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server_document_lane as doc_lane
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB

API_KEY = "sk-test"
SECRET = "Secret word: PINEAPPLE-7742"
PDF_MIME = "application/pdf"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
DOC_MIME = "application/msword"
XLS_MIME = "application/vnd.ms-excel"
PPT_MIME = "application/vnd.ms-powerpoint"
ZIP_MIME = "application/zip"

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGPkEpFjYGBgYgAD"
    "AALmAEAUQs4PAAAAAElFTkSuQmCC")
FORWARD = "A 1x1 test image."


# --------------------------------------------------------------------------- asset builders

def _pdf(lines, *, flate: bool) -> bytes:
    """A one-page PDF whose /Contents stream draws ``lines`` (optionally ``/FlateDecode``)."""
    stream = ("BT /F1 12 Tf 10 50 Td " + " ".join(f"({ln}) Tj" for ln in lines) + " ET\n").encode()
    body = zlib.compress(stream) if flate else stream
    filt = " /Filter /FlateDecode" if flate else ""
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 100]"
        b" /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d%s >>\nstream\n%s\nendstream" % (len(body), filt.encode(), body),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    for i, obj in enumerate(objs, start=1):
        out += b"%d 0 obj\n%s\nendobj\n" % (i, obj)
    return bytes(out) + b"trailer\n<< /Size 6 /Root 1 0 R >>\n%%EOF\n"


def _docx(paragraphs) -> bytes:
    body = "".join(
        f'<w:p><w:r><w:t xml:space="preserve">{p}</w:t></w:r></w:p>' for p in paragraphs)
    document = ('<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                f"<w:body>{body}</w:body></w:document>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("word/document.xml", document)
    return buf.getvalue()


def _xlsx(rows) -> bytes:
    shared: list[str] = []
    for row in rows:
        for cell in row:
            if cell not in shared:
                shared.append(cell)
    sheet = ('<?xml version="1.0" encoding="UTF-8"?>'
             '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
             "<sheetData>"
             + "".join(
                 f'<row r="{r}">'
                 + "".join(f'<c r="{chr(65 + c)}{r}" t="s"><v>{shared.index(cell)}</v></c>'
                           for c, cell in enumerate(row))
                 + "</row>"
                 for r, row in enumerate(rows, start=1))
             + "</sheetData></worksheet>")
    sst = ('<?xml version="1.0" encoding="UTF-8"?>'
           '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
           f'count="{len(shared)}" uniqueCount="{len(shared)}">'
           + "".join(f"<si><t>{cell}</t></si>" for cell in shared) + "</sst>")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/sharedStrings.xml", sst)
        z.writestr("xl/worksheets/sheet1.xml", sheet)
    return buf.getvalue()


def _zip_with(members) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in members.items():
            z.writestr(name, content)
    return buf.getvalue()


# --- legacy OLE (Compound File Binary) asset builders --------------------------------------- #
#
# A .doc/.xls/.ppt is a CFB container, so the fixtures have to BE one: the helpers below write a
# valid v3 compound file (512-byte sectors, a FAT, 128-byte directory entries and a mini stream for
# anything under the 4096-byte cutoff, which is where a small Word document's text really lives) and
# then put a real FIB / BIFF8 / PowerPoint record tree inside it. Generating the container rather
# than shipping a binary fixture keeps the sample reviewable.

_END, _FREE = 0xFFFFFFFE, 0xFFFFFFFF
_SECTOR, _MINI = 512, 64


def _cfb(streams: dict) -> bytes:
    """A minimal valid CFB v3 file holding ``streams`` (small ones in the mini stream)."""
    big = {n: d for n, d in streams.items() if len(d) >= 4096}
    small = {n: d for n, d in streams.items() if n not in big}
    mini_data = bytearray()
    mini_start: dict = {}
    for name, data in small.items():
        mini_start[name] = len(mini_data) // _MINI
        mini_data += data + b"\x00" * (-len(data) % _MINI)
    mini_chains = {name: list(range(mini_start[name],
                                    mini_start[name] + max(1, -(-len(data) // _MINI))))
                   for name, data in small.items()}
    mini_fat = [_FREE] * (len(mini_data) // _MINI)
    for chain in mini_chains.values():
        for a, b in zip(chain, chain[1:]):
            mini_fat[a] = b
        mini_fat[chain[-1]] = _END

    sectors: list = []
    big_chains: dict = {}
    for name, data in big.items():
        padded = data + b"\x00" * (-len(data) % _SECTOR)
        big_chains[name] = list(range(len(sectors),
                                      len(sectors) + max(1, len(padded) // _SECTOR)))
        for i in range(0, len(padded), _SECTOR):
            sectors.append(padded[i:i + _SECTOR])
    mini_fat_bytes = b"".join(struct.pack("<I", v) for v in mini_fat)
    mini_fat_bytes += b"\xff" * (-len(mini_fat_bytes) % _SECTOR)
    mini_fat_sector = len(sectors)
    for i in range(0, max(len(mini_fat_bytes), 1), _SECTOR):
        sectors.append(mini_fat_bytes[i:i + _SECTOR] or b"\x00" * _SECTOR)
    mini_fat_sectors = len(sectors) - mini_fat_sector
    mini_stream_data = bytes(mini_data) + b"\x00" * (-len(mini_data) % _SECTOR)
    mini_stream_sector = len(sectors)
    for i in range(0, max(len(mini_stream_data), 1), _SECTOR):
        sectors.append(mini_stream_data[i:i + _SECTOR] or b"\x00" * _SECTOR)
    mini_stream_sectors = len(sectors) - mini_stream_sector

    def entry(name, kind, start, size):
        raw = bytearray(128)
        raw[:len(name) * 2 + 2] = name.encode("utf-16-le") + b"\x00\x00"
        struct.pack_into("<H", raw, 64, len(name) * 2 + 2)
        if kind:
            raw[66] = kind
        struct.pack_into("<I", raw, 116, start)
        struct.pack_into("<Q", raw, 120, size)
        struct.pack_into("<III", raw, 68, _FREE, _FREE, _FREE)
        return bytes(raw)

    entries = [entry("Root Entry", 5, mini_stream_sector if mini_data else _END, len(mini_data))]
    for name, data in streams.items():
        start = big_chains.get(name, mini_chains.get(name, [_END]))[0] if data else _END
        entries.append(entry(name, 2, start, len(data)))
    while len(entries) % 4:
        entries.append(b"\x00" * 128)
    dir_bytes = b"".join(entries)
    dir_sector = len(sectors)
    for i in range(0, len(dir_bytes), _SECTOR):
        sectors.append(dir_bytes[i:i + _SECTOR])
    dir_sectors = len(sectors) - dir_sector

    fat_count = max(1, -(-(len(sectors) + 1) // 128))
    fat_sector = len(sectors)
    for _ in range(fat_count):
        sectors.append(b"\x00" * _SECTOR)
    fat = [_FREE] * (fat_count * 128)
    for chain in big_chains.values():
        for a, b in zip(chain, chain[1:]):
            fat[a] = b
        fat[chain[-1]] = _END
    fat[mini_fat_sector] = _END
    for i in range(mini_stream_sectors):
        fat[mini_stream_sector + i] = (mini_stream_sector + i + 1
                                       if i + 1 < mini_stream_sectors else _END)
    for i in range(dir_sectors):
        fat[dir_sector + i] = dir_sector + i + 1 if i + 1 < dir_sectors else _END
    for i in range(fat_count):
        fat[fat_sector + i] = fat_sector + i + 1 if i + 1 < fat_count else _END
    fat_bytes = b"".join(struct.pack("<I", v) for v in fat)
    for i in range(fat_count):
        sectors[fat_sector + i] = fat_bytes[i * _SECTOR:(i + 1) * _SECTOR]

    header = bytearray(512)
    header[0:8] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    struct.pack_into("<HHHH", header, 24, 0x003E, 3, 0xFFFE, 9)
    struct.pack_into("<H", header, 32, 6)
    struct.pack_into("<I", header, 44, fat_count)
    struct.pack_into("<I", header, 48, dir_sector)
    struct.pack_into("<I", header, 56, 4096)
    struct.pack_into("<I", header, 60, mini_fat_sector)
    struct.pack_into("<I", header, 64, mini_fat_sectors)
    struct.pack_into("<I", header, 68, _END)
    struct.pack_into("<109I", header, 76, fat_sector, *([_FREE] * 108))
    return bytes(header) + b"".join(sectors)


def _word_doc(text: str) -> bytes:
    """A Word 6/95/97 ``.doc`` whose FIB points ``fcMin..fcMac`` at UTF-16 text (not complex)."""
    body = text.encode("utf-16-le")
    stream = bytearray(0x800 + len(body))
    struct.pack_into("<H", stream, 0, 0xA5EC)            # wIdent
    struct.pack_into("<H", stream, 2, 193)               # nFib: Word 97
    struct.pack_into("<H", stream, 0x0A, 0x1000)         # fExtChar, fComplex clear
    struct.pack_into("<I", stream, 0x18, 0x800)          # fcMin
    struct.pack_into("<I", stream, 0x1C, 0x800 + len(body))   # fcMac
    stream[0x800:] = body
    return _cfb({"WordDocument": bytes(stream)})


def _word_doc_complex(text: str) -> bytes:
    """A complex ``.doc``: the text is only reachable through the 0Table piece table (``Clx``)."""
    body = text.encode("utf-16-le")
    stream = bytearray(0x800 + len(body))
    struct.pack_into("<H", stream, 0, 0xA5EC)
    struct.pack_into("<H", stream, 2, 193)
    struct.pack_into("<H", stream, 0x0A, 0x0004)         # fComplex
    struct.pack_into("<I", stream, 0x18, 0x800)
    struct.pack_into("<I", stream, 0x1C, 0x800)
    stream[0x800:] = body
    clx = bytearray(b"\x01") + struct.pack("<h", 4) + b"\x00" * 4     # a Prc to skip
    plc = struct.pack("<II", 0, len(text)) + struct.pack("<HHI", 0, 0x800, 0)
    clx += b"\x02" + struct.pack("<I", len(plc)) + plc
    struct.pack_into("<II", stream, 0x01A2, 0, len(clx))             # fcClx, lcbClx
    return _cfb({"WordDocument": bytes(stream), "0Table": bytes(clx)})


def _xls_book(rows, *, split_sst: bool = False) -> bytes:
    """A BIFF8 workbook: globals (BOUNDSHEET + SST) then one worksheet of labels + a number."""
    def rec(rectype: int, payload: bytes) -> bytes:
        return struct.pack("<HH", rectype, len(payload)) + payload

    def bof(dt: int) -> bytes:
        return rec(0x0809, struct.pack("<HHHHII", 0x0600, dt, 0x0DBB, 0x07CC, 0x41, 6))

    shared: list = []
    for row in rows:
        for cell in row:
            if cell not in shared:
                shared.append(cell)
    sst = bytearray(struct.pack("<II", sum(len(r) for r in rows), len(shared)))
    for value in shared:
        sst += struct.pack("<HB", len(value), 0x01) + value.encode("utf-16-le")
    if split_sst:
        head, tail = bytes(sst[:-4]), bytes(sst[-4:])
        # BIFF8: a CONTINUE that resumes a string re-states its compression flag as its first byte
        records = [bof(0x0005), rec(0x00FC, head), rec(0x003C, b"\x01" + tail)]
    else:
        records = [bof(0x0005), rec(0x00FC, bytes(sst))]
    name = "Sheet1"
    records.append(rec(0x0085, struct.pack("<IBBB", 0, 0, 0, len(name)) + b"\x01"
                       + name.encode("utf-16-le")))
    records.append(rec(0x000A, b""))
    records.append(bof(0x0010))
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            records.append(rec(0x00FD, struct.pack("<HHHI", r, c, 0, shared.index(cell))))
    records.append(rec(0x0203, struct.pack("<HHH", len(rows), 0, 0) + struct.pack("<d", 42.5)))
    records.append(rec(0x000A, b""))
    return _cfb({"Workbook": b"".join(records)})


def _ppt_deck(lines) -> bytes:
    """A ``.ppt`` record tree: Document > Slide containers, each line a TextCharsAtom (UTF-16LE)."""
    def rec(rectype: int, payload: bytes, ver: int = 0) -> bytes:
        # the 16-bit record header packs recVer in its low 4 bits, recInstance in the high 12
        return struct.pack("<HHI", ver, rectype, len(payload)) + payload

    atoms = b"".join(rec(0x0FA0, line.encode("utf-16-le")) for line in lines)
    return _cfb({"PowerPoint Document": rec(0x03E8, rec(0x03EE, atoms, ver=0xF), ver=0xF)})


def _docx_package(paragraph: str) -> bytes:
    document = ('<?xml version="1.0" encoding="UTF-8"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                f"<w:body><w:p><w:r><w:t>{paragraph}</w:t></w:r></w:p></w:body></w:document>")
    return _zip_with({"word/document.xml": document})


# --------------------------------------------------------------------------- extractor units

def test_pdf_with_an_uncompressed_content_stream_is_read():
    note = doc_lane.extract_note(_pdf([SECRET], flate=False), PDF_MIME, "secret.pdf")
    assert SECRET in note
    assert note.startswith("[Document contents (extracted,")


def test_pdf_with_a_flate_compressed_content_stream_is_read():
    """The shape every real PDF has: a zlib-deflated content stream (the naive byte grep fails)."""
    note = doc_lane.extract_note(_pdf(["Quarterly report", SECRET], flate=True), PDF_MIME, "r.pdf")
    assert SECRET in note
    assert "Quarterly report" in note


def test_pdf_multiline_text_keeps_lines_apart():
    note = doc_lane.extract_note(_pdf(["title", SECRET], flate=True), PDF_MIME, "r.pdf")
    assert "title" in note and SECRET in note


def test_pdf_with_no_readable_stream_says_so_instead_of_claiming_a_reading():
    note = doc_lane.extract_note(b"%PDF-1.4\nnot really a pdf\n%%EOF\n", PDF_MIME, "broken.pdf")
    assert "text could not be extracted from this application/pdf" in note
    assert "PINEAPPLE" not in note


def test_pdf_with_a_stream_but_no_text_is_called_out_as_image_only():
    """A scan: the document WAS read and has no text layer — the model is told that, not nothing."""
    scan = _pdf([], flate=True)
    note = doc_lane.extract_note(scan, PDF_MIME, "scan.pdf")
    assert "0 chars" in note
    assert "scan or an image-only page" in note


def test_docx_paragraphs_are_extracted():
    note = doc_lane.extract_note(_docx(["Internal memo", SECRET, "Owner: Samer"]), DOCX_MIME,
                                 "memo.docx")
    assert SECRET in note
    assert "Internal memo" in note and "Owner: Samer" in note


def test_xlsx_rows_are_extracted_through_shared_strings():
    note = doc_lane.extract_note(
        _xlsx([["item", "value"], ["secret", SECRET]]), "application/octet-stream", "sheet.xlsx")
    assert SECRET in note
    assert "item\tvalue" in note


def test_mislabelled_or_damaged_package_says_so():
    bogus = b"PK\x03\x04" + b"\x00" * 24 + b"word/document.xml"
    note = doc_lane.extract_note(bogus, "application/octet-stream", "memo.docx")
    assert "not a readable DOCX package" in note
    assert "text could not be extracted from this" in note


def test_zip_package_without_the_document_part_says_so():
    note = doc_lane.extract_note(_zip_with({"readme.txt": "hi"}), "application/octet-stream",
                                 "memo.docx")
    assert "has no readable document part" in note


@pytest.mark.parametrize("mime,filename,payload", [
    ("text/plain", "notes.txt", f"line one\n{SECRET}\n"),
    ("text/markdown", "notes.md", f"# Title\n\n{SECRET}\n"),
    ("text/csv", "rows.csv", f'item,value\nsecret,"{SECRET}"\n'),
    ("application/json", "payload.json", json.dumps({"secret": SECRET})),
    ("application/xml", "doc.xml", f"<doc><secret>{SECRET}</secret></doc>"),
])
def test_text_family_documents_are_read(mime, filename, payload):
    note = doc_lane.extract_note(payload.encode(), mime, filename)
    assert SECRET in note, note


def test_utf16_text_document_is_decoded():
    raw = ("café " + SECRET).encode("utf-16")
    note = doc_lane.extract_note(raw, "text/plain", "u.txt")
    assert SECRET in note and "café" in note


# --------------------------------------------------------------------------- zip archives

def test_zip_archive_members_are_extracted_with_their_names():
    note = doc_lane.extract_note(
        _zip_with({"notes.txt": f"hello\n{SECRET}\n", "readme": "an extensionless member\n"}),
        ZIP_MIME, "bundle.zip")
    assert SECRET in note
    assert "--- notes.txt (text/plain) ---" in note
    assert "an extensionless member" in note, "an extensionless text member is still read"


def test_zip_archive_states_the_caps_it_runs_under():
    """A partly-read archive must SAY what it may have left out, in the turn."""
    note = doc_lane.extract_note(_zip_with({"a.txt": SECRET}), ZIP_MIME, "a.zip")
    assert f"at most {doc_lane.MAX_ARCHIVE_ENTRIES} entries" in note
    assert f"{doc_lane.MAX_ARCHIVE_DEPTH} nested zip level(s)" in note
    assert f"{doc_lane.MAX_ARCHIVE_TEXT_CHARS} characters of text" in note


def test_a_zip_inside_a_zip_is_walked_and_the_nesting_is_capped():
    deep = _zip_with({"deep.txt": SECRET})
    for level in range(4):
        deep = _zip_with({f"level{level}.zip": deep})
    note = doc_lane.extract_note(deep, ZIP_MIME, "deep.zip")
    assert "nested zip" in note
    assert (f"[not opened: the {doc_lane.MAX_ARCHIVE_DEPTH}-level nested-zip limit for this "
            f"upload was reached.]") in note, note


def test_an_office_package_inside_a_zip_is_read_by_its_own_reader():
    note = doc_lane.extract_note(_zip_with({"memo.docx": _docx_package(SECRET)}), ZIP_MIME,
                                 "package.zip")
    assert SECRET in note
    assert "--- memo.docx (application/vnd.openxmlformats-officedocument.wordprocessingml" in note


def test_a_zip_bomb_shaped_archive_is_capped_and_says_what_it_skipped(monkeypatch):
    """Many entries plus a member that inflates hugely: capped, stated, and it does not wedge."""
    monkeypatch.setattr(doc_lane, "MAX_ARCHIVE_ENTRIES", 5)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bomb.txt", b"\x00" * (12 * 1024 * 1024))     # ~12 KB on the wire
        for i in range(40):
            archive.writestr(f"f{i:02d}.txt", f"entry {i}\n")
    started = time.time()
    note = doc_lane.extract_note(buf.getvalue(), ZIP_MIME, "bomb.zip")
    assert time.time() - started < 20, "a bomb-shaped archive must not hold the turn"
    assert "at most 5 entries" in note
    assert "caps above were reached" in note
    assert "5 of its 41 entries were read" in note
    assert "\x00" not in note, "a wall of NULs is not text and must not be passed to the model"
    assert "no extractable text in this entry" in note
    assert len(note) <= doc_lane.MAX_EXTRACT_CHARS + 2000


def test_a_zip_of_nothing_readable_says_so_instead_of_guessing():
    note = doc_lane.extract_note(_zip_with({"blob.bin": bytes(range(256)) * 4}), ZIP_MIME, "b.zip")
    assert "holds no extractable text" in note
    assert "text could not be extracted from this application/zip" in note


def test_an_empty_zip_and_a_damaged_zip_say_what_is_wrong():
    assert "contains no entries" in doc_lane.extract_note(_zip_with({}), ZIP_MIME, "e.zip")
    damaged = doc_lane.extract_note(b"PK\x03\x04" + b"\x00" * 40, ZIP_MIME, "bad.zip")
    assert "not a readable zip archive" in damaged
    assert "text could not be extracted from this application/zip" in damaged


# --------------------------------------------------------------------------- legacy OLE binaries

@pytest.mark.parametrize("filler", ["", "x" * 5000])
def test_legacy_doc_text_is_extracted(filler):
    """Both container shapes: a small stream lives in the mini stream, a big one in the main FAT."""
    note = doc_lane.extract_note(_word_doc(f"{SECRET}\n{filler}"), DOC_MIME, "old.doc")
    assert SECRET in note
    assert note.startswith("[Document contents (extracted,")


def test_legacy_doc_piece_table_is_walked():
    """Word's complex/Fast-Saved shape: the text is only reachable via the Clx piece table."""
    note = doc_lane.extract_note(_word_doc_complex(f"Piece table: {SECRET}"), DOC_MIME, "c.doc")
    assert SECRET in note


def test_legacy_xls_cells_sheet_name_and_numbers_are_extracted():
    note = doc_lane.extract_note(_xls_book([["item", "value"], ["secret", SECRET]]), XLS_MIME,
                                 "old.xls")
    assert "[Sheet1]" in note
    assert f"secret\t{SECRET}" in note
    assert "42.5" in note


def test_legacy_xls_shared_strings_survive_a_continue_split():
    """BIFF8 splits the SST across CONTINUE records, re-stating the string's flag byte at the seam."""
    note = doc_lane.extract_note(
        _xls_book([["item", "value"], ["secret", SECRET]], split_sst=True), XLS_MIME, "old.xls")
    assert f"secret\t{SECRET}" in note


def test_legacy_ppt_text_atoms_are_extracted_from_the_record_tree():
    note = doc_lane.extract_note(_ppt_deck(["Deck title", SECRET]), PPT_MIME, "old.ppt")
    assert SECRET in note and "Deck title" in note


def test_a_mislabelled_ole_file_says_which_stream_is_missing():
    """A real OLE container that is not Word: it says so BY NAME, it does not fail silently."""
    note = doc_lane.extract_note(_cfb({"Data": b"not word" * 700}), DOC_MIME, "memo.doc")
    assert "no WordDocument stream" in note
    assert "text could not be extracted from this application/msword" in note
    assert "PINEAPPLE" not in note


def test_a_truncated_ole_file_is_refused_by_name():
    good = _word_doc(SECRET)
    for cut, reason in ((512, "truncated"), (1024, "FAT")):
        note = doc_lane.extract_note(good[:cut], DOC_MIME, "t.doc")
        assert "not extracted" in note, note
        assert reason in note, note
        assert "PINEAPPLE" not in note


def test_a_doc_that_is_not_a_compound_file_at_all_says_so():
    note = doc_lane.extract_note(b"%PDF-1.4 not actually word", DOC_MIME, "fake.doc")
    assert "not a compound (OLE) file" in note
    assert "text could not be extracted from this application/msword" in note


def test_a_doc_with_a_damaged_word_stream_is_refused_not_guessed():
    note = doc_lane.extract_note(_cfb({"WordDocument": b"\x00" * 6000}), DOC_MIME, "d.doc")
    assert "no Word FIB signature" in note


def test_an_xls_and_a_ppt_without_their_streams_say_so():
    assert "no Workbook stream" in doc_lane.extract_note(_cfb({"Data": b"x" * 6000}), XLS_MIME,
                                                         "a.xls")
    assert "'PowerPoint Document' stream" in doc_lane.extract_note(_cfb({"Data": b"x" * 6000}),
                                                                   PPT_MIME, "a.ppt")


def test_unreadable_type_gets_a_stated_reason_not_a_silent_omission():
    assert "no extractor is available" in doc_lane.extract_note(b"MZ\x90\x00",
                                                               "application/x-msdownload", "a.exe")
    assert "the upload was empty" in doc_lane.extract_note(b"", "text/plain", "e.txt")


def test_truncation_is_labelled_at_the_cap_and_the_extractor_never_exceeds_it():
    long_text = ("filler line\n" * 5000).encode()          # 60 000 chars
    note = doc_lane.extract_note(long_text, "text/plain", "big.txt")
    assert "extracted, 40000 chars" in note
    assert "truncated at 40000 chars" in note
    assert "20000 more characters were not included" in note


def test_oversize_source_is_skipped_with_the_limit_in_the_turn(monkeypatch):
    monkeypatch.setattr(doc_lane, "MAX_SOURCE_BYTES", 1024)
    note = doc_lane.extract_note(b"x" * 2048, "text/plain", "big.txt")
    assert "not extracted" in note
    assert "extraction limit" in note, "the skip states the limit it hit"
    assert "2.0 KB" in note
    assert "text could not be extracted from this text/plain" in note


def test_a_broken_extractor_still_produces_a_turn_line(monkeypatch):
    def boom(_raw):
        raise RuntimeError("pdf parser exploded")

    monkeypatch.setitem(doc_lane._EXTRACTORS, PDF_MIME, boom)
    note = doc_lane.extract_note(b"%PDF-1.4", PDF_MIME, "x.pdf")
    assert "extraction failed (RuntimeError" in note
    assert "text could not be extracted from this application/pdf" in note


def test_extraction_never_raises_on_random_bytes():
    junk = bytes(range(256)) * 40
    for mime, filename in ((PDF_MIME, "a.pdf"), (DOCX_MIME, "a.docx"), (XLSX_MIME, "a.xlsx"),
                           (ZIP_MIME, "a.zip"), ("application/x-zip-compressed", "a.zip"),
                           (DOC_MIME, "a.doc"), (XLS_MIME, "a.xls"), (PPT_MIME, "a.ppt"),
                           ("text/plain", "a.txt")):
        note = doc_lane.extract_note(junk, mime, filename)
        assert note.startswith("[Document contents")
        assert len(note) <= doc_lane.MAX_EXTRACT_CHARS + 600


# --------------------------------------------------------------------------- route end to end

@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db, tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", str(tmp_path / "images"))
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", str(tmp_path / "documents"))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": API_KEY}))
    adapter._session_db = session_db
    return adapter


@pytest.fixture
def session_id(session_db):
    return session_db.create_session("doc-text-session", "api_server")


def _client(adapter: APIServerAdapter) -> TestClient:
    app = web.Application()
    app["api_server_adapter"] = adapter
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    return TestClient(TestServer(app))


def _headers(session_id: str) -> dict:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
        "X-Hermes-Session-Key": f"android:default:{session_id}",
    }


def _upload_form(*, message: str, mime: str, data: bytes, filename: str) -> FormData:
    form = FormData()
    form.add_field("message", message)
    form.add_field("filename", filename)
    form.add_field("mime_type", mime)
    form.add_field("file", data, filename=filename, content_type=mime)
    return form


class _CapturingAgent:
    """Minimal AIAgent stand-in: records exactly what ``run_conversation`` was handed."""

    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_total_tokens = 0

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.provider = "main-provider"
        self.model = "main-model"
        self.requested_provider = "main-provider"
        self.calls: list[dict] = []

    def run_conversation(self, **kwargs):
        self.calls.append(kwargs)
        return {"final_response": FORWARD, "session_id": self.session_id}


def _turn_text(message) -> str:
    if isinstance(message, str):
        return message
    return "\n".join(str(p.get("text") or "") for p in message if isinstance(p, dict))


@pytest.mark.asyncio
async def test_uploaded_pdf_puts_its_text_in_the_turn(adapter, session_id):
    """The live bug, as a test: a PDF whose only text is a secret must REACH the model's context."""
    agent = _CapturingAgent(session_id)
    with patch.object(adapter, "_create_agent", return_value=agent), patch(
            "agent.image_routing.decide_image_input_mode") as decide:
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/attachments",
                data=_upload_form(
                    message="What secret word is written inside this document? "
                            "Answer with just the word.",
                    mime=PDF_MIME, data=_pdf([SECRET], flate=True), filename="secret.pdf"),
                headers=_headers(session_id))
            assert resp.status == 200, await resp.text()
            assert (await resp.json())["object"] == "hermes.session.chat.completion"

    text = _turn_text(agent.calls[0]["user_message"])
    assert "[Document contents (extracted," in text
    assert SECRET in text, text
    assert "[Document attached: secret.pdf (application/pdf)]" in text
    assert "[Document attached at: " in text, "the path handle stays alongside the text"
    assert not decide.called, "a document must not pay for the image-routing decision"


@pytest.mark.asyncio
async def test_uploaded_markdown_text_reaches_the_turn(adapter, session_id):
    agent = _CapturingAgent(session_id)
    with patch.object(adapter, "_create_agent", return_value=agent):
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/attachments",
                data=_upload_form(message="read this", mime="text/markdown",
                                  data=f"# Notes\n\n{SECRET}\n".encode(), filename="notes.md"),
                headers=_headers(session_id))
            assert resp.status == 200, await resp.text()

    assert SECRET in _turn_text(agent.calls[0]["user_message"])


@pytest.mark.asyncio
async def test_uploaded_zip_puts_its_members_text_in_the_turn(adapter, session_id):
    agent = _CapturingAgent(session_id)
    with patch.object(adapter, "_create_agent", return_value=agent):
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/attachments",
                data=_upload_form(message="What is in this archive?", mime=ZIP_MIME,
                                  data=_zip_with({"notes.txt": f"hello\n{SECRET}\n"}),
                                  filename="bundle.zip"),
                headers=_headers(session_id))
            assert resp.status == 200, await resp.text()

    text = _turn_text(agent.calls[0]["user_message"])
    assert "[Document contents (extracted," in text
    assert SECRET in text
    assert "[Archive limits for this upload:" in text


@pytest.mark.asyncio
async def test_uploaded_legacy_doc_puts_its_text_in_the_turn(adapter, session_id):
    agent = _CapturingAgent(session_id)
    with patch.object(adapter, "_create_agent", return_value=agent):
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/attachments",
                data=_upload_form(
                    message="What secret word is written inside this document? "
                            "Answer with just the word.",
                    mime=DOC_MIME, data=_word_doc(f"old memo\n{SECRET}\n"), filename="old.doc"),
                headers=_headers(session_id))
            assert resp.status == 200, await resp.text()

    text = _turn_text(agent.calls[0]["user_message"])
    assert SECRET in text, text
    assert "[Document attached: old.doc (application/msword)]" in text


@pytest.mark.asyncio
async def test_uploaded_legacy_spreadsheet_and_deck_reach_the_turn(adapter, session_id):
    agent = _CapturingAgent(session_id)
    with patch.object(adapter, "_create_agent", return_value=agent):
        async with _client(adapter) as cli:
            for mime, filename, data in (
                    (XLS_MIME, "old.xls", _xls_book([["secret", SECRET]])),
                    (PPT_MIME, "old.ppt", _ppt_deck([SECRET]))):
                resp = await cli.post(
                    f"/api/sessions/{session_id}/chat/attachments",
                    data=_upload_form(message="read this", mime=mime, data=data, filename=filename),
                    headers=_headers(session_id))
                assert resp.status == 200, await resp.text()

    for call in agent.calls:
        assert SECRET in _turn_text(call["user_message"])


@pytest.mark.asyncio
async def test_a_binary_zip_upload_runs_a_turn_that_says_it_could_not_be_read(adapter,
                                                                              session_id):
    """The honest failure path, through the route: 200, a turn, and no invented contents."""
    agent = _CapturingAgent(session_id)
    with patch.object(adapter, "_create_agent", return_value=agent):
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/attachments",
                data=_upload_form(message="read this", mime=ZIP_MIME,
                                  data=_zip_with({"blob.bin": bytes(range(256)) * 4}),
                                  filename="files.zip"),
                headers=_headers(session_id))
            assert resp.status == 200, await resp.text()

    text = _turn_text(agent.calls[0]["user_message"])
    assert "[Document contents not extracted:" in text
    assert "text could not be extracted from this application/zip" in text


@pytest.mark.asyncio
async def test_an_image_upload_is_untouched_by_the_document_lane(adapter, session_id):
    """The image path must not change: one inline image part, no ``[Document contents`` note."""
    agent = _CapturingAgent(session_id)
    with patch.object(adapter, "_create_agent", return_value=agent), patch(
            "agent.image_routing.decide_image_input_mode", return_value="native"):
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/attachments",
                data=_upload_form(message="What is this?", mime="image/png", data=PNG_BYTES,
                                  filename="shot.png"),
                headers=_headers(session_id))
            assert resp.status == 200, await resp.text()

    message = agent.calls[0]["user_message"]
    assert isinstance(message, list)
    parts = [p for p in message if isinstance(p, dict) and p.get("type") == "image_url"]
    assert len(parts) == 1
    assert base64.b64decode(parts[0]["image_url"]["url"].split(",", 1)[1]) == PNG_BYTES
    assert "[Document contents" not in _turn_text(message)


@pytest.mark.asyncio
async def test_a_long_document_reaches_the_turn_truncated_and_labelled(adapter, session_id):
    agent = _CapturingAgent(session_id)
    with patch.object(adapter, "_create_agent", return_value=agent):
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/attachments",
                data=_upload_form(message="summarise", mime="text/plain",
                                  data=("row\n" * 20000).encode(), filename="big.txt"),
                headers=_headers(session_id))
            assert resp.status == 200, await resp.text()

    text = _turn_text(agent.calls[0]["user_message"])
    assert "truncated at 40000 chars" in text


@pytest.mark.asyncio
async def test_an_unsupported_payload_still_gets_415(adapter, session_id):
    with patch.object(adapter, "_create_agent") as create:
        async with _client(adapter) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/attachments",
                data=_upload_form(message="run this", mime="application/x-nonsense",
                                  data=b"nope", filename="payload.xyz"),
                headers=_headers(session_id))
            assert resp.status == 415, await resp.text()
    assert not create.called, "a refused upload must not run a turn"
