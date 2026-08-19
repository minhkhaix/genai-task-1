#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vn_legal_chunker.py
===================
Chunk Vietnamese legal documents (Luật / Nghị định / Thông tư) into one chunk
per "Điều" (Article), ready for LLM fine-tuning or RAG indexing.

Core technique (as specified):
    * re.finditer()          -> stream every article without building a huge list of splits
    * re.DOTALL              -> "." also matches "\\n", so an article body may span many lines
    * positive lookahead     -> (?=...) marks where the NEXT article starts without consuming it,
                                so the boundary character stays available for the next match

Output: list[dict] with keys "dieu" (title) and "content" (body), JSON-ready.

Author's notes on real-world Vietnamese legal text are in README_chunker.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from typing import Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# 1. The article marker
# ---------------------------------------------------------------------------
# "Điều" spelling variants seen in scraped corpora (thuvienphapluat, vbpl, OCR'd PDFs):
#   Điều  - correct (Đ = U+0110)
#   Ðiều  - "Eth" (Ð = U+00D0) instead of D-with-stroke; extremely common in old exports
#   Điền  - OCR/typo slip ("u" read as "n"). Nghị định 26/2019 really does contain "Điền 29."
#   ĐIỀU  - all-caps headings
_DIEU_WORD = r"(?:Điều|Ðiều|ĐIỀU|ÐIỀU|Điền|Ðiền)"

# Article number: 1, 2, ... plus the "inserted article" form 24a / 24b used by amendments.
_DIEU_NUM = r"\d{1,3}[a-zA-Zđ]?"

# Guard against cross-references such as "khoản 3 Điều 13", "tại Điều 20", "và Điều 18".
# A real heading is preceded by end-of-sentence punctuation, an UPPERCASE letter
# (".. NHỮNG QUY ĐỊNH CHUNG Điều 1."), a newline, or start-of-string --- never by a
# lowercase word or a digit. Fixed width (2 chars), so Python's re accepts the lookbehind.
_LOWER = (
    "a-z0-9"
    "àảãáạăằẳẵắặâầẩẫấậ"
    "èẻẽéẹêềểễếệ"
    "ìỉĩíị"
    "òỏõóọôồổỗốộơờởỡớợ"
    "ùủũúụưừửữứự"
    "ỳỷỹýỵ"
    "đ"
)
_NOT_A_CROSS_REF = rf"(?<![{_LOWER}] )"

# The heading itself: "Điều 1." / "Điều 2:" / "Điều 24a ."
MARKER = rf"{_NOT_A_CROSS_REF}{_DIEU_WORD}\s+{_DIEU_NUM}\s*[.:]"

# finditer pattern: grab the heading, then everything (DOTALL) lazily up to the
# lookahead for the next heading -- or \Z, the end of the text.
ARTICLE_RE = re.compile(
    rf"(?P<marker>{MARKER})"      # "Điều 1."
    rf"(?P<rest>.*?)"             # title + body, may span newlines thanks to re.DOTALL
    rf"(?={MARKER}|\Z)",          # positive lookahead: stop right before the next Điều
    re.DOTALL,
)

# Structural headings that sit BETWEEN two articles and would otherwise be glued
# onto the tail of the previous article's content.
CHUONG_RE = re.compile(r"Chương\s+[IVXLCDM]+\b")
MUC_RE = re.compile(r"Mục\s+\d+\s*[.:]")   # the dot matters: "Mục 1. ĐỒNG QUẢN LÝ" is a
                                           # heading, "tại Mục 1 Phụ lục VI" is a reference
# NOTE the leading "\s" inside the negated class. Without it the class also matches
# a space, so "\s+" and "[^...]+" can both consume the same whitespace -- the classic
# (a+)* ambiguity. Every failed match then re-partitions the tail exponentially:
# measured on this very pattern, a 104-char tail took 90 ms, 119 chars 731 ms and
# 134 chars 5.8 s. A 300-char uppercase chapter title never returns, which looks
# exactly like a corpus run "stuck at 1%". Excluding \s makes the split unique and
# the scan linear.
_TRAILING_STRUCTURE_RE = re.compile(
    r"(?:\s|^)(?:Chương\s+[IVXLCDM]+|Mục\s+\d+\s*[.:]?)"
    r"(?:\s+[^\sa-zàảãáạăằẳẵắặâầẩẫấậèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợùủũúụưừửữứựỳỷỹýỵđ.;,]+)*\s*$"
)

# Boilerplate after the last article (signature block, distribution list, attachments).
DEFAULT_STOP_MARKERS = (
    "Nơi nhận:",
    "TM. CHÍNH PHỦ",
    "TM. CHINH PHU",
    "KT. BỘ TRƯỞNG",
    "FILE ĐƯỢC ĐÍNH KÈM",
    "PHỤ LỤC I",
)


# ---------------------------------------------------------------------------
# 2. Cleaning helpers
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """NFC-normalise (Vietnamese diacritics arrive both pre-composed and decomposed
    from different scrapers -- 'Điều' would otherwise not match itself) and drop
    zero-width / non-breaking junk."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")   # Windows / old-Mac line endings
    text = text.replace(" ", " ").replace("﻿", "")
    text = re.sub(r"[​-‏‪-‮]", "", text)
    return text


_WS_RUN_RE = re.compile(r"\s+")


def clean_content(text: str) -> str:
    """Requirement 3: consecutive newlines inside the body collapse to ONE space.
    Also collapses the runs of spaces that produces, so the chunk is a single
    tidy paragraph -- which is what an LLM tokenizer wants."""
    # One pass, one unambiguous pattern. The earlier two-step version
    # (r"[ \t]*\n[ \t\n]*" then r"[ \t]{2,}") is quadratic on a long run of spaces
    # with no newline in it -- the engine re-scans the whole run from every start
    # position and fails: 8k spaces took 77 ms, 32k took 1.2 s, 128k took 19 s.
    # Scraped HTML tables produce exactly those runs. "\s+" cannot backtrack.
    return _WS_RUN_RE.sub(" ", text).strip()


def _strip_trailing_structure(body: str) -> str:
    """Remove a 'Chương II ...' / 'Mục 3. ...' heading that trails an article body.

    Those headings live between two articles, so the lookahead hands them to the
    PREVIOUS article. Left alone they poison the last sentence of every chunk that
    happens to sit at a chapter boundary.
    """
    prev = None
    while prev != body:
        prev = body
        body = _TRAILING_STRUCTURE_RE.sub("", body).rstrip()
    return body


def _cut_at_stop_marker(body: str, stop_markers: Iterable[str]) -> str:
    """Trim the signature / distribution boilerplate glued to the final article."""
    cut = len(body)
    for marker in stop_markers:
        idx = body.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    if cut == len(body):
        return body                      # nothing trimmed -> leave the text untouched
    return body[:cut].rstrip(" -–—/;")   # tidy the dangling "./." separator only


# ---------------------------------------------------------------------------
# 3. Title / body separation
# ---------------------------------------------------------------------------
# In a well-formed document the heading owns its own line:
#     Điều 1. Phạm vi điều chỉnh\nLuật này quy định...
# In a flattened corpus (everything on one line) nothing separates the title from
# the body, so we need heuristics. Two signals, in priority order:
#   (1) a body opener phrase  -- "Đối tượng áp dụng | Nghị định này áp dụng..."
#   (2) the khoản-1 marker    -- "Công nhận và giao quyền... | 1. Hồ sơ đề nghị..."
_CLAUSE_ONE_RE = re.compile(r"(?<=\s)1\s*[.)](?=\s)")

_BODY_PHRASES = (
    r"(?:Nghị định|Luật|Thông tư|Quyết định)\s+này",
    r"Trong\s+(?:Nghị định|Luật|Thông tư)\s+này",
    r"Tổ chức,\s*cá nhân",
    r"Điểm\s+[a-zđ]\b",
    r"Khoản\s+\d+\b",
    r"Cơ quan\s+(?:có thẩm quyền|quản lý)",
    r"Bộ\s+(?:trưởng|Nông nghiệp)",
    r"Tàu cá\b",
)
_BODY_PHRASE_RE = re.compile(r"(?<=\s)(?:%s)" % "|".join(_BODY_PHRASES))

# If the phrase is preceded by one of these, it is still part of the title
# ("...thủ tục hành chính trong Nghị định này"), not the start of the body.
_FUNCTION_WORDS = {
    "trong", "tại", "theo", "của", "và", "với", "về", "cho", "từ", "đến",
    "kèm", "bởi", "do", "hoặc", "như", "bằng",
}


def _phrase_cut(rest: str, limit: int) -> Optional[int]:
    for m in _BODY_PHRASE_RE.finditer(rest[:limit]):
        if m.start() == 0:
            continue
        prev_token = rest[: m.start()].rstrip().rsplit(" ", 1)[-1].strip(",;:.").lower()
        if prev_token in _FUNCTION_WORDS:
            continue                     # a preposition -> we are mid-title
        return m.start()
    return None


def _split_title_body(rest: str, max_title_chars: int = 200) -> tuple[str, str, bool]:
    """Return (title_text, body_text, confident)."""
    rest = rest.strip()
    if not rest:
        return "", "", True

    # --- Case A: the heading owns its line (exact, no guessing) ---
    if "\n" in rest:
        title, body = rest.split("\n", 1)
        return title.strip(), body.strip(), True

    # --- Case B: flattened single-line text -> heuristics, never "confident" ---
    limit = max_title_chars + 120
    phrase = _phrase_cut(rest, limit)
    clause = _CLAUSE_ONE_RE.search(rest[:limit])
    clause_at = clause.start() if clause else None

    cut = None
    if phrase is not None and (clause_at is None or phrase < clause_at):
        cut = phrase
    elif clause_at is not None:
        cut = clause_at

    if cut:
        return rest[:cut].strip(), rest[cut:].strip(), False

    # No usable signal. Never guess a title at the cost of losing text: keep the
    # whole span as content and let "dieu" be the bare marker ("Điều 75.").
    return "", rest.strip(), False


# ---------------------------------------------------------------------------
# 4. The main function
# ---------------------------------------------------------------------------
def extract_dieu(
    text: str,
    *,
    max_title_chars: int = 200,
    strip_structure: bool = True,
    stop_markers: Iterable[str] = DEFAULT_STOP_MARKERS,
    keep_empty: bool = False,
    with_metadata: bool = False,
    max_chunk_chars: Optional[int] = None,
    min_chunk_chars: int = 250,
    repeat_lead: bool = False,
    prefix_title: bool = False,
) -> List[Dict[str, object]]:
    """Split a Vietnamese legal document into one chunk per "Điều".

    Parameters
    ----------
    text
        Plain-text document. Line breaks are preserved as information: if the
        heading sits on its own line, the title/body split is exact.
    max_title_chars
        Safety cap for the flattened-text heuristic.
    strip_structure
        Drop a trailing "Chương .../ Mục ..." heading from an article body.
    stop_markers
        Substrings that end the operative text (signature block, "Nơi nhận:" ...).
    keep_empty
        Keep articles whose body is empty (heading-only fragments).
    with_metadata
        Also emit so_dieu, chuong, muc, char_len, title_confident.
    max_chunk_chars
        If set, articles longer than this are sub-split at khoản boundaries
        (see split_by_khoan); the title is repeated on every piece.

    Returns
    -------
    list of {"dieu": <title>, "content": <body>} -- JSON-ready.
    """
    text = normalize_text(text)

    # Track which Chương / Mục each article belongs to (useful context for training).
    chuong_marks = [(m.start(), m.group(0).strip()) for m in CHUONG_RE.finditer(text)]
    muc_marks = [(m.start(), m.group(0).strip()) for m in MUC_RE.finditer(text)]

    def _context(pos: int, marks) -> Optional[str]:
        found = None
        for start, label in marks:
            if start < pos:
                found = label
            else:
                break
        return found

    results: List[Dict[str, object]] = []

    for match in ARTICLE_RE.finditer(text):
        marker = match.group("marker").strip()          # "Điều 1."
        rest = match.group("rest")

        # Trim the between-articles rubbish BEFORE splitting title from body,
        # otherwise a short last article swallows the signature block as its "title".
        if strip_structure:
            rest = _strip_trailing_structure(rest)
        rest = _cut_at_stop_marker(rest, stop_markers)

        title_text, body, confident = _split_title_body(rest, max_title_chars)

        dieu = f"{marker} {title_text}".strip() if title_text else marker
        dieu = clean_content(dieu)
        content = clean_content(body)

        if not content and not keep_empty:
            continue

        chunk: Dict[str, object] = {"dieu": dieu, "content": content}

        if with_metadata:
            num = re.search(rf"{_DIEU_NUM}", marker)
            chunk.update(
                so_dieu=num.group(0) if num else None,
                chuong=_context(match.start(), chuong_marks),
                muc=_context(match.start(), muc_marks),
                char_len=len(content),
                title_confident=confident,
            )

        results.append(chunk)

    if max_chunk_chars:
        expanded: List[Dict[str, object]] = []
        for chunk in results:
            expanded.extend(
                split_by_khoan(
                    chunk,
                    max_chars=max_chunk_chars,
                    min_chars=min_chunk_chars,
                    repeat_lead=repeat_lead,
                    prefix_title=prefix_title,
                )
            )
        return expanded

    return results


# ---------------------------------------------------------------------------
# 4b. Sub-splitting long articles at "khoản" (and "điểm") boundaries
# ---------------------------------------------------------------------------
# Điều 44 / Điều 63 / Điều 65 run past 6.000 characters while Điều 24 is under 300.
# For training or embedding you want even pieces -- split at the legal seams, not
# at an arbitrary character count, and repeat the article title on every piece so
# each one is still a valid citation on its own.
#
# The trap: "1." / "2." also appear as "Mẫu số 09.BT", "ngày 01 tháng 01 năm 2020.",
# "từ điểm 01 đến điểm 18.", "hệ số 0,3". A pattern alone cannot tell them apart.
# The fix is that khoản numbering is SEQUENTIAL: only accept a candidate whose
# number is exactly the next one expected (1, then 2, then 3 ...). Everything else
# is prose that happens to contain a digit and a dot.

_UPPER = "A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-ỹ"
_KHOAN_CAND_RE = re.compile(rf"(?:^|(?<=\s))(\d{{1,2}})\s*\.\s+(?=[{_UPPER}\d\"“(])")

# Vietnamese điểm lettering: no f/j/w/z, and "đ" sits between d and e.
_DIEM_SEQUENCE = list("abcd") + ["đ"] + list("eghiklmnopqrstu") + ["ư"] + list("vxy")
_DIEM_CAND_RE = re.compile(r"(?:^|(?<=\s))([a-zđư])\)\s+")

_SENTENCE_END_RE = re.compile(r"(?<=[.;:])\s+")


def _sequential_spans(text: str, cand_re: re.Pattern, sequence: List[str]) -> List[tuple]:
    """Return [(label, start, end)] for markers that follow `sequence` in order."""
    candidates = list(cand_re.finditer(text))
    if not candidates:
        return []

    # Normally the numbering starts at 1 (or "a"). If the text is a fragment that
    # opens on a different marker -- "3. Tuyến bờ ..." -- anchor to that instead,
    # but only when it sits at position 0, so prose digits can never seed the run.
    expected = 0
    first = candidates[0].group(1).lstrip("0") or "0"
    if candidates[0].start() == 0 and first in sequence:
        expected = sequence.index(first)

    hits = []
    for m in candidates:
        token = m.group(1).lstrip("0") or "0"
        if expected < len(sequence) and token == sequence[expected]:
            hits.append((token, m.start()))
            expected += 1
    spans = []
    for i, (label, start) in enumerate(hits):
        end = hits[i + 1][1] if i + 1 < len(hits) else len(text)
        spans.append((label, start, end))
    return spans


def _khoan_spans(text: str) -> List[tuple]:
    return _sequential_spans(text, _KHOAN_CAND_RE, [str(i) for i in range(1, 100)])


def _diem_spans(text: str) -> List[tuple]:
    return _sequential_spans(text, _DIEM_CAND_RE, _DIEM_SEQUENCE)


def _hard_split(text: str, max_chars: int) -> List[str]:
    """Last resort: break an over-long segment at the strongest boundary available.

    Delegates to smart_split, which protects formulas, rating scales, key-value
    runs, tables and reference lists from being cut through, and which never
    treats a colon as a sentence end. The old implementation split on
    ``(?<=[.;:])\\s+``, which severed "Cấp 3:" from its value and "Bảng C.1 -
    Mức giới hạn:" from its rows -- both observed in the TCVN sample.

    Falls back to the original sentence-only behaviour if smart_split is not
    importable, so this module keeps working standalone.
    """
    if len(text) <= max_chars:
        return [text]
    try:
        from smart_split import split_text
        return split_text(text, max_chars=max_chars, min_chars=0)
    except ImportError:                                   # pragma: no cover
        out, cur = [], ""
        for sentence in _SENTENCE_END_RE.split(text):
            if cur and len(cur) + 1 + len(sentence) > max_chars:
                out.append(cur.strip())
                cur = ""
            cur = f"{cur} {sentence}".strip()
        if cur:
            out.append(cur.strip())
        return out


def _segments(content: str, max_chars: int) -> List[tuple]:
    """Cut `content` into (label, text) pieces at khoản seams, then điểm seams."""
    spans = _khoan_spans(content)
    if not spans:
        return [(None, part) for part in _hard_split(content, max_chars)]

    segs: List[tuple] = []
    lead = content[: spans[0][1]].strip()          # text before khoản 1 ("... như sau:")
    if lead:
        segs.append((None, lead))

    for label, start, end in spans:
        body = content[start:end].strip()
        if len(body) <= max_chars:
            segs.append((label, body))
            continue
        # too long -> try the điểm level (a), b), c) ...
        diem = _diem_spans(body)
        if diem:
            head = body[: diem[0][1]].strip()
            if head:
                segs.append((label, head))
            for dlabel, dstart, dend in diem:
                piece = body[dstart:dend].strip()
                for k, sub in enumerate(_hard_split(piece, max_chars)):
                    segs.append((f"{label}{dlabel}" if k == 0 else f"{label}{dlabel}+", sub))
        else:
            for k, sub in enumerate(_hard_split(body, max_chars)):
                segs.append((label if k == 0 else f"{label}+", sub))
    return segs


def _pack(segments: List[tuple], max_chars: int, min_chars: int) -> List[tuple]:
    """Greedily glue consecutive segments together while they fit."""
    packed: List[tuple] = []
    labels: List[str] = []
    buf = ""
    for label, text in segments:
        if buf and len(buf) + 1 + len(text) > max_chars:
            packed.append((labels, buf))
            labels, buf = [], ""
        if label:
            labels.append(label)
        buf = f"{buf} {text}".strip()
    if buf:
        # Don't leave a stub -- but don't blow the budget to avoid one either.
        # The original merged unconditionally, which pushed the final piece up to
        # min_chars past max_chars; on a mixed corpus that was 429 chunks over
        # budget at a mean of 973 characters, silently truncated at index time.
        if (packed and len(buf) < min_chars
                and len(packed[-1][1]) + 1 + len(buf) <= max_chars):
            plabels, ptext = packed[-1]
            packed[-1] = (plabels + labels, f"{ptext} {buf}".strip())
        else:
            packed.append((labels, buf))
    return packed


def split_by_khoan(
    chunk: Dict[str, object],
    *,
    max_chars: int = 1500,
    min_chars: int = 250,
    repeat_lead: bool = False,
    prefix_title: bool = False,
) -> List[Dict[str, object]]:
    """Split one article chunk into pieces at khoản boundaries.

    The "dieu" title is repeated on every piece, so a retrieved fragment still
    carries its citation. Short articles come back unchanged (single-item list).

    Parameters
    ----------
    max_chars    target ceiling per piece (soft: a single indivisible khoản may exceed it)
    min_chars    a trailing piece shorter than this is merged back into the previous one
    repeat_lead  also copy the article's lead-in sentence ("... được quy định như sau:")
                 onto every piece -- useful for RAG, costs a little duplication
    prefix_title prepend "Điều N. Title — " to the text of each piece as well

    Returns
    -------
    list of chunks with the same keys plus: khoan, part, n_parts.
    """
    content = str(chunk.get("content", ""))
    title = str(chunk.get("dieu", ""))

    # --prefix-title prepends "Điều 63. … — " to the text AFTER splitting, so the
    # budget has to be charged for it up front: the embedder sees the final
    # string, not the body. Without this every prefixed chunk ran over by the
    # length of its own title.
    if prefix_title and title:
        max_chars = max(200, max_chars - len(title) - 3)
        min_chars = min(min_chars, max_chars // 2)

    if len(content) <= max_chars:
        out = dict(chunk)
        out.update(khoan=None, part=1, n_parts=1)
        if prefix_title and title:
            out["content"] = f"{title} — {content}"
            out["char_len"] = len(out["content"])
        return [out]

    segments = _segments(content, max_chars)

    lead = ""
    if repeat_lead and segments and segments[0][0] is None:
        lead = segments[0][1]
        if len(lead) > max_chars // 3:      # a long lead-in is a paragraph, not a header
            lead = ""

    packed = _pack(segments, max_chars, min_chars)

    pieces: List[Dict[str, object]] = []
    for i, (labels, text) in enumerate(packed, start=1):
        if lead and i > 1:
            text = f"{lead} {text}"
        if prefix_title and title:
            text = f"{title} — {text}"
        piece = dict(chunk)
        khoan = None
        if labels:
            khoan = labels[0] if len(labels) == 1 else f"{labels[0]}–{labels[-1]}"
        piece.update(
            dieu=title,                     # <- the title is repeated on every piece
            content=text,
            khoan=khoan,
            part=i,
            n_parts=len(packed),
        )
        if "char_len" in piece:
            piece["char_len"] = len(text)
        pieces.append(piece)
    return pieces


def chunk_plain_text(
    text: str,
    *,
    max_chars: int = 1500,
    min_chars: int = 250,
    title: str = "",
) -> List[Dict[str, object]]:
    """Fallback for documents that contain no "Điều" at all.

    A large slice of a Vietnamese legal corpus is NOT article-structured:
    TCVN / QCVN technical standards, công văn, công điện, biểu mẫu, phụ lục.
    They use "Phụ lục B", "B.1", "Cấp 1:", "Bảng C.1" instead. Running only
    extract_dieu() on those returns [] and the whole document disappears --
    silently, which is the worst way to lose training data. This windows the
    text on sentence boundaries instead, so the document still contributes.
    """
    # This path carries the TCVN / QCVN standards, phụ lục and biểu mẫu -- which
    # is precisely where the tables, rating scales and formulas live. It gets
    # smart_split directly rather than _hard_split + _pack: smart_split already
    # fills to the budget, so packing on top of it would re-merge across the
    # structural boundaries it just protected.
    content = clean_content(normalize_text(text))
    if not content:
        return []

    try:
        from smart_split import split_text
        packed = [(None, p) for p in split_text(content, max_chars=max_chars,
                                                min_chars=min_chars)]
    except ImportError:                                   # pragma: no cover
        packed = _pack([(None, p) for p in _hard_split(content, max_chars)],
                       max_chars, min_chars)
    return [
        {
            "dieu": title,
            "content": body,
            "khoan": None,
            "part": i,
            "n_parts": len(packed),
            "so_dieu": None,
            "chuong": None,
            "muc": None,
            "char_len": len(body),
            "title_confident": False,
        }
        for i, (_labels, body) in enumerate(packed, start=1)
    ]


# ---------------------------------------------------------------------------
# 5. Corpus helpers (for records shaped like {"id", "link", "name", "passage"})
# ---------------------------------------------------------------------------
def chunk_record(record: Dict[str, object], text_field: str = "passage", **kwargs) -> List[Dict[str, object]]:
    """Chunk one corpus record and carry the document metadata onto every chunk."""
    text = str(record.get(text_field, "") or "")
    chunks = extract_dieu(text, with_metadata=True, **kwargs)
    doc_id = record.get("id")
    out = []
    for i, chunk in enumerate(chunks, start=1):
        chunk_id = f"{doc_id}_{chunk.get('so_dieu') or i}"
        if chunk.get("n_parts", 1) > 1:
            chunk_id += f"_p{chunk['part']}"
        out.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "doc_name": record.get("name"),
                "link": record.get("link"),
                **chunk,
            }
        )
    return out


def _iter_records(path: str) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as fh:
        if path.endswith(".jsonl"):
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
        else:
            data = json.load(fh)
            yield from (data if isinstance(data, list) else [data])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Chunk Vietnamese legal documents by Điều.")
    ap.add_argument("input", help="input .json / .jsonl / .txt")
    ap.add_argument("-o", "--output", default="chunks.jsonl", help="output .jsonl")
    ap.add_argument("--text-field", default="passage")
    ap.add_argument("--min-chars", type=int, default=0, help="drop chunks shorter than this")
    ap.add_argument("--max-chars", type=int, default=None,
                    help="sub-split articles longer than this at khoản boundaries")
    ap.add_argument("--repeat-lead", action="store_true",
                    help="copy the article lead-in sentence onto every sub-chunk")
    ap.add_argument("--prefix-title", action="store_true",
                    help="prepend 'Điều N. Title — ' to the text of every chunk")
    args = ap.parse_args(argv)

    split_opts = dict(
        max_chunk_chars=args.max_chars,
        repeat_lead=args.repeat_lead,
        prefix_title=args.prefix_title,
    )
    total = 0
    with open(args.output, "w", encoding="utf-8") as out:
        if args.input.endswith(".txt"):
            with open(args.input, "r", encoding="utf-8") as fh:
                records = [{"id": None, "name": args.input, "passage": fh.read()}]
        else:
            records = _iter_records(args.input)

        for record in records:
            for chunk in chunk_record(record, text_field=args.text_field, **split_opts):
                if len(chunk["content"]) < args.min_chars:
                    continue
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total += 1

    print(f"{total} chunks -> {args.output}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# 6. Demo
# ---------------------------------------------------------------------------
SAMPLE = """
Điều 1. Phạm vi điều chỉnh
Luật này quy định về xử lý vi phạm hành chính.
1. Tổ chức cá nhân...
2. Các hành vi...
Điều 2: Đối tượng áp dụng
Luật này áp dụng đối với:
a) Cơ quan nhà nước;
b) Tổ chức, cá nhân.
"""

if __name__ == "__main__":
    if len(sys.argv) > 1:
        raise SystemExit(main())
    print(json.dumps(extract_dieu(SAMPLE), ensure_ascii=False, indent=2))