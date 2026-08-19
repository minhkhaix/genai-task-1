#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smart_split.py — structure-aware recursive splitting for Vietnamese legal and
technical text, with atomic-block protection.

Replaces the last-resort splitter in vn_legal_chunker (`_hard_split`) and the
whole plain-text path (`chunk_plain_text`), which cut on a naive
`(?<=[.;:])\\s+` rule. That rule treats every colon as a sentence end, which is
exactly wrong for this corpus: it severs `Cấp 3:` from its value, `Bảng C.1 -
Mức giới hạn:` from its rows, and `2.` from the khoản it introduces.

Three ideas, in order of importance:

1. ATOMIC BLOCKS. Formulas, rating scales, key-value runs, table bodies and
   reference lists are found first and marked unsplittable. No cut position may
   fall inside one. This is what keeps a table together; a separator hierarchy
   alone will not.

2. A BOUNDARY LATTICE, not a recursive descent. Every separator level proposes
   candidate cut positions over the whole text at once. Splitting then walks
   forward and, for each piece, takes the *highest-level* boundary that lands
   inside the budget window. That is recursive character splitting's idea
   without its weakness: it never over-fragments, because it always fills to the
   budget before cutting, and it degrades one level at a time only where needed.

3. A SENTENCE SPLITTER THAT KNOWS THE DOMAIN. `B.1`, `09.BT`, `0,05 mg/kg`,
   `26/2019/NĐ-CP`, `QCVN 01-38:2010/BNNPTNT`, `TS.`, `v.v.` are not sentence
   ends. Neither is a colon.

Public API
----------
    smart_split(text, max_chars=900, min_chars=250, overlap=0) -> list[Piece]
    detect_atomic_spans(text)  -> list[(start, end, kind)]
    split_sentences_vi(text)   -> list[str]
    measure_breakage(chunks)   -> dict     # so the improvement is measured, not claimed
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 0. character classes
# ---------------------------------------------------------------------------
_VN_LOWER = ("a-z0-9àảãáạăằẳẵắặâầẩẫấậèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợ"
             "ùủũúụưừửữứựỳỷỹýỵđ")
_VN_UPPER = "A-ZÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬÈẺẼÉẸÊỀỂỄẾỆÌỈĨÍỊÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỚỢÙỦŨÚỤƯỪỬỮỨỰỲỶỸÝỴĐ"

# Module-level switch, because the call chain into _hard_split runs three frames
# deep inside vn_legal_chunker and threading a flag through all of it would be
# worse than one documented global. chunk_corpus.py --no-protect-structure sets it.
PROTECT_DEFAULT = True

# ---------------------------------------------------------------------------
# 1. atomic blocks — regions no cut may fall inside
# ---------------------------------------------------------------------------
# Tokens that must never be split, however tight the budget gets.
_TOKEN_PATTERNS = [
    (r"\d{1,4}/\d{4}/[A-ZĐ]+(?:-[A-ZĐ]+)*", "doc_code"),      # 26/2019/NĐ-CP
    (r"\b[A-Z]{2,6}\s?\d{1,5}(?:-\d{1,4})?:\d{4}(?:/[A-ZĐ\-]+)?", "standard_code"),  # QCVN 01-38:2010/BNNPTNT
    (r"\b\d{1,2}/\d{1,2}/\d{4}\b", "date"),                    # 10/12/2010
    (r"\b\d+[.,]\d+(?:\s*%|\s*mg/kg|\s*m2|\s*kg)?", "number"),  # 0,05 mg/kg
    (r"\b[A-ZĐ]\.\d+(?:\.\d+)*", "outline_id"),                # B.1, C.1.2
    (r"\b(?:TCVN|QCVN|ISO|TCN)\s*[\d\-:./]+", "standard_ref"),
]

# Multi-item structures. Each must match at least TWO items, so ordinary prose
# containing one colon is never swallowed.
_SCALE_ITEM = rf"(?:Cấp|Mức|Loại|Hạng|Bậc|Nhóm)\s+\d+\s*(?:\([^)]{{0,30}}\))?\s*:"
_BLOCK_PATTERNS = [
    # a formula: a stretch containing "=" up to the next sentence end
    (rf"[^.\n]{{0,150}}=\s*[^.\n]{{1,250}}(?:\.|$)", "formula"),
    # a rating scale: "Cấp 1: … Cấp 3: … Cấp 5: …"
    (rf"{_SCALE_ITEM}(?:(?!{_SCALE_ITEM}).)*(?:{_SCALE_ITEM}(?:(?!{_SCALE_ITEM}).)*){{1,}}", "scale"),
    # a key-value run: "Chỉ tiêu Asen 0,5 mg/kg; Chỉ tiêu Chì 0,3 mg/kg; …"
    (rf"(?:[{_VN_UPPER}][^;:\n]{{2,70}}[:\s][^;\n]{{1,90}};\s*){{2,}}[^;\n]{{0,90}}\.?", "kv_run"),
    # a table caption and the run that follows it
    (rf"(?:Bảng|Biểu|Phụ lục|Mẫu số)\s+[A-ZĐ0-9IVX][\w.\-]*\s*[-–:]?[^.\n]{{0,400}}", "table"),
    # a reference list: "[1] … [2] …"
    (r"(?:\[\d{1,3}\][^\[\n]{1,250}){2,}", "refs"),
]

# Inside a protected block, these are the seams BETWEEN items. A rating scale of
# 40 rows must not be indivisible just because it is one block -- the guarantee
# is "never cut through a row", not "never cut a long run at all".
_ITEM_CUT_RE = re.compile(
    r"(?:(?<=\s)|^)(?:(?:Cấp|Mức|Loại|Hạng|Bậc|Nhóm)\s+\d+\s*[(:]"
    r"|Chỉ tiêu\b|Bảng\s|Biểu\s|Phụ lục\s|\[\d{1,3}\])"
    r"|(?<=;)\s+"
)

_TOKEN_RE = re.compile("|".join(f"(?:{p})" for p, _ in _TOKEN_PATTERNS))
_COMPILED_BLOCKS = [(re.compile(p, re.DOTALL), kind) for p, kind in _BLOCK_PATTERNS]


def _merge_spans(spans: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """Overlapping protected regions become one region."""
    if not spans:
        return []
    spans = sorted(spans)
    out = [list(spans[0])]
    for start, end, kind in spans[1:]:
        # strict overlap only. Two formulas that merely TOUCH stay separate --
        # merging them would protect the boundary between them, which is exactly
        # where the splitter should be allowed to cut.
        if start < out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
            if kind not in out[-1][2].split("+"):
                out[-1][2] += "+" + kind
        else:
            out.append([start, end, kind])
    return [(a, b, k) for a, b, k in out]


def detect_atomic_spans(text: str, max_atomic: int = 2000) -> List[Tuple[int, int, str]]:
    """Regions that must not be cut through.

    `max_atomic` is a safety valve: a "table" that swallowed half the document
    is more likely a false positive than a real table, and protecting it would
    defeat the budget entirely. Oversized matches are dropped, not honoured.
    """
    spans: List[Tuple[int, int, str]] = []
    for m in _TOKEN_RE.finditer(text):
        spans.append((m.start(), m.end(), "token"))
    for rx, kind in _COMPILED_BLOCKS:
        for m in rx.finditer(text):
            if 0 < m.end() - m.start() <= max_atomic:
                spans.append((m.start(), m.end(), kind))
    return _merge_spans(spans)


# ---------------------------------------------------------------------------
# 2. a sentence splitter that knows the domain
# ---------------------------------------------------------------------------
# Only abbreviations that genuinely appear MID-sentence. Organisation codes and
# units are deliberately absent: "... của Bộ NNPTNT." and "... là 0,05 mg/kg."
# really do end sentences, and listing them here would glue the next sentence on.
_ABBREV = ("TS", "ThS", "GS", "PGS", "BS", "KS", "TP", "Q", "P", "Nxb", "St",
           "v.v", "vv", "tr")
_ABBREV_RE = re.compile(r"(?:(?<![\w])|^)(?:" + "|".join(re.escape(a) for a in _ABBREV) + r")\.$")

# A single initial: "B." in "mục B.1", not the final T of "NNPTNT."
_INITIAL_RE = re.compile(rf"(?:^|[^\w])[{_VN_UPPER}]\.$")

# A sentence ends at . ! ? followed by whitespace and an uppercase letter, a
# digit, or a quote/bracket. Colons and semicolons are NOT sentence ends here --
# treating them as such is what severs "Cấp 3:" from its value.
_SENT_END_RE = re.compile(rf"(?<=[.!?])\s+(?=[{_VN_UPPER}\d\"“(\[])")


def _is_real_sentence_end(text: str, pos: int) -> bool:
    """`pos` is the index just after the punctuation."""
    # A 24-character window, not the whole prefix: every pattern below is
    # anchored to the end and none is longer than "BNNPTNT." Slicing the prefix
    # here would make sentence splitting quadratic in document length.
    left = text[max(0, pos - 24):pos]
    if _ABBREV_RE.search(left):                       # "TS." / "v.v."
        return False
    if _INITIAL_RE.search(left):                      # a single initial: "B." "C."
        return False
    if re.search(r"\d\.$", left) and re.match(r"\s*\d", text[pos:]):
        return False                                   # "1. 2" -- a numbered run, not a sentence
    return True


def split_sentences_vi(text: str) -> List[str]:
    """Sentence boundaries that survive Vietnamese legal and technical text."""
    cuts = [0]
    for m in _SENT_END_RE.finditer(text):
        if _is_real_sentence_end(text, m.start()):
            cuts.append(m.end())
    cuts.append(len(text))
    return [text[a:b].strip() for a, b in zip(cuts, cuts[1:]) if text[a:b].strip()]


# ---------------------------------------------------------------------------
# 3. the separator hierarchy
# ---------------------------------------------------------------------------
# Each level proposes candidate cut positions. Lower index = stronger boundary.
# A cut position is the index at which the NEXT piece begins.
_KHOAN_AT = rf"(?:(?<=\s)|^)\d{{1,2}}\.\s+(?=[{_VN_UPPER}\d])"
_DIEM_AT = rf"(?:(?<=\s)|^)[a-zđư]\)\s+"

LEVELS: List[Tuple[str, str]] = [
    ("dieu",      rf"(?:(?<=\s)|^)(?:Điều|Ðiều|ĐIỀU|Điền)\s+\d{{1,3}}[a-zđ]?\s*[.:]"),
    ("structure", r"(?:(?<=\s)|^)(?:Chương\s+[IVXLCDM]+|Mục\s+\d+\s*[.:]|Phụ lục\s+[A-ZĐIVX0-9]+)"),
    ("paragraph", r"\n[ \t]*\n"),
    ("khoan",     _KHOAN_AT),
    ("line",      r"\n"),
    ("diem",      _DIEM_AT),
    ("sentence",  rf"(?<=[.!?])\s+(?=[{_VN_UPPER}\d\"“(\[])"),
    ("clause",    r"(?<=;)\s+"),
    ("colon",     r"(?<=:)\s+"),        # last structural resort: it does sever a label
    # NOTE there is deliberately no "word" level. Every space in the document
    # would become a candidate position -- ~400.000 of them in a 2,8 MB document,
    # which is most of the cost of building and searching the lattice for a
    # boundary that is only ever used as a last resort. The fallback below finds
    # it with a single str.rfind instead.
]
_COMPILED_LEVELS = [(name, re.compile(rx)) for name, rx in LEVELS]


_LEAD_IN_RE = re.compile(r"[:：]\s*$")


def _bind_lead_ins(pieces: List["Piece"], max_chars: int, min_chars: int) -> List["Piece"]:
    """A piece ending on a colon introduces whatever comes next. Left alone it is
    a caption with no table ("Bảng C.1 - Mức giới hạn:") or a lead-in with no
    list ("... được quy định như sau:"). Merge it forward.

    Allowed to exceed the budget only when the following piece is already an
    atomic overflow -- in that case the budget is lost either way and keeping
    the caption attached is strictly better.
    """
    if len(pieces) < 2:
        return pieces
    out: List[Piece] = []
    i = 0
    while i < len(pieces):
        cur = pieces[i]
        if (i + 1 < len(pieces) and _LEAD_IN_RE.search(cur.text)
                and len(cur) < max(min_chars, max_chars // 3)):
            nxt = pieces[i + 1]
            fits = len(cur) + 1 + len(nxt) <= max_chars
            if fits or nxt.cut_by == "atomic_overflow":
                out.append(Piece(f"{cur.text} {nxt.text}".strip(), cur.start, nxt.end,
                                 nxt.cut_by,
                                 sorted(set(cur.protected) | set(nxt.protected))))
                i += 2
                continue
        out.append(cur)
        i += 1
    return out


@dataclass
class Piece:
    text: str
    start: int
    end: int
    cut_by: str = "budget"          # which level produced the boundary AFTER this piece
    protected: List[str] = field(default_factory=list)   # atomic kinds inside

    def __len__(self) -> int:
        return len(self.text)


def _ends_on_colon(text: str, pos: int) -> bool:
    """Walk backwards over whitespace. Do NOT write this as
    ``text[:pos].rstrip().endswith(':')`` -- that copies the entire prefix on
    every call, and it is called once per candidate position. On a 2,8 MB
    document with ~50.000 candidates that is 7e10 character copies, which is
    what turned a 6-second document into a 5-minute one."""
    i = pos - 1
    while i >= 0 and text[i].isspace():
        i -= 1
    return i >= 0 and text[i] in ":："


class SpanIndex:
    """Disjoint, sorted spans with O(log n) containment.

    _merge_spans folds overlapping regions, so what comes out never overlaps --
    which means one binary search answers "is this position inside a block?".
    The first version scanned the span list for every candidate position; on a
    2,8 MB document that is ~400.000 positions times ~9.000 spans, and it is why
    a single large document could stall a run for minutes.
    """

    __slots__ = ("starts", "ends", "spans")

    def __init__(self, spans: Sequence[Tuple[int, int, str]]):
        self.spans = list(spans)
        self.starts = [a for a, _, _ in self.spans]
        self.ends = [b for _, b, _ in self.spans]

    def containing(self, pos: int) -> Optional[Tuple[int, int, str]]:
        i = bisect.bisect_right(self.starts, pos) - 1
        if i >= 0 and pos < self.ends[i]:
            return self.spans[i]
        return None

    def strictly_inside(self, pos: int) -> bool:
        s = self.containing(pos)
        return s is not None and s[0] < pos < s[1]

    def straddling(self, pos: int) -> Optional[Tuple[int, int, str]]:
        s = self.containing(pos)
        return s if s is not None and s[0] < pos < s[1] else None


def _candidate_cuts(text: str, atomic: List[Tuple[int, int, str]]) -> Dict[int, int]:
    """position -> best (lowest) level index. Positions inside an atomic block
    are dropped, which is the whole point."""
    index = SpanIndex(atomic)
    inside = index.strictly_inside

    cuts: Dict[int, int] = {}
    for level, (name, rx) in enumerate(_COMPILED_LEVELS):
        for m in rx.finditer(text):
            pos = m.start() if name in ("dieu", "structure", "khoan", "diem") else m.end()
            if pos <= 0 or pos >= len(text) or inside(pos):
                continue
            # Never end a piece on a colon except at the dedicated last-resort
            # level. A cut before "Cấp 1:" is a good boundary, but not when the
            # text just before it is "... Phân cấp hại đối với rầy nâu:" -- that
            # leaves the lead-in stranded from the list it introduces.
            if name != "colon" and _ends_on_colon(text, pos):
                continue
            if pos not in cuts or level < cuts[pos]:
                cuts[pos] = level
    return cuts


# ---------------------------------------------------------------------------
# 4. the splitter
# ---------------------------------------------------------------------------
def smart_split(
    text: str,
    *,
    max_chars: int = 900,
    min_chars: int = 250,
    overlap: int = 0,
    fill_ratio: float = 0.55,
    protect: Optional[bool] = None,
) -> List[Piece]:
    """Split `text` into pieces of at most `max_chars`, cutting at the strongest
    structural boundary available inside the budget and never inside an atomic
    block.

    fill_ratio
        A cut is preferred only once the piece is at least this fraction of the
        budget, so a strong-but-early boundary does not produce a stub. Below
        that threshold the search widens and takes the best boundary it can.
    overlap
        Characters of the previous piece to repeat at the start of the next one,
        applied ONLY when the boundary was weak (`word` or `budget`), where the
        context loss is real. Structural cuts get no overlap: an article and a
        khoản are complete on their own.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [Piece(text, 0, len(text), "whole")]

    if protect is None:
        protect = PROTECT_DEFAULT
    atomic = detect_atomic_spans(text) if protect else []
    cuts = _candidate_cuts(text, atomic)
    sorted_positions = sorted(cuts)
    index = SpanIndex(atomic)
    small_index = SpanIndex([s for s in atomic if s[1] - s[0] <= 220])

    pieces: List[Piece] = []
    pos = 0
    n = len(text)

    while pos < n:
        if n - pos <= max_chars:
            pieces.append(Piece(text[pos:].strip(), pos, n, "end"))
            break

        hard_limit = pos + max_chars
        soft_floor = pos + int(max_chars * fill_ratio)

        # binary search, not a scan of every candidate in the document
        hi = bisect.bisect_right(sorted_positions, hard_limit)
        window = sorted_positions[bisect.bisect_left(sorted_positions, soft_floor):hi]
        if not window:
            window = sorted_positions[bisect.bisect_right(sorted_positions, pos):hi]

        if window:
            best_level = min(cuts[p] for p in window)
            # among the strongest boundaries available, take the furthest one
            cut = max(p for p in window if cuts[p] == best_level)
            level_name = _COMPILED_LEVELS[best_level][0]
        else:
            # No boundary at all inside the budget. Either an atomic block spans
            # it -- in which case we let the block through whole, deliberately
            # over budget, because half a formula is worth nothing -- or the text
            # is one unbroken run and we cut on the budget.
            covering = index.containing(pos)
            straddling = index.straddling(hard_limit)
            if covering is not None and covering[1] > pos:
                # the piece STARTS inside a protected block: run to the end of
                # that block, over budget. Half a formula is worth nothing.
                block_end = covering[1]
                cut, level_name = block_end, "atomic_overflow"
                if block_end - pos > max_chars:
                    # ... unless the block is long enough to be worth cutting
                    # BETWEEN its items. A 40-row scale is not one indivisible
                    # thing; a single row is.
                    seams = [pos + m.start()
                             for m in _ITEM_CUT_RE.finditer(text[pos:min(block_end,
                                                                         pos + max_chars + 1)])
                             if m.start() > 0]
                    seams = [s for s in seams
                             if not small_index.strictly_inside(s)   # never inside one row
                             and not _ends_on_colon(text, s)]
                    if seams:
                        cut, level_name = max(seams), "atomic_item"
            elif straddling is not None and straddling[0] > pos:
                # the budget lands in the middle of a block that starts later:
                # stop just BEFORE it, so the block opens the next piece whole.
                cut = straddling[0]
                level_name = "before_atomic"
            else:
                cut, level_name = hard_limit, "budget"
                j = bisect.bisect_right(sorted_positions, hard_limit)
                if j < len(sorted_positions) and sorted_positions[j] - pos <= max_chars * 1.5:
                    cut, level_name = sorted_positions[j], "stretch"
                else:
                    # last resort: the nearest word boundary, found directly
                    space = text.rfind(" ", pos + max_chars // 2, hard_limit)
                    if space > pos:
                        cut, level_name = space + 1, "word"

        body = text[pos:cut].strip()
        if body:
            kinds = sorted({k for a, b, k in atomic if a >= pos and b <= cut})
            pieces.append(Piece(body, pos, cut, level_name, kinds))
        pos = cut

    # --- deal with a trailing stub -------------------------------------------
    # Merge it back if it fits. If it does not, REBALANCE the last two pieces
    # rather than leaving a 40-character orphan: a stub that small is usually
    # dropped downstream by --min-chars, and then a reference or a final clause
    # is silently gone from the corpus.
    if len(pieces) > 1 and len(pieces[-1]) < min_chars:
        merged = f"{pieces[-2].text} {pieces[-1].text}".strip()
        if len(merged) <= max_chars:
            pieces[-2] = Piece(merged, pieces[-2].start, pieces[-1].end,
                               pieces[-2].cut_by, pieces[-2].protected)
            pieces.pop()
        else:
            half = len(merged) // 2 + 1
            rebalanced = smart_split(merged, max_chars=max(half, min_chars),
                                     min_chars=0, fill_ratio=fill_ratio,
                                     protect=protect)
            # the sub-split ran with min_chars=0 to keep the recursion one level
            # deep, so fold any stub it produced back in -- now against the real
            # budget, where it comfortably fits.
            folded: List[Piece] = []
            for p in rebalanced:
                if (folded and len(p) < min_chars
                        and len(folded[-1]) + 1 + len(p) <= max_chars):
                    folded[-1] = Piece(f"{folded[-1].text} {p.text}".strip(),
                                       folded[-1].start, p.end, folded[-1].cut_by,
                                       sorted(set(folded[-1].protected) | set(p.protected)))
                else:
                    folded.append(p)
            floor = min(min_chars, 120)
            if 1 < len(folded) <= 3 and all(len(p) >= floor for p in folded):
                base = pieces[-2].start
                pieces[-2:] = [Piece(p.text, base, base + len(p.text), p.cut_by,
                                     p.protected) for p in folded]

    # Lead-in binding runs LAST: rebalancing can leave a piece ending on a colon,
    # and re-cutting is what created the orphan in the first place.
    pieces = _bind_lead_ins(pieces, max_chars, min_chars)

    # --- overlap, only where the boundary was weak ---------------------------
    if overlap > 0:
        for i in range(1, len(pieces)):
            if pieces[i - 1].cut_by in ("word", "budget"):
                tail = pieces[i - 1].text[-overlap:]
                pieces[i] = Piece(f"{tail} {pieces[i].text}".strip(), pieces[i].start,
                                  pieces[i].end, pieces[i].cut_by, pieces[i].protected)
    return pieces


# ---------------------------------------------------------------------------
# 5. optional semantic pass
# ---------------------------------------------------------------------------
def semantic_merge(
    pieces: Sequence[Piece],
    encode: Callable[[List[str]], "list"],
    *,
    max_chars: int = 900,
    threshold: float = 0.82,
) -> List[Piece]:
    """Merge adjacent pieces whose embeddings are very similar and that still fit.

    `encode` is any callable mapping a list of strings to a list of vectors --
    pass your own SentenceTransformer, so this module keeps no model dependency.

    Use this as a POLISH, never as the primary mechanism. On legal text the
    structural boundary is a far stronger signal than cosine similarity: two
    adjacent khoản of one article are near-identical in style and wording while
    stating different rules, so a purely similarity-driven splitter merges things
    that must stay apart and cuts things that must stay together. It is also
    ~4 orders of magnitude slower than the rule pass.
    """
    if len(pieces) < 2:
        return list(pieces)

    vecs = encode([p.text for p in pieces])

    def cos(u, v) -> float:
        dot = sum(a * b for a, b in zip(u, v))
        nu = sum(a * a for a in u) ** 0.5
        nv = sum(b * b for b in v) ** 0.5
        return dot / (nu * nv) if nu and nv else 0.0

    out = [pieces[0]]
    for i in range(1, len(pieces)):
        prev, cur = out[-1], pieces[i]
        if (len(prev) + 1 + len(cur) <= max_chars
                and cos(vecs[i - 1], vecs[i]) >= threshold):
            out[-1] = Piece(f"{prev.text} {cur.text}".strip(), prev.start, cur.end,
                            "semantic", sorted(set(prev.protected) | set(cur.protected)))
        else:
            out.append(cur)
    return out


# ---------------------------------------------------------------------------
# 6. measuring the result
# ---------------------------------------------------------------------------
# ":" is deliberately NOT terminal. A chunk that ends on a colon has been severed
# from the list, table or definition the colon introduces.
_TERMINAL = (".", "!", "?", ";", "…")
_LABEL_TAIL_RE = re.compile(r"(?:Cấp|Mức|Loại|Hạng|Bậc|Nhóm|Chỉ tiêu|Bảng|Biểu|Điểm)\s*[\w.]*\s*:\s*$")
_VALUE_HEAD_RE = re.compile(r"^\s*(?:[<>≤≥]|Từ\s|Trên\s|Dưới\s|\d+[.,]?\d*\s*(?:%|mg|kg|m2|ngày))")
# A TRUNCATED code, not a complete one. "… ngày 10/12/2010" is a whole date and
# must not be flagged; "… Thông tư 71/2010/" and "… QCVN 01-38:" are cut short.
_CODE_TAIL_RE = re.compile(r"(?:\d{1,4}/\d{4}/$|/[A-ZĐ]{1,3}$|\b[A-Z]{3,5}\s*\d+-\d*:?$)")

# A chunk that ends without punctuation is fine if the next one opens a new list
# item -- that is a clean seam, not a severed sentence.
_ITEM_HEAD_RE = re.compile(
    r"^\s*(?:\[\d{1,3}\]|(?:Cấp|Mức|Loại|Hạng|Bậc|Nhóm)\s+\d+\s*[(:]|Chỉ tiêu\b"
    r"|Bảng\s|Biểu\s|Phụ lục\s|[a-zđư]\)|\d{1,2}\.\s)")
# An orphan marker is a khoản/điểm marker left with nothing after it. It must be
# at the START of an item, so it follows a sentence end or opens the chunk --
# otherwise "… quy đổi x 5." trips it, and that trailing "5." is just prose.
_MARKER_TAIL_RE = re.compile(r"(?:^|(?<=[.!?;])\s)(?:\d{1,2}\.|[a-zđư]\))\s*$")


def count_atomic_breaks(source: str, chunks: Sequence[str]) -> Dict[str, object]:
    """How many protected regions of `source` were cut through by these chunks.

    This is the check that answers the complaint directly: a formula, a rating
    scale or a table body that starts in one chunk and finishes in another. It
    works by locating each chunk's end in the source and asking whether any
    atomic span straddles that position.
    """
    spans = detect_atomic_spans(source)
    boundaries, cursor = [], 0
    for c in chunks[:-1]:
        tail = c.strip()[-40:]
        idx = source.find(tail, cursor)
        if idx == -1:                       # whitespace was normalised away; approximate
            cursor += len(c)
            boundaries.append(min(cursor, len(source)))
        else:
            cursor = idx + len(tail)
            boundaries.append(cursor)

    # A cut BETWEEN items of a long run is allowed -- the guarantee is "never cut
    # through a row / a formula / a key-value pair", not "never divide a 40-row
    # scale at all". Seam positions are therefore exempt.
    seams = set()
    for m in _ITEM_CUT_RE.finditer(source):
        seams.update(range(max(0, m.start() - 2), m.end() + 3))

    broken = [(a, b, k) for a, b, k in spans
              if any(a < pos < b and pos not in seams for pos in boundaries)]
    return {
        "atomic_blocks": len(spans),
        "atomic_broken": len(broken),
        "broken_kinds": sorted({k for _, _, k in broken}),
        "examples": [source[a:b][:70] for a, b, _ in broken[:3]],
    }


def measure_breakage(chunks: Sequence[str]) -> Dict[str, object]:
    """Count the specific ways a chunk boundary damages meaning.

    Every one of these was observed in the output of the previous splitter, so
    none of them is hypothetical. `chunks_with_defects` is the headline: a chunk
    can trip several checks for one bad boundary, and counting it once keeps the
    before/after comparison honest.
    """
    kinds = ("ends_mid_sentence", "starts_mid_sentence", "orphan_lead_in",
             "orphan_label", "dangling_value", "split_code", "orphan_marker")
    counts = {k: 0 for k in kinds}
    damaged = 0

    for i, c in enumerate(chunks):
        c = c.strip()
        if not c:
            continue
        nxt = chunks[i + 1].strip() if i + 1 < len(chunks) else ""
        hit = set()
        if not c.endswith(_TERMINAL) and not _ITEM_HEAD_RE.match(nxt):
            hit.add("ends_mid_sentence")
        if c.endswith(":"):
            hit.add("orphan_lead_in")
        first = next((ch for ch in c if ch.isalpha()), "")
        if c[0] in ")];,-–" or (first and first.islower()):
            hit.add("starts_mid_sentence")
        if _LABEL_TAIL_RE.search(c):
            hit.add("orphan_label")
        if _VALUE_HEAD_RE.match(c):
            hit.add("dangling_value")
        if _CODE_TAIL_RE.search(c):
            hit.add("split_code")
        if _MARKER_TAIL_RE.search(c):
            hit.add("orphan_marker")
        for k in hit:
            counts[k] += 1
        if hit:
            damaged += 1

    counts["chunks"] = len(chunks)
    counts["chunks_with_defects"] = damaged
    counts["defect_rate"] = round(100 * damaged / max(len(chunks), 1), 1)
    return counts


# ---------------------------------------------------------------------------
def split_text(text: str, **kw) -> List[str]:
    """Convenience: just the strings."""
    return [p.text for p in smart_split(text, **kw)]


if __name__ == "__main__":
    import json
    import sys
    src = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else __doc__
    for p in smart_split(src, max_chars=600):
        print(f"[{p.cut_by:<16}] {len(p):4d}  {p.protected}  {p.text[:90]}")