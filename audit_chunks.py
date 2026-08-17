#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audit_chunks.py — measure whether chunks_output.jsonl is actually fit to index
and to train on, instead of eyeballing the first twenty lines.

    python audit_chunks.py chunks_output.jsonl \
        --corpus corpus.jsonl \
        --tokenizer Qwen/Qwen2.5-3B --max-tokens 512 \
        --report audit.json --samples audit_samples.jsonl \
        --emit-clean chunks_clean.jsonl

Three groups of checks:

  FATAL      breaks the index or the training run outright
             duplicate chunk_id, empty content, char_len disagreeing with content
  RAG        the chunk retrieves badly or cannot be cited
             over the embedder's token budget, no Điều title, no link, opens
             mid-sentence, is a table of contents, is a signature block
  TRAINING   the chunk teaches the model something you did not intend
             exact and near duplicates, boilerplate repetition, junk/table text,
             text that is not really Vietnamese prose

plus FIDELITY against the source corpus, which is the check that actually matters
and that almost nobody runs: did every document survive, and did every character?

Nothing here needs the network. Token counts use your own tokenizer when you pass
--tokenizer (transformers must be installed and the model cached locally); without
it the report gives characters, words and UTF-8 bytes, which are exact, plus a
clearly-labelled token estimate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import unicodedata
from array import array
from collections import Counter, defaultdict
from typing import Dict, Iterator, List, Optional

# ---------------------------------------------------------------------------
# severities
FATAL, RAG, TRAIN, INFO = "FATAL", "RAG", "TRAINING", "INFO"

CHECKS = {
    # id -> (severity, one-line meaning)
    "dup_chunk_id":      (FATAL, "two chunks share a chunk_id; the later one overwrites the earlier on upsert"),
    "empty_content":     (FATAL, "content is empty or whitespace"),
    "char_len_mismatch": (FATAL, "char_len does not equal len(content)"),
    "bad_json":          (FATAL, "line is not valid JSON"),

    "over_budget":       (RAG,   "longer than the embedder's window; the tail is silently truncated at index time"),
    "no_dieu":           (RAG,   "no Điều title, so a retrieved hit cannot be cited"),
    "title_guessed":     (RAG,   "title/body split was heuristic (flattened source), the title may be wrong"),
    "no_link":           (RAG,   "no link/name, so the answer cannot show a source"),
    "fragment_start":    (RAG,   "opens mid-sentence; reads as a fragment out of context"),
    "orphan_part":       (RAG,   "part 2+ of an article without the title repeated in the text"),
    "toc":               (RAG,   "table of contents: a list of Điều headings, not legal text"),
    "boilerplate":       (RAG,   "signature block / distribution list / attachment notice"),
    "too_short":         (RAG,   "below the useful floor; adds noise to the index"),

    "exact_dup":         (TRAIN, "identical text already seen"),
    "near_dup":          (TRAIN, "identical after normalising case, punctuation and whitespace"),
    "template_dup":      (INFO,  "identical once numbers are folded too: either a repeated form, "
                                 "or two clauses that differ only in a figure -- read before dropping"),
    "junk_ratio":        (TRAIN, "mostly digits/punctuation: a table, a form, or dotted leaders"),
    "all_caps":          (TRAIN, "all upper case: a heading, not prose"),
    "not_vietnamese":    (TRAIN, "few common Vietnamese words; OCR garbage or another language"),
    "self_repeating":    (TRAIN, "the same phrase repeats inside one chunk"),
}

# ---------------------------------------------------------------------------
# text probes
_VN_WORDS = ("của", "và", "các", "được", "quy định", "tại", "theo", "này",
             "trong", "đối với", "cơ quan", "thực hiện", "khoản", "điều")

# A table of contents lists Điều as HEADINGS. Ordinary legal text mentions Điều as
# CROSS-REFERENCES -- "khoản 10 Điều 10", "điểm b khoản 3 Điều 13" -- and Điều 1 of
# Nghị định 26/2019 has eight of those in one sentence. Same lookbehind the chunker
# uses: a heading never follows a lowercase word or a digit.
_LOWER_CLS = ("a-z0-9àảãáạăằẳẵắặâầẩẫấậèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợ"
              "ùủũúụưừửữứựỳỷỹýỵđ")
_DIEU_HEADING_RE = re.compile(rf"(?<![{_LOWER_CLS}] )Điều\s+\d{{1,3}}\s*[.:]")
_SENTENCE_RE = re.compile(r"[.;:!?]")
_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_DIGIT_PUNCT_RE = re.compile(r"[\d.,;:/()\[\]%–—-]")
_NUM_RE = re.compile(r"\d+")
_NONWORD_RE = re.compile(r"[\W_]+", re.UNICODE)
# Uppercase only, and no "./." -- that terminator is glued to the end of Điều 75
# ("... chịu trách nhiệm thi hành Nghị định này./."), which is real legal text.
# Strip the marker, don't throw the article away.
_BOILERPLATE = ("Nơi nhận:", "TM. CHÍNH PHỦ", "KT. BỘ TRƯỞNG", "THỦ TƯỚNG CHÍNH PHỦ",
                "FILE ĐƯỢC ĐÍNH KÈM", "Lưu: VT", "(Đã ký)")
# Punctuation a chunk should never open with. Do NOT put connective WORDS here:
# "Trong Nghị định này, các từ ngữ..." is the correct opening of Điều 3, and a
# lowercase word list matched against lowercased text flags it every time.
# In Vietnamese legal prose a sentence always starts upper case, so the case of
# the first letter is the signal.
_FRAGMENT_PUNCT = (")", ";", ",", "-", "–", "+")


def _norm_for_dup(text: str) -> str:
    """Case, diacritic composition, punctuation and whitespace folded away --
    but NOT numbers. In legal text the numbers are the meaning: "phụ cấp bằng 20%"
    and "phụ cấp bằng 30%" are two different rules, not one clause seen twice."""
    return _NONWORD_RE.sub("", unicodedata.normalize("NFC", text).lower())


def _norm_for_template(text: str) -> str:
    """The aggressive fold, numbers included. Two chunks that collide here but not
    in _norm_for_dup are either a form repeated per row (drop them) or a genuine
    pair of clauses differing only in a figure (keep them) -- hence INFO, not a
    verdict. Read the samples before deciding."""
    return _NUM_RE.sub("0", _norm_for_dup(text))


def _fp(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


def _first_alpha(text: str) -> str:
    for ch in text:
        if ch.isalpha():
            return ch
    return ""


def _shingle_diversity(text: str, n: int = 8) -> float:
    """unique n-word shingles / all of them. Legal prose sits near 1.0; a block the
    scraper duplicated, or a form repeated per row, drops well below it. Counting
    only the single most frequent shingle is not enough -- technical standards
    legitimately repeat one phrase ("Tổng số ... điều tra") without being junk."""
    words = text.split()
    if len(words) < n * 4:
        return 1.0
    shingles = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(shingles)) / len(shingles)


def probe(content: str) -> Dict[str, bool]:
    """Every content-only signal, computed once per chunk."""
    letters = _LETTER_RE.findall(content)
    n_letters = len(letters)
    upper = sum(1 for c in letters if c.isupper())
    low = content.lower()
    headings = len(_DIEU_HEADING_RE.findall(content))
    stripped = content.lstrip()
    first = _first_alpha(content)

    return {
        "toc": headings >= 4,
        "boilerplate": any(b in content for b in _BOILERPLATE) and len(content) < 600,
        "junk_ratio": len(content) > 80
                      and len(_DIGIT_PUNCT_RE.findall(content)) / len(content) > 0.35,
        "all_caps": n_letters > 40 and upper / max(n_letters, 1) > 0.75,
        "not_vietnamese": len(content) > 200
                          and sum(1 for w in _VN_WORDS if w in low) < 2,
        "self_repeating": _shingle_diversity(content) < 0.7,
        # opens mid-sentence: leading punctuation, or a lowercase first letter
        # (") Thuyền viên ...", "a) Sau 5 năm ...").
        "fragment_start": stripped.startswith(_FRAGMENT_PUNCT)
                          or (bool(first) and first.islower()),
    }


# ---------------------------------------------------------------------------
# tokenizer
class Sizer:
    """Real tokenizer when you have one, exact byte/word counts always."""

    def __init__(self, name: str = "", chars_per_token: float = 0.0):
        self.tok = None
        self.name = name
        self.estimated = True
        self.ratio = chars_per_token or 2.4
        if name:
            try:
                from transformers import AutoTokenizer
                self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=False)
                self.estimated = False
            except Exception as exc:                       # offline, not cached, no transformers
                print(f"!  could not load tokenizer {name!r} ({type(exc).__name__}); "
                      f"falling back to a character estimate at {self.ratio} chars/token.\n"
                      f"   Run this again on the machine that has the model cached to get "
                      f"exact token counts.", file=sys.stderr)

    def tokens(self, text: str) -> int:
        if self.tok is not None:
            return len(self.tok(text, add_special_tokens=False)["input_ids"])
        return max(1, int(round(len(text) / self.ratio)))


# ---------------------------------------------------------------------------
def read_jsonl(path: str) -> Iterator[tuple]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError:
                yield lineno, None


def pct(values: array, q: float) -> int:
    if not len(values):
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(q / 100 * len(ordered))) - 1))
    return ordered[idx]


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Audit chunk quality for RAG indexing and fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("chunks", help="chunks_output.jsonl from chunk_corpus.py")
    ap.add_argument("--corpus", default="", help="the source .jsonl, for the fidelity pass")
    ap.add_argument("--text-field", default="passage")
    ap.add_argument("--id-field", default="id")
    ap.add_argument("--tokenizer", default="",
                    help="HF tokenizer name or local path, e.g. Qwen/Qwen2.5-3B")
    ap.add_argument("--chars-per-token", type=float, default=0.0,
                    help="fallback ratio when no tokenizer is available (Vietnamese, byte-BPE)")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="the embedder's window; anything above it is truncated at index time")
    ap.add_argument("--min-chars", type=int, default=120, help="useful floor for a chunk")
    ap.add_argument("--fidelity-sample", type=int, default=300,
                    help="documents to verify character-for-character against the source")
    ap.add_argument("--samples-per-check", type=int, default=5)
    ap.add_argument("--report", default="audit.json")
    ap.add_argument("--samples", default="audit_samples.jsonl",
                    help="worst offenders, one per line, so you can read them")
    ap.add_argument("--emit-clean", default="",
                    help="write a filtered copy with FATAL and --drop defects removed")
    ap.add_argument("--profile", choices=("rag", "train"), default="rag",
                    help="which --drop preset to use; 'train' additionally drops chunks that "
                         "are not Vietnamese prose or that repeat themselves, because those "
                         "teach the model something you did not intend")
    ap.add_argument("--drop", default="",
                    help="comma-separated check ids to drop in --emit-clean (overrides --profile)")
    args = ap.parse_args(argv)

    PROFILES = {
        "rag":   "exact_dup,near_dup,toc,boilerplate,all_caps,junk_ratio,too_short",
        "train": "exact_dup,near_dup,toc,boilerplate,all_caps,junk_ratio,too_short,"
                 "not_vietnamese,self_repeating",
    }
    if not args.drop:
        args.drop = PROFILES[args.profile]

    sizer = Sizer(args.tokenizer, args.chars_per_token)

    # --- fidelity pass 1: remember how big each source document was ----------
    src_chars: Dict[str, int] = {}
    src_text: Dict[str, str] = {}
    if args.corpus:
        keep_every = None
        for lineno, rec in read_jsonl(args.corpus):
            if not isinstance(rec, dict):
                continue
            text = rec.get(args.text_field) or ""
            if not isinstance(text, str):
                continue
            doc_id = str(rec.get(args.id_field, "") or f"line{lineno}")
            flat = _NONWORD_RE.sub("", unicodedata.normalize("NFC", text).lower())
            src_chars[doc_id] = len(flat)
            if len(src_text) < args.fidelity_sample:
                src_text[doc_id] = flat
        print(f"corpus: {len(src_chars):,} documents "
              f"({len(src_text):,} kept in full for the containment check)", file=sys.stderr)

    # --- main pass over the chunks -------------------------------------------
    flags = Counter()
    samples: Dict[str, list] = defaultdict(list)
    chars, tokens = array("i"), array("i")
    seen_ids, seen_exact, seen_near, seen_tpl = set(), set(), set(), set()
    per_doc_chars: Dict[str, int] = defaultdict(int)
    per_doc_chunks: Counter = Counter()
    per_doc_text: Dict[str, list] = defaultdict(list)
    structures, n_rows = Counter(), 0
    clean_out = open(args.emit_clean, "w", encoding="utf-8") if args.emit_clean else None
    drop = {c.strip() for c in args.drop.split(",") if c.strip()}
    kept = 0

    for lineno, row in read_jsonl(args.chunks):
        n_rows += 1
        if row is None or not isinstance(row, dict):
            flags["bad_json"] += 1
            continue

        content = str(row.get("content", "") or "")
        cid = str(row.get("chunk_id", "") or f"line{lineno}")
        doc_id = str(row.get("doc_id", "") or "")
        hit = set()

        # --- FATAL ---------------------------------------------------------
        if not content.strip():
            hit.add("empty_content")
        if cid in seen_ids:
            hit.add("dup_chunk_id")
        seen_ids.add(cid)
        if "char_len" in row and row["char_len"] != len(content):
            hit.add("char_len_mismatch")

        if content.strip():
            n_tok = sizer.tokens(content)
            chars.append(len(content))
            tokens.append(n_tok)
            structures[str(row.get("structure", "?"))] += 1
            # --prefix-title / --repeat-lead deliberately copy the article title (and
            # the lead-in) onto every piece. That text is NOT extra source material:
            # counting it would push retention above 1.0 and break containment, since
            # "Điều 63. … — 2. Chế độ…" is not a substring of the document. Strip the
            # repeated head before measuring, so fidelity means what it says.
            body = _NONWORD_RE.sub("", content.lower())
            head = _NONWORD_RE.sub("", str(row.get("dieu", "") or "").lower())
            if head and body.startswith(head):
                body = body[len(head):]
            per_doc_chars[doc_id] += len(body)
            per_doc_chunks[doc_id] += 1
            if doc_id in src_text and body:
                per_doc_text[doc_id].append(body)

            # --- duplicates -------------------------------------------------
            fp = _fp(content)
            if fp in seen_exact:
                hit.add("exact_dup")
            else:
                seen_exact.add(fp)
                nfp = _fp(_norm_for_dup(content))
                if nfp in seen_near:
                    hit.add("near_dup")
                else:
                    seen_near.add(nfp)
                    tfp = _fp(_norm_for_template(content))
                    if tfp in seen_tpl:
                        hit.add("template_dup")
                    else:
                        seen_tpl.add(tfp)

            # --- size -------------------------------------------------------
            if n_tok > args.max_tokens:
                hit.add("over_budget")
            if len(content) < args.min_chars:
                hit.add("too_short")

            # --- citability -------------------------------------------------
            dieu = str(row.get("dieu", "") or "")
            if not dieu.strip():
                hit.add("no_dieu")
            if row.get("title_confident") is False:
                hit.add("title_guessed")
            if not (row.get("link") or row.get("name")):
                hit.add("no_link")
            try:
                part = int(row.get("part", 1) or 1)
            except (TypeError, ValueError):
                part = 1
            if part > 1 and dieu and dieu[:24] not in content[:len(dieu) + 40]:
                hit.add("orphan_part")

            # --- content probes ----------------------------------------------
            for name, tripped in probe(content).items():
                if tripped:
                    hit.add(name)

        for name in hit:
            flags[name] += 1
            if len(samples[name]) < args.samples_per_check:
                samples[name].append({"check": name, "chunk_id": cid,
                                      "severity": CHECKS.get(name, (INFO, ""))[0],
                                      "why": CHECKS.get(name, (INFO, ""))[1],
                                      "chars": len(content),
                                      "dieu": str(row.get("dieu", "") or "")[:120],
                                      "content": content[:400]})

        if clean_out is not None:
            fatal = {k for k, (sev, _) in CHECKS.items() if sev == FATAL}
            if not (hit & (fatal | drop)):
                clean_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1

    if clean_out is not None:
        clean_out.close()

    # --- fidelity ------------------------------------------------------------
    fidelity: Dict[str, object] = {}
    if args.corpus:
        missing = [d for d in src_chars if d not in per_doc_chunks]
        retention, thin = [], []
        for doc_id, src_n in src_chars.items():
            if not src_n or doc_id not in per_doc_chars:
                continue
            r = per_doc_chars[doc_id] / src_n
            retention.append(r)
            if r < 0.5:
                thin.append({"doc_id": doc_id, "kept": round(r, 3),
                             "source_chars": src_n, "chunks": per_doc_chunks[doc_id]})
        contained = mangled = 0
        for doc_id, parts in per_doc_text.items():
            whole = src_text.get(doc_id, "")
            for p in parts:
                if p and p in whole:
                    contained += 1
                else:
                    mangled += 1
        fidelity = {
            "source_documents": len(src_chars),
            "documents_with_chunks": len(per_doc_chunks),
            "documents_with_no_chunks": len(missing),
            "documents_with_no_chunks_sample": missing[:20],
            "median_character_retention": round(statistics.median(retention), 3) if retention else None,
            "mean_character_retention": round(statistics.fmean(retention), 3) if retention else None,
            "documents_under_50pct_retained": len(thin),
            "documents_under_50pct_sample": sorted(thin, key=lambda d: d["kept"])[:20],
            "containment_checked": contained + mangled,
            "containment_ok": contained,
            "containment_failed": mangled,
        }

    # --- report --------------------------------------------------------------
    n = len(chars) or 1
    report = {
        "chunks_file": args.chunks,
        "rows_read": n_rows,
        "chunks_scored": len(chars),
        "kept_in_clean_file": kept if clean_out is not None else None,
        "by_structure": dict(structures),
        "length": {
            "unit": "characters",
            "min": min(chars) if len(chars) else 0,
            "p25": pct(chars, 25), "median": pct(chars, 50),
            "p75": pct(chars, 75), "p95": pct(chars, 95),
            "max": max(chars) if len(chars) else 0,
            "mean": round(sum(chars) / n),
        },
        "tokens": {
            "source": ("tokenizer " + sizer.name) if not sizer.estimated
                      else f"ESTIMATE at {sizer.ratio} chars/token -- pass --tokenizer for exact counts",
            "median": pct(tokens, 50), "p95": pct(tokens, 95),
            "max": max(tokens) if len(tokens) else 0,
            "total": sum(tokens),
            "over_budget_at": args.max_tokens,
        },
        "findings": {},
        "fidelity": fidelity,
    }
    for name, (sev, why) in CHECKS.items():
        count = flags.get(name, 0)
        report["findings"][name] = {"severity": sev, "count": count,
                                    "share": round(100 * count / n, 2), "meaning": why}

    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    if args.samples:
        with open(args.samples, "w", encoding="utf-8") as fh:
            for name in CHECKS:
                for s in samples.get(name, []):
                    fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    # --- human summary --------------------------------------------------------
    L, T = report["length"], report["tokens"]
    print(f"\n{len(chars):,} chunks from {n_rows:,} rows")
    print(f"  characters  min {L['min']}  p25 {L['p25']}  median {L['median']}  "
          f"p75 {L['p75']}  p95 {L['p95']}  max {L['max']}")
    print(f"  tokens      median {T['median']}  p95 {T['p95']}  max {T['max']}  "
          f"total {T['total']:,}")
    print(f"              {T['source']}")
    if structures:
        print(f"  structure   " + "  ".join(f"{k}={v:,}" for k, v in structures.items()))

    print(f"\n{'':<4}{'check':<20}{'count':>10}{'share':>9}   meaning")
    for sev in (FATAL, RAG, TRAIN, INFO):
        rows = [(k, v) for k, v in report["findings"].items() if v["severity"] == sev and v["count"]]
        if not rows:
            print(f"\n  {sev}: clean")
            continue
        print(f"\n  {sev}")
        for name, v in sorted(rows, key=lambda kv: -kv[1]["count"]):
            print(f"{'':<4}{name:<20}{v['count']:>10,}{v['share']:>8.2f}%   {v['meaning']}")

    if fidelity:
        f = fidelity
        print(f"\n  FIDELITY vs {args.corpus}")
        print(f"{'':<4}{'documents':<20}{f['documents_with_chunks']:>10,} of "
              f"{f['source_documents']:,} produced chunks "
              f"({f['documents_with_no_chunks']:,} produced none)")
        print(f"{'':<4}{'text retained':<20}{f['median_character_retention']!s:>10} median   "
              f"({f['documents_under_50pct_retained']:,} documents kept under half their text)")
        print(f"{'':<4}{'containment':<20}{f['containment_ok']:>10,} of "
              f"{f['containment_checked']:,} sampled chunks are verbatim substrings of their source"
              + ("" if not f["containment_failed"] else
                 f"  <-- {f['containment_failed']:,} FAILED"))

    print(f"\nreport  -> {args.report}")
    if args.samples:
        print(f"samples -> {args.samples}   (read these; the counts only tell you where to look)")
    if clean_out is not None:
        print(f"clean   -> {args.emit_clean}   {kept:,} of {len(chars):,} chunks kept")

    fatal_total = sum(v["count"] for v in report["findings"].values() if v["severity"] == FATAL)
    return 1 if fatal_total else 0


if __name__ == "__main__":
    raise SystemExit(main())