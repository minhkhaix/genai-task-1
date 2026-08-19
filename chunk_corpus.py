#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chunk_corpus.py — stream a huge legal-document .jsonl through the Điều chunker.

    python chunk_corpus.py corpus.jsonl -o chunks_output.jsonl --max-chars 900

Memory stays flat no matter how large the input is: one line in, N chunks out,
nothing accumulated. Tested on 8.852-document corpora shaped like

    {"id": "...", "link": "...", "name": "...", "passage": "<the whole document>"}

Every key other than the text field is copied onto every chunk it produces.

Why the --fallback flag matters
-------------------------------
Not every Vietnamese legal document has articles. TCVN / QCVN standards, công
văn, biểu mẫu and phụ lục use "Phụ lục B", "B.1", "Cấp 1:", "Bảng C.1" instead.
extract_dieu() returns [] on those, and with `--fallback skip` the document
vanishes without a trace. The default `--fallback window` keeps it, chunked on
sentence boundaries and tagged `"structure": "plain"`, so you can decide later
whether to train on it. Either way the run report tells you how many there were.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import sys
import time
from collections import Counter
from typing import Dict, Iterator, List, Optional

from vn_legal_chunker import chunk_plain_text, extract_dieu

try:
    from tqdm import tqdm
except ImportError:                                    # pragma: no cover
    print("! tqdm not installed  ->  pip install tqdm   (running without a bar)", file=sys.stderr)

    def tqdm(iterable=None, **kwargs):                 # minimal stand-in
        return iterable if iterable is not None else iter(())


# ---------------------------------------------------------------------------
def count_lines(path: str, buf_size: int = 1 << 20) -> Optional[int]:
    """Fast first pass just to give tqdm a total. Reads bytes, never decodes."""
    try:
        total = 0
        last = b"\n"
        with open(path, "rb") as fh:
            while True:
                buf = fh.read(buf_size)
                if not buf:
                    break
                total += buf.count(b"\n")
                last = buf[-1:]
        if last not in (b"\n", b""):
            total += 1                                  # file without trailing newline
        return total
    except OSError:
        return None


def read_lines(path: str) -> Iterator[str]:
    """Line-by-line generator -- never readlines(), never .read()."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:                                 # the whole point: streaming
            yield line


def _fingerprint(text: str) -> bytes:
    """8-byte digest: the only structure that grows with corpus size, so keep it
    small -- ~65 bytes per unique chunk, i.e. ~35 MB for half a million chunks."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


class DocTimeout(Exception):
    """One document took longer than --doc-timeout."""


@contextlib.contextmanager
def time_limit(seconds: float):
    """Abandon a single document instead of hanging the whole run.

    A regex that backtracks runs inside C code, but CPython's engine polls for
    signals while it does, so SIGALRM really does break it out (verified: a
    runaway pattern was cut off after exactly 1.0 s). Unix + main thread only;
    everywhere else this is a no-op rather than an error.
    """
    if not seconds or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _fire(signum, frame):
        raise DocTimeout()

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def load_id_filter(spec: str) -> Optional[set]:
    """Accept a comma-separated list, a file of ids, or a previous report.json."""
    if not spec:
        return None
    if os.path.exists(spec):
        with open(spec, "r", encoding="utf-8") as fh:
            head = fh.read()
        try:
            report = json.loads(head)
            ids = {str(d.get("id")) for d in report.get("slow_documents", [])
                   if d.get("id") is not None}
            if ids:
                print(f"retrying {len(ids)} document(s) from {spec}: "
                      f"{', '.join(sorted(ids)[:10])}", file=sys.stderr)
                return ids
        except json.JSONDecodeError:
            pass
        return {line.strip() for line in head.splitlines() if line.strip()}
    return {part.strip() for part in spec.split(",") if part.strip()}


def _name_from_link(link: str) -> str:
    """thuvienphapluat URLs carry a readable slug; use it when "name" is empty."""
    if not link:
        return ""
    slug = link.rstrip("/").rsplit("/", 1)[-1]
    for suffix in (".aspx", ".html", ".htm"):
        if slug.endswith(suffix):
            slug = slug[: -len(suffix)]
    return slug.replace("-", " ").strip()


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Stream a legal-document .jsonl into Điều / khoản chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input", help="input corpus .jsonl (one JSON document per line)")
    ap.add_argument("-o", "--output", default="chunks_output.jsonl")
    ap.add_argument("--text-field", default="passage", help="key holding the document text")
    ap.add_argument("--id-field", default="id")
    ap.add_argument("--max-chars", type=int, default=900,
                    help="sub-split articles longer than this at khoản boundaries")
    ap.add_argument("--min-chars", type=int, default=80,
                    help="drop chunks shorter than this (junk / stray headings)")
    ap.add_argument("--merge-below", type=int, default=250,
                    help="a trailing sub-chunk shorter than this is merged back")
    ap.add_argument("--fallback", choices=("window", "skip"), default="window",
                    help="what to do with documents that contain no Điều")
    ap.add_argument("--dedup", action="store_true",
                    help="drop chunks whose text was already written (duplicates are "
                         "counted and reported either way)")
    ap.add_argument("--fill-name", action="store_true",
                    help="derive a document name from the link when 'name' is empty")
    ap.add_argument("--prefix-title", action="store_true",
                    help="prepend 'Điều N. Title — ' to each chunk's text")
    ap.add_argument("--repeat-lead", action="store_true",
                    help="copy the article lead-in onto every sub-chunk")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N documents (0 = all) -- use for a dry run first")
    ap.add_argument("--doc-timeout", type=float, default=60.0,
                    help="give up on a single document after this many seconds "
                         "(0 = never); the run continues and the id is reported")
    ap.add_argument("--slow-seconds", type=float, default=5.0,
                    help="print the id of any document that takes longer than this")
    ap.add_argument("--no-count", action="store_true",
                    help="skip the line-counting pass (bar shows count, not percentage)")
    ap.add_argument("--only-ids", default="",
                    help="process ONLY these document ids: a comma-separated list, or the "
                         "path to a file with one id per line, or the path to a report.json "
                         "whose slow_documents / timed-out ids should be retried. Use this to "
                         "re-run the handful of documents a previous pass abandoned instead "
                         "of redoing the whole corpus")
    ap.add_argument("--report", default="", help="write run statistics to this JSON file")
    ap.add_argument("--no-protect-structure", action="store_true",
                    help="disable smart_split's atomic-block protection. Chunks then never "
                         "exceed --max-chars, at the cost of cutting through formulas, "
                         "rating scales and table bodies")
    args = ap.parse_args(argv)

    if args.no_protect_structure:
        try:
            import smart_split
            smart_split.PROTECT_DEFAULT = False
            print("!  structure protection OFF: tables and formulas may be split",
                  file=sys.stderr)
        except ImportError:
            pass

    if not os.path.exists(args.input):
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2

    id_filter = load_id_filter(args.only_ids)
    total_lines = None if args.no_count else count_lines(args.input)
    if total_lines:
        print(f"{args.input}: {total_lines:,} documents", file=sys.stderr)

    stats = Counter()
    len_sum = 0
    seen: set = set()
    structures = Counter()
    slow: List[dict] = []
    started = time.time()

    bar = tqdm(
        read_lines(args.input),
        total=total_lines,
        unit=" doc",
        desc="chunking",
        dynamic_ncols=True,
    )
    # Show which document is in flight, so a stall names its culprit on screen
    # instead of just freezing the percentage.
    set_status = getattr(bar, "set_postfix_str", lambda *a, **k: None)

    with open(args.output, "w", encoding="utf-8") as fout:
        for lineno, line in enumerate(bar, start=1):
            if args.limit and stats["docs"] >= args.limit:
                break

            line = line.strip()
            if not line:
                stats["blank_lines"] += 1
                continue

            # --- parse -----------------------------------------------------
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                stats["bad_json"] += 1
                if stats["bad_json"] <= 5:
                    print(f"  line {lineno}: bad JSON ({exc.msg}) -- skipped", file=sys.stderr)
                continue
            if not isinstance(record, dict):
                stats["not_an_object"] += 1
                continue

            if id_filter is not None:
                if str(record.get(args.id_field, "")) not in id_filter:
                    stats["docs_not_selected"] += 1
                    continue

            stats["docs"] += 1
            text = record.get(args.text_field) or ""
            if not isinstance(text, str) or not text.strip():
                stats["empty_text"] += 1
                continue

            # --- metadata: everything except the raw text ------------------
            meta = {k: v for k, v in record.items() if k != args.text_field}
            if args.fill_name and not meta.get("name"):
                meta["name"] = _name_from_link(str(meta.get("link", "")))
            doc_id = str(record.get(args.id_field, "") or f"line{lineno}")

            # --- chunk, under a watchdog -----------------------------------
            # One pathological document must never stall an 8.000-document run.
            set_status(f"doc {doc_id} ({len(text):,}c)", refresh=False)
            structure = "dieu"
            t_doc = time.time()
            try:
                with time_limit(args.doc_timeout):
                    chunks = extract_dieu(
                        text,
                        with_metadata=True,
                        max_chunk_chars=args.max_chars,
                        min_chunk_chars=args.merge_below,
                        prefix_title=args.prefix_title,
                        repeat_lead=args.repeat_lead,
                    )
                    if not chunks:
                        stats["docs_without_dieu"] += 1
                        if args.fallback == "window":
                            chunks = chunk_plain_text(
                                text,
                                max_chars=args.max_chars,
                                min_chars=args.merge_below,
                                title=str(meta.get("name") or ""),
                            )
                            structure = "plain"
            except DocTimeout:
                stats["docs_timed_out"] += 1
                stats["docs_skipped"] += 1
                slow.append({"line": lineno, "id": doc_id, "chars": len(text),
                             "seconds": args.doc_timeout, "timed_out": True,
                             "link": str(meta.get("link", ""))})
                print(f"  line {lineno}: doc {doc_id} ({len(text):,} chars) exceeded "
                      f"{args.doc_timeout}s -- abandoned", file=sys.stderr)
                continue

            took = time.time() - t_doc
            if args.slow_seconds and took >= args.slow_seconds:
                slow.append({"line": lineno, "id": doc_id, "chars": len(text),
                             "seconds": round(took, 1), "timed_out": False,
                             "link": str(meta.get("link", ""))})
                print(f"  line {lineno}: doc {doc_id} ({len(text):,} chars) took "
                      f"{took:.1f}s", file=sys.stderr)

            if not chunks:
                stats["docs_skipped"] += 1
                continue

            # --- emit ------------------------------------------------------
            used_ids: Counter = Counter()
            wrote_any = False
            for index, chunk in enumerate(chunks, start=1):
                content = str(chunk.get("content", ""))
                if len(content) < args.min_chars:
                    stats["chunks_too_short"] += 1
                    continue

                fp = _fingerprint(content)
                if fp in seen:
                    stats["duplicates"] += 1
                    if args.dedup:
                        continue
                else:
                    seen.add(fp)

                so_dieu = chunk.get("so_dieu")
                base = f"{doc_id}_{so_dieu}" if so_dieu else doc_id
                if chunk.get("n_parts", 1) > 1:
                    base += f"_p{chunk['part']}"
                elif not so_dieu:
                    base += f"_{index}"
                used_ids[base] += 1
                chunk_id = base if used_ids[base] == 1 else f"{base}_{used_ids[base]}"

                out = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    **meta,                       # <- original metadata, propagated
                    "structure": structure,       # "dieu" | "plain"
                    "dieu": chunk.get("dieu", ""),
                    "khoan": chunk.get("khoan"),
                    "part": chunk.get("part", 1),
                    "n_parts": chunk.get("n_parts", 1),
                    "chuong": chunk.get("chuong"),
                    "muc": chunk.get("muc"),
                    "content": content,
                    "char_len": len(content),
                }
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")

                stats["chunks"] += 1
                if len(content) > args.max_chars:
                    # deliberate: a protected block (formula, rating scale, table
                    # body) was kept whole rather than cut. Reported, never silent.
                    stats["chunks_over_budget"] += 1
                    stats["over_budget_chars"] += len(content)
                structures[structure] += 1
                len_sum += len(content)
                wrote_any = True

            if not wrote_any:
                stats["docs_producing_nothing"] += 1

    elapsed = time.time() - started
    docs = stats["docs"] or 1
    summary = {
        "input": args.input,
        "output": args.output,
        "documents_read": stats["docs"],
        "chunks_written": stats["chunks"],
        "chunks_per_document": round(stats["chunks"] / docs, 2),
        "mean_chunk_chars": round(len_sum / (stats["chunks"] or 1)),
        "by_structure": dict(structures),
        "documents_without_dieu": stats["docs_without_dieu"],
        "documents_skipped": stats["docs_skipped"],
        "chunks_over_budget": stats["chunks_over_budget"],
        "mean_over_budget_chars": (round(stats["over_budget_chars"] /
                                         stats["chunks_over_budget"])
                                   if stats["chunks_over_budget"] else None),
        "documents_timed_out": stats["docs_timed_out"],
        "slow_documents": slow[:50],
        "documents_producing_nothing": stats["docs_producing_nothing"],
        "empty_text": stats["empty_text"],
        "bad_json_lines": stats["bad_json"],
        "blank_lines": stats["blank_lines"],
        "duplicate_chunks": stats["duplicates"],
        "duplicates_dropped": args.dedup,
        "seconds": round(elapsed, 1),
        "docs_per_second": round(stats["docs"] / elapsed, 1) if elapsed else None,
    }

    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2), file=sys.stderr)
    if stats["docs_without_dieu"]:
        share = 100 * stats["docs_without_dieu"] / docs
        verb = "windowed as plain text" if args.fallback == "window" else "DROPPED"
        print(f"\n!  {stats['docs_without_dieu']:,} documents ({share:.1f}%) contain no "
              f"'Điều' and were {verb}.", file=sys.stderr)
    if stats["docs_timed_out"]:
        print(f"\n!  {stats['docs_timed_out']:,} documents exceeded --doc-timeout and were "
              f"abandoned; their ids are in 'slow_documents'.", file=sys.stderr)
    if stats["duplicates"] and not args.dedup:
        print(f"!  {stats['duplicates']:,} duplicate chunks were kept "
              f"(re-run with --dedup to drop them).", file=sys.stderr)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        print(f"report -> {args.report}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())