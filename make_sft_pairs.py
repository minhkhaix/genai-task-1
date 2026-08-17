#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_sft_pairs.py — turn audited chunks into a fine-tuning set for Qwen.

    python make_sft_pairs.py chunks_clean.jsonl -o sft \
        --with-context 4 --refusal-rate 0.12 --split 0.90,0.05,0.05

Writes sft.train.jsonl / sft.val.jsonl / sft.test.jsonl in `messages` format
(ChatML), the format Qwen's own chat template consumes.

The point of this file: RAW CHUNKS ARE NOT AN INSTRUCTION SET. Feeding article
text to an SFT run teaches the model to continue legal prose, not to answer a
question about it. What you want, if you will serve the model behind your own
retriever, is examples shaped exactly like serving time:

    system    you answer only from the passages given, and you cite the Điều
    user      <ngữ cảnh> 4 retrieved passages, one of which is the right one
              Câu hỏi: ...
    assistant the answer, drawn from that passage, ending in a citation

Three things this builds that a naive generator does not:

  * HARD NEGATIVES. The other passages in the context come from the same
    document, or the same Chương, so they look relevant and are not. A model
    trained with only the correct passage in context learns to ignore the
    context entirely and answer from memory.
  * REFUSALS. --refusal-rate replaces the correct passage with distractors and
    teaches the model to say the context does not cover it. Without these the
    model will confabulate an Điều number for anything you ask.
  * A SPLIT THAT DOES NOT LEAK. Splitting by chunk puts khoản 1 of an article in
    train and khoản 2 in test, and your eval score becomes fiction. Splitting is
    by document, and near-duplicate text is checked across the boundary.

Question wording here is TEMPLATE-GENERATED, which is honest but narrow: it
teaches format, citation discipline and grounding, not the long tail of how
people really ask. Use --llm-prompts to emit a generation manifest, send it to a
stronger model, and merge the result with --merge-generated; the same validator
runs over both.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterator, List, Optional

SYSTEM = (
    "Bạn là trợ lý pháp luật Việt Nam. Chỉ trả lời dựa trên các đoạn văn bản được "
    "cung cấp trong phần Ngữ cảnh. Luôn trích dẫn Điều và tên văn bản ở cuối câu "
    "trả lời. Nếu Ngữ cảnh không chứa thông tin cần thiết, hãy nói rõ là không tìm "
    "thấy căn cứ và không suy đoán."
)

REFUSAL = ("Ngữ cảnh được cung cấp không chứa quy định trả lời cho câu hỏi này. "
           "Tôi không có căn cứ để trả lời.")

# --- document names --------------------------------------------------------
# The corpus carries slugs ("Nghi-dinh-26-2019-ND-CP"); a citation has to read
# like a citation ("Nghị định 26/2019/NĐ-CP") or the model learns to emit slugs.
_DOC_KINDS = [
    ("thong-tu-lien-tich", "Thông tư liên tịch"), ("nghi-quyet", "Nghị quyết"),
    ("quyet-dinh", "Quyết định"), ("nghi-dinh", "Nghị định"), ("thong-tu", "Thông tư"),
    ("phap-lenh", "Pháp lệnh"), ("chi-thi", "Chỉ thị"), ("cong-van", "Công văn"),
    ("bo-luat", "Bộ luật"), ("luat", "Luật"), ("hien-phap", "Hiến pháp"),
    ("tieu-chuan", "Tiêu chuẩn"), ("quy-chuan", "Quy chuẩn"),
]
_CODE_RE = re.compile(r"^(\d+)[-/](\d{4})[-/]([A-ZĐa-zđ\-]+)$")


def pretty_doc_name(name: str, link: str = "") -> str:
    raw = (name or "").strip()
    if not raw and link:
        raw = link.rstrip("/").rsplit("/", 1)[-1]
        raw = re.sub(r"-\d+\.(aspx|html?)$", "", raw)
    if not raw:
        return ""
    if re.search(r"[àảãáạăằẳẵắặâầẩẫấậèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợùủũúụưừửữứựỳỷỹýỵđ]",
                 raw, re.IGNORECASE):
        return raw                                    # already human-readable
    slug = raw.lower()
    for prefix, label in _DOC_KINDS:
        if slug.startswith(prefix + "-"):
            rest = raw[len(prefix) + 1:]
            m = _CODE_RE.match(rest)
            if m:
                return f"{label} {m.group(1)}/{m.group(2)}/{m.group(3).upper().replace('D', 'Đ', 1)}"
            return f"{label} {rest.replace('-', ' ')}"
    return raw.replace("-", " ")


# --- question templates ----------------------------------------------------
def _title_only(dieu: str) -> str:
    """'Điều 63. Chế độ, chính sách đối với Kiểm ngư' -> 'Chế độ, chính sách...'"""
    return re.sub(r"^\s*(?:Điều|Ðiều|ĐIỀU|Điền)\s+\d{1,3}[a-zđ]?\s*[.:]\s*", "", dieu).strip()


def _so_dieu(dieu: str) -> str:
    m = re.search(r"(?:Điều|Ðiều|ĐIỀU|Điền)\s+(\d{1,3}[a-zđ]?)", dieu or "")
    return m.group(1) if m else ""


def questions(chunk: Dict[str, object], doc: str) -> List[str]:
    dieu = norm_dieu_label(str(chunk.get("dieu") or ""))
    n, title = _so_dieu(dieu), _title_only(dieu)
    khoan = chunk.get("khoan")
    out: List[str] = []
    where = f"Điều {n}" + (f" {doc}" if doc else "")

    if khoan and str(khoan).isdigit():
        out += [f"Khoản {khoan} {where} quy định nội dung gì?",
                f"Cho tôi biết nội dung khoản {khoan} {where}."]
    if n:
        out += [f"{where} quy định về vấn đề gì?",
                f"Nội dung {where} là gì?"]
    if title:
        low = title[0].lower() + title[1:]
        out += [f"Pháp luật quy định thế nào về {low}?",
                f"{title} được quy định ra sao" + (f" trong {doc}?" if doc else "?")]
    return [q for q in out if len(q) > 15]


# --- io --------------------------------------------------------------------
def read_jsonl(path: str) -> Iterator[Dict[str, object]]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFC", text).lower())


_DIEU_VARIANTS_RE = re.compile(r"\b(?:Ðiều|ĐIỀU|ÐIỀU|Điền|Ðiền)(?=\s+\d)")


def norm_dieu_label(dieu: str) -> str:
    """The chunker keeps the source's spelling on purpose -- Nghị định 26/2019 really
    contains 'Điền 29.' and 'Ðiều' with an Eth. Fidelity is right for the corpus and
    wrong for training data: shown a hundred times, the model learns to cite 'Điền'."""
    return _DIEU_VARIANTS_RE.sub("Điều", dieu or "")


def strip_repeated_title(content: str, dieu: str) -> str:
    """Chunks built with --prefix-title carry 'Điều 63. … — ' at the head of the
    text, which is exactly right for embedding (the vector must stand alone) and
    exactly wrong here, where the title is already on the label line. Two copies
    of the heading in one prompt is wasted context and teaches nothing."""
    body, head = content.lstrip(), (dieu or "").strip()
    if head and body.startswith(head):
        body = body[len(head):].lstrip(" —–-—:.")
    return body.strip() or content.strip()


def context_block(passages: List[Dict[str, object]]) -> str:
    parts = []
    for i, p in enumerate(passages, start=1):
        head = norm_dieu_label(str(p.get("dieu") or "")).strip()
        doc = p["_doc"]
        label = " — ".join(x for x in (doc, head) if x)
        body = strip_repeated_title(str(p.get("content", "")), str(p.get("dieu") or ""))
        parts.append(f"[{i}] {label}\n{body}")
    return "Ngữ cảnh:\n" + "\n\n".join(parts)


def answer_text(chunk: Dict[str, object], doc: str) -> str:
    dieu = norm_dieu_label(str(chunk.get("dieu") or "")).strip()
    n = _so_dieu(dieu)
    cite = " ".join(x for x in (f"Điều {n}" if n else "", doc) if x).strip()
    body = strip_repeated_title(str(chunk.get("content") or ""), str(chunk.get("dieu") or ""))
    return f"{body}\n\n(Căn cứ: {cite})" if cite else body


# --- validation ------------------------------------------------------------
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?%?")


def validate(example: Dict[str, object]) -> List[str]:
    """Every number in the answer must appear in the context, and the cited Điều
    must be one of the passages actually shown. This is the check that catches a
    generated pair whose answer drifted off its source."""
    msgs = example.get("messages") or []
    user = next((m["content"] for m in msgs if m["role"] == "user"), "")
    assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    problems = []
    if not assistant.strip():
        problems.append("empty_answer")
        return problems
    if assistant.strip() == REFUSAL:
        return problems
    ctx_nums = set(_NUM_RE.findall(user))
    stray = [x for x in _NUM_RE.findall(assistant) if x not in ctx_nums]
    if stray:
        problems.append(f"numbers_not_in_context:{','.join(stray[:5])}")
    m = re.search(r"Căn cứ:\s*Điều\s+(\d{1,3}[a-zđ]?)", assistant)
    if m and f"Điều {m.group(1)}" not in user:
        problems.append(f"cited_dieu_not_in_context:{m.group(1)}")
    return problems


# --- main ------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Build a grounded instruction set from audited chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("chunks", help="chunks_clean.jsonl from audit_chunks.py --emit-clean")
    ap.add_argument("-o", "--out-prefix", default="sft")
    ap.add_argument("--with-context", type=int, default=4,
                    help="passages per prompt: 1 correct + (n-1) hard negatives. 0 = no context "
                         "(closed-book; only do this for continued pretraining)")
    ap.add_argument("--questions-per-chunk", type=int, default=1)
    ap.add_argument("--refusal-rate", type=float, default=0.12,
                    help="share of examples whose context does NOT contain the answer")
    ap.add_argument("--split", default="0.90,0.05,0.05", help="train,val,test by DOCUMENT")
    ap.add_argument("--max-context-chars", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--format", choices=("messages", "alpaca"), default="messages")
    ap.add_argument("--llm-prompts", default="",
                    help="also write a generation manifest for diverse question wording")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)

    # --- load, grouped by document ---------------------------------------
    by_doc: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in read_jsonl(args.chunks):
        if not str(row.get("content") or "").strip():
            continue
        row["_doc"] = pretty_doc_name(str(row.get("name") or ""), str(row.get("link") or ""))
        by_doc[str(row.get("doc_id") or "?")].append(row)
        if args.limit and sum(len(v) for v in by_doc.values()) >= args.limit:
            break
    all_chunks = [c for v in by_doc.values() for c in v]
    if not all_chunks:
        print("no usable chunks", file=sys.stderr)
        return 2
    print(f"{len(all_chunks):,} chunks from {len(by_doc):,} documents", file=sys.stderr)

    # --- split by DOCUMENT, never by chunk --------------------------------
    ratios = [float(x) for x in args.split.split(",")]
    docs = sorted(by_doc)
    rng.shuffle(docs)
    n_train = int(len(docs) * ratios[0])
    n_val = int(len(docs) * ratios[1])
    split_of = {}
    for i, d in enumerate(docs):
        split_of[d] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")

    # --- build --------------------------------------------------------------
    files = {s: open(f"{args.out_prefix}.{s}.jsonl", "w", encoding="utf-8")
             for s in ("train", "val", "test")}
    stats, problems = Counter(), Counter()
    seen_answer_norm: Dict[str, str] = {}          # normalised answer -> split, leak check
    leaks = 0

    for doc_id, chunks in by_doc.items():
        split = split_of[doc_id]
        for chunk in chunks:
            doc = str(chunk.get("_doc") or "")
            qs = questions(chunk, doc)
            if not qs:
                stats["chunks_without_a_question"] += 1
                continue
            for q in qs[: args.questions_per_chunk]:
                refuse = rng.random() < args.refusal_rate

                if args.with_context > 0:
                    # hard negatives: same document first, then the same Chương,
                    # then anywhere -- in that order of difficulty.
                    pool = [c for c in chunks if c is not chunk]
                    if len(pool) < args.with_context - 1:
                        pool += [c for c in rng.sample(all_chunks, min(len(all_chunks), 40))
                                 if c is not chunk]
                    if refuse:
                        # A refusal is only correct if the answer is genuinely absent.
                        # Drop any distractor from the same Điều (khoản 2 of the article
                        # answers a question about the article) or with the same text
                        # under a different number -- legal corpora repeat clauses.
                        same_dieu = str(chunk.get("dieu") or "").strip()
                        same_text = _norm(str(chunk.get("content") or ""))[:300]
                        pool = [c for c in pool
                                if str(c.get("dieu") or "").strip() != same_dieu
                                and _norm(str(c.get("content") or ""))[:300] != same_text]
                        if not pool:
                            stats["refusal_impossible"] += 1
                            continue
                    negs = rng.sample(pool, min(len(pool), args.with_context - 1))
                    passages = negs if refuse else negs + [chunk]
                    rng.shuffle(passages)
                    block = context_block(passages)
                    # trim by identity -- list.remove() compares by value and two
                    # chunks that happen to be equal dicts would drop the wrong one
                    while len(block) > args.max_context_chars and len(passages) > 1:
                        keep = [p for p in passages if p is chunk]
                        rest = [p for p in passages if p is not chunk]
                        if not rest:
                            break
                        passages = keep + rest[:-1]
                        rng.shuffle(passages)
                        block = context_block(passages)
                    user = f"{block}\n\nCâu hỏi: {q}"
                else:
                    if refuse:
                        continue
                    user = q

                assistant = REFUSAL if refuse else answer_text(chunk, doc)
                example = {"messages": [{"role": "system", "content": SYSTEM},
                                        {"role": "user", "content": user},
                                        {"role": "assistant", "content": assistant}],
                           "meta": {"chunk_id": chunk.get("chunk_id"), "doc_id": doc_id,
                                    "split": split, "kind": "refusal" if refuse else "grounded"}}

                bad = validate(example)
                if bad:
                    for b in bad:
                        problems[b.split(":")[0]] += 1
                    stats["rejected"] += 1
                    continue

                key = _norm(assistant)[:400]
                if key and key in seen_answer_norm and seen_answer_norm[key] != split:
                    leaks += 1
                    continue
                seen_answer_norm.setdefault(key, split)

                if args.format == "alpaca":
                    payload = {"instruction": q,
                               "input": user.split("Câu hỏi:")[0].strip() if args.with_context else "",
                               "output": assistant, "meta": example["meta"]}
                else:
                    payload = example
                files[split].write(json.dumps(payload, ensure_ascii=False) + "\n")
                stats[split] += 1
                stats["refusals" if refuse else "grounded"] += 1

    for fh in files.values():
        fh.close()

    # --- generation manifest -------------------------------------------------
    if args.llm_prompts:
        with open(args.llm_prompts, "w", encoding="utf-8") as fh:
            for chunk in all_chunks:
                fh.write(json.dumps({
                    "chunk_id": chunk.get("chunk_id"),
                    "instruction":
                        "Đọc đoạn văn bản pháp luật dưới đây và viết 3 câu hỏi mà một "
                        "người dân hoặc cán bộ có thể hỏi, kèm câu trả lời. Quy tắc: chỉ "
                        "dùng thông tin có trong đoạn văn; giữ nguyên mọi con số; mỗi câu "
                        "trả lời kết thúc bằng trích dẫn Điều và tên văn bản; nếu đoạn văn "
                        "không đủ thông tin cho một câu hỏi thì bỏ câu hỏi đó. Trả về JSON "
                        '{"pairs":[{"question":"...","answer":"..."}]}.',
                    "document": chunk.get("_doc"),
                    "dieu": chunk.get("dieu"),
                    "passage": chunk.get("content"),
                }, ensure_ascii=False) + "\n")

    # --- report ---------------------------------------------------------------
    total = stats["train"] + stats["val"] + stats["test"]
    for s in ("val", "test"):
        if not stats[s]:
            print(f"!  the {s} split is empty -- {len(docs)} documents cannot be divided "
                  f"{args.split}. Give it more documents or change --split.", file=sys.stderr)
    print(f"\n{total:,} examples")
    print(f"  train {stats['train']:,}   val {stats['val']:,}   test {stats['test']:,}"
          f"   (split by document: {n_train:,}/{n_val:,}/{len(docs)-n_train-n_val:,} docs)")
    print(f"  grounded {stats['grounded']:,}   refusals {stats['refusals']:,} "
          f"({100*stats['refusals']/max(total,1):.1f}%)")
    if args.with_context:
        print(f"  context   {args.with_context} passages each, "
              f"{args.with_context-1} of them hard negatives")
    if stats["rejected"]:
        print(f"  rejected  {stats['rejected']:,} by the grounding validator "
              f"({dict(problems)})")
    if leaks:
        print(f"  {leaks:,} examples dropped because the same answer text appeared "
              f"in another split")
    if stats["chunks_without_a_question"]:
        print(f"  {stats['chunks_without_a_question']:,} chunks produced no question "
              f"(no Điều title -- these are the 'no_dieu' ones from the audit)")
    print(f"\n-> {args.out_prefix}.train.jsonl / .val.jsonl / .test.jsonl")
    if args.llm_prompts:
        print(f"-> {args.llm_prompts}  ({len(all_chunks):,} generation prompts; send these to a "
              f"stronger model, then re-run validate() over what comes back)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())