#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Seeded-defect test for audit_chunks.py.

A quality checker is only worth running if it finds what it claims to find AND
stays quiet on good data. So: build 40 clean chunks, assert the audit reports
nothing at all, then plant exactly one instance of every defect and assert each
one is caught.
"""

import io
import json
import contextlib
import os
import sys

import audit_chunks

TITLE = "Điều {n}. Quy định về quản lý hoạt động thủy sản"
BODY = ("Tổ chức, cá nhân hoạt động thủy sản phải tuân thủ quy định của Luật Thủy sản "
        "và các văn bản hướng dẫn thi hành. Cơ quan quản lý nhà nước về thủy sản có "
        "trách nhiệm kiểm tra, giám sát việc thực hiện theo thẩm quyền được giao. "
        "Trường hợp phát hiện vi phạm thì xử lý theo quy định của pháp luật hiện hành. ")


SUBJECTS = ("giống thủy sản", "thức ăn thủy sản", "tàu cá", "khu bảo tồn biển",
            "kiểm ngư", "nuôi trồng thủy sản", "khai thác thủy sản", "cảng cá")
VERBS = ("đăng ký", "cấp phép", "kiểm tra chất lượng", "công bố hợp quy",
         "thu hồi giấy chứng nhận", "gia hạn", "đánh giá định kỳ", "giám sát")
CLAUSES = (
    "Hồ sơ nộp trực tiếp hoặc qua dịch vụ bưu chính đến cơ quan có thẩm quyền.",
    "Thời hạn giải quyết tính từ ngày nhận đủ hồ sơ hợp lệ theo quy định.",
    "Kết quả được trả cho tổ chức, cá nhân đề nghị bằng văn bản hoặc bản điện tử.",
    "Trường hợp không đủ điều kiện thì phải nêu rõ lý do bằng văn bản.",
    "Chi phí thực hiện do tổ chức, cá nhân đề nghị chi trả theo biểu mức hiện hành.",
    "Cơ quan tiếp nhận có trách nhiệm công khai trình tự trên cổng thông tin điện tử.",
    "Việc kiểm tra thực địa được tiến hành khi hồ sơ chưa đủ căn cứ kết luận.",
)


def chunk(n, **over):
    # Every clean chunk must be genuinely different text, otherwise the near-dup
    # check is right to complain and the test is testing the fixture, not the tool.
    s, v = SUBJECTS[n % len(SUBJECTS)], VERBS[(n * 3) % len(VERBS)]
    a, b = CLAUSES[n % len(CLAUSES)], CLAUSES[(n * 5 + 2) % len(CLAUSES)]
    content = over.pop("content", f"{n % 9 + 1}. Việc {v} đối với {s} thực hiện theo "
                                  f"trình tự quy định tại điều này. {a} {b} " + BODY)
    row = {"chunk_id": f"doc1_{n}", "doc_id": "doc1",
           "link": "https://thuvienphapluat.vn/van-ban/x-356284.aspx",
           "name": "Nghi-dinh-26-2019-ND-CP", "structure": "dieu",
           "dieu": TITLE.format(n=n), "khoan": None, "part": 1, "n_parts": 1,
           "chuong": "Chương II", "muc": None, "title_confident": True,
           "content": content, "char_len": len(content)}
    row.update(over)
    if "content" in over and "char_len" not in over:
        row["char_len"] = len(row["content"])
    return row


def write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def run(path, **kw):
    argv = [path, "--report", "/tmp/_a.json", "--samples", "", "--min-chars", "120",
            "--max-tokens", "512"]
    for k, v in kw.items():
        argv += [f"--{k}", str(v)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        audit_chunks.main(argv)
    return json.load(open("/tmp/_a.json", encoding="utf-8"))


# --- 1. clean data must produce a silent report ------------------------------
clean = [chunk(n) for n in range(1, 41)]
write("/tmp/clean.jsonl", clean)
rep = run("/tmp/clean.jsonl")
noisy = {k: v["count"] for k, v in rep["findings"].items()
         if v["count"] and v["severity"] != audit_chunks.INFO}
info = {k: v["count"] for k, v in rep["findings"].items()
        if v["count"] and v["severity"] == audit_chunks.INFO}
print(f"1) 40 clean chunks -> actionable findings: {noisy or 'none'}   (INFO: {info or 'none'})")
assert not noisy, f"false positives on clean data: {noisy}"
print(f"   median {rep['length']['median']} chars / {rep['tokens']['median']} tokens\n")

# --- 2. one planted instance of every defect ---------------------------------
CAPS = "QUẢN LÝ NHÀ NƯỚC VỀ THỦY SẢN KHU BẢO TỒN BIỂN NGUỒN LỢI THỦY SẢN GIỐNG THỦY SẢN "
planted = {
    "dup_chunk_id":      chunk(3),                                        # id doc1_3 already used
    "empty_content":     chunk(101, content="   ", char_len=3),
    "char_len_mismatch": chunk(102, char_len=999999),
    "over_budget":       chunk(103, content=BODY * 12),
    "no_dieu":           chunk(104, dieu=""),
    "title_guessed":     chunk(105, title_confident=False),
    "no_link":           chunk(106, link="", name=""),
    "fragment_start":    chunk(107, content="a) " + BODY),
    "orphan_part":       chunk(108, part=2, n_parts=3),
    "toc":               chunk(109, content=" ".join(
                             f"Điều {i}. Quy định chung về hoạt động thủy sản." for i in range(1, 8))),
    "boilerplate":       chunk(110, content="Nơi nhận: - Ban Bí thư Trung ương Đảng; "
                                            "- Thủ tướng, các Phó Thủ tướng Chính phủ; - Lưu: VT."),
    "too_short":         chunk(111, content="Điều này có hiệu lực thi hành."),
    "exact_dup":         chunk(112, content=clean[0]["content"]),
    # differs from clean[1] ONLY in spacing and punctuation -- numbers intact,
    # because a chunk that differs in a figure is a different rule, not a copy
    "near_dup":          chunk(113, content=clean[1]["content"]
                                    .replace(". ", " .  ").replace(", ", " ,  ")),
    "junk_ratio":        chunk(114, content="1.1. 2,50 3.75 (4) [5] 6/7 8-9 10% 11.12 13,14 " * 8),
    "all_caps":          chunk(115, content=CAPS * 3),
    "not_vietnamese":    chunk(116, content="This chunk is English boilerplate that a scraper "
                                            "picked up from the page footer. It carries no legal "
                                            "content whatsoever and should never reach the index "
                                            "or the training set, yet it is long enough to look "
                                            "plausible at a glance in a spot check."),
    "self_repeating":    chunk(117, content=(BODY.split(".")[0] + ". ") * 14),
}
write("/tmp/planted.jsonl", clean + list(planted.values()))
rep = run("/tmp/planted.jsonl")

print("2) one planted instance of each defect")
missed, extra = [], []
for name in audit_chunks.CHECKS:
    if name == "bad_json":
        continue
    count = rep["findings"][name]["count"]
    if name in planted:
        print(f"   {'ok ' if count else 'MISS'}  {name:<18} count={count}")
        if not count:
            missed.append(name)
    elif count:
        extra.append((name, count))
assert not missed, f"the audit missed: {missed}"
assert not extra, f"unplanted checks fired: {extra}"

# --- 3. --emit-clean actually removes them -----------------------------------
for profile, dropped in (("rag", "exact_dup near_dup toc boilerplate all_caps junk_ratio too_short"),
                         ("train", "exact_dup near_dup toc boilerplate all_caps junk_ratio "
                                   "too_short not_vietnamese self_repeating")):
    out = f"/tmp/cleaned_{profile}.jsonl"
    run("/tmp/planted.jsonl", profile=profile, **{"emit-clean": out})
    survivors = [json.loads(l)["chunk_id"] for l in open(out, encoding="utf-8")]
    lost = {c["chunk_id"] for c in clean} - set(survivors)
    assert not lost, f"--profile {profile} dropped good chunks: {sorted(lost)}"
    rep2 = run(out)
    must_be_gone = set(dropped.split()) | {k for k, (s, _) in audit_chunks.CHECKS.items()
                                           if s == audit_chunks.FATAL}
    still = {k: v["count"] for k, v in rep2["findings"].items()
             if v["count"] and k in must_be_gone}
    print(f"\n3) --profile {profile}: kept {len(survivors)} of {len(clean) + len(planted)} rows, "
          f"all {len(clean)} clean ones survived; residual targeted defects: {still or 'none'}")
    assert not still, still

print("\n✓ audit finds every planted defect and stays quiet on clean data")