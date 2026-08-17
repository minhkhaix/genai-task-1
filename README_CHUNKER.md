# Chunking văn bản pháp luật Việt Nam theo "Điều"

`vn_legal_chunker.py` — one chunk per Article, JSON-ready for LLM training / RAG.
`chunk_corpus.py` — streams a whole `.jsonl` corpus through it.
`audit_chunks.py` — measures whether the output is fit to index and to train on.
`make_sft_pairs.py` — turns audited chunks into a grounded instruction set.
`diagnose_hang.py` — finds the document, and the line, if a run ever stalls.

**Chunking is step one.** What "good chunks" means, and how to check it rather
than eyeball it, is in [RAG_FINETUNE_GUIDE.md](RAG_FINETUNE_GUIDE.md).

## Running the whole corpus

```bash
pip install tqdm
python chunk_corpus.py corpus.jsonl -o chunks_output.jsonl --max-chars 900 --fill-name --dedup
```

Reads one line at a time (`for line in fh:` — never `readlines()`, never `.read()`),
parses it with `json.loads`, chunks it, and writes each chunk immediately with
`json.dumps`. Measured on a synthetic 8.852-document / 202 MB corpus:
**52 s, peak RSS 27 MB** — memory does not track file size. A 564 MB file peaked at
24 MB.

Every key of the source object except the text field is copied onto every chunk:

```json
{"chunk_id": "60011_63_p2", "doc_id": "60011", "id": "60011", "link": "https://…",
 "name": "Nghi-dinh-26-2019-ND-CP", "structure": "dieu",
 "dieu": "Điều 63. Chế độ, chính sách đối với Kiểm ngư", "khoan": "2",
 "part": 2, "n_parts": 5, "chuong": "Chương VI", "muc": null,
 "content": "2. Chế độ phụ cấp trách nhiệm …", "char_len": 593}
```

Useful flags: `--limit 200` (dry run first), `--fallback window|skip`,
`--dedup`, `--fill-name`, `--prefix-title`, `--min-chars`, `--report stats.json`,
`--text-field`/`--id-field` if your keys differ, `--no-count` to skip the
line-counting pass tqdm uses for the percentage.

The run ends with a JSON report: documents read, chunks written, chunks per
document, mean length, counts by structure, documents without any Điều,
duplicates, bad JSON lines, throughput.

### If the bar freezes (it did — this is fixed)

On the real 8.532-document corpus the run reached 1% and stopped: no error, no
traceback, CPU still pinned. Nothing was wrong with the corpus size or the
streaming — one single document was still inside one `re.sub()` call.

Two patterns were super-linear. Both are now fixed and both are covered by
`test_backtracking.py`.

**1. `_TRAILING_STRUCTURE_RE` backtracked exponentially.** The negated character
class did not exclude whitespace, so `\s+` and `[^…]+` could both consume the same
space — the classic `(a+)*` ambiguity. Whenever the overall match failed, the
engine re-partitioned the tail every possible way:

| all-caps words after `Chương V` | chars | before | after |
|---|---|---|---|
| 14 | 98 | 8.4 ms | 0.008 ms |
| 28 | 169 | **83 s** | 0.021 ms |
| 56 | 295 | hours | 0.034 ms |
| 1.100 | 6.700 | — | 0.9 ms |

The trigger is an all-caps chapter heading or table of contents — a long run with
no lowercase letter and no `.` `;` `,` — followed by a lowercase word. Very common
in scraped `Chương`/`Mục` blocks, which is exactly what this pattern hunts for.
Adding `\s` to the class makes the split unique and the scan linear.

**2. `clean_content` was quadratic on long space runs.** `[ \t]*\n[ \t\n]*` re-scans
the whole run from every start position and fails when there is no newline in it —
8k spaces 75 ms, 32k spaces 1,2 s, 128k spaces 19 s, and it runs twice per article.
Scraped HTML tables produce runs like that. One `\s+` pass does the same job and
cannot backtrack.

**A watchdog now makes this impossible to miss.** `--doc-timeout` (default 60 s)
abandons a single document instead of hanging the run, `--slow-seconds` (default 5)
prints the id of anything unusually slow, and the progress bar shows the document
in flight, so a stall names its culprit on screen:

```
chunking:  1%|▌  | 48/8532 [00:01<03:08, 45.05 doc/s, doc 60011b (2,671c)]
```

Abandoned ids land in `documents_timed_out` / `slow_documents` in the report.
SIGALRM does interrupt a runaway regex — CPython polls for signals inside the
matching loop — so the timeout works even when the process is stuck in C code.

Ten adversarial documents, before and after — note that the three that hang are
the *small* ones, and that nothing else in the chunker is super-linear:

| adversarial document | chars | before | after |
|---|---|---|---|
| caps `Chương` block then lowercase, inside one article | 891 | **hangs** | 0.001 s |
| caps `Mục 1.` block then lowercase | 888 | **hangs** | 0.000 s |
| 1 MB of spaces inside an article | 1.000.036 | **hangs** | 0.094 s |
| one 500k article, no khoản, no sentence ends | 500.018 | 0.073 s | 0.065 s |
| one 500k article of normal prose | 504.018 | 0.105 s | 0.118 s |
| 20.000 cross-references `khoản 3 Điều 13` | 600.018 | 0.118 s | 0.130 s |
| 10.000 khoản markers | 457.805 | 0.097 s | 0.100 s |
| 5.000 điểm markers in one khoản | 95.078 | 0.023 s | 0.024 s |
| 20.000 `Chương I` mentions + 300 articles | 279.384 | 0.141 s | 0.140 s |
| no Điều at all, 1 MB (plain-text fallback) | 1.024.000 | 0.104 s | 0.131 s |

### `diagnose_hang.py` — if it still stalls

```bash
python diagnose_hang.py corpus.jsonl
```

Three answers, in order, each conclusive on its own:

1. **Which copy of the code is actually imported.** Prints the resolved path of
   `vn_legal_chunker.py` and whether each fix is present. Python takes the first
   match on `sys.path`, so an older copy in the working directory silently wins
   over a fixed one installed elsewhere — this line settles that in one look.
2. **The live stack of the stuck thread**, via `faulthandler`, every `--stall`
   seconds. A frozen regex prints as the exact function and line:
   `File "vn_legal_chunker.py", line 139 in _strip_trailing_structure`.
3. **Per-document timing** for the whole corpus, each document abandoned after
   `--timeout`, ending in a table of the slowest documents by id, line number and
   length, plus `hang_report.json`.

```
  !! line 1  id=POISON  201,542 chars  -- still running after 10.0s, abandoned
   seconds     line      chars  id
    10.000        1    201,542  POISON  <-- STALLED
     0.001        2      2,671  60011
```

### Documents with no "Điều" — the thing to watch

Your corpus is not uniform. The TCVN 13268 sample you sent contains **zero**
`Điều`: it is built from `Phụ lục B`, `B.1`, `Cấp 1:`, `Bảng C.1`. `extract_dieu()`
returns `[]` for documents like that, and a naive loop would drop them without a
word. Default `--fallback window` keeps them, windowed on sentence boundaries and
tagged `"structure": "plain"`, so you can filter on that field later. `--fallback
skip` drops them on purpose. Either way the report prints how many there were —
check that number before you train on the output.

Bad JSON lines, empty `passage` values and blank lines are counted and skipped,
never fatal: one corrupt line out of 8.852 should not kill a 50-second job.
Duplicate chunk text is always counted (your TCVN sample repeats its opening
block verbatim); `--dedup` also drops it.

## Quick start

```python
from vn_legal_chunker import extract_dieu

chunks = extract_dieu(text)
# [{"dieu": "Điều 1. Phạm vi điều chỉnh", "content": "Luật này quy định ..."}, ...]
```

For a whole corpus use `chunk_corpus.py` (above). `vn_legal_chunker.py` also has a
small built-in CLI for one-off files:

```bash
python vn_legal_chunker.py corpus.json -o chunks.jsonl --min-chars 50
python vn_legal_chunker.py vanban.txt  -o chunks.jsonl        # plain text too
```

## How the 4 requirements are met

| # | Requirement | Where |
|---|---|---|
| 1 | `re.DOTALL` + positive lookahead `(?=...)` up to the next Điều or `\Z` | `ARTICLE_RE` |
| 2 | `Điều 1.` / `Điều 2:` — word, spaces, number, dot or colon | `MARKER` |
| 3 | Title split from body; runs of `\n` → one space | `_split_title_body`, `clean_content` |
| 4 | `list[dict]` with `"dieu"` / `"content"` | return of `extract_dieu` |

The pattern, in one line:

```python
ARTICLE_RE = re.compile(rf"(?P<marker>{MARKER})(?P<rest>.*?)(?={MARKER}|\Z)", re.DOTALL)
```

`.*?` is lazy so it stops at the *first* following heading; the lookahead does not
consume that heading, so `finditer` starts the next match exactly there — no
`re.split`, no off-by-one, no second pass.

## Five traps in real Vietnamese legal text (all handled)

1. **Cross-references look like headings.** Điều 1 of Nghị định 26/2019 contains
   *"khoản 10 Điều 10, điểm b khoản 3 Điều 13, ..."* — 30 of them. A naive
   `Điều\s+\d+\s*[.:]` would shred the article. `MARKER` carries a fixed-width
   negative lookbehind `(?<![lowercase/digit] )`: a real heading follows a period,
   an UPPERCASE word or a newline, never a lowercase word or a digit.
   Demo: naive pattern → 3 chunks on the trap sentence, this one → 2.
2. **Your `passage` field is flattened** (every `\n` destroyed by the scraper), so
   nothing separates title from body. Fallback heuristics: body-opener phrase
   ("Nghị định này…", "Bộ trưởng…") or the `1.` khoản marker, whichever fits;
   `title_confident: false` flags those chunks. **If you can re-scrape keeping line
   breaks, do — the split then becomes exact.** Nothing is ever dropped: if no
   signal is found, the whole span stays in `content`.
3. **`Chương` / `Mục` headings sit between articles**, so the lookahead hands them
   to the *previous* article's tail. They are stripped and re-exposed as `chuong` /
   `muc` metadata instead.
4. **OCR typos in the source.** Nghị định 26/2019 really contains `Điền 29.`
   (n instead of u) — Article 29 would silently vanish into Article 28. `_DIEU_WORD`
   accepts `Điều | Ðiều (Eth U+00D0) | ĐIỀU | Điền`.
5. **Boilerplate after the last article** (`Nơi nhận:`, signature, `FILE ĐƯỢC ĐÍNH KÈM`)
   would be glued to Điều 75 — trimmed via `DEFAULT_STOP_MARKERS`.

Also: input is NFC-normalised (scrapers emit both composed and decomposed
diacritics — decomposed `Điều` does not match composed `Điều`), and CRLF is handled.

## Options

```python
extract_dieu(
    text,
    max_title_chars=200,     # safety cap for flattened text
    strip_structure=True,    # drop trailing Chương/Mục heading
    stop_markers=DEFAULT_STOP_MARKERS,
    keep_empty=False,        # keep heading-only articles
    with_metadata=True,      # + so_dieu, chuong, muc, char_len, title_confident
)
```

## Sub-splitting long articles at `khoản`

Article length is very uneven — Điều 44 and Điều 65 run past 6.000 characters
while Điều 24 is under 300. `split_by_khoan()` cuts the long ones at the legal
seams and **repeats the `dieu` title on every piece**, so a retrieved fragment is
still a valid citation.

```python
pieces = extract_dieu(text, with_metadata=True, max_chunk_chars=900)
# or on a single chunk:
pieces = split_by_khoan(chunk, max_chars=1500, min_chars=250)
```

```bash
python vn_legal_chunker.py corpus.json -o chunks.jsonl --max-chars 900
```

```
60011_63_p1   khoản 1     352 chars   Điều 63. Chế độ, chính sách đối với Kiểm ngư
60011_63_p2   khoản 2     593 chars   Điều 63. Chế độ, chính sách đối với Kiểm ngư
60011_63_p3   khoản 3–4   757 chars   Điều 63. Chế độ, chính sách đối với Kiểm ngư
60011_63_p4   khoản 5     712 chars   Điều 63. Chế độ, chính sách đối với Kiểm ngư
60011_63_p5   khoản 6–7   423 chars   Điều 63. Chế độ, chính sách đối với Kiểm ngư
```

Each piece adds `khoan`, `part`, `n_parts`; `chunk_id` gains a `_p{n}` suffix.
Articles under the budget pass through untouched (`n_parts: 1`).

**How the seams are found.** `1.` `2.` `3.` also occur as prose: *"từ điểm 01 đến
điểm 18. Tọa độ..."*, *"ngày 01 tháng 01 năm 2020."*, *"hệ số 0,3 mức lương"*,
*"Mẫu số 09.BT"*. No pattern can tell those apart — but khoản numbering is
**sequential**, so a candidate is accepted only if it is exactly the next expected
number. On the real Điều 3 sentence above, a plain digit-dot pattern finds
`['3', '18', '4']`; the sequence guard returns `['3', '4']`.

Fallback ladder, in order:

1. cut at `khoản` boundaries, then greedily glue consecutive khoản up to `max_chars`
2. a single khoản still too long → cut at `điểm` boundaries (`a) b) c) …`, using
   the Vietnamese letter sequence `a b c d đ e g h i k l m n o p q …` — no f/j/w/z)
3. still too long → break on sentence boundaries
4. a trailing piece under `min_chars` is merged back into the previous one

Two options worth knowing:

* `repeat_lead=True` — copies the article's lead-in (*"…được quy định như sau:"*)
  onto every piece. Costs duplication, helps retrieval when the lead-in carries
  the meaning.
* `prefix_title=True` — also prepends `Điều 63. … — ` to the *text*, so each piece
  is self-contained when fed to a model that only sees `content`.

One known imperfection on flattened text: a closing paragraph that follows the
last `điểm` of a khoản (e.g. the "Phụ cấp đặc biệt … chia cho 22 ngày" sentence at
the end of Điều 63 khoản 5) is attached to that last điểm rather than to the khoản
as a whole. Line-structured input does not have this problem.
