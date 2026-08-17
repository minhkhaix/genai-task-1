# Making the chunks fit to index and fit to train on

"Perfect" is not a thing you can eyeball at the top of a 500.000-line file. What
you can do is *measure* the ways a chunk set goes wrong, fix the ones that
matter, and know the number for the rest. That is what `audit_chunks.py` is for.

```bash
# 1. re-chunk with the settings below
python chunk_corpus.py corpus.jsonl -o chunks.jsonl \
    --max-chars 900 --min-chars 150 --prefix-title --fill-name --dedup \
    --report chunk_report.json

# 2. measure it, against the source
python audit_chunks.py chunks.jsonl --corpus corpus.jsonl \
    --tokenizer Qwen/Qwen2.5-3B --max-tokens 512 \
    --profile rag --emit-clean chunks_rag.jsonl \
    --report audit.json --samples audit_samples.jsonl

# 3. read the samples. The counts tell you where to look; the text tells you
#    whether it is really a defect.
head -40 audit_samples.jsonl

# 4. build the instruction set from the audited chunks
python audit_chunks.py chunks.jsonl --profile train --emit-clean chunks_train.jsonl
python make_sft_pairs.py chunks_train.jsonl -o sft \
    --with-context 4 --refusal-rate 0.12 --split 0.90,0.05,0.05
```

---

## 1. RAG and fine-tuning do not want the same chunk

| | RAG index | Fine-tuning set |
|---|---|---|
| unit | one retrievable, citable passage | one *question and answer*, not a passage |
| size | fits the embedder with room to spare | fits the training context with the retrieved passages |
| duplicates | waste index space, skew similarity | far worse: the model memorises the repeated text |
| a fragment ("b) Từ năm thứ sáu…") | useless — nothing to cite | useless — nothing to learn |
| a table of contents | pure noise in the top-k | teaches the model to emit lists of headings |
| non-Vietnamese junk | rarely retrieved, harmless | actively harmful |

Same source chunks, two filters. That is why `audit_chunks.py` has
`--profile rag` and `--profile train`: `train` additionally drops chunks that are
not Vietnamese prose and chunks that repeat themselves.

---

## 2. Size, for Qwen

Qwen's context is not the binding constraint — 32k tokens means chunk size is
driven by *retrieval quality*, not by a cliff. (Some popular Vietnamese encoders
do have a cliff: `bkai-foundation-models/vietnamese-bi-encoder` truncates at 256
tokens, and 900-character chunks would lose their tails silently. If you ever
switch to one, re-run the audit with `--max-tokens 256` first.)

What actually goes wrong at each size:

* **too small** — the answer spans two chunks and neither one retrieves. Vietnamese
  legal answers usually live in a whole khoản; cutting inside one is the main
  cause of "the right article ranked #1 but the answer was not in it".
* **too large** — one embedding averages several unrelated rules, and the vector
  stops discriminating. An article covering seven khoản about seven different
  allowances embeds to mush.

`--max-chars 900` with khoản-boundary splitting is a good default: it cuts at the
legal seam rather than at a character count, so a piece is one or a few whole
khoản. Raise to 1200 if recall is poor; drop to 600 if the top-k comes back
topically right but factually vague.

**Token counts need your tokenizer, not mine.** Run the audit with
`--tokenizer Qwen/Qwen2.5-3B` on the machine that has the model cached — the
`tokens` block then reports exact counts. Without it the report falls back to a
character estimate and says so on the line.

---

## 3. The settings to change right now

**`--prefix-title` is the important one.** Without it, part 2 of a split article
is a bare "2. Chế độ phụ cấp trách nhiệm…" with nothing saying which article it
belongs to. It embeds badly (the topic words are in the title, which is missing)
and it cannot be cited when retrieved. The audit reports this as `orphan_part`;
on the test corpus it was **6 of 18 chunks**. With `--prefix-title` the title is
repeated into the text of every piece and the count goes to zero.

The cost is duplication — the title is in the vector for every part. That is the
right trade for legal retrieval, where the article title carries most of the
topical signal. `make_sft_pairs.py` strips the repeated title again when it builds
prompts, so you do not pay for it twice in the training context.

**`--min-chars 150`** (up from 80). Below that you get stray headings and
one-line cross-references that occupy an index slot and never answer anything.

**Keep `--dedup`.** Vietnamese legal corpora repeat boilerplate heavily, and
scrapers duplicate opening blocks.

**Decide about `structure: "plain"`.** TCVN/QCVN standards, công văn and biểu mẫu
have no Điều, and `--fallback window` keeps them as sentence-windowed chunks. They
are fine to retrieve and bad to train citation behaviour on, because there is no
Điều to cite. Filter on the `structure` field: keep them in the index, exclude
them from the instruction set.

---

## 4. What the audit checks, and what to do about each

**FATAL — fix before doing anything else**

| check | what it means |
|---|---|
| `dup_chunk_id` | two chunks share an id; on upsert the later silently overwrites the earlier and you lose a passage |
| `empty_content` | empty chunk |
| `char_len_mismatch` | the metadata disagrees with the text; something rewrote one and not the other |

**RAG**

| check | what to do |
|---|---|
| `over_budget` | above `--max-tokens`; the tail is dropped at index time and you never see it. Lower `--max-chars`. |
| `orphan_part` | re-run with `--prefix-title` |
| `no_dieu` | plain-text chunks; keep for retrieval, exclude from citation training |
| `title_guessed` | the source was flattened so the title/body split was inferred. If you can re-scrape keeping line breaks, this number goes to zero. |
| `no_link` | you cannot show a source; use `--fill-name` |
| `fragment_start` | opens mid-sentence — usually an điểm-level split. Raise `--max-chars` so fewer khoản need cutting. |
| `toc`, `boilerplate`, `too_short` | drop (both profiles do) |

**TRAINING**

`exact_dup`, `near_dup`, `junk_ratio`, `all_caps`, `not_vietnamese`,
`self_repeating` — all dropped by `--profile train`. `near_dup` deliberately keeps
numbers intact: "phụ cấp bằng 20%" and "phụ cấp bằng 30%" are two different rules,
not one clause seen twice.

**INFO**

`template_dup` — identical once numbers are folded too. Could be a form repeated
per row (drop it) or two genuinely different clauses (keep them). Read the samples
before deciding; nothing is dropped automatically.

**FIDELITY — the check nobody runs**

With `--corpus` the audit answers three questions the chunk count cannot:

* **did every document survive?** `documents_with_no_chunks` names the ones that
  vanished.
* **did every character survive?** `median_character_retention`. Below 1.0 by a
  few percent is expected and correct — that is the Chương headings, the header
  block and the `Nơi nhận:` boilerplate being stripped. A document at 0.4 means
  the chunker lost most of it; `documents_under_50pct_sample` lists them.
* **is the text still verbatim?** Every sampled chunk is checked to be a literal
  substring of its source document. `containment_failed > 0` means something is
  mangling text, and that is the one number that should always be zero.

---

## 5. Fine-tuning: raw chunks are not an instruction set

Feeding article text to an SFT run teaches the model to *continue legal prose*.
It does not teach it to answer a question, and it does not teach it to use a
retrieved passage. If you will serve the model behind your own retriever, the
training examples have to look like serving time:

```
system     answer only from the passages given, and cite the Điều
user       Ngữ cảnh: [1] … [2] … [3] … [4] …
           Câu hỏi: Khoản 2 Điều 63 Nghị định 26/2019/NĐ-CP quy định gì?
assistant  2. Chế độ phụ cấp trách nhiệm theo nghề … (Căn cứ: Điều 63 Nghị định 26/2019/NĐ-CP)
```

Three things `make_sft_pairs.py` does that a naive generator does not:

1. **Hard negatives.** The other three passages come from the same document, so
   they look relevant and are not. Train with only the correct passage in context
   and the model learns to ignore the context and answer from memory — which is
   exactly the failure you built RAG to avoid.
2. **Refusals** (`--refusal-rate 0.12`). The correct passage is removed and the
   answer becomes "không có căn cứ". Without these the model invents an Điều
   number for anything you ask, which in a legal assistant is the worst possible
   failure. The builder checks that a refusal is *actually* correct: it removes
   any distractor from the same Điều, or with the same text under a different
   number, before deciding the answer is absent.
3. **A split that does not leak.** Splitting by chunk puts khoản 1 of an article
   in train and khoản 2 in test, and your eval number becomes fiction. The split
   is by **document**, and answer text is checked across the boundary as well —
   on a 300-document run that dropped a further 120 examples.

Every generated pair goes through a grounding validator: every number in the
answer must appear in the context, and the cited Điều must be one of the passages
actually shown. On a 12.562-example run it reported zero failures — which is what
you want from template-generated data, and is exactly the check you need when the
data is *not* template-generated.

**On question diversity — the honest limitation.** The questions here are
template-generated. They teach format, citation discipline and grounding; they do
not teach the long tail of how people actually ask ("bố tôi đi biển 3 tháng thì
được phụ cấp gì"). Use `--llm-prompts gen.jsonl` to emit one generation prompt per
chunk, send those to a stronger model, and merge the returned pairs. The same
`validate()` runs over them — and there it is doing real work, because a generated
answer can drift off its source in a way a template answer cannot.

---

## 6. Before you train, and how you will know it worked

Before:

- [ ] `audit.json` shows **zero FATAL**
- [ ] `containment_failed` is **0**
- [ ] `median_character_retention` is where you expect (~0.9 with boilerplate
      stripping; investigate anything much lower)
- [ ] `documents_with_no_chunks` is a number you have looked at, not a surprise
- [ ] `over_budget` is 0 at your real `--max-tokens`, measured with your tokenizer
- [ ] you have actually read `audit_samples.jsonl`

After — these three numbers matter more than training loss:

1. **Retrieval recall@k.** Hold out questions whose answering article you know
   (the `test` split gives you these for free: each example's `meta.chunk_id` is
   the passage that should come back). If recall@5 is poor, the chunking is
   wrong, not the model.
2. **Citation accuracy.** Of the answers that cite an Điều, how many cite the one
   the answer actually came from? `validate()` in `make_sft_pairs.py` is the same
   check, and you can run it over the model's outputs.
3. **Refusal rate on unanswerable questions.** Ask questions whose article you
   deliberately kept out of the context. A model that answers anyway has learned
   to ignore retrieval, and no amount of extra training data fixes that — you add
   refusal examples instead.
