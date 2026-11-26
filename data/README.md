# Corpora

The BERT examples use the two TSVs in this directory by default. Regenerate
them from their original sources with `scripts/prepare_datasets.py`.

| File | Rows | Format | Used by | Provenance |
|---|---:|---|---|---|
| `eng-it-full.tsv` | 239,245 | `english<TAB>italian` | MLM demo (default) | [`eng-it-full.md`](eng-it-full.md) |
| `sms-spam-full.tsv` | 5,574 | `ham\|spam<TAB>message` | Spam demo (default) | [`sms-spam-full.md`](sms-spam-full.md) |

The OPUS archive contains 540,976 raw Tatoeba pairs; 239,245 pass the corpus
quality filters. The resulting file is about 15 MB. The SMS file keeps the
source corpus's natural class imbalance: 747 spam and 4,827 ham messages.

## Regenerating

```bash
python scripts/prepare_datasets.py all
```

Individual corpora and deterministic subsets are also supported:

```bash
python scripts/prepare_datasets.py tatoeba
python scripts/prepare_datasets.py sms --limit 50_000
```

`--limit N` applies a reservoir sample with seed `42`, so repeated runs produce
the same subset. Each generated corpus gets a companion `.md` note with its
source URL, SHA-256 checksum, license, and citation.

## Running the defaults

```bash
cargo run --release -- --cpu
cargo run --release -- --cpu --spam
```

The defaults are 20,000 MLM steps / 1,000 spam steps, `d_model=128`, `d_ff=256`,
4 heads, 2 layers, `max_len=64`, and `dropout=0.1`. MLM validation uses a fixed
sample of 512 held-out examples so longer runs stay practical. Pass positional
`[DATA_TSV] [STEPS]` or capacity flags to override them.
