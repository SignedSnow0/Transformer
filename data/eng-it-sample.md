# Bundled corpora

This directory contains two small educational samples:

- `eng-it-sample.tsv`: 2,000 English–Italian sentence pairs for the default
  masked-language-model demo.
- `sms-spam-sample.tsv`: 1,200 labelled SMS messages for the `--spam` demo.

## English–Italian sample corpus

`eng-it-sample.tsv` is a deterministic sample of **2,000 English–Italian
sentence pairs** from [Tatoeba](https://tatoeba.org/), distributed by
[OPUS](https://opus.nlpl.eu/legacy/Tatoeba.php) as the `Tatoeba v2026-07-08`
Moses package (`en-it.txt.zip`).

## Provenance and license

- Source: `https://object.pouta.csc.fi/OPUS-Tatoeba/v2026-07-08/moses/en-it.txt.zip`
- Original files: `Tatoeba.en-it.en` and `Tatoeba.en-it.it` (540,976 pairs each)
- License: [CC BY 2.0 FR](https://creativecommons.org/licenses/by/2.0/fr/)
- Copyright and terms: <https://tatoeba.org/eng/terms_of_use>

Tatoeba sentences are contributed by volunteers. The sample keeps the
attribution and license requirements of the original corpus. If you reuse the
data, cite:

> J. Tiedemann, 2012. *Parallel Data, Tools and Interfaces in OPUS.* In
> Proceedings of the 8th International Conference on Language Resources and
> Evaluation (LREC 2012).

## How the sample was produced

A reservoir sample of 2,000 pairs was drawn with seed `42` from the following
filtered candidates:

- both sides non-empty and 12–180 characters long,
- 3–18 whitespace-delimited words per side,
- no URLs, email addresses, or markup characters (`<>{}[]`),
- the two sides differ,
- English sentences unique under case-insensitive comparison.

The result was sorted lexicographically and encoded as UTF-8 without a BOM.
Each line is `english<TAB>italian`. This is a small educational sample, not a
representative benchmark.

## Regenerating

Download and extract the OPUS archive, then use the filtering rules above.
The bundled `src/bert/examples/mlm.rs` example loads this exact format; any other
two-column TSV with English in the first column and Italian in the second works
as well.

## SMS spam sample corpus

`sms-spam-sample.tsv` is a deterministic, class-balanced sample of **1,200 SMS
messages** (600 ham and 600 spam) from the
[SMS Spam Collection v.1](https://archive.ics.uci.edu/dataset/228/sms+spam+collection),
distributed by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/).

### Provenance and license

- Source archive: `https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip`
- Original file: `SMSSpamCollection` (5,574 messages: 4,827 ham and 747 spam)
- Copyright: Tiago Agostinho de Almeida and José María Gómez Hidalgo
- License: the corpus's custom permissive terms, reproduced in
  `sms-spam-collection-README.txt`. Redistribution is allowed; attribution to
  the corpus and its authors is requested, and the dataset is provided "as is"
  without warranty.

Please cite the corpus when reusing it:

> Almeida, T.A., Gómez Hidalgo, J.M., and Yamakami, A. 2011. *Contributions to
> the study of SMS spam filtering: new collection and results.* In Proceedings
> of the 2011 ACM Symposium on Document Engineering (ACM DOCENG'11), Mountain
> View, CA, USA.

### How the sample was produced

The original corpus was downloaded from UCI. Lines were split at the first tab
into `label` and `message`; empty and malformed lines were rejected. The ham and
spam rows were shuffled independently with seed `42`, the first 600 rows of each
class were selected, and the combined 1,200 rows were shuffled again with the
same seed. The result was written as UTF-8 without a BOM, one
`ham<TAB>message` or `spam<TAB>message` record per line. A few messages in the
source use UTF-8-encoded pound signs, which are preserved in the sample.

Every tenth record of the shuffled sample is held out by the demo, giving 120
evaluation messages (64 spam, 56 ham) and 1,080 training messages. The training
split alone is used to build the word-level spam vocabulary. The MLM example
also holds out every tenth sentence pair and trains its BPE merges on the
training split only.

### Regenerating

Download and extract the UCI archive, apply the deterministic sampling procedure
above, and keep `sms-spam-collection-README.txt` alongside the sample so the
license terms remain available. The `src/bert/examples/spam.rs` example accepts any
`label<TAB>message` TSV with `ham` or `spam` in the first column.
