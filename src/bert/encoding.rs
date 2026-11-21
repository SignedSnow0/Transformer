//! Text encoding and decoding for the BERT examples.
//!
//! This module owns the whitespace/punctuation pre-tokenizer, the fixed
//! special-token IDs, and [`BertVocab`], which maps tokens to IDs and back.
//! [`EncodingStrategy`] selects either whole-word lookup or a small BPE model
//! learned from the training corpus. The MLM example uses BPE so inflected and
//! rare words become combinations of known subword pieces instead of `[UNK]`.

use std::collections::HashMap;

/// One English–Italian sentence pair, used to build the vocabulary and as
/// training data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParallelPair {
    /// English sentence.
    pub english: String,
    /// Italian translation.
    pub italian: String,
}

impl ParallelPair {
    /// Creates a sentence pair.
    pub fn new(english: impl Into<String>, italian: impl Into<String>) -> Self {
        Self {
            english: english.into(),
            italian: italian.into(),
        }
    }
}

/// Unknown-token ID.
pub const BERT_UNK_TOKEN: usize = 1;
/// Start-of-sequence/classification token ID.
pub const BERT_CLS_TOKEN: usize = 2;
/// Segment separator token ID.
pub const BERT_SEP_TOKEN: usize = 3;
/// Masked-token ID.
pub const BERT_MASK_TOKEN: usize = 4;

/// Special tokens, in ID order.
pub const BERT_SPECIAL_TOKENS: [&str; 5] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"];

/// Number of reserved special tokens.
pub const BERT_NUM_SPECIAL_TOKENS: usize = BERT_SPECIAL_TOKENS.len();

/// End-of-word marker appended to each pre-token before BPE merging.
///
/// It is part of subword pieces such as `gatto</w>` so the decoder can recover
/// word boundaries. The standalone marker is not a valid MLM prediction target.
pub const BPE_END_OF_WORD: &str = "</w>";

/// Minimum training occurrences for a BPE piece to be an MLM target.
///
/// Rare pieces remain available as context, but masking them in a short run
/// makes the objective needlessly noisy. Punctuation-only pieces are excluded so
/// the metric reflects word prediction rather than a punctuation prior. Smaller
/// corpora lower the cutoff to their most frequent word-like piece.
const BPE_MIN_MASK_FREQUENCY: usize = 50;

/// Vocabulary encoding strategy used by [`BertVocab`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingStrategy {
    /// Look up each whitespace/punctuation token as one vocabulary entry.
    Word,
    /// Split each pre-token into BPE pieces learned from the training corpus.
    Bpe,
}

/// Lower-cases and splits text into word-level and punctuation tokens.
///
/// Consecutive alphanumeric characters, including accented letters and
/// apostrophes, form one token. Other non-whitespace characters become
/// individual punctuation tokens.
pub fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for character in text.chars() {
        let character = match character {
            '\u{2018}' | '\u{2019}' => '\'',
            other => other,
        };

        if character.is_alphanumeric() || character == '\'' {
            for lowercase in character.to_lowercase() {
                current.push(lowercase);
            }
        } else {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
            if !character.is_whitespace() {
                tokens.push(character.to_string());
            }
        }
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

/// Vocabulary with BERT's special tokens at fixed IDs and a selectable
/// [`EncodingStrategy`].
#[derive(Debug, Clone)]
pub struct BertVocab {
    tokens: Vec<String>,
    index: HashMap<String, usize>,
    strategy: EncodingStrategy,
    /// Whether each ID is a useful MLM prediction target.
    maskable: Vec<bool>,
    /// BPE merge ranks. Empty for the word-level strategy.
    merges: HashMap<(String, String), usize>,
}

impl BertVocab {
    /// Builds a vocabulary with an explicit encoding strategy.
    pub fn build_with_strategy(
        pairs: &[ParallelPair],
        max_size: usize,
        min_frequency: usize,
        strategy: EncodingStrategy,
    ) -> Self {
        let sentences = pairs
            .iter()
            .flat_map(|pair| [pair.english.as_str(), pair.italian.as_str()]);
        Self::build_from_texts_with_strategy(sentences, max_size, min_frequency, strategy)
    }

    /// Builds a word-level vocabulary from an arbitrary collection of texts.
    ///
    /// This is used by the spam detector, whose corpus has one text per record
    /// rather than aligned sentence pairs. The ranking rules are identical to
    /// [`BertVocab::build_with_strategy`] with [`EncodingStrategy::Word`].
    pub fn build_from_texts<'a>(
        texts: impl IntoIterator<Item = &'a str>,
        max_size: usize,
        min_frequency: usize,
    ) -> Self {
        Self::build_from_texts_with_strategy(texts, max_size, min_frequency, EncodingStrategy::Word)
    }

    /// Builds a vocabulary from texts using an explicit encoding strategy.
    pub fn build_from_texts_with_strategy<'a>(
        texts: impl IntoIterator<Item = &'a str>,
        max_size: usize,
        min_frequency: usize,
        strategy: EncodingStrategy,
    ) -> Self {
        match strategy {
            EncodingStrategy::Word => Self::build_word_vocab(texts, max_size, min_frequency),
            EncodingStrategy::Bpe => Self::build_bpe_vocab(texts, max_size, min_frequency),
        }
    }

    fn build_word_vocab<'a>(
        texts: impl IntoIterator<Item = &'a str>,
        max_size: usize,
        min_frequency: usize,
    ) -> Self {
        let max_size = max_size.max(BERT_NUM_SPECIAL_TOKENS);
        let min_frequency = min_frequency.max(1);
        let mut counts: HashMap<String, usize> = HashMap::new();

        for sentence in texts {
            for token in tokenize(sentence) {
                if !BERT_SPECIAL_TOKENS.contains(&token.as_str()) {
                    *counts.entry(token).or_default() += 1;
                }
            }
        }

        let mut words: Vec<(String, usize)> = counts
            .into_iter()
            .filter(|(_, count)| *count >= min_frequency)
            .collect();
        words.sort_by(|(left_word, left_count), (right_word, right_count)| {
            right_count
                .cmp(left_count)
                .then_with(|| left_word.cmp(right_word))
        });
        words.truncate(max_size.saturating_sub(BERT_NUM_SPECIAL_TOKENS));

        let mut tokens = special_tokens();
        tokens.extend(words.iter().map(|(word, _)| word.clone()));
        let maskable = (0..tokens.len())
            .map(|id| id >= BERT_NUM_SPECIAL_TOKENS)
            .collect();

        Self::from_parts(tokens, maskable, EncodingStrategy::Word, HashMap::new())
    }

    /// Trains BPE merges on the training texts, then fills the vocabulary with
    /// the learned pieces. Only the training split should be passed here.
    fn build_bpe_vocab<'a>(
        texts: impl IntoIterator<Item = &'a str>,
        max_size: usize,
        min_frequency: usize,
    ) -> Self {
        let max_size = max_size.max(BERT_NUM_SPECIAL_TOKENS);
        let min_frequency = min_frequency.max(1);
        let mut word_counts: HashMap<String, usize> = HashMap::new();

        for sentence in texts {
            for token in tokenize(sentence) {
                if !BERT_SPECIAL_TOKENS.contains(&token.as_str()) {
                    *word_counts.entry(token).or_default() += 1;
                }
            }
        }
        word_counts.retain(|_, count| *count >= min_frequency);

        // Start with the characters seen in training plus the word boundary.
        // The most frequent symbols are kept when `max_size` is very small.
        let mut character_counts: HashMap<String, usize> = HashMap::new();
        for (word, count) in &word_counts {
            for character in word.chars() {
                *character_counts.entry(character.to_string()).or_default() += count;
            }
            *character_counts
                .entry(BPE_END_OF_WORD.to_string())
                .or_default() += count;
        }
        let mut characters: Vec<(String, usize)> = character_counts.into_iter().collect();
        characters.sort_by(|(left, left_count), (right, right_count)| {
            right_count.cmp(left_count).then_with(|| left.cmp(right))
        });
        characters.truncate(max_size.saturating_sub(BERT_NUM_SPECIAL_TOKENS));

        let mut tokens = special_tokens();
        tokens.extend(characters.iter().map(|(symbol, _)| symbol.clone()));

        let mut words: Vec<(Vec<String>, usize)> = word_counts
            .iter()
            .map(|(word, count)| (bpe_symbols(word), *count))
            .collect();
        words.sort_by(|(left, _), (right, _)| left.cmp(right));

        let mut merges: HashMap<(String, String), usize> = HashMap::new();
        while tokens.len() < max_size {
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
            for (symbols, count) in &words {
                for pair in symbols.windows(2) {
                    *pair_counts
                        .entry((pair[0].clone(), pair[1].clone()))
                        .or_default() += count;
                }
            }

            let Some((pair, frequency)) = pair_counts.into_iter().max_by(
                |(left_pair, left_count), (right_pair, right_count)| {
                    left_count
                        .cmp(right_count)
                        .then_with(|| right_pair.cmp(left_pair))
                },
            ) else {
                break;
            };
            if frequency == 0 {
                break;
            }

            let merged = format!("{}{}", pair.0, pair.1);
            if merges.contains_key(&pair) || tokens.contains(&merged) {
                break;
            }
            merges.insert(pair.clone(), merges.len());
            tokens.push(merged);
            for (symbols, _) in &mut words {
                merge_bpe_pair(symbols, &pair);
            }
        }

        // Rare pieces are still useful context, but they make poor MLM targets
        // in a short run. Count final pieces so masking can prefer frequent
        // ones while rare words remain representable as subwords.
        let mut piece_counts: HashMap<&str, usize> = HashMap::new();
        for (symbols, count) in &words {
            for symbol in symbols {
                *piece_counts.entry(symbol.as_str()).or_default() += count;
            }
        }
        // Small custom corpora may not have any piece with 50 occurrences, so
        // lower the cutoff to the most frequent piece instead of failing.
        let max_piece_count = tokens
            .iter()
            .filter(|token| is_word_piece(token))
            .filter_map(|token| piece_counts.get(token.as_str()).copied())
            .max()
            .unwrap_or(1);
        let mask_frequency = BPE_MIN_MASK_FREQUENCY.min(max_piece_count.max(1));
        let maskable = tokens
            .iter()
            .map(|token| {
                is_word_piece(token)
                    && piece_counts
                        .get(token.as_str())
                        .is_some_and(|&count| count >= mask_frequency)
            })
            .collect();

        Self::from_parts(tokens, maskable, EncodingStrategy::Bpe, merges)
    }

    fn from_parts(
        tokens: Vec<String>,
        maskable: Vec<bool>,
        strategy: EncodingStrategy,
        merges: HashMap<(String, String), usize>,
    ) -> Self {
        let index = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id))
            .collect();
        Self {
            tokens,
            index,
            strategy,
            maskable,
            merges,
        }
    }

    /// Returns the number of tokens in the vocabulary.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Returns the encoding strategy used by this vocabulary.
    pub fn strategy(&self) -> EncodingStrategy {
        self.strategy
    }

    /// Returns the ID for an exact vocabulary token, falling back to `[UNK]`.
    pub fn id(&self, token: &str) -> usize {
        self.index.get(token).copied().unwrap_or(BERT_UNK_TOKEN)
    }

    /// Returns the token for an ID.
    pub fn token(&self, id: usize) -> Option<&str> {
        self.tokens.get(id).map(String::as_str)
    }

    /// Returns whether an ID is a valid MLM prediction target.
    ///
    /// Special tokens are always excluded. A BPE vocabulary also excludes the
    /// word boundary, punctuation-only pieces, single characters, and pieces
    /// below the frequency cutoff: they remain useful context but make poor
    /// short-run prediction targets.
    pub fn is_maskable(&self, id: usize) -> bool {
        self.maskable.get(id).copied().unwrap_or(false)
    }

    /// Encodes pre-tokenized words using this vocabulary's strategy.
    pub fn encode_tokens(&self, tokens: &[String]) -> Vec<usize> {
        match self.strategy {
            EncodingStrategy::Word => tokens.iter().map(|token| self.id(token)).collect(),
            EncodingStrategy::Bpe => tokens
                .iter()
                .flat_map(|token| self.encode_bpe_word(token))
                .collect(),
        }
    }

    /// Decodes token IDs into a readable, space-separated string.
    pub fn decode(&self, ids: &[usize]) -> String {
        match self.strategy {
            EncodingStrategy::Word => ids
                .iter()
                .map(|&id| self.token(id).unwrap_or("[UNK]"))
                .collect::<Vec<_>>()
                .join(" "),
            EncodingStrategy::Bpe => {
                let mut text = String::new();
                for &id in ids {
                    let Some(token) = self.token(id) else {
                        continue;
                    };
                    if BERT_SPECIAL_TOKENS.contains(&token) {
                        text.push(' ');
                        text.push_str(token);
                        text.push(' ');
                    } else if let Some(word) = token.strip_suffix(BPE_END_OF_WORD) {
                        text.push_str(word);
                        text.push(' ');
                    } else {
                        text.push_str(token);
                    }
                }
                text.split_whitespace().collect::<Vec<_>>().join(" ")
            }
        }
    }

    fn encode_bpe_word(&self, word: &str) -> Vec<usize> {
        let mut symbols = bpe_symbols(word);
        if symbols
            .iter()
            .any(|symbol| !self.index.contains_key(symbol))
        {
            return vec![BERT_UNK_TOKEN];
        }

        loop {
            let mut best_rank = usize::MAX;
            let mut best_position = None;
            for position in 0..symbols.len().saturating_sub(1) {
                let pair = (symbols[position].clone(), symbols[position + 1].clone());
                if let Some(&rank) = self.merges.get(&pair)
                    && rank < best_rank
                {
                    best_rank = rank;
                    best_position = Some(position);
                }
            }

            let Some(position) = best_position else {
                break;
            };
            let merged = format!("{}{}", symbols[position], symbols[position + 1]);
            symbols.splice(position..position + 2, [merged]);
        }

        symbols.iter().map(|symbol| self.id(symbol)).collect()
    }
}

fn special_tokens() -> Vec<String> {
    BERT_SPECIAL_TOKENS
        .iter()
        .map(|token| (*token).to_string())
        .collect()
}

fn bpe_symbols(word: &str) -> Vec<String> {
    let mut symbols: Vec<String> = word
        .chars()
        .map(|character| character.to_string())
        .collect();
    symbols.push(BPE_END_OF_WORD.to_string());
    symbols
}

fn is_word_piece(token: &str) -> bool {
    let word = token.strip_suffix(BPE_END_OF_WORD).unwrap_or(token);
    word.chars().count() > 1 && word.chars().any(char::is_alphabetic)
}

fn merge_bpe_pair(symbols: &mut Vec<String>, pair: &(String, String)) {
    let mut merged_symbols = Vec::with_capacity(symbols.len());
    let mut position = 0;
    while position < symbols.len() {
        if position + 1 < symbols.len()
            && symbols[position] == pair.0
            && symbols[position + 1] == pair.1
        {
            merged_symbols.push(format!("{}{}", pair.0, pair.1));
            position += 2;
        } else {
            merged_symbols.push(symbols[position].clone());
            position += 1;
        }
    }
    *symbols = merged_symbols;
}
