import re
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

# capitalize a word and turn it into a number pattern.
# uppercase letters become X's
# lowercase letters become x's
# numbers become d's
# if there is a continuous run of repeated symbols, this function basically collapses it so that words with the same capitalization pattern can be identified by the model
def word_shape(word):
    chars = []
    for c in word:
        if c.isupper():
            chars.append("X")
        elif c.islower():
            chars.append("x")
        elif c.isdigit():
            chars.append("d")
        else:
            chars.append(c)

    s = "".join(chars)
    result = ""
    for c in s:
        if not result.endswith(c * 2):
            result += c
    return result

# because a word's meaning usually depends on where it sits in a sentence, we need a sort of "dictionary" to deduce each word's meaning in the sentence.
# this is where features comes in

# These features give hints on what the current word may be
# - feats["is_punct"] if the token has no letters or digits return 1, 0 otherwise
# - feats["is_single_cap"] if the token has only 1 capital letter return 1, 0 otherwise
# - feats["cap_not_first"] if the word starts with a capital letter and is NOT the first word in the sentence return 1, otherwise 0
# - feats["all_caps"] if the word is all caps return 1, 0 otherwise
# - feats["length"] measures how long a word is, turns the length into a decimal between 0 and 1
# - feats["shape"] this is based on the word_shaping we did a while ago, basically maps out a word's shape to a specific value (like Xxx = 1)
# - feats["word"] this is a recording of the actual word itself.

# These features give hints on what the previous word may be
# - feats["BOS"] if the word is at the start of the sentence return 1, 0 otherwise
# - feats["prev_words"] tells the model what word came before it (its based on feats["word"])
# - feats["prev_suffix3="] set to 1 for the last 3 characters of the previous word
# - feats["prev_is_cap"] if the previous word starts with a capital letter return 1, 0 otherwise
# - feats["prev_is_punct"] if the previous token is a punctuation return 1, otherwise 0

# These features give hints on what the next word may be
# - feats["EOS"] if the current token is the very last one in the sentence return 1
# - feats["next_words"] tells the model what word comes after it
# - feats["next_suffix3"] set to 1 for the last 3 characters of the next word
# - feats["next_is_cap"] if the next word starts with a capital letter, otherwise 0
# - feats["next_is_punct"] if the next token is a punctuation return 1, 0 otherwise
def token_features(tokens, i):
    word = tokens[i]
    feats = {}

    # is_punct
    if len(word) == 0:
        feats["is_punct"] = 0
    else:
        has_letter_or_digit = False
        for char in word:
            if char.isalnum():
                has_letter_or_digit = True
                break

        if has_letter_or_digit == False:
            feats["is_punct"] = 1
        else:
            feats["is_punct"] = 0

    # is_single_cap
    if len(word) == 1 and word.isupper():
        feats["is_single_cap"] = 1
    else:
        feats["is_single_cap"] = 0

    # cap_not_first
    if len(word) > 0 and word[0].isupper() and i > 0:
        feats["cap_not_first"] = 1
    else:
        feats["cap_not_first"] = 0

    # all_caps
    has_letter = False
    for char in word:
        if char.isalpha():
            has_letter = True
            break

    if has_letter == True and word.isupper():
        feats["all_caps"] = 1
    else:
        feats["all_caps"] = 0

    # length
    word_len = len(word)
    if word_len > 15:
        capped_len = 15
    else:
        capped_len = word_len

    feats["length"] = capped_len / 15.0

    # shape
    shape_pattern = word_shape(word)
    feats["shape=" + shape_pattern] = 1

    # word
    lower_word = word.lower()
    feats["word=" + lower_word] = 1

    # Previous token clues
    if i > 0:
        prev_word = tokens[i - 1]
        prev_lower = prev_word.lower()

        feats["prev_word=" + prev_lower] = 1
        feats["prev_suffix3=" + prev_lower[-3:]] = 1

        if len(prev_word) > 0 and prev_word[0].isupper():
            feats["prev_is_cap"] = 1
        else:
            feats["prev_is_cap"] = 0

        if len(prev_word) == 0:
            feats["prev_is_punct"] = 0
        else:
            prev_has_alnum = False
            for char in prev_word:
                if char.isalnum():
                    prev_has_alnum = True
                    break

            if prev_has_alnum == False:
                feats["prev_is_punct"] = 1
            else:
                feats["prev_is_punct"] = 0

    else:
        feats["BOS"] = 1

    # Next token clues
    last_index = len(tokens) - 1
    if i < last_index:
        next_word = tokens[i + 1]
        next_lower = next_word.lower()

        feats["next_word=" + next_lower] = 1
        feats["next_suffix3=" + next_lower[-3:]] = 1

        if len(next_word) > 0 and next_word[0].isupper():
            feats["next_is_cap"] = 1
        else:
            feats["next_is_cap"] = 0

        if len(next_word) == 0:
            feats["next_is_punct"] = 0
        else:
            next_has_alnum = False
            for char in next_word:
                if char.isalnum():
                    next_has_alnum = True
                    break

            if next_has_alnum == False:
                feats["next_is_punct"] = 1
            else:
                feats["next_is_punct"] = 0

    else:
        feats["EOS"] = 1

    return feats

# converts a word into lowercase and wraps them in '^' and '$'
# ^ means start
# $ means end
# These markers let the model tell prefixes apart from suffixes
def char_text(word):
    return "^" + word.lower() + "$"

# creates two "translators" of some sort. It turns text clues into numbers that the models can read.
# char_vec = chops each ^word$ string into every 1-4 letter chunk and counts them
# dict_vec = turns the features dictionary into numeric columns
def make_vectorizers():
    char_vec = CountVectorizer(analyzer="char", ngram_range=(1,4))
    dict_vec = DictVectorizer()

    return char_vec, dict_vec

# loop through each word in a sentence and run char_text and the features on it. Feed those words into the translators and add them to a matrix.
def build_matrix(sentence_token_lists, char_vec, dict_vec, fit=False):
    words = []
    dicts = []

    for tokens in sentence_token_lists:
        for i in range(len(tokens)):
            words.append(char_text(tokens[i]))
            dicts.append(token_features(tokens, i))

    if fit:
        char_part = char_vec.fit_transform(words)
        dict_part = dict_vec.fit_transform(dicts)
    else:
        char_aprt = char_vec.transform(words)
        dict_part = dict_vec.transform(dicts)

    return sp.hstack([char_part, dict_part]).tocsr()


    


        