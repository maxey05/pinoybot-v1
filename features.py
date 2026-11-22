# Features of the previous word that you already computed for the current one
# Features of the next word that you already computed for the current one (last two help since adjacent words tend to share labels)


# just a punctuation (almost always OTH)
def feature1(word):
    to_check = [".", "?", "!", ","]
    if all(ch in to_check for ch in word):
        return 1
    return 0

# hypen or apostrophe appearing in a word (mostly FIL)
def feature2(word):
    to_check = ["-","'"]
    if any(ch in word for ch in to_check):
        return 1
    return 0

# ratio of capital letters (helps for acronyms and capitalize words)
def feature3(word):
    count = 0
    for char in word:
        if char.isupper():
            count += 1
    return count / len(word)

# sentence postion (helps for diffentiating of proper names and starting words)
def feature4(index,sentence_words):
    if sentence_words > 1: 
        return index / (sentence_words - 1)
    return 0

# character 3-gram (helps find patterns between words of languages) FIL
def feature5(word):
    three_word = ["ang", "ing", "nag", "pag", "mag"]
    count = 0
    lower_word = word.lower()
    if len(lower_word) >= 3:
        for i in range(len(lower_word) - 2):
            gram = lower_word[i : i + 3]
            if gram in three_word:
                return 1
    return 0

# ENG ver
def feature6(word):
    three_word = ["ion", "ght", "ent", "the", "and"]
    count = 0
    lower_word = word.lower()
    if len(lower_word) >= 3:
        for i in range(len(lower_word) - 2):
            gram = lower_word[i : i + 3]
            if gram in three_word:
                return 1
    return 0

def get_features(word,index,sentence_words):
    f1 = feature1(word)
    f2 = feature2(word)
    f3 = feature3(word)
    f4 = feature4(index,sentence_words)
    f5 = feature5(word)
    f6 = feature6(word)

    features = [f1, f2, f3, f4, f5,f6]

    return features