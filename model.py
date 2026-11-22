import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/dataset.csv"
# locks the random shuffler so it deals the same sentences, 42 is arbitrary
RANDOM_STATE = 42 

# Reads the CSV, sorts and combines words with the same sentence_id as 
# one sentence, then returns a list of (tokens, labels) pairs.
def load_sentences(path):
    df = pd.read_csv(path, dtype = str, keep_default_na=False)

    print(df.shape)
    print(df["label"].value_counts(normalize=True))

    df["word_id"] = df["word_id"].astype(int)
    df["sentence_id"] = df["sentence_id"].astype(int)
    df = df.sort_values(["sentence_id", "word_id"])

    sentences = []

    for _, group in df.groupby("sentence_id", sort=False):
        list_of_words = group["word"].tolist()
        list_of_labels = group["label"].tolist()
        sentences.append((list_of_words, list_of_labels))

    return sentences

# Split sentences with a 70/15/15 split, 70% train, 15% validate, and 15% test
def split_sentences(sentences):
    train, rest = train_test_split(sentences, test_size = 0.30, random_state=RANDOM_STATE)
    val, test = train_test_split(rest, test_size = 0.50, random_state=RANDOM_STATE)

    return train, val, test

# Extract the label list from all sentences and combines the tags into a flat list
def flatten_labels(sentences):
    flat_list = []

    for _, labels in sentences:
        for label in labels:
            flat_list.append(label)

    return flat_list

# Create a baseline by tagging every token as FIL (since its the most common)
# It's effectively a dumb benchmark since we just tag everything as FIL.
def baseline(val):
    y_true = flatten_labels(val) # annonated labels
    y_pred = ["FIL"] * len(y_true) # model's predicted labels
    
    print("Baseline Accuracy: ", accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred, zero_division=0))

# Test
if __name__ == "__main__":
    sentences = load_sentences(DATA_PATH)

    # print total sentences
    print(f"\nTotal sentences loaded: {len(sentences)}")

    # print first 3 sentences to inspect the structure
    for i, (tokens, labels) in enumerate(sentences[:3]):
        print(f"\nSentence {i + 1}:")
        print("Tokens: ", tokens)
        print("Labels: ", labels)

    #print and see how the sentences are split (70/15/15)
    train, val, test = split_sentences(sentences)
    print(f"Train: {len(train)}, Validate: {len(val)}, Test: {len(test)}")

    baseline(val)
    
