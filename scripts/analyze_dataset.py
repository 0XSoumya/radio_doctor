import pandas as pd
import numpy as np

def analyze():
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")

    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Sample submission shape: {sample_sub.shape}")

    print("\nTrain columns:", train_df.columns.tolist())
    print("Test columns:", test_df.columns.tolist())
    print("Sample sub columns:", sample_sub.columns.tolist())

    print("\nTrain null counts:\n", train_df.isnull().sum())
    print("\nTest null counts:\n", test_df.isnull().sum())

    print("\nModalities in Train:\n", train_df['modality'].value_counts())
    print("\nModalities in Test:\n", test_df['modality'].value_counts())

    print("\nTop 5 Body Parts in Train:\n", train_df['body_part'].value_counts().head(5))
    print("\nTop 5 Body Parts in Test:\n", test_df['body_part'].value_counts().head(5))

    # Template overlap
    train_templates = set(train_df['template_content'].unique())
    test_templates = set(test_df['template_content'].unique())
    overlap = test_templates.intersection(train_templates)

    print(f"\nUnique templates in Train: {len(train_templates)}")
    print(f"Unique templates in Test: {len(test_templates)}")
    print(f"Test templates present in Train: {len(overlap)} / {len(test_templates)} ({len(overlap)/len(test_templates)*100:.1f}%)")

    test_in_train_rows = test_df['template_content'].isin(train_templates).sum()
    print(f"Test cases with exact template in Train: {test_in_train_rows} / {len(test_df)} ({test_in_train_rows/len(test_df)*100:.1f}%)")

    # Dictation lengths
    train_dict_words = train_df['dictation'].fillna('').apply(lambda s: len(s.split()))
    test_dict_words = test_df['dictation'].fillna('').apply(lambda s: len(s.split()))

    print(f"\nDictation word count train: min={train_dict_words.min()}, median={train_dict_words.median()}, mean={train_dict_words.mean():.1f}, max={train_dict_words.max()}")
    print(f"Dictation word count test: min={test_dict_words.min()}, median={test_dict_words.median()}, mean={test_dict_words.mean():.1f}, max={test_dict_words.max()}")

    # Distribution of empty/short dictations
    print(f"Train dictation <= 5 words: {(train_dict_words <= 5).sum()} ({(train_dict_words <= 5).mean()*100:.1f}%)")
    print(f"Test dictation <= 5 words: {(test_dict_words <= 5).sum()} ({(test_dict_words <= 5).mean()*100:.1f}%)")
    print(f"Train dictation > 30 words: {(train_dict_words > 30).sum()} ({(train_dict_words > 30).mean()*100:.1f}%)")
    print(f"Test dictation > 30 words: {(test_dict_words > 30).sum()} ({(test_dict_words > 30).mean()*100:.1f}%)")

if __name__ == "__main__":
    analyze()
