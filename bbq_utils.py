import json
import pandas as pd
from datasets import Dataset, Features, Value

def load_dataset_splits(dataset_name, bbq_category):
    """
    Centralized data loading.
    """
    data_dir = 'data'
    dataset_name = dataset_name.lower()
    
    if "bbq" in dataset_name:
        path = f"{data_dir}/bbq/combined.jsonl"
        data = load_bbq(path, bbq_category)
        full_df = data if isinstance(data, pd.DataFrame) else data.to_pandas()
        full_df = full_df.sample(frac=1)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return full_df

def format_bbq_input(row):
    """
    Formats the BBQ context, question, and answers into a single string.
    """
    return (
        f"The following is a multiple-choice question. Select the best option (A, B, or C).\n\n"
        f"Context: {row['context']}\n"
        f"Question: {row['question']}\n\n"
        f"(A) {row['ans0']}\n"
        f"(B) {row['ans1']}\n"
        f"(C) {row['ans2']}\n\n"
        f"Correct answer is: (" 
    )

def bbq_df_to_hf(df):
    df = df.copy()
    df['text'] = df.apply(format_bbq_input, axis=1)
    df['label'] = df['label'].astype(int)

    # HF Dataset doesn't accept native dicts, so answer_info / additional_metadata
    # need to be JSON-encoded strings before conversion.
    if 'answer_info' in df.columns:
        df['answer_info'] = df['answer_info'].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else (x if isinstance(x, str) else '{}')
        )

    if 'additional_metadata' in df.columns:
        df['additional_metadata'] = df['additional_metadata'].apply(
            lambda x: json.dumps(x) if isinstance(x, dict) else (x if isinstance(x, str) else '{}')
        )

    features_dict = {
        "text":               Value("string"),
        "label":              Value("int64"),
        "example_id":         Value("int64"),
        "question_index":     Value("string"),
        "category":           Value("string"),
        "context_condition":  Value("string"),
        "question_polarity":  Value("string"),
        "context":            Value("string"),
        "question":           Value("string"),
        "ans0":               Value("string"),
        "ans1":               Value("string"),
        "ans2":               Value("string"),
        "answer_info":        Value("string"),   # JSON-encoded string
        "additional_metadata": Value("string"),  # JSON-encoded string
        # Optional fields present in some combined.jsonl builds
        "protected_group":    Value("string"),
        "secondary_group":    Value("string"),
        "target_role":        Value("string"),
        "ans0_role":          Value("string"),
        "ans1_role":          Value("string"),
        "ans2_role":          Value("string"),
        "ans0_demo":          Value("string"),
        "ans1_demo":          Value("string"),
        "ans2_demo":          Value("string"),
    }

    cols_to_keep = [k for k in features_dict.keys() if k in df.columns]
    df = df[cols_to_keep]

    return Dataset.from_pandas(
        df,
        features=Features({k: features_dict[k] for k in cols_to_keep}),
        preserve_index=False,
    )


def load_bbq(path, category=None, return_df=False):
    """
    Load BBQ rows from a combined JSONL file, optionally filtered by category.
    """
    line_dicts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)

            if category is not None and data.get("category") != category:
                continue

            line_dicts.append(data)

    df = pd.DataFrame(line_dicts)
    return df if return_df else bbq_df_to_hf(df)