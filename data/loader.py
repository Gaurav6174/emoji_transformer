import pandas as pd
from datasets import load_dataset

def load_tweet_eval() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Laoding dataset..." )
    dataset = load_dataset("cardiffnlp/tweet_eval", "emoji")

    train_df = _to_dataframe(dataset["train"])
    val_df   = _to_dataframe(dataset["validation"])
    test_df  = _to_dataframe(dataset["test"])
    
    
    #to see the number of Data
    print(f"  Train : {len(train_df):,} samples")
    print(f"  Val   : {len(val_df):,} samples")
    print(f"  Test  : {len(test_df):,} samples")
    print("Done.\n")
    
    return train_df, val_df, test_df

def _to_dataframe(split) -> pd.DataFrame:
    #Converts a HuggingFace dataset split into a pandas DataFrame.
    return pd.DataFrame({
        "text": split["text"],
        "label": split["label"]
    })
    
    