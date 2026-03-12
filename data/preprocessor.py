import re
import pandas as pd

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_text"] = df["text"].apply(_clean)
    return df

def _clean(text: str) -> str:
    
    #1. lowercase everything
    text = text.lower()
    
    #2. ereplace URLS with single @url token
    text = re.sub(r"http\S+|www\S+", "@url", text)
    
    #3. replace @mentions with single @user token
    text = re.sub(r"@\w+", "@user", text)
    
    #4. strip the # symbol from the word
    text = re.sub(r"#(\w+)", r"\1", text)
    
    #5. remove non-ACSII characters (we train the model from emoji model)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    
    #6.collapse multiple space into one
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


