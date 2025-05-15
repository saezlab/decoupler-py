import pandas as pd


def _infer_dtypes(
    df: pd.DataFrame
) -> pd.DataFrame:
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
            continue
        except ValueError:
            pass
        if df[col].dtype == 'string':
            df[col] = df[col].astype(str)
        lowered = df[col].str.lower()
        if lowered.isin(["true", "false"]).all():
            df[col] = lowered == "true"
            continue
    return df
