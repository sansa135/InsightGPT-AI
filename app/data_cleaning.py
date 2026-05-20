import pandas as pd


def clean_data(df):

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Numeric columns
    numeric_cols = df.select_dtypes(
        include='number'
    ).columns

    # Fill numeric missing values
    for col in numeric_cols:

        df[col].fillna(
            df[col].mean(),
            inplace=True
        )

    # Object columns
    object_cols = df.select_dtypes(
        include='object'
    ).columns

    # Fill text missing values
    for col in object_cols:

        df[col].fillna(
            "Unknown",
            inplace=True
        )

    return df