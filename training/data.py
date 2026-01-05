import pandas as pd


def load_dataset(path="data/problems_data.jsonl"):
    

    # -----------------------------
    # Load dataset
    # -----------------------------
    df = pd.read_json(path, lines=True)

    # -----------------------------
    # Keep only required columns
    # -----------------------------
    df = df[
        [
            "description",
            "input_description",
            "output_description",
            "problem_class",
            "problem_score"
        ]
    ]

    # -----------------------------
    # Drop missing values
    # -----------------------------
    df = df.dropna(
        subset=[
            "description",
            "input_description",
            "output_description",
            "problem_class",
            "problem_score"
        ]
    )

    # -----------------------------
    # Combine text fields
    # -----------------------------
    df["text"] = (
        df["description"] + " " +
        df["input_description"] + " " +
        df["output_description"]
    )

    # -----------------------------
    # Normalize text (for duplicates)
    # -----------------------------
    df["text_norm"] = (
        df["text"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # -----------------------------
    # Remove duplicate problems
    # -----------------------------
    before = len(df)
    df = df.drop_duplicates(subset="text_norm")
    after = len(df)

    print(f"[Data] Removed {before - after} duplicate problems")

    # -----------------------------
    # Clean labels
    # -----------------------------
    df["problem_class"] = (
        df["problem_class"]
        .str.strip()
        .str.lower()
    )

    # -----------------------------
    # Final cleanup
    # -----------------------------
    df = df.drop(columns=["text_norm"])
    df = df.reset_index(drop=True)

    return df
