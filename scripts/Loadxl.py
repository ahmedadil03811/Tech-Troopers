import pandas as pd
import re

def load_and_clean_data(file_path):
    # Load the Excel file
    data = pd.read_excel(file_path)

    # Drop unnecessary columns (e.g., "PDF PATH")
    if "PDF Path" in data.columns:
        data = data.drop(columns=["PDF Path"])
    
    # Clean text
    def clean_text(text):
        text = re.sub(r"[^\w\s€$/,]", "", text)  # Remove special characters
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        return text.strip()

    data["Text"] = data["Text"].apply(clean_text)
    return data

wiki_data = load_and_clean_data("data/wikileaks_parsed.xlsx")
print(wiki_data.head())