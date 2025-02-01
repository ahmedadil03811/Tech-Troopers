import pandas as pd
import spacy

# Load trained model
nlp = spacy.load("./model/model-best") 


df = pd.read_csv("data/training_data/extracted_texts.csv") 

if "PDF Path" in df.columns:
        data = df.drop(columns=["PDF Path"])



def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


df["entities"] = df["Text"].apply(extract_entities)

df.to_csv("annotated_text.csv", index=False)

# First Few Rows
print(df.head())