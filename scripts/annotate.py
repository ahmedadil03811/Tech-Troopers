import spacy
from spacy.tokens import DocBin
from Loadxl import wiki_data
from spacy.symbols import ORTH
# Load spaCy's large eng model
nlp = spacy.load("en_core_web_lg")

tokenizer = nlp.tokenizer
tokenizer.add_special_case("€1,530", [{ORTH: "€1,530"}])  
tokenizer.add_special_case("2023-09-30", [{ORTH: "2023-09-30"}])


patterns = [
    {"label": "CASE", "pattern": [{"TEXT": {"REGEX": r"Case\s+\d+/\d+"}}]},
    {"label": "ORG", "pattern": [{"TEXT": {"REGEX": r"Vendor\s+\d+"}}]},
    {"label": "DATE", "pattern": [{"TEXT": {"REGEX": r"\d{1,2}(st|nd|rd|th)?\s+\w+\s+\d{4}"}}]},
    {"label": "MONEY", "pattern": [{"TEXT": {"REGEX": r"€\d+([,.]\d+)?"}}]},
]


ruler = nlp.add_pipe("entity_ruler")
ruler.add_patterns(patterns)

doc_bin = DocBin()
for text in wiki_data["Text"]:
    doc = nlp(text)
    doc_bin.add(doc)

doc_bin.to_disk("data/preannotated.spacy")

doc_bin = DocBin().from_disk("data/preannotated.spacy")
print(f"Number of training examples: {len(doc_bin)}")