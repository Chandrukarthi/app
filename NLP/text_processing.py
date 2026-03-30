import spacy

nlp = spacy.load("en_core_web_sm")

text = """
Microsoft is expanding its cloud services in Hyderabad, India.
CEO Satya Nadella announced the new development center in 2025 during an event in Seattle.
"""

doc = nlp(text)

print("Sentences:")
for sent in doc.sents:
    print("-", sent.text)

print("\n------------------")

print("Keywords:")
keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]
print(keywords)

print("\n------------------")

print("Named Entities:")
for ent in doc.ents:
    print(ent.text, "-", ent.label_)