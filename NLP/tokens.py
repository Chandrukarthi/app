import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required datasets
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

text = "Google is developing advanced AI technologies in India and expanding its research in 2026."

tokens = word_tokenize(text)

print("Tokens:")
print(tokens)

print("\n------------------")

stop_words = set(stopwords.words('english'))

filtered_words = []
for word in tokens:
    if word.lower() not in stop_words:
        filtered_words.append(word)

print("After Removing Stop Words:")
print(filtered_words)