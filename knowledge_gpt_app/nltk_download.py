"""Download required NLTK resources."""
import nltk

RESOURCES = [
    "punkt",
    "stopwords",
    "averaged_perceptron_tagger",
    "wordnet",
    "omw-1.4",
]

for resource in RESOURCES:
    nltk.download(resource)

print("NLTK resource download completed.")
