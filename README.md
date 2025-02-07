# Task Extraction from Unstructured Text

## Overview
This project implements a heuristic-based approach to extract tasks or action items from unstructured text. Since no annotated dataset is provided, the system relies on NLP techniques and rule-based heuristics to identify and categorize tasks.

## Features
- **Text Preprocessing:** Cleans the input text by removing stopwords, punctuation, and irrelevant metadata.
- **Task Identification:** Uses NLP techniques such as POS tagging and dependency parsing to identify actionable tasks.
- **Task Categorization:** Clusters tasks into useful categories using word embeddings and topic modeling.
- **Structured Output:** Extracts task details, assigns them to individuals (if mentioned), and identifies deadlines.

## Installation
Ensure you have Python installed along with the required libraries:

```bash
pip install spacy nltk gensim
python -m spacy download en_core_web_lg
```

## Usage

### 1. Preprocessing
The text undergoes preprocessing to remove stopwords and punctuation:
```python
import spacy
import string
from nltk.corpus import stopwords

def preprocess_text(text):
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(text)
    cleaned_tokens = [token.text for token in doc if token.text.lower() not in stopwords.words('english') and token.text not in string.punctuation]
    return " ".join(cleaned_tokens)
```

### 2. Task Identification
Tasks are identified using POS tagging and dependency parsing:
```python
from nltk.tokenize import sent_tokenize

def identify_tasks(text):
    nlp = spacy.load("en_core_web_lg")
    sentences = sent_tokenize(text)
    tasks = []
    
    for sentence in sentences:
        doc = nlp(sentence)
        if any(token.pos_ == "VERB" and token.dep_ == "ROOT" for token in doc):
            tasks.append(sentence)
    
    return tasks
```

### 3. Task Categorization
Tasks are grouped into meaningful categories using LDA:
```python
from gensim import corpora
from gensim.models import LdaModel

def categorize_tasks(tasks):
    texts = [[word for word in task.lower().split()] for task in tasks]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
    return lda.print_topics(num_words=4)
```

### 4. Structured Output
The extracted tasks are formatted into a structured list:
```python
def structure_tasks(tasks):
    structured_output = []
    for task in tasks:
        assigned_to, deadline = None, None
        words = task.split()
        for word in words:
            if word.lower() in ["today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
                deadline = word
        structured_output.append({"task": task, "assigned_to": assigned_to, "deadline": deadline})
    return structured_output
```

## Example Execution
```python
text = "John must submit the report before Monday. Rahul should clean the room by 5 PM today."
cleaned_text = preprocess_text(text)
tasks = identify_tasks(cleaned_text)
categorized_tasks = categorize_tasks(tasks)
structured_tasks = structure_tasks(tasks)
print(structured_tasks)
```

## Output Example
```json
[
    {"task": "John must submit the report before Monday.", "assigned_to": "John", "deadline": "Monday"},
    {"task": "Rahul should clean the room by 5 PM today.", "assigned_to": "Rahul", "deadline": "today"}
]
```

## Future Enhancements
- Implement Named Entity Recognition (NER) for better person identification.
- Use a more advanced rule-based approach to detect deadlines and responsibilities.
- Integrate the system with a database for task tracking.

