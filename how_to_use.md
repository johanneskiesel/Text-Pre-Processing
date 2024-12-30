## Environment Setup


1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```


## How to Use

## Import the Functions

```python
from pre_processing import *
```

## Function Details

### 1. `choose_spacy_model(language)`

This function simplifies choosing the appropriate spaCy language model based on the input language code.

```python
model = choose_spacy_model('en')  # Choose spaCy English model
```

### 2. `tokenize_text(text)`

Tokenizes the input text using spaCy.
This function provides a list of tokens.

```python
text = "This is a sample text."
tokens = tokenize_text(text)
print(tokens)
# Output: ['This', 'is', 'a', 'sample', 'text', '.']
```

### 3. `remove_stopwords(text)`

Removes stopwords from the input text using spaCy, returning a list of tokens without stopwords.

```python
text = "Remove these stopwords from the text."
filtered_tokens = remove_stopwords(text)
print(filtered_tokens)
# Output: ['Remove', 'stopwords', 'text', '.']
```

### 4. `lemmatize_text(text)`

Lemmatizes the input text using spaCy, providing a list of lemmatized tokens.

```python
text = "Lemmatize this text."
lemmatized_tokens = lemmatize_text(text)
print(lemmatized_tokens)
# Output: ['lemmatize', 'this', 'text', '.']
```


### 5. `split_sentences(text)`

Split the input text in its sentences using spaCy.

```python
text = "This is a sample text. Remove these stopwords from the text. Lemmatize this text."
list_of_sentences = split_sentences(text)
print(list_of_sentences)
# Output: ['This is a sample text.', 'Remove these stopwords from the text.', 'Lemmatize this text.']
```


## Input/Output Specification
### Input Dataset/Corpus (if applicable): 
This method operates on any text corpus, make sure to load a spacy model that fits your language.

## Repo Structure
The repository contains one python script with text pre-processing functions.
Furthermore, it contains one notebook file which displays test usages of th available functions.
