# Tutorial On Text-Pre-Processing
## Learning Objective 
The objective of this tutorial is to provide a comprehensive introduction to text pre-processing techniques using Python and spaCy. You will learn how to implement essential text cleaning and preparation functions including tokenization, stopword removal, lemmatization, and sentence splitting. By the end of this tutorial, you will be able to build a complete text pre-processing pipeline suitable for natural language processing tasks and social science research applications.

## How This Tutorial Benefits Social Scientists

1. **Survey Response Analysis:** Social scientists can preprocess open-ended survey responses to clean and standardize text, making it easier to identify common themes and perform quantitative text analysis.

2. **Interview Transcript Preparation:** By removing stopwords and lemmatizing words in interview transcripts, researchers can focus on the core content, facilitating qualitative coding and thematic analysis.

3. **Social Media Data Mining:** The tutorial's techniques help in cleaning and structuring social media posts, enabling sentiment analysis, topic modeling, and trend detection in large-scale digital communication datasets.



## Environment Setup Guide

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to use

To use the provided functions, simply import them into your Python script or Jupyter notebook as shown above. You can chain the functions together for a complete text pre-processing pipeline, for example:

```python
text = "Text pre-processing is essential for NLP tasks."
tokens = tokenize_text(text)
filtered_tokens = remove_stopwords(' '.join(tokens))
lemmatized_tokens = lemmatize_text(' '.join(filtered_tokens))
print(lemmatized_tokens)
```

This modular approach allows you to customize your workflow based on your project's requirements.

## Social Science Use Case

Text pre-processing is a foundational step in social science research involving textual data. For example, researchers analyzing survey responses, interview transcripts, or social media posts can use these functions to:

- Clean and standardize text data for qualitative analysis.
- Tokenize and lemmatize responses to identify common themes or topics.
- Remove stopwords to focus on meaningful content.
- Split large documents into sentences for sentiment or discourse analysis.

By streamlining text preparation, these tools enable social scientists to extract insights from unstructured data more efficiently and accurately.
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
