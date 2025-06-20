# Text Preprocessing

## Description
The method performs necessary text preprocessing functions that are necessary before using the data for a downstream task. The method offer multiple preprocessing functions i.e., text tokenization, text lemmatization, remove stopwords, and document documents into sentences 

## Use Cases

1. **Survey Response Analysis:** Social scientists can preprocess open-ended survey responses to clean and standardize text, making it easier to identify common themes and perform quantitative text analysis.

2. **Interview Transcript Preparation:** By removing stopwords and lemmatizing words in interview transcripts, researchers can focus on the core content, facilitating qualitative coding and thematic analysis.

3. **Social Media Data Mining:** The Method's techniques help in cleaning and structuring social media posts, enabling sentiment analysis, topic modeling, and trend detection in large-scale digital communication datasets.

## Input Data
This method operates on any text corpus of textual data. Please make sure to load a spacy model that fits your language.

## Output Data
Preprocessed textual data

## Hardware Requirements
The method runs on a cheap virtual machine provided by cloud computing company (2 x86 CPU core, 4 GB RAM, 40GB HDD).

## Environment Setup

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

Import the functions
```python
from pre_processing import *
```

## Technical Details

1. `choose_spacy_model(language)`

This function simplifies choosing the appropriate spaCy language model based on the input language code.

```python
model = choose_spacy_model('en')  # Choose spaCy English model
```

2. `tokenize_text(text)`

Tokenizes the input text using spaCy.
This function provides a list of tokens.

```python
text = "This is a sample text."
tokens = tokenize_text(text)
print(tokens)
# Output: ['This', 'is', 'a', 'sample', 'text', '.']
```

3. `remove_stopwords(text)`

Removes stopwords from the input text using spaCy, returning a list of tokens without stopwords.

```python
text = "Remove these stopwords from the text."
filtered_tokens = remove_stopwords(text)
print(filtered_tokens)
# Output: ['Remove', 'stopwords', 'text', '.']
```

4. `lemmatize_text(text)`

Lemmatizes the input text using spaCy, providing a list of lemmatized tokens.

```python
text = "Lemmatize this text."
lemmatized_tokens = lemmatize_text(text)
print(lemmatized_tokens)
# Output: ['lemmatize', 'this', 'text', '.']
```


5. `split_sentences(text)`

Split the input text in its sentences using spaCy.

```python
text = "This is a sample text. Remove these stopwords from the text. Lemmatize this text."
list_of_sentences = split_sentences(text)
print(list_of_sentences)
# Output: ['This is a sample text.', 'Remove these stopwords from the text.', 'Lemmatize this text.']
```

This Method has introduced essential text pre-processing techniques using Python and spaCy, providing practical tools for cleaning, tokenizing, lemmatizing, splitting text data and much more. By applying these methods, social scientists and researchers can efficiently prepare textual datasets for analysis, ensuring more accurate and meaningful results in their studies. The modular functions and examples provided serve as a foundation for building robust NLP pipelines tailored to diverse research needs.

## Contact Details
[Stephan.Linzbach@gesis.org](mailto:Stephan.Linzbach@gesis.org)
