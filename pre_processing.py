import spacy
import os
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

def get_model(language):
    try:
        return spacy.load(f"{language}_core_web_sm")
    except OSError:
        print(f"Model '{language}_core_news_sm' not found. Downloading...")
        try:
            download_command = f"python -m spacy download {language}_core_web_sm"
            exit_code = os.system(download_command)
        except:
            raise ValueError(f"Language '{language}' is not supported.")
        return spacy.load(f"{language}_core_web_sm")
        
def choose_spacy_model(language):
    """
    Choose the appropriate spaCy language model based on the input language.

    Parameters:
    - language (str): Language code (e.g., 'en' for English, 'fr' for French, 'de' for German).

    Returns:
    - spaCy language model
    """
    return get_model(language)
        
def tokenize_text(text):
    """
    Tokenize the input text using spaCy.

    Parameters:
    - text (str): Input text.

    Returns:
    - list: List of tokens.
    """
    doc = nlp(text)
    tokens = [token.text for token in doc]
    return tokens

def remove_stopwords(text):
    """
    Remove stopwords from the input text using spaCy.

    Parameters:
    - text (str): Input text.

    Returns:
    - list: List of tokens without stopwords.
    """
    doc = nlp(text)
    tokens_without_stopwords = [token.text for token in doc if not token.is_stop]
    return tokens_without_stopwords

def lemmatize_text(text):
    """
    Lemmatize the input text using spaCy.

    Parameters:
    - text (str): Input text.

    Returns:
    - list: List of lemmatized tokens.
    """
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    return lemmatized_tokens

def split_sentences(text):
    """
    Split the input text in its sentences using spaCy.

    Parameters:
    - text (str): Input text.

    Returns:
    - list: List of sentences.
    """
    doc = nlp(text)
    assert doc.has_annotation("SENT_START")
    splitted_sentences = [sentence.text for sentence in doc.sents]
    return splitted_sentences

