import unittest
import spacy
import matplotlib.pyplot as plt
from pre_processing import (
    get_model,
    choose_spacy_model,
    tokenize_text,
    remove_stopwords,
    lemmatize_text,
    ,)

class TestTextProcessingToolkit(unittest.TestCase):
    def setUp(self):
        self.nlp = spacy.load("en_core_web_sm")

    def test_tokenize_text(self):
        text = "This is a sample text."
        tokens = tokenize_text(text)
        self.assertEqual(tokens, ['This', 'is', 'a', 'sample', 'text', '.'])

    def test_remove_stopwords(self):
        text = "Remove these stopwords from the text."
        filtered_tokens = remove_stopwords(text)
        self.assertEqual(filtered_tokens, ['Remove', 'stopwords', 'text', '.'])

    def test_lemmatize_text(self):
        text = "Lemmatize this text."
        lemmatized_tokens = lemmatize_text(text)
        self.assertEqual(lemmatized_tokens, ['lemmatize', 'this', 'text', '.'])

if __name__ == '__main__':
    unittest.main()
