import unittest
import spacy
import matplotlib.pyplot as plt
from pre_processing import *

# Assuming the functions from the pre_processing module are imported correctly
class TestTextProcessingToolkit(unittest.TestCase):
    """Unit tests for the Text Processing Toolkit using spaCy.
    """
    def setUp(self):
        self.nlp = spacy.load("en_core_web_sm")

    def test_tokenize_text(self):
        """Test the tokenize_text function."""
        text = "This is a sample text."
        tokens = tokenize_text(text)
        self.assertEqual(tokens, ['This', 'is', 'a', 'sample', 'text', '.'])

    def test_remove_stopwords(self):
        """Test the remove_stopwords function."""
        text = "Remove these stopwords from the text."
        filtered_tokens = remove_stopwords(text)
        self.assertEqual(filtered_tokens, ['Remove', 'stopwords', 'text', '.'])

    def test_lemmatize_text(self):
        """Test the lemmatize_text function."""
        text = "Lemmatize this text."
        lemmatized_tokens = lemmatize_text(text)
        self.assertEqual(lemmatized_tokens, ['lemmatize', 'this', 'text', '.'])
    
    def test_split_sentences(self):
        """Test the split_sentences function."""
        text = "This is the first sentence. This is the second sentence."
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertIn("This is the first sentence.", sentences)
        self.assertIn("This is the second sentence.", sentences)
    
    def test_extract_named_entities(self):
        """Test the extract_named_entities function."""
        text = "Apple Inc. is located in Cupertino, California."
        entities = extract_named_entities(text)
        self.assertIsInstance(entities, list)
        self.assertGreater(len(entities), 0)
    
    def test_extract_keywords_tfidf(self):
        """Test the extract_keywords_tfidf function."""
        texts = ["This is a sample document.", "This is another document.", "Sample text here."]
        keywords = extract_keywords_tfidf(texts, max_features=10)
        self.assertIsInstance(keywords, list)
        self.assertLessEqual(len(keywords), 10)
    
    def test_analyze_sentiment_basic(self):
        """Test the analyze_sentiment_basic function."""
        positive_text = "This is a wonderful day!"
        negative_text = "This is terrible."
        neutral_text = "This is a statement."
        
        pos_sentiment = analyze_sentiment_basic(positive_text)
        neg_sentiment = analyze_sentiment_basic(negative_text)
        neutral_sentiment = analyze_sentiment_basic(neutral_text)
        
        self.assertIn(pos_sentiment['sentiment'], ['positive', 'negative', 'neutral'])
        self.assertIn(neg_sentiment['sentiment'], ['positive', 'negative', 'neutral'])
        self.assertIn(neutral_sentiment['sentiment'], ['positive', 'negative', 'neutral'])

    def test_get_text_statistics(self):
        """Test the get_text_statistics function."""
        text = "This is a sample text with multiple words."
        stats = get_text_statistics(text)
        self.assertIsInstance(stats, dict)
        self.assertIn('word_count', stats)
        self.assertIn('character_count', stats)
        self.assertIn('sentence_count', stats)
        self.assertGreater(stats['word_count'], 0)

    
    def test_compare_texts_vocabulary(self):
        """Test the compare_texts_vocabulary function."""
        text1 = "This is the first text sample."
        text2 = "This is the second text example."
        comparison = compare_texts_vocabulary(text1, text2)
        self.assertIsInstance(comparison, dict)
        self.assertIn('common_words', comparison)
        self.assertIn('similarity_score', comparison)
        self.assertGreaterEqual(comparison['similarity_score'], 0)
        self.assertLessEqual(comparison['similarity_score'], 100)

if __name__ == '__main__':
    # this function is used to run the tests when the script is executed
    # it will automatically discover and run all the test cases defined in the class
    unittest.main()
