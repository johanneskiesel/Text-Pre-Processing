import spacy
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

def choose_spacy_model(language):
    """
    Loads a spaCy language model for the specified language. If the model is not found,
    attempts to download it and then load it.
    Args:
        language (str): The language code (e.g., 'en', 'fr', 'de') for which to load the spaCy model.
    Returns:
        spacy.language.Language: The loaded spaCy language model.
    Raises:
        ValueError: If the specified language is not supported or the model cannot be downloaded.
    """

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

def tokenize(nlp, text):
    """
    Tokenize the input text using spaCy.

    Parameters:
    - text (str): Input text.

    Returns:
    - list: List of tokens.
    """
    document = nlp(text)
    tokens = [token.text for token in document]
    return tokens

def tokenize_without_stopwords(nlp, text):
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

def tokenize_and_lemmatize(nlp, text):
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

def split_sentences(nlp, text):
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

def extract_named_entities(nlp, text):
    """
    Extract named entities (people, organizations, locations, etc.) from text.
    
    This function identifies and extracts important entities mentioned in your text,
    such as person names, company names, geographical locations, dates, and monetary values.
    This is particularly useful for analyzing political speeches, news articles, or 
    interview transcripts where you want to identify key actors and locations.

    Parameters:
    - text (str): The input text to analyze (e.g., "Barack Obama visited Paris in 2015")

    Returns:
    - list: List of dictionaries, each containing:
            - 'text': the entity text (e.g., "Barack Obama")
            - 'label': the entity type (e.g., "PERSON", "GPE" for geopolitical entity)
            - 'description': human-readable description of the entity type
    
    Example:
    Input: "Apple Inc. was founded by Steve Jobs in California."
    Output: [{'text': 'Apple Inc.', 'label': 'ORG', 'description': 'Organization'},
                {'text': 'Steve Jobs', 'label': 'PERSON', 'description': 'Person'},
                {'text': 'California', 'label': 'GPE', 'description': 'Geopolitical entity'}]
    """
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_,
            'description': spacy.explain(ent.label_)
        })
    return entities

def extract_keywords_tfidf(nlp, texts, max_features=20, ngram_range=(1, 2)):
    """
    Extract the most important keywords from a collection of texts using TF-IDF analysis.
    
    TF-IDF (Term Frequency-Inverse Document Frequency) helps identify words that are
    important in individual documents but not too common across all documents.
    This is excellent for finding distinctive themes in survey responses, interview
    transcripts, or comparing different groups' language use.

    Parameters:
    - texts (list): List of text documents to analyze (e.g., survey responses)
    - max_features (int): Maximum number of top keywords to return (default: 20)
    - ngram_range (tuple): Range of n-grams to consider. (1,1) for single words,
                            (1,2) for single words and two-word phrases (default: (1,2))

    Returns:
    - list: List of tuples containing (keyword, importance_score)
            Keywords are sorted by importance (highest first)
    
    Example:
    For analyzing political speeches, this might return:
    [('economic policy', 0.45), ('healthcare reform', 0.38), ('job creation', 0.32), ...]
    
    Note: You need at least 2 documents for meaningful TF-IDF analysis.
    """
    if len(texts) < 2:
        raise ValueError("TF-IDF analysis requires at least 2 documents for comparison.")
    
    # Preprocess texts by removing stopwords and lemmatizing
    processed_texts = []
    for text in texts:
        doc = nlp(text)
        processed_text = ' '.join([token.lemma_.lower() for token in doc 
                                    if not token.is_stop and not token.is_punct 
                                    and len(token.text) > 2])
        processed_texts.append(processed_text)
    
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    
    feature_names = vectorizer.get_feature_names_out()
    mean_scores = [float(number) for number in tfidf_matrix.mean(axis=0).A1]

    keywords_scores = list(zip(feature_names, mean_scores))
    keywords_scores.sort(key=lambda x: x[1], reverse=True)
    
    return keywords_scores

def analyze_sentiment_basic(text):
    """
    Perform basic sentiment analysis to determine if text expresses positive or negative emotions.
    
    This function analyzes the emotional tone of text by looking for positive and negative
    words. While not as sophisticated as machine learning approaches, it provides a quick
    way to gauge overall sentiment in survey responses, social media posts, or interviews.
    
    Useful for: analyzing public opinion, customer feedback, political discourse, or
    any text where emotional tone matters for your research.

    Parameters:
    - text (str): The text to analyze for sentiment

    Returns:
    - dict: Dictionary containing:
            - 'sentiment': overall sentiment ('positive', 'negative', or 'neutral')
            - 'positive_words': list of positive words found
            - 'negative_words': list of negative words found
            - 'score': numerical score (positive = above 0, negative = below 0)
    
    Example:
    Input: "I love this new policy but I hate the implementation."
    Output: {'sentiment': 'neutral', 'positive_words': ['love'], 
                'negative_words': ['hate'], 'score': 0}
    """
    # Basic positive and negative word lists (you might want to expand these)
    positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                        'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'positive',
                        'benefit', 'advantage', 'success', 'improve', 'better', 'best'}
    
    negative_words = {'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'angry',
                        'sad', 'disappointed', 'frustrated', 'negative', 'problem', 'issue',
                        'difficult', 'hard', 'impossible', 'fail', 'failure', 'worse', 'worst'}
    
    doc = nlp(text.lower())
    
    found_positive = []
    found_negative = []
    
    for token in doc:
        if token.lemma_ in positive_words:
            found_positive.append(token.text)
        elif token.lemma_ in negative_words:
            found_negative.append(token.text)
    
    score = len(found_positive) - len(found_negative)
    
    if score > 0:
        sentiment = 'positive'
    elif score < 0:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return {
        'sentiment': sentiment,
        'positive_words': found_positive,
        'negative_words': found_negative,
        'score': score
    }

def get_text_statistics(text):
    """
    Calculate comprehensive statistics about a text document.
    
    This function provides detailed quantitative measures of text complexity and structure.
    These statistics are valuable for comparing different types of documents, analyzing
    readability, or understanding the linguistic characteristics of different speakers
    or writers in your research.

    Parameters:
    - text (str): The text to analyze

    Returns:
    - dict: Dictionary containing detailed statistics:
            - 'word_count': total number of words
            - 'sentence_count': total number of sentences
            - 'character_count': total characters (including spaces)
            - 'avg_words_per_sentence': average sentence length
            - 'avg_characters_per_word': average word length
            - 'unique_words': number of unique words (vocabulary richness)
            - 'lexical_diversity': ratio of unique words to total words (0-1 scale)
            - 'pos_distribution': distribution of parts of speech (nouns, verbs, etc.)
    
    Example use cases:
    - Compare complexity of political speeches across different candidates
    - Analyze linguistic development in student essays
    - Study vocabulary richness in interview responses across different demographics
    """
    doc = nlp(text)
    
    # Basic counts
    words = [token for token in doc if not token.is_space and not token.is_punct]
    sentences = list(doc.sents)
    
    word_count = len(words)
    sentence_count = len(sentences)
    character_count = len(text)
    
    # Calculate averages
    avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
    word_lengths = [len(token.text) for token in words]
    avg_characters_per_word = sum(word_lengths) / len(word_lengths) if word_lengths else 0
    
    # Vocabulary analysis
    word_texts = [token.text.lower() for token in words if token.is_alpha]
    unique_words = len(set(word_texts))
    lexical_diversity = unique_words / len(word_texts) if word_texts else 0
    
    # Parts of speech distribution
    pos_counts = {}
    for token in words:
        pos = token.pos_
        pos_counts[pos] = pos_counts.get(pos, 0) + 1
    
    # Convert to percentages
    pos_distribution = {pos: (count/word_count)*100 for pos, count in pos_counts.items()}
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'character_count': character_count,
        'avg_words_per_sentence': round(avg_words_per_sentence, 2),
        'avg_characters_per_word': round(avg_characters_per_word, 2),
        'unique_words': unique_words,
        'lexical_diversity': round(lexical_diversity, 3),
        'pos_distribution': pos_distribution
    }

def compare_texts_vocabulary(text1, text2, top_n=10):
    """
    Compare the vocabulary usage between two texts to identify similarities and differences.
    
    This function is particularly useful for comparative analysis in social science research,
    such as comparing political speeches from different parties, analyzing language differences
    between demographic groups, or studying how language use changes over time.

    Parameters:
    - text1 (str): First text for comparison
    - text2 (str): Second text for comparison  
    - top_n (int): Number of top unique words to return for each text (default: 10)

    Returns:
    - dict: Dictionary containing:
            - 'common_words': words that appear in both texts with their frequencies
            - 'unique_to_text1': words that appear only in the first text
            - 'unique_to_text2': words that appear only in the second text
            - 'similarity_score': percentage of vocabulary overlap (0-100)
    
    Example use cases:
    - Compare campaign speeches from different political candidates
    - Analyze language differences between age groups in survey responses
    - Study evolution of terminology in policy documents over time
    """
    # Process both texts
    doc1 = nlp(text1.lower())
    doc2 = nlp(text2.lower())
    
    # Extract meaningful words (no stopwords, punctuation, or short words)
    words1 = [token.lemma_ for token in doc1 if not token.is_stop and not token.is_punct 
                and token.is_alpha and len(token.text) > 2]
    words2 = [token.lemma_ for token in doc2 if not token.is_stop and not token.is_punct 
                and token.is_alpha and len(token.text) > 2]
    
    # Count word frequencies
    freq1 = Counter(words1)
    freq2 = Counter(words2)
    
    # Find common and unique words
    set1 = set(freq1.keys())
    set2 = set(freq2.keys())
    
    common_words = {}
    for word in set1.intersection(set2):
        common_words[word] = {'text1_freq': freq1[word], 'text2_freq': freq2[word]}
    
    unique_to_text1 = {word: freq1[word] for word in set1 - set2}
    unique_to_text2 = {word: freq2[word] for word in set2 - set1}
    
    # Sort by frequency and get top N
    unique_to_text1 = dict(sorted(unique_to_text1.items(), key=lambda x: x[1], reverse=True)[:top_n])
    unique_to_text2 = dict(sorted(unique_to_text2.items(), key=lambda x: x[1], reverse=True)[:top_n])
    
    # Calculate similarity score
    total_unique_words = len(set1.union(set2))
    similarity_score = (len(common_words) / total_unique_words * 100) if total_unique_words > 0 else 0
    
    return {
        'common_words': common_words,
        'unique_to_text1': unique_to_text1,
        'unique_to_text2': unique_to_text2,
        'similarity_score': round(similarity_score, 2)
    }
