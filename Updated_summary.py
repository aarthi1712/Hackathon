import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from datetime import timedelta
import gensim
from gensim.models import Word2Vec
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')

def summarize_text(text):
    sentences = sent_tokenize(text)
    if len(sentences) <= 2:
        return text  # Small text, return as is

    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    tfidf_matrix = vectorizer.fit_transform(sentences)

    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix).flatten()
    sorted_indices = cosine_similarities.argsort()[::-1]
    
    summary_sentences = [sentences[i] for i in sorted_indices[:3]]
    return ' '.join(summary_sentences)

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity

    if sentiment_score > 0.1:
        return "Positive"
    elif sentiment_score < -0.1:
        return "Negative"
    else:
        return "Neutral"

def calculate_actual_duration(text):
    words = word_tokenize(text)
    avg_wpm = 150  # average words spoken per minute
    minutes = len(words) / avg_wpm
    return timedelta(minutes=minutes)

def train_word2vec_model(corpus):
    sentences = [word_tokenize(doc.lower()) for doc in corpus]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

def get_topic_similarity(model, topic_name, transcript):
    topic_words = word_tokenize(topic_name.lower())
    transcript_words = word_tokenize(transcript.lower())
    
    # Compute similarity of each word in transcript to the topic name using Word2Vec
    similarity_scores = []
    for word in transcript_words:
        if word in model.wv:
            word_vector = model.wv[word]
            topic_similarity = np.mean([model.wv.similarity(word, topic_word) for topic_word in topic_words if topic_word in model.wv])
            similarity_scores.append(topic_similarity)
    if len(similarity_scores) == 0:
        return 0
    return np.mean(similarity_scores)

def process_meeting(meeting_id, agenda, transcripts):
    topics = {item['topic_name']: {"duration": item['duration'], "content": ""} for item in agenda}

    # Collect all transcript content
    corpus = [entry['content'] for entry in transcripts]
    
    # Train Word2Vec model on the collected corpus
    word2vec_model = train_word2vec_model(corpus)

    off_topic_transcripts = []

    for entry in transcripts:
        matched = False
        max_similarity = -1
        matched_topic = None
        for topic in topics:
            similarity = get_topic_similarity(word2vec_model, topic, entry['content'])
            if similarity > max_similarity:
                max_similarity = similarity
                matched_topic = topic

        if matched_topic:
            topics[matched_topic]['content'] += " " + entry['content']
        else:
            off_topic_transcripts.append(entry['content'])

    results = []
    total_scheduled = timedelta()
    total_actual = timedelta()

    for topic_name, info in topics.items():
        scheduled_duration = timedelta(minutes=info['duration'])
        total_scheduled += scheduled_duration

        content = info['content']
        if content.strip():
            summary = summarize_text(content)
            sentiment = analyze_sentiment(summary)
            actual_duration = calculate_actual_duration(content)
        else:
            summary = "No discussion found."
            sentiment = "Neutral"  # Default to neutral if no content
            actual_duration = timedelta(minutes=0)

        total_actual += actual_duration

        results.append({
            "topic_name": topic_name,
            "scheduled_duration": str(scheduled_duration),
            "actual_duration": str(actual_duration),
            "summary": summary,
            "sentiment": sentiment
        })

    off_topic_text = " ".join(off_topic_transcripts)
    if off_topic_text.strip():
        off_summary = summarize_text(off_topic_text)
        off_sentiment = analyze_sentiment(off_summary)
    else:
        off_summary = ""
        off_sentiment = "Neutral"

    status = "In-Time" if total_scheduled >= total_actual else "Exceeded"

    return {
        "meeting_id": meeting_id,
        "topics": results,
        "off_topic_summary": off_summary,
        "off_topic_sentiment": off_sentiment,
        "total_scheduled_duration": str(total_scheduled),
        "total_actual_duration": str(total_actual),
        "status": status
    }
