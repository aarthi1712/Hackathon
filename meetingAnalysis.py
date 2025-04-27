import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from datetime import timedelta

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
    return blob.sentiment.polarity

def calculate_actual_duration(text):
    words = word_tokenize(text)
    avg_wpm = 150  # average words spoken per minute
    minutes = len(words) / avg_wpm
    return timedelta(minutes=minutes)

def process_meeting(meeting_id, agenda, transcripts):
    topics = {item['topic_name']: {"duration": item['duration'], "content": ""} for item in agenda}

    off_topic_transcripts = []

    for entry in transcripts:
        matched = False
        for topic in topics:
            if topic.lower() in entry['content'].lower():
                topics[topic]['content'] += " " + entry['content']
                matched = True
                break
        if not matched:
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
            sentiment = 0.0
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
        off_sentiment = 0.0

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
