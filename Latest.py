import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.summarization import summarize
from datetime import timedelta
from tinydb import TinyDB
import uuid

nltk.download('punkt')

db = TinyDB('meetings.json')

FILLER_WORDS = ['um', 'uh', 'hmm', 'yeah', 'ya', 'you know', 'like', 'okay', 'ok', 'huh', 'right', 'actually']

def preprocess_text(text):
    text = text.lower()
    for filler in FILLER_WORDS:
        pattern = r'\b' + re.escape(filler) + r'\b'
        text = re.sub(pattern, '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    print(f"[Preprocessed Text]: {text}")
    return text

def calculate_actual_duration(text):
    words = word_tokenize(text)
    avg_wpm = 150
    minutes = len(words) / avg_wpm
    return timedelta(minutes=minutes)

def summarize_text(text):
    try:
        # Gensim's summarize requires text with enough sentences
        if len(sent_tokenize(text)) < 3:
            return text
        return summarize(text, word_count=50)
    except ValueError:
        return text

def process_meeting(meeting_id, agenda, transcripts, meeting_name, meeting_date):
    topics = {item['topic_name']: {"duration": item['duration'], "content": ""} for item in agenda}
    off_topic_transcripts = []

    for entry in transcripts:
        preprocessed_content = preprocess_text(entry['content'])
        matched = False
        for topic in topics:
            topic_words = set(word_tokenize(topic.lower()))
            content_words = set(word_tokenize(preprocessed_content))
            common_words = topic_words.intersection(content_words)
            similarity_score = len(common_words) / max(len(topic_words), 1)

            print(f"[Topic Match]: Topic '{topic}' with Preprocessed Content => Similarity: {similarity_score}")

            if similarity_score > 0.2:
                topics[topic]['content'] += " " + preprocessed_content
                matched = True
                break
        if not matched:
            off_topic_transcripts.append(preprocessed_content)

    results = []
    total_scheduled = timedelta()
    total_actual = timedelta()

    for topic_name, info in topics.items():
        scheduled_duration = timedelta(minutes=info['duration'])
        total_scheduled += scheduled_duration

        content = info['content']
        if content.strip():
            summary = summarize_text(content)
            actual_duration = calculate_actual_duration(content)
        else:
            summary = "No discussion found."
            actual_duration = timedelta(minutes=0)

        total_actual += actual_duration

        results.append({
            "topic_name": topic_name,
            "scheduled_duration": str(scheduled_duration),
            "actual_duration": str(actual_duration),
            "summary": summary
        })

    off_topic_text = " ".join(off_topic_transcripts)
    if off_topic_text.strip():
        off_summary = summarize_text(off_topic_text)
    else:
        off_summary = ""

    status = "In-Time" if total_scheduled >= total_actual else "Exceeded"

    meeting_entry = {
        "id": str(uuid.uuid4()),
        "meeting_id": meeting_id,
        "meeting_name": meeting_name,
        "meeting_date": meeting_date,
        "topics": results,
        "off_topic_summary": off_summary,
        "total_scheduled_duration": str(total_scheduled),
        "total_actual_duration": str(total_actual),
        "status": status
    }

    db.insert(meeting_entry)
    return meeting_entry

def get_recent_meetings(meeting_name):
    meetings = db.search(lambda m: m['meeting_name'] == meeting_name)
    # Sort by meeting_date descending
    meetings_sorted = sorted(meetings, key=lambda x: x['meeting_date'], reverse=True)
    return meetings_sorted[:3]
