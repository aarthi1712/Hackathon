import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from datetime import timedelta
from tinydb import TinyDB
import uuid

nltk.download('punkt')

db = TinyDB('meetings.json')

FILLER_WORDS = ['um', 'uh', 'hmm', 'yeah', 'ya', 'you know', 'like', 'okay', 'ok', 'huh', 'right', 'actually']

def preprocess_text(text):
    """
    Preprocess the transcript by removing filler words.
    Converts text to lowercase and removes unwanted words like "um", "uh", etc.
    """
    text = text.lower()
    for filler in FILLER_WORDS:
        pattern = r'\b' + re.escape(filler) + r'\b'
        text = re.sub(pattern, '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    print(f"[Preprocessed Text]: {text}")
    return text

def calculate_actual_duration(text):
    """
    Calculates the estimated time a speaker would take to say the given text,
    assuming an average words per minute rate.
    """
    words = word_tokenize(text)
    avg_wpm = 150  # Average words spoken per minute
    minutes = len(words) / avg_wpm
    return timedelta(minutes=minutes)

def summarize_text_sumy(text, sentence_count=3):
    """
    Summarizes the text using the TextRank algorithm from the sumy library.
    This method breaks the text into sentences, processes it, and returns a concise summary.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, sentence_count)
    return ' '.join(str(sentence) for sentence in summary)

def process_meeting(meeting_id, agenda, transcripts, meeting_name, meeting_date):
    """
    Processes a meeting, extracting topics, calculating durations, and generating summaries.
    It stores the results into a TinyDB database.
    """
    topics = {item['topic_name']: {"duration": item['duration'], "content": ""} for item in agenda}
    off_topic_transcripts = []

    # Process each transcript entry
    for entry in transcripts:
        preprocessed_content = preprocess_text(entry['content'])
        matched = False
        for topic in topics:
            topic_words = set(word_tokenize(topic.lower()))
            content_words = set(word_tokenize(preprocessed_content))
            common_words = topic_words.intersection(content_words)
            similarity_score = len(common_words) / max(len(topic_words), 1)

            print(f"[Topic Match]: Topic '{topic}' with Preprocessed Content => Similarity: {similarity_score}")

            # If similarity is greater than a threshold (e.g., 0.2), it's considered a match
            if similarity_score > 0.2:
                topics[topic]['content'] += " " + preprocessed_content
                matched = True
                break
        if not matched:
            off_topic_transcripts.append(preprocessed_content)

    # Summarize topics and calculate actual durations
    results = []
    total_scheduled = timedelta()
    total_actual = timedelta()

    for topic_name, info in topics.items():
        scheduled_duration = timedelta(minutes=info['duration'])
        total_scheduled += scheduled_duration

        content = info['content']
        if content.strip():
            summary = summarize_text_sumy(content)  # Use sumy TextRank for summary
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

    # Process off-topic content (drift)
    off_topic_text = " ".join(off_topic_transcripts)
    if off_topic_text.strip():
        off_summary = summarize_text_sumy(off_topic_text)  # Use sumy TextRank for off-topic summary
    else:
        off_summary = ""

    # Determine whether the meeting was "In-Time" or "Exceeded" based on scheduled vs. actual time
    status = "In-Time" if total_scheduled >= total_actual else "Exceeded"

    # Create a new entry for the meeting in the TinyDB database
    meeting_entry = {
        "id": str(uuid.uuid4()),  # Unique ID for each entry
        "meeting_id": meeting_id,
        "meeting_name": meeting_name,
        "meeting_date": meeting_date,
        "topics": results,
        "off_topic_summary": off_summary,
        "total_scheduled_duration": str(total_scheduled),
        "total_actual_duration": str(total_actual),
        "status": status
    }

    db.insert(meeting_entry)  # Store the result in TinyDB
    return meeting_entry

def get_recent_meetings(meeting_name):
    """
    Retrieves the most recent meetings by a given meeting name from TinyDB,
    returning the top 3 most recent meetings.
    """
    meetings = db.search(lambda m: m['meeting_name'] == meeting_name)
    # Sort meetings by the meeting date in descending order (most recent first)
    meetings_sorted = sorted(meetings, key=lambda x: x['meeting_date'], reverse=True)
    return meetings_sorted[:3]
