from flask import Flask, request, jsonify
from models import save_meeting, get_meetings_by_date_range
from meeting_analysis import process_meeting
from datetime import datetime

app = Flask(__name__)

@app.route('/analyze_meeting', methods=['POST'])
def analyze_meeting():
    data = request.get_json()

    meeting_id = data['meeting_id']
    agenda = data['agenda']
    transcripts = data['transcripts']

    result = process_meeting(meeting_id, agenda, transcripts)

    # Add date for retrieval
    result['date'] = datetime.now().isoformat()

    # Save to TinyDB
    save_meeting(result)

    return jsonify(result)

@app.route('/meeting_summary', methods=['POST'])
def meeting_summary():
    data = request.get_json()

    meeting_id = data['meeting_id']
    start_date = datetime.strptime(data['start_date'], "%Y-%m-%d")
    end_date = datetime.strptime(data['end_date'], "%Y-%m-%d")

    meetings = get_meetings_by_date_range(start_date, end_date, meeting_id)

    combined_summary = {
        "meeting_id": meeting_id,
        "total_meetings": len(meetings),
        "topics": [],
        "off_topics": []
    }

    for meeting in meetings:
        for topic in meeting.get('topics', []):
            combined_summary["topics"].append({
                "topic_name": topic["topic_name"],
                "summary": topic["summary"],
                "sentiment": topic["sentiment"]
            })
        combined_summary["off_topics"].append({
            "off_topic_summary": meeting.get('off_topic_summary', ""),
            "off_topic_sentiment": meeting.get('off_topic_sentiment', 0.0)
        })

    return jsonify(combined_summary)

if __name__ == "__main__":
    app.run(debug=True)
