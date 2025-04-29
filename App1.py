from flask import Flask, request, jsonify
from meeting_analysis import process_meeting, get_recent_meetings

app = Flask(__name__)

@app.route('/analyze_meeting', methods=['POST'])
def analyze_meeting():
    data = request.get_json()
    meeting_id = data['meeting_id']
    agenda = data['agenda']
    transcripts = data['transcripts']
    meeting_name = data['meeting_name']
    meeting_date = data['meeting_date']

    result = process_meeting(meeting_id, agenda, transcripts, meeting_name, meeting_date)
    return jsonify(result)

@app.route('/get_weekly_drift', methods=['POST'])
def get_weekly_drift():
    data = request.get_json()
    meeting_name = data['meeting_name']

    recent_meetings = get_recent_meetings(meeting_name)
    return jsonify(recent_meetings)

if __name__ == '__main__':
    app.run(debug=True)
