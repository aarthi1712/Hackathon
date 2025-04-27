from tinydb import TinyDB, Query
from datetime import datetime

# Initialize DB
db = TinyDB('meetings.json')

def save_meeting(meeting_data):
    db.insert(meeting_data)

def get_meetings_by_date_range(start_date, end_date, meeting_id):
    Meeting = Query()
    results = db.search(
        (Meeting.meeting_id == meeting_id) &
        (Meeting.date.test(lambda d: start_date <= datetime.fromisoformat(d) <= end_date))
    )
    return results
