import sqlite3
import json

def init_db():
    conn = sqlite3.connect('lti_demo.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS lti_sessions
                 (session_id TEXT PRIMARY KEY, 
                  user_id TEXT, 
                  role TEXT, 
                  course_id TEXT,
                  assignment_id TEXT,
                  data TEXT)''')
    conn.commit()
    conn.close()

def save_session(session_id, user_id, role, course_id, assignment_id, data):
    conn = sqlite3.connect('lti_demo.db')
    c = conn.cursor()
    c.execute('INSERT OR REPLACE INTO lti_sessions VALUES (?,?,?,?,?,?)',
              (session_id, user_id, role, course_id, assignment_id, json.dumps(data)))
    conn.commit()
    conn.close()

def get_session(session_id):
    conn = sqlite3.connect('lti_demo.db')
    c = conn.cursor()
    c.execute('SELECT * FROM lti_sessions WHERE session_id=?', (session_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            'session_id': row[0],
            'user_id': row[1],
            'role': row[2],
            'course_id': row[3],
            'assignment_id': row[4],
            'data': json.loads(row[5]) if row[5] else {}
        }
    return None