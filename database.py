import sqlite3
import json
import os

# Database file path
DB_PATH = os.environ.get('DB_PATH', 'lti_demo.db')

def init_db():
    """Initialize the database with required tables"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS lti_sessions
                     (session_id TEXT PRIMARY KEY, 
                      user_id TEXT, 
                      role TEXT, 
                      course_id TEXT,
                      assignment_id TEXT,
                      data TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization error: {str(e)}")
        raise

def save_session(session_id, user_id, role, course_id, assignment_id, data):
    """Save LTI session to database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('INSERT OR REPLACE INTO lti_sessions (session_id, user_id, role, course_id, assignment_id, data) VALUES (?,?,?,?,?,?)',
                  (session_id, user_id, role, course_id, assignment_id, json.dumps(data)))
        conn.commit()
        conn.close()
        print(f"✅ Session saved: {session_id}")
    except Exception as e:
        print(f"❌ Error saving session: {str(e)}")
        raise

def get_session(session_id):
    """Retrieve LTI session from database"""
    try:
        conn = sqlite3.connect(DB_PATH)
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
    except Exception as e:
        print(f"❌ Error getting session: {str(e)}")
        return None
