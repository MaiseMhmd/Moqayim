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
```

---

## Step 3: Set Environment Variables in Render

After you push these updates to GitHub and Render redeploys:

1. **Go to your Render dashboard**
2. **Click on your "Moqayim" service**
3. **Go to "Environment" tab**
4. **Add these environment variables:**
```
STREAMLIT_URL = https://your-app.streamlit.app
```
(Replace with your actual Streamlit URL once you deploy it)

You can add more variables later if needed:
```
FLASK_SECRET_KEY = your-random-secret-key-here
DB_PATH = lti_demo.db
```

---

## Step 4: What You Need to Give Blackboard Admin

**After deployment succeeds**, provide these to your admin:
```
Provider Domain: moqayim.onrender.com
(WITHOUT https://)

Tool Provider Key: moqayim_key_12345

Tool Provider Secret: moqayim_secret_abc789xyz

Launch URL: https://moqayim.onrender.com/lti/launch
