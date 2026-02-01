from flask import Flask, request, redirect, render_template_string, jsonify
import secrets
import os
from database import init_db, save_session, get_session

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-change-this')

# Initialize database
init_db()

# LTI Configuration
# IMPORTANT: Change these to your own values!
CONSUMERS = {
    'moqayim_key': {  # This is your Tool Provider Key
        'secret': 'moqayim_secret'  # This is your Tool Provider Secret
    }
}

# Get Streamlit URL from environment or use default
STREAMLIT_URL = os.environ.get('STREAMLIT_URL', 'https://mogayim.streamlit.app/')

@app.route('/')
def home():
    return "✅ Moqayim LTI Server is running!", 200

@app.route('/lti/config', methods=['GET'])
def lti_config():
    """LTI configuration XML for Blackboard"""
    # This will use your actual Render URL automatically
    base_url = request.host_url.rstrip('/')
    
    config_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
    <cartridge_basiclti_link xmlns="http://www.imsglobal.org/xsd/imslticc_v1p0">
        <title>Moqayim Grading Tool</title>
        <description>AI-powered short answer grading</description>
        <launch_url>{base_url}/lti/launch</launch_url>
    </cartridge_basiclti_link>'''
    return config_xml, 200, {'Content-Type': 'application/xml'}

@app.route('/lti/launch', methods=['POST', 'GET'])
def lti_launch():
    """Receive LTI launch from Blackboard"""
    
    try:
        # For testing, accept both POST and GET
        if request.method == 'GET':
            return "✅ LTI endpoint is working! Use POST with LTI parameters to launch.", 200
        
        # Extract LTI parameters from POST data
        user_id = request.form.get('user_id', 'demo_user')
        roles = request.form.get('roles', 'Student')
        course_id = request.form.get('context_id', 'demo_course')
        assignment_id = request.form.get('resource_link_id', 'demo_assignment')
        
        print(f"📥 LTI Launch received:")
        print(f"   User: {user_id}")
        print(f"   Role: {roles}")
        print(f"   Course: {course_id}")
        print(f"   Assignment: {assignment_id}")
        
        # Determine if instructor or student
        is_instructor = 'Instructor' in roles or 'Administrator' in roles
        role = 'instructor' if is_instructor else 'student'
        
        # Create session
        session_id = secrets.token_urlsafe(16)
        save_session(session_id, user_id, role, course_id, assignment_id, {})
        
        print(f"✅ Session created: {session_id}")
        
        # Redirect to Streamlit with session ID
        streamlit_url = f"{STREAMLIT_URL}?session={session_id}&role={role}"
        
        # Show redirect page
        return render_template_string('''
            <html>
            <head>
                <title>Launching Moqayim...</title>
                <meta http-equiv="refresh" content="0;url={{ url }}">
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                    }
                    .container {
                        text-align: center;
                        padding: 40px;
                        background: rgba(255, 255, 255, 0.1);
                        border-radius: 15px;
                        backdrop-filter: blur(10px);
                    }
                    .spinner {
                        border: 4px solid rgba(255, 255, 255, 0.3);
                        border-radius: 50%;
                        border-top: 4px solid white;
                        width: 40px;
                        height: 40px;
                        animation: spin 1s linear infinite;
                        margin: 20px auto;
                    }
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                    a {
                        color: #ffd700;
                        text-decoration: none;
                        font-weight: bold;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>🚀 Launching Moqayim Grading Tool...</h2>
                    <div class="spinner"></div>
                    <p>Session: {{ session }}</p>
                    <p>Role: {{ role }}</p>
                    <p style="margin-top: 30px; font-size: 14px;">
                        If not redirected automatically, <a href="{{ url }}">click here</a>.
                    </p>
                </div>
            </body>
            </html>
        ''', url=streamlit_url, session=session_id, role=role), 200
        
    except Exception as e:
        print(f"❌ Error in LTI launch: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 500

@app.route('/api/session/<session_id>', methods=['GET'])
def api_get_session(session_id):
    """API endpoint for Streamlit to fetch session"""
    try:
        session_data = get_session(session_id)
        if session_data:
            return jsonify(session_data), 200
        return jsonify({'error': 'Session not found'}), 404
    except Exception as e:
        print(f"❌ Error fetching session: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/lti/grade', methods=['POST'])
def grade_passback():
    """Receive grades from Streamlit and send back to Blackboard"""
    try:
        data = request.json
        print(f"📊 Grade passback received: {data}")
        return {'status': 'success'}, 200
    except Exception as e:
        print(f"❌ Grade passback error: {str(e)}")
        return {'status': 'error', 'message': str(e)}, 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return {'status': 'healthy', 'app': 'moqayim-lti', 'streamlit_url': STREAMLIT_URL}, 200

if __name__ == '__main__':
    print("="*60)
    print("🚀 Moqayim LTI Server Starting...")
    print("="*60)
    print(f"📍 Streamlit URL: {STREAMLIT_URL}")
    print("="*60)
    # Render uses PORT environment variable
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)


