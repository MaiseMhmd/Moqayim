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
STREAMLIT_URL = os.environ.get('STREAMLIT_URL', 'https://mogayim.streamlit.app')

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
        <extensions platform="bb.com">
            <property name="launch_target">window</property>
        </extensions>
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
        
        # Show redirect page with improved redirect logic
        return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Launching Moqayim...</title>
                <style>
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        margin: 0;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                    }
                    .container {
                        text-align: center;
                        padding: 50px;
                        background: rgba(255, 255, 255, 0.15);
                        border-radius: 20px;
                        backdrop-filter: blur(10px);
                        max-width: 500px;
                        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
                    }
                    h2 {
                        margin: 0 0 20px 0;
                        font-size: 28px;
                    }
                    .spinner {
                        border: 4px solid rgba(255, 255, 255, 0.3);
                        border-radius: 50%;
                        border-top: 4px solid white;
                        width: 50px;
                        height: 50px;
                        animation: spin 1s linear infinite;
                        margin: 30px auto;
                    }
                    @keyframes spin {
                        0% { transform: rotate(0deg); }
                        100% { transform: rotate(360deg); }
                    }
                    .launch-button {
                        display: inline-block;
                        margin-top: 25px;
                        padding: 15px 40px;
                        background: #ffd700;
                        color: #333;
                        text-decoration: none;
                        font-weight: bold;
                        font-size: 16px;
                        border-radius: 30px;
                        transition: all 0.3s;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                        border: none;
                        cursor: pointer;
                    }
                    .launch-button:hover {
                        background: #ffed4e;
                        transform: translateY(-2px);
                        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
                    }
                    .info {
                        font-size: 13px;
                        margin-top: 25px;
                        opacity: 0.9;
                        line-height: 1.6;
                    }
                    .status {
                        margin: 15px 0;
                        font-size: 15px;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>🚀 Launching Moqayim</h2>
                    <div class="spinner"></div>
                    <div class="status" id="status">Preparing your session...</div>
                    <a href="{{ url }}" class="launch-button" target="_top" id="launchBtn" style="display: none;">
                        Launch Moqayim
                    </a>
                    <div class="info">
                        <div>Session ID: <code>{{ session }}</code></div>
                        <div>Role: <strong>{{ role }}</strong></div>
                    </div>
                </div>
                
                <script>
                    const targetUrl = "{{ url }}";
                    let redirectAttempted = false;
                    
                    // Update status message
                    function updateStatus(message) {
                        document.getElementById('status').textContent = message;
                    }
                    
                    // Strategy 1: Try to break out of iframe and redirect top window
                    function redirectToTop() {
                        if (redirectAttempted) return;
                        redirectAttempted = true;
                        
                        try {
                            updateStatus('Redirecting to Moqayim...');
                            
                            // Check if we're in an iframe
                            if (window.top !== window.self) {
                                console.log('In iframe, attempting to redirect parent window');
                                // Try to redirect the top window
                                window.top.location.href = targetUrl;
                            } else {
                                console.log('Not in iframe, redirecting directly');
                                window.location.href = targetUrl;
                            }
                        } catch (e) {
                            // Cross-origin iframe prevents access to parent
                            console.error('Cannot access parent window:', e);
                            updateStatus('Opening in new window...');
                            openInNewWindow();
                        }
                    }
                    
                    // Strategy 2: Open in new window
                    function openInNewWindow() {
                        const newWindow = window.open(targetUrl, '_blank', 'noopener,noreferrer');
                        
                        if (newWindow) {
                            updateStatus('Moqayim opened in new window!');
                            document.getElementById('launchBtn').style.display = 'none';
                        } else {
                            // Popup blocked
                            showManualLaunch();
                        }
                    }
                    
                    // Strategy 3: Show manual launch button
                    function showManualLaunch() {
                        updateStatus('Please click the button below:');
                        const btn = document.getElementById('launchBtn');
                        btn.style.display = 'inline-block';
                        document.querySelector('.spinner').style.display = 'none';
                    }
                    
                    // Attempt automatic redirect after short delay
                    setTimeout(() => {
                        redirectToTop();
                    }, 800);
                    
                    // Manual button click handler
                    document.getElementById('launchBtn').addEventListener('click', function(e) {
                        e.preventDefault();
                        
                        try {
                            if (window.top !== window.self) {
                                window.top.location.href = targetUrl;
                            } else {
                                window.location.href = targetUrl;
                            }
                        } catch (err) {
                            // If top window access fails, open in new tab
                            window.open(targetUrl, '_blank');
                        }
                    });
                    
                    // Show manual button after 3 seconds as fallback
                    setTimeout(() => {
                        if (!redirectAttempted) {
                            showManualLaunch();
                        }
                    }, 3000);
                </script>
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
