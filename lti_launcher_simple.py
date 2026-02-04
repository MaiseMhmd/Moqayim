from flask import Flask, request, redirect, render_template_string, jsonify, make_response
import secrets
import os
import hashlib
import hmac
import time
from urllib.parse import quote, unquote
from database import init_db, save_session, get_session

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-change-this')

# Initialize database
init_db()

# LTI Configuration
# IMPORTANT: These must match what you gave to Blackboard admin
CONSUMERS = {
    'moqayim_key': {  # This is your Tool Provider Key
        'secret': 'moqayim_secret'  # This is your Tool Provider Secret
    }
}

# Get Streamlit URL from environment or use default
STREAMLIT_URL = os.environ.get('STREAMLIT_URL', 'https://mogayim.streamlit.app')

def verify_lti_request(request_data, consumer_secret):
    """Verify LTI OAuth signature (simplified for basic validation)"""
    # For production, you should use a proper OAuth library
    # This is a simplified version for basic LTI 1.1
    oauth_signature = request_data.get('oauth_signature', '')
    return True  # For now, accept all requests. TODO: Implement proper OAuth validation

@app.route('/')
def home():
    return "✅ Moqayim LTI Server is running!", 200

@app.route('/lti/config', methods=['GET'])
def lti_config():
    """LTI configuration XML for Blackboard"""
    base_url = request.host_url.rstrip('/')
    
    # This XML tells Blackboard to open in a new window, not iframe
    config_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<cartridge_basiclti_link xmlns="http://www.imsglobal.org/xsd/imslticc_v1p0"
    xmlns:blti="http://www.imsglobal.org/xsd/imsbasiclti_v1p0"
    xmlns:lticm="http://www.imsglobal.org/xsd/imslticm_v1p0"
    xmlns:lticp="http://www.imsglobal.org/xsd/imslticp_v1p0"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.imsglobal.org/xsd/imslticc_v1p0 http://www.imsglobal.org/xsd/lti/ltiv1p0/imslticc_v1p0.xsd
    http://www.imsglobal.org/xsd/imsbasiclti_v1p0 http://www.imsglobal.org/xsd/lti/ltiv1p0/imsbasiclti_v1p0p1.xsd
    http://www.imsglobal.org/xsd/imslticm_v1p0 http://www.imsglobal.org/xsd/lti/ltiv1p0/imslticm_v1p0.xsd
    http://www.imsglobal.org/xsd/imslticp_v1p0 http://www.imsglobal.org/xsd/lti/ltiv1p0/imslticp_v1p0.xsd">
    <blti:title>Moqayim Grading Tool</blti:title>
    <blti:description>AI-powered short answer grading tool for instructors and students</blti:description>
    <blti:launch_url>{base_url}/lti/launch</blti:launch_url>
    <blti:secure_launch_url>{base_url}/lti/launch</blti:secure_launch_url>
    <blti:icon>{base_url}/static/icon.png</blti:icon>
    <blti:vendor>
        <lticp:code>moqayim</lticp:code>
        <lticp:name>Moqayim</lticp:name>
        <lticp:description>AI-powered grading assistance</lticp:description>
    </blti:vendor>
    <blti:custom>
        <lticm:property name="launch_presentation_return_url">$ToolConsumerProfile.url</lticm:property>
    </blti:custom>
    <cartridge_bundle identifierref="BLTI001_Bundle"/>
    <cartridge_icon identifierref="BLTI001_Icon"/>
</cartridge_basiclti_link>'''
    
    response = make_response(config_xml)
    response.headers['Content-Type'] = 'application/xml'
    response.headers['Content-Disposition'] = 'attachment; filename=moqayim_lti_config.xml'
    return response

@app.route('/lti/launch', methods=['POST', 'GET'])
def lti_launch():
    """Receive LTI launch from Blackboard"""
    
    try:
        # For testing with GET
        if request.method == 'GET':
            test_html = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Moqayim LTI Test</title>
                <style>
                    body { font-family: Arial; padding: 40px; background: #f5f5f5; }
                    .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                    h2 { color: #667eea; }
                    .status { padding: 15px; background: #e3f2fd; border-left: 4px solid #2196f3; margin: 20px 0; }
                    .info { background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }
                    code { background: #f5f5f5; padding: 2px 6px; border-radius: 3px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>✅ Moqayim LTI Endpoint is Active</h2>
                    <div class="status">
                        <strong>Status:</strong> Ready to receive LTI launches from Blackboard
                    </div>
                    <div class="info">
                        <p><strong>Configuration:</strong></p>
                        <ul>
                            <li>Launch URL: <code>''' + request.host_url.rstrip('/') + '''/lti/launch</code></li>
                            <li>Tool Provider Key: <code>moqayim_key</code></li>
                            <li>Tool Provider Secret: <code>moqayim_secret</code></li>
                            <li>Target App: <code>''' + STREAMLIT_URL + '''</code></li>
                        </ul>
                    </div>
                    <p>This endpoint expects POST requests from Blackboard with LTI parameters.</p>
                </div>
            </body>
            </html>
            '''
            return test_html, 200
        
        # Log all received parameters for debugging
        print("\n" + "="*60)
        print("📥 LTI LAUNCH REQUEST RECEIVED")
        print("="*60)
        print("Method:", request.method)
        print("Headers:", dict(request.headers))
        print("\nForm Data:")
        for key, value in request.form.items():
            print(f"   {key}: {value}")
        print("="*60 + "\n")
        
        # Extract LTI parameters
        oauth_consumer_key = request.form.get('oauth_consumer_key', '')
        user_id = request.form.get('user_id', request.form.get('lis_person_sourcedid', 'demo_user'))
        roles = request.form.get('roles', 'Student')
        course_id = request.form.get('context_id', 'demo_course')
        course_title = request.form.get('context_title', 'Course')
        assignment_id = request.form.get('resource_link_id', 'demo_assignment')
        assignment_title = request.form.get('resource_link_title', 'Assignment')
        user_email = request.form.get('lis_person_contact_email_primary', '')
        user_name = request.form.get('lis_person_name_full', user_id)
        
        # Verify consumer key
        if oauth_consumer_key and oauth_consumer_key not in CONSUMERS:
            print(f"❌ Invalid consumer key: {oauth_consumer_key}")
            return "Invalid consumer key", 403
        
        if oauth_consumer_key:
            consumer_secret = CONSUMERS[oauth_consumer_key]['secret']
            # In production, verify OAuth signature here
            # verify_lti_request(request.form, consumer_secret)
        
        print(f"✅ LTI Parameters:")
        print(f"   User: {user_name} ({user_id})")
        print(f"   Email: {user_email}")
        print(f"   Role: {roles}")
        print(f"   Course: {course_title} ({course_id})")
        print(f"   Assignment: {assignment_title} ({assignment_id})")
        
        # Determine if instructor or student
        is_instructor = any(role in roles for role in ['Instructor', 'Administrator', 'ContentDeveloper', 'TeachingAssistant'])
        role = 'instructor' if is_instructor else 'student'
        
        # Create session with all metadata
        session_id = secrets.token_urlsafe(16)
        session_metadata = {
            'user_name': user_name,
            'user_email': user_email,
            'course_title': course_title,
            'assignment_title': assignment_title,
            'roles': roles
        }
        save_session(session_id, user_id, role, course_id, assignment_id, session_metadata)
        
        print(f"✅ Session created: {session_id}")
        print(f"   Role: {role}")
        
        # Build Streamlit URL with session
        streamlit_url = f"{STREAMLIT_URL}?session={session_id}&role={role}"
        
        # Return HTML that forces a full-page redirect (not iframe)
        return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Launching Moqayim...</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .container {
            text-align: center;
            padding: 60px 40px;
            background: rgba(255, 255, 255, 0.15);
            border-radius: 24px;
            backdrop-filter: blur(12px);
            max-width: 550px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.25);
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        h1 {
            font-size: 32px;
            margin-bottom: 10px;
            font-weight: 600;
        }
        .subtitle {
            font-size: 16px;
            opacity: 0.9;
            margin-bottom: 30px;
        }
        .spinner {
            border: 5px solid rgba(255, 255, 255, 0.25);
            border-radius: 50%;
            border-top: 5px solid white;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
            margin: 30px auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .status {
            font-size: 16px;
            margin: 25px 0;
            font-weight: 500;
        }
        .launch-button {
            display: inline-block;
            margin-top: 30px;
            padding: 16px 45px;
            background: #ffd700;
            color: #333;
            text-decoration: none;
            font-weight: 700;
            font-size: 17px;
            border-radius: 30px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            border: none;
            cursor: pointer;
            display: none;
        }
        .launch-button:hover {
            background: #ffed4e;
            transform: translateY(-3px);
            box-shadow: 0 6px 25px rgba(0,0,0,0.3);
        }
        .info {
            font-size: 12px;
            margin-top: 30px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            line-height: 1.6;
        }
        .info-item {
            margin: 8px 0;
        }
        code {
            background: rgba(0,0,0,0.2);
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 11px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Launching Moqayim</h1>
        <p class="subtitle">AI-Powered Grading Assistant</p>
        <div class="spinner"></div>
        <div class="status" id="status">Connecting to your session...</div>
        <button class="launch-button" id="launchBtn" onclick="launchApp()">
            Launch Moqayim Now
        </button>
        <div class="info">
            <div class="info-item"><strong>User:</strong> {{ user_name }}</div>
            <div class="info-item"><strong>Role:</strong> {{ role|upper }}</div>
            <div class="info-item"><strong>Session:</strong> <code>{{ session }}</code></div>
        </div>
    </div>
    
    <script>
        const TARGET_URL = {{ url|tojson }};
        let redirectAttempted = false;
        
        function updateStatus(message, showButton = false) {
            const statusEl = document.getElementById('status');
            const buttonEl = document.getElementById('launchBtn');
            const spinnerEl = document.querySelector('.spinner');
            
            statusEl.textContent = message;
            
            if (showButton) {
                buttonEl.style.display = 'inline-block';
                spinnerEl.style.display = 'none';
            }
        }
        
        function launchApp() {
            if (redirectAttempted) return;
            redirectAttempted = true;
            
            try {
                // Method 1: Try to break out of iframe
                if (window.top && window.top !== window.self) {
                    console.log('In iframe - attempting to redirect parent');
                    updateStatus('Opening in main window...');
                    window.top.location.replace(TARGET_URL);
                    return;
                }
                
                // Method 2: Direct redirect
                console.log('Direct redirect');
                updateStatus('Redirecting now...');
                window.location.replace(TARGET_URL);
                
            } catch (error) {
                console.error('Redirect failed:', error);
                
                // Method 3: Open in new window
                updateStatus('Opening in new window...');
                const newWindow = window.open(TARGET_URL, '_blank', 'noopener,noreferrer');
                
                if (newWindow) {
                    updateStatus('✅ Opened successfully!', false);
                    setTimeout(() => {
                        updateStatus('You can close this window', false);
                    }, 2000);
                } else {
                    // Popup blocked - show button
                    updateStatus('⚠️ Please click the button below:', true);
                }
            }
        }
        
        // Automatic launch after 1 second
        setTimeout(() => {
            updateStatus('Launching now...');
            launchApp();
        }, 1000);
        
        // Show manual button after 4 seconds if still here
        setTimeout(() => {
            if (!redirectAttempted || window.location.href.indexOf('moqayim') === -1) {
                updateStatus('Click below to launch:', true);
            }
        }, 4000);
        
        // Prevent browser back button issues
        window.history.pushState(null, '', window.location.href);
        window.addEventListener('popstate', function() {
            window.history.pushState(null, '', window.location.href);
        });
    </script>
</body>
</html>
        ''', 
        url=streamlit_url, 
        session=session_id, 
        role=role,
        user_name=user_name), 200
        
    except Exception as e:
        print(f"❌ ERROR in LTI launch: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error</title>
            <style>
                body { font-family: Arial; padding: 40px; background: #f5f5f5; }
                .error { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; border-left: 4px solid #f44336; }
                h2 { color: #f44336; }
                pre { background: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <div class="error">
                <h2>⚠️ Launch Error</h2>
                <p>There was an error launching Moqayim:</p>
                <pre>{{ error }}</pre>
                <p>Please contact your administrator or try again.</p>
            </div>
        </body>
        </html>
        ''', error=str(e)), 500

@app.route('/api/session/<session_id>', methods=['GET'])
def api_get_session(session_id):
    """API endpoint for Streamlit to fetch session"""
    try:
        session_data = get_session(session_id)
        if session_data:
            print(f"✅ Session fetched: {session_id}")
            return jsonify(session_data), 200
        print(f"❌ Session not found: {session_id}")
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
        # TODO: Implement actual grade passback to Blackboard
        return jsonify({'status': 'success', 'message': 'Grade received'}), 200
    except Exception as e:
        print(f"❌ Grade passback error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'app': 'moqayim-lti',
        'streamlit_url': STREAMLIT_URL,
        'version': '2.0'
    }), 200

@app.route('/test-redirect')
def test_redirect():
    """Test the redirect functionality"""
    test_session = secrets.token_urlsafe(16)
    streamlit_url = f"{STREAMLIT_URL}?session={test_session}&role=instructor"
    
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head><title>Testing Redirect</title></head>
    <body style="font-family: Arial; padding: 40px;">
        <h2>Testing Redirect to Moqayim</h2>
        <p>Target URL: <code>{{ url }}</code></p>
        <button onclick="window.open('{{ url }}', '_blank')">Open in New Tab</button>
        <button onclick="window.location.href='{{ url }}'">Redirect Current Tab</button>
        <script>
            setTimeout(() => {
                console.log('Auto-redirecting...');
                window.location.href = '{{ url }}';
            }, 2000);
        </script>
    </body>
    </html>
    ''', url=streamlit_url)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 MOQAYIM LTI SERVER STARTING")
    print("="*70)
    print(f"📍 Streamlit App URL: {STREAMLIT_URL}")
    print(f"🔑 LTI Provider Key: moqayim_key")
    print(f"🔐 LTI Provider Secret: moqayim_secret")
    print("="*70 + "\n")
    
    # Render uses PORT environment variable
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
