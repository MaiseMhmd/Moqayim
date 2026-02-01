from flask import Flask, request, redirect
import os

app = Flask(__name__)

# Get Streamlit URL from environment or use default
STREAMLIT_URL = os.environ.get('STREAMLIT_URL', 'https://mogayim.streamlit.app')

@app.route('/')
def home():
    return "✅ Moqayim LTI Server is running!", 200

@app.route('/lti/launch', methods=['POST', 'GET'])
def lti_launch():
    """Just redirect to Streamlit - no session needed"""
    
    # If GET request (testing in browser)
    if request.method == 'GET':
        return redirect(STREAMLIT_URL)
    
    # If POST request (from Blackboard)
    # Just redirect directly to Streamlit
    return redirect(STREAMLIT_URL)

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return {'status': 'healthy', 'streamlit_url': STREAMLIT_URL}, 200

if __name__ == '__main__':
    print("🚀 Moqayim LTI - Simple Redirect Server")
    print(f"🔗 Will redirect to: {STREAMLIT_URL}")
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
