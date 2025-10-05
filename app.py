from flask import Flask, request, jsonify
from flask_cors import CORS
from chatbot import get_response  # Your chatbot logic
from dotenv import load_dotenv  # Add this
import traceback
import os

load_dotenv()  # Add this: Loads .env

# Heroku/Render sets PORT; default 5000 locally
port = int(os.environ.get('PORT', 5000))

app = Flask(__name__)
CORS(app)

@app.route("/get_response", methods=["POST"])
def bot_response():
    try:
        user_input = request.form.get("user_input", "").strip()
        if user_input:
            response = get_response(user_input)
        else:
            response = "Please enter a message to chat."
        return jsonify({"response": response})
    except Exception as e:
        error_msg = f"Chatbot error: {str(e)}"
        print(f"Error in /get_response: {traceback.format_exc()}")
        return jsonify({"response": "Sorry, something went wrong. Try again or email admissions@nhck.in."})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "NHCK Chatbot API is running! Version: 1.0"})

# Optional home route for basic UI (add if testing locally)
@app.route('/')
def home():
    return '''
    <html><body>
        <h1>NHCK AI College Chatbot</h1>
        <form id="chat-form">
            <input type="text" name="user_input" placeholder="Ask about NHCK admissions..." required>
            <button type="submit">Send</button>
        </form>
        <div id="response"></div>
        <script>
            document.getElementById('chat-form').addEventListener('submit', function(e) {
                e.preventDefault();
                const formData = new FormData(this);
                fetch('/get_response', {method: 'POST', body: formData})
                .then(r => r.json()).then(data => {
                    document.getElementById('response').innerHTML += '<p><strong>Bot:</strong> ' + data.response + '</p>';
                });
            });
        </script>
    </body></html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)