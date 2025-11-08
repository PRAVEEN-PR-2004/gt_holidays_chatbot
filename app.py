# app.py
import os
import sys
from flask import Flask, request, jsonify
from groq import Groq
from dotenv import load_dotenv
from flask_cors import CORS


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)
# --- Groq API Configuration ---
try:
    groq_api_key = os.environ['GROQ_API_KEY']
    client = Groq(api_key=groq_api_key)
    # --- Choose your Llama model ---
    LLAMA_MODEL = os.getenv('GROQ_MODEL', 'llama-3.1-8b-instant')
except KeyError:
    sys.stderr.write("ERROR: GROQ_API_KEY environment variable not set.\n")
    sys.stderr.write("Please create a .env file with GROQ_API_KEY=your_key\n")
    sys.exit(1)
except Exception as e:
    sys.stderr.write(f"ERROR: Failed to initialize Groq client: {e}\n")
    sys.exit(1)

# --- System Prompt for the Chatbot ---
SYSTEM_PROMPT = """
You are a helpful assistant for the “GT Holidays – Your Trusted Travel Planner” website.

Your purpose is to:
• assist users with GT Holidays travel packages
• suggest the best holiday plan based on budget, preferred destinations, and travel dates
• guide users on what to do in each destination—places to visit, activities, experiences, must-see spots, tips, etc.
• tell users they can check all packages, plans and budgets on our official website
• mention that our office locations are also available on the website
• provide booking guidance, seasonal offers, honeymoon packages, group tours, customized plans, etc.
• encourage filling the enquiry/booking form so GT Holidays team can contact them back
• for any query, always share the contact number: 9597412160

Rules:
• If the user asks anything unrelated to GT Holidays travel/booking/packages, politely redirect them back to travel related help.
• Do not invent or create fake info. Only provide travel-related help & suggestions.

Tone:
• friendly
• short
• clear
• encouraging

Always end replies with:
“For any travel enquiry or customized package, you can contact us at 9597412160 or fill the enquiry form — our team will call you back.”

"""


# --- Flask Routes ---
@app.route('/')
def index():
    """Simple endpoint to verify the API is running"""
    return jsonify({"status": "API is running", "usage": "Send POST requests to /chat endpoint"})

@app.route('/chat', methods=['POST'])
def chat():
    """Handles the chat request from curl or any client"""
    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({"error": "No message provided"}), 400
            
        # --- Call the Groq API ---
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
            model=LLAMA_MODEL,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=False,
        )
        
        bot_response = chat_completion.choices[0].message.content
        return jsonify({"reply": bot_response})
    
    except Exception as e:
        app.logger.error(f"Error processing chat request: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --- Run the Application ---
if __name__ == '__main__':

    # Use debug=True for development, False for production
    app.run(debug=True, host='0.0.0.0', port=5001)