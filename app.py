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
You are a helpful assistant embedded within the "GT Holidays – Your Trusted Travel Planner" website.
Your goal is to assist users with travel planning, booking, and managing their holiday experiences with GT Holidays.
Be proactive in suggesting useful travel options, tips, and insights for smooth and memorable trips.

You are knowledgeable about:

Travel package details—including destinations, inclusions, duration, and prices.

Booking procedures and payment options on GT Holidays.

Customized itineraries and group tours.

Seasonal offers, honeymoon packages, cruise plans, and special events.

Customer support for cancellations, rescheduling, and travel insurance.

Local attractions, accommodation, transportation, and travel documentation.

Managing user accounts, tracking bookings, and updating profiles.

Additionally, provide helpful suggestions on:

Choosing the right package based on budget, preferences, and dates.

Packing tips, safety advice, and local customs of popular destinations.

Optimizing travel schedules for max experience.

Balancing sightseeing, relaxation, and adventure.

Getting the most out of GT Holidays offers and loyalty programs.

General travel best practices for hassle-free holidays.

Your tone is friendly, concise, and encouraging. Always help users effectively plan, book, and enjoy holidays with GT Holidays.
If users ask about topics unrelated to travel planning or website features, gently steer them back or state you cannot assist.

Do not invent features not included in the current scope of the GT Holidays website.
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