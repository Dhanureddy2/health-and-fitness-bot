from flask import Flask, request, jsonify, render_template
import os
import json
import logging
import traceback
from gpt4all import GPT4All

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)

# Load JSON dataset for predefined responses
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "medical_chatbot_conversations.json")

# Load chat data with error handling
try:
    with open(JSON_PATH, "r", encoding="utf-8") as file:
        chat_data = json.load(file)
except FileNotFoundError:
    logging.error(f"JSON file not found at {JSON_PATH}. Ensure the file exists.")
    chat_data = []  # Fallback to an empty dataset
except json.JSONDecodeError as e:
    logging.error(f"Error decoding JSON file: {e}")
    chat_data = []  # Fallback to an empty dataset

# Load GPT4All model
MODEL_PATH = os.path.join(BASE_DIR, "models", "mistral-7b-instruct-v0.2.Q4_0.gguf")  # Adjust as needed

if not os.path.exists(MODEL_PATH):
    logging.error(f"Model file not found at {MODEL_PATH}. Ensure the model is downloaded.")
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

try:
    model = GPT4All(MODEL_PATH)
except Exception as e:
    logging.error(f"Error loading GPT4All model: {e}")
    raise RuntimeError("Failed to load GPT4All model.")

def get_predefined_response(user_query):
    user_query = user_query.lower().strip()

    # Check for greeting keywords
    greetings = ["hello", "hi", "hey", "greetings", "good morning", "good evening"]
    if any(greet in user_query for greet in greetings):
        return "Hello! How can I assist you with your health and fitness today? 😊"

    # Check predefined responses from JSON dataset
    for conv in chat_data:
        for msg in conv["messages"]:
            if msg["role"] == "user":
                question = msg["text"].lower().strip()
                if question in user_query or user_query in question:
                    return conv["messages"][1]["text"]  # Return the bot's response

    return None  # No predefined response found

def get_gpt4all_response(user_input):
    try:
        response = model.generate(user_input, max_tokens=150, temp=0.7)
        return response
    except Exception as e:
        logging.error(f"GPT4All Error: {e}")
        return "Sorry, I couldn't generate a response at the moment."

@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Extract user input
        data = request.json
        user_input = data.get("message", "").strip()

        if not user_input:
            return jsonify({"error": "No input provided"}), 400

        # Check for predefined responses
        predefined_response = get_predefined_response(user_input)
        if predefined_response:
            logging.info("Served predefined response.")
            return jsonify({"response": predefined_response})

        # Generate response using GPT4All
        logging.info("Sending user-defined query to GPT4All: %s", user_input)
        reply = get_gpt4all_response(user_input)
        logging.info("GPT4All response: %s", reply)

        return jsonify({"response": reply})

    except KeyError:
        logging.error("Invalid JSON structure in request.")
        return jsonify({"error": "Invalid JSON structure in request."}), 400
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": "An unexpected error occurred. Please try again later."}), 500

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
