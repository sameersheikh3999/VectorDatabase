import os
from datetime import datetime
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
from ratelimit import limits, sleep_and_retry
from http import HTTPStatus

from google.oauth2 import service_account
from sqlalchemy import create_engine
from jsonschema import validate, ValidationError

# --- Load .env (Windows full path) ---
env_path = r"C:\Users\HP\Videos\AI Projects\Chatbot LLama\.env"
load_dotenv(dotenv_path=env_path)

# --- ENV Vars ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY_2")
PORT = int(os.getenv("PORT", 5000))
RATE_LIMIT_CALLS = int(os.getenv("RATE_LIMIT_CALLS", 60))
RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", 60))

# --- BigQuery ENV Vars ---
credentials_path = os.getenv("bq_service_accout_key")
project_id = os.getenv("bq_project_id")

# --- BigQuery Auth ---
credentials = service_account.Credentials.from_service_account_file(credentials_path)
engine = create_engine(
    f"bigquery://{project_id}",
    credentials_path=credentials_path
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')]
)
logger = logging.getLogger(__name__)

# --- Flask App ---
app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})

# --- Groq Client ---
if not GROQ_API_KEY:
    logger.error("Missing GROQ_API_KEY_2 environment variable")
    raise ValueError("Missing GROQ_API_KEY_2 environment variable")

try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {str(e)}")
    raise

chat_histories = {}

# --- Supported Chart Types ---
supported_chart_types = [
    "bar", "line", "pie", "doughnut", "bubble", "polarArea", "radar", "scatter", "area", "mixed"
]

# --- JSON Schema ---
chart_json_schema = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": supported_chart_types},
        "data": {
            "type": "object",
            "properties": {
                "labels": {"type": "array", "items": {"type": "string"}},
                "datasets": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "data": {"type": "array", "items": {"type": "number"}},
                            "backgroundColor": {"type": "array", "items": {"type": "string"}},
                            "borderColor": {"type": "array", "items": {"type": "string"}},
                            "borderWidth": {"type": "number"}
                        },
                        "required": ["label", "data", "backgroundColor"]
                    }
                }
            },
            "required": ["labels", "datasets"]
        },
        "options": {
            "type": "object",
            "properties": {
                "responsive": {"type": "boolean"},
                "plugins": {
                    "type": "object",
                    "properties": {
                        "legend": {
                            "type": "object",
                            "properties": {
                                "position": {"type": "string"}
                            },
                            "required": ["position"]
                        },
                        "title": {
                            "type": "object",
                            "properties": {
                                "display": {"type": "boolean"},
                                "text": {"type": "string"}
                            },
                            "required": ["display", "text"]
                        }
                    },
                    "required": ["legend", "title"]
                },
                "scales": {
                    "type": "object",
                    "properties": {
                        "y": {
                            "type": "object",
                            "properties": {
                                "beginAtZero": {"type": "boolean"}
                            },
                            "required": ["beginAtZero"]
                        }
                    },
                    "required": ["y"]
                }
            },
            "required": ["responsive", "plugins", "scales"]
        }
    },
    "required": ["type", "data", "options"]
}

# --- Validation Function ---
def validate_chart_json(chart_json_str):
    try:
        chart_obj = json.loads(chart_json_str)
        validate(instance=chart_obj, schema=chart_json_schema)
        return True, chart_obj
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error(f"Chart JSON validation failed: {str(e)}")
        return False, str(e)

# --- Groq Chat Function ---
@sleep_and_retry
@limits(calls=RATE_LIMIT_CALLS, period=RATE_LIMIT_PERIOD)
def get_chat_response(messages: list) -> str:
    try:
        system_message = {
            "role": "system",
            "content": (
                "You are an intelligent assistant that generates valid Chart.js JSON for the following chart types: bar, line, pie, doughnut, bubble, polarArea, radar, scatter, area, and mixed."
                " Format your output as a JSON object: {type, data, options} where each follows Chart.js structure. Return only JSON."
            )
        }
        full_messages = [system_message] + messages
        logger.info("Sending request to Groq API with messages: %s", full_messages)
        chat_completion = client.chat.completions.create(
            messages=full_messages,
            model="llama3-70b-8192",
            stream=False,
            max_tokens=1000,
            temperature=0.7
        )
        response = chat_completion.choices[0].message.content
        logger.info("Received response from Groq API: %s", response)
        return response
    except Exception as e:
        logger.error(f"Error calling Groq API: {str(e)}")
        raise ValueError(f"Error calling Groq API: {str(e)}")

# --- Save Chat History ---
def save_chat_history_to_file(client_ip: str):
    try:
        folder = "chat_logs"
        os.makedirs(folder, exist_ok=True)
        filename = f"{client_ip}_chat.json"
        filepath = os.path.join(folder, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(chat_histories[client_ip], f, indent=2, ensure_ascii=False)
        logger.info(f"Chat history saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save chat history: {str(e)}")

# --- Chat Endpoint ---
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json(silent=True)
        if not data or 'message' not in data:
            return jsonify({"error": "Invalid or missing 'message' field"}), HTTPStatus.BAD_REQUEST

        user_message = data['message']
        client_ip = request.remote_addr
        if client_ip not in chat_histories:
            chat_histories[client_ip] = []
        chat_histories[client_ip] = chat_histories[client_ip][-10:]

        bq_context = (
            "Context: The BigQuery table `tbproddb.Teachers_Data` contains information about teachers, including user_id, name, gender (Male/Female), "
            "signed_in_status ('Signed In', 'Not Signed In'), assigned coach, sector, and EMIS. "
            "The sectors are: Tarnol, Nilore, B.K (Barakahu), Sihala, Urban-I, and Urban-II. "
            "To count total teachers or schools, use COUNT(DISTINCT user_id). Now, here's my actual question: "
        )

        full_user_message = bq_context + user_message
        chat_histories[client_ip].append({"role": "user", "content": full_user_message})

        ai_response = get_chat_response(chat_histories[client_ip])
        is_valid, parsed_or_error = validate_chart_json(ai_response)
        if not is_valid:
            return jsonify({"error": "Invalid chart JSON format", "details": parsed_or_error}), HTTPStatus.BAD_REQUEST

        chat_histories[client_ip].append({"role": "assistant", "content": ai_response})
        save_chat_history_to_file(client_ip)

        return jsonify({"response": ai_response}), HTTPStatus.OK

    except Exception as e:
        logger.error(f"Unexpected error in /chat: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), HTTPStatus.INTERNAL_SERVER_ERROR

# --- Run App ---
if __name__ == "__main__":
    logger.info(f"Starting Flask app on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
