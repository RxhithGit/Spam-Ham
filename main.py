from flask import Flask, request, jsonify, render_template
from spam_detection import classify_message  # This function is defined in your spam detection file
from organize_messages import categorize_message  # This function is defined in your organizing file
from flask_cors import CORS # type: ignore


app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    message = data.get("text", "")
    if not message.strip():
        return jsonify({"error": "Empty message"}), 400

    result = classify_message(message)  # "Spam" or "Ham"
    return jsonify({"result": result.lower()})  # Return as "spam" or "ham"

@app.route('/categories')
def categories_page():
    return render_template("categories.html")

@app.route('/categorize', methods=['POST'])
def categorize():
    data = request.get_json()
    message = data.get("text", "")
    if not message.strip():
        return jsonify({"error": "Empty message"}), 400

    category = categorize_message(message)  # "bank", "social", or "other"
    return jsonify({"category": category})


if __name__ == "__main__":
    app.run(debug=True)
