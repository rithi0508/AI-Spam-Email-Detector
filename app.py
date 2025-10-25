import os
import re
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained model & vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'spam_model.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'vectorizer.pkl'))

# Common spam words for highlighting
SPAM_WORDS = ["win", "free", "money", "offer", "click", "urgent", "buy", "limited"]

def highlight_spam_words(text, spam_words):
    """Highlight spammy words using <mark>."""
    if not text:
        return ""
    for word in spam_words:
        text = re.sub(rf'\b({re.escape(word)})\b', r'<mark>\1</mark>', text, flags=re.IGNORECASE)
    return text

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    detected_words = []
    highlighted_message = ""
    
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        
        if message:
            transformed = vectorizer.transform([message])
            pred_raw = model.predict(transformed)[0]
            prediction = "SPAM" if pred_raw == 1 else "HAM"
            confidence = round(model.predict_proba(transformed).max() * 100, 2)
            detected_words = [w for w in SPAM_WORDS if w.lower() in message.lower()]
            highlighted_message = highlight_spam_words(message, detected_words)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        detected_words=detected_words,
        highlighted_message=highlighted_message
    )

if __name__ == "__main__":
    app.run(debug=True)
