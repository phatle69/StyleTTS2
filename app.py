from flask import Flask, request, jsonify
from Inference import TTS

app = Flask(__name__)

# Khởi tạo mô hình StyleTTS 2
tts = TTS(checkpoint="path/to/model.pth", config="path/to/config.yml")

@app.route("/", methods=["GET"])
def home():
    return "StyleTTS 2 API is running!"

@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "Missing text"}), 400

    # Chuyển văn bản thành giọng nói
    output_path = "output.wav"
    tts.synthesize(text, output_path)

    return jsonify({"audio": output_path})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
