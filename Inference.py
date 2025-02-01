import torch
import torchaudio
import librosa
import numpy as np
import yaml
from munch import Munch
from models import StyleEncoder  # Điều chỉnh nếu tên model khác
from utils import TextCleaner

# Khởi tạo text cleaner
text_cleaner = TextCleaner()

# Tải cấu hình
with open("config.yaml", "r") as f:
    config = Munch(yaml.safe_load(f))

# Load mô hình StyleTTS2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StyleTTS2(config.model).to(device)
model.load_state_dict(torch.load("styletts2.pth", map_location=device))
model.eval()

def preprocess_text(text):
    """ Tiền xử lý văn bản """
    return text_cleaner.clean(text)

def text_to_speech(text, save_path="output.wav"):
    """ Chuyển văn bản thành giọng nói """
    text = preprocess_text(text)
    with torch.no_grad():
        audio = model.infer(text)  # Giả định model có phương thức infer
    torchaudio.save(save_path, audio.cpu(), sample_rate=22050)
    print(f"File âm thanh lưu tại {save_path}")

if __name__ == "__main__":
    sample_text = "Hello, this is a test for StyleTTS2 deployment."
    text_to_speech(sample_text)
