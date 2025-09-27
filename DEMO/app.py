import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Tải stop words và punkt
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')
    stop_words = set(stopwords.words('english'))

# Hàm tiền xử lý văn bản
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Tải mô hình đã lưu
try:
    loaded_model = load_model('sentiment_lstm_model.h5')
except FileNotFoundError:
    print("Error: 'sentiment_lstm_model.h5' not found. Please ensure the model file is in the same directory.")
    exit()

# Cấu hình tokenizer
max_words = 10000  # Số từ tối đa trong vocabulary
max_len = 100    # Độ dài chuỗi đầu vào

# Khởi tạo tokenizer
tokenizer = Tokenizer(num_words=max_words)

# Dữ liệu mẫu để fit tokenizer (nếu không có file dữ liệu gốc)
# Trong thực tế, bạn cần fit tokenizer trên dữ liệu huấn luyện gốc
sample_texts = [
    "This product is amazing and I love it so much",
    "The service was okay but nothing special",
    "I hate this product, worst experience ever",
    "Neutral comment about the weather today",
    "Not bad at all, quite good actually"
]
tokenizer.fit_on_texts([clean_text(text) for text in sample_texts])

# Hàm dự đoán cảm xúc
def predict_sentiment(text):
    # Tiền xử lý văn bản
    cleaned_text = clean_text(text)
    # Chuyển văn bản thành chuỗi số
    seq = tokenizer.texts_to_sequences([cleaned_text])
    # Pad chuỗi
    padded_seq = pad_sequences(seq, maxlen=max_len)
    # Dự đoán
    prediction = loaded_model.predict(padded_seq, verbose=0)
    # Lấy nhãn dự đoán
    predicted_class = np.argmax(prediction, axis=1)[0]
    # Ánh xạ nhãn
    sentiment_map = {0: 'Tiêu cực', 1: 'Trung lập', 2: 'Tích cực'}
    return sentiment_map[predicted_class], prediction[0]

# Ứng dụng chính
def main():
    print("=== Ứng dụng Phân loại Cảm xúc ===")
    print("Nhập 'quit' để thoát.")
    while True:
        user_input = input("\nNhập comment của bạn: ")
        if user_input.lower() == 'quit':
            print("Tạm biệt!")
            break
        if not user_input.strip():
            print("Vui lòng nhập comment hợp lệ.")
            continue
        sentiment, probabilities = predict_sentiment(user_input)
        print(f"\nComment: '{user_input}'")
        print(f"Cảm xúc dự đoán: {sentiment}")
        print(f"Xác suất (Tiêu cực, Trung lập, Tích cực): {probabilities}")

if __name__ == "__main__":
    main()