🎬 IMDB Movie Review Sentiment Analysis

Deep Learning • Transformers • TensorFlow • Flask UI • Docker Deployment

This project performs sentiment analysis (Positive 👏 / Negative 👎) on IMDB movie reviews using DistilBERT, a state-of-the-art Transformer model from Hugging Face.
The model is trained, evaluated, visualized, and deployed as an interactive sentiment prediction web app.

🧠 Project Highlights
Feature	Description
🔠 NLP Task	Sentiment Classification on IMDB Dataset
🧩 Model Used	DistilBERT (HuggingFace Transformers)
📊 Training Framework	TensorFlow + Keras
📉 Tracking & Metrics	Accuracy, Loss, plots saved
🌐 Web App	Flask + HTML UI
📦 Deployment	Docker container (local hosting)
🗃️ Tokenizer	Custom-tokenized & saved locally
📝 Explainability	Predictions returned with probability scores
📁 Project Structure
IMDB_Sentiment_Analysis/
│
├── src/                    # Data processing + training pipeline
│   ├── data.py             # Dataset load + tokenization
│   ├── model.py            # Model creation code
│   ├── utils_logging.py    # Plots + logging utilities
│   ├── run_quickly.py      # Training script
│
├── tf_distilbert_imdb/     # Saved trained model + tokenizer
│
├── templates/
│   └── index.html          # Web UI
│
├── app.py                  # Flask API + frontend serving
├── Dockerfile.cpu          # Docker deployment file
├── requirements.txt        # Project dependencies
└── README.md               # You're here!

🚀 How to Run Locally
1️⃣ Create & Activate Virtual Environment
python -m venv imdb_venv
source imdb_venv/Scripts/activate       # Windows

2️⃣ Install Requirements
pip install -r requirements.txt

3️⃣ Run Training Script (optional – already trained)
python src/run_quickly.py

4️⃣ Start Web App
python app.py


Now open UI in browser 👉 http://127.0.0.1:5000/ui

🐳 Run with Docker (Optional)

Build image:

docker build -f Dockerfile.cpu -t imdb-sentiment:cpu .


Run container:

docker run --rm -it -p 5000:5000 imdb-sentiment:cpu


Open ➝ http://127.0.0.1:5000/ui

📊 Model Performance
Metric	Value
Test Accuracy	⭐ ~92%
Model Type	DistilBERT Fine-Tuned
Epochs	2–4 (configurable)

Training logs + charts saved in runs/ folder:

accuracy.png

loss.png

history.json/csv

✨ UI Preview

✔ Enter any movie review text
✔ Click Predict
✔ Instantly get Sentiment + Confidence score

User-friendly interface built with Flask + HTML.

🌟 Skills Demonstrated

✔ NLP + Transformers
✔ TensorFlow fine-tuning
✔ Docker Containerization
✔ Full-stack ML deployment
✔ Git & GitHub version control
✔ Model evaluation & visualization

📌 Future Enhancements

🔹 Add LSTM / BERT comparison
🔹 Add confusion matrix UI
🔹 Deploy on Render / HuggingFace Spaces
🔹 Add batch prediction upload (CSV)

🙌 Author

Shravan Adapa
AI/ML & Data Science Enthusiast
📧 Open for collaborations & feedback!
