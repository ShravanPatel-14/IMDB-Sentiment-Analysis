import gradio as gr
from transformers import pipeline
import torch

torch.set_num_threads(1)
torch.set_grad_enabled(False)

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device="cpu"
)

def predict_sentiment(text):
    if not text.strip():
        return "Please enter some text."
    result = classifier(text)[0]
    return f"Sentiment: {result['label']} | Confidence: {round(result['score']*100, 2)}%"

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Enter a movie review here..."),
    outputs="text",
    title="IMDB Sentiment Analyzer (DistilBERT)",
    description="Real-time NLP Sentiment Analysis using Hugging Face Transformers"
)

if __name__ == "__main__":
    demo.launch()
