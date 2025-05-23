import yt_dlp
from pydub import AudioSegment
import whisper
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from tkinter import messagebox, simpledialog
import os
import sys
import re
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai  # For GPT-based chatbot responses
from fpdf import FPDF  # For PDF generation

def download_youtube_mp3(url, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s')
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Downloading audio from the provided URL...")
            ydl.download([url])
        print("Download complete!")
    except Exception as e:
        print(f"Error occurred during download: {e}")

def convert_mp3_to_wav(mp3_file, wav_file):
    try:
        audio = AudioSegment.from_mp3(mp3_file)
        audio.export(wav_file, format="wav")
        print(f"Conversion to WAV complete: {wav_file}")
    except Exception as e:
        print(f"Error occurred during conversion to WAV: {e}")

def transcribe_audio_with_whisper(wav_file, model_name="base"):
    try:
        print(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)
        print("Transcribing audio with Whisper...")
        result = model.transcribe(wav_file)
        return result["text"]
    except Exception as e:
        print(f"Error occurred during transcription with Whisper: {e}")
        return None

def preprocess_transcription(text):
    text = re.sub(r"(\s+)", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1. \2", text)
    if not text.endswith("."):
        text += "."
    return text.strip()

def summarize_text_with_limit(text, max_words):
    try:
        model_name = "google/pegasus-cnn_dailymail"
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)
        processed_text = preprocess_transcription(text)
        inputs = tokenizer(processed_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_words,
            min_length=max(10, max_words // 2),
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        print(f"Error in summarizing text: {e}")
        return None

def generate_pdf(summary, output_filename="summary.pdf"):
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Video Summary", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, summary)
        pdf.output(output_filename)
        print(f"Summary saved as PDF: {output_filename}")
    except Exception as e:
        print(f"Error generating PDF: {e}")

def chatbot_interface():
    while True:
        user_query = input("Ask the chatbot a question (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        print("Chatbot:", chatbot_response(user_query))

def chatbot_response(query):
    try:
        openai.api_key = "your_openai_api_key"  # Replace with actual API key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": query}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error in chatbot response: {e}")
        return "Sorry, I couldn't process that."

def main():
    url = input("Enter the YouTube video URL: ")
    output_dir = input("Enter the directory to save the MP3 file: ")
    download_youtube_mp3(url, output_dir)
    mp3_files = [f for f in os.listdir(output_dir) if f.endswith('.mp3')]
    if not mp3_files:
        print("No MP3 files found.")
        sys.exit()
    mp3_file = os.path.join(output_dir, mp3_files[0])
    wav_file = mp3_file.replace('.mp3', '.wav')
    convert_mp3_to_wav(mp3_file, wav_file)
    transcription = transcribe_audio_with_whisper(wav_file)
    if not transcription or transcription.strip() == "":
        print("Error: Transcription failed.")
        sys.exit()
    print("\nTranscription:\n", transcription)
    max_words = 100
    summary = summarize_text_with_limit(transcription, max_words)
    if not summary:
        print("Error: Summary generation failed.")
        sys.exit()
    print("\nSummary:\n", summary)
    generate_pdf(summary)
    chatbot_interface()

if __name__ == "__main__":
    main()