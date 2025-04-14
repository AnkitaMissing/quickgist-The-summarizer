import yt_dlp
from pydub import AudioSegment
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import os
import sys
import re
import textwrap
from fpdf import FPDF
from sentence_transformers import SentenceTransformer
import openai
from sklearn.metrics.pairwise import cosine_similarity
from faster_whisper import WhisperModel
import os
import urllib.request

def download_font():
    font_url = "https://github.com/googlefonts/noto-fonts/blob/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf?raw=true"
    font_path = "NotoSans-Regular.ttf"

    if not os.path.exists(font_path):
        print("Downloading DejaVuSans.ttf font...")
        urllib.request.urlretrieve(font_url, font_path)
        print("Font downloaded.")

# ========== Download YouTube MP3 ==========
def download_youtube_mp3(url, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'progress_hooks': [lambda d: print(f"Downloading: {d['_percent_str']}") if d['status'] == 'downloading' else None]
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Downloading audio from the provided URL...")
            ydl.download([url])
        print("Download complete!")
    except Exception as e:
        print(f"Error occurred during download: {e}")

# ========== Convert MP3 to WAV ==========
def convert_mp3_to_wav(mp3_file, wav_file):
    try:
        audio = AudioSegment.from_mp3(mp3_file)
        audio.export(wav_file, format="wav")
        print(f"Conversion to WAV complete: {wav_file}")
    except Exception as e:
        print(f"Error occurred during conversion to WAV: {e}")

# ========== Transcribe Audio with Whisper ==========
def transcribe_audio_with_whisper(wav_file, model_name="base", language="hi"):
    try:
        print(f"Loading Whisper model: {model_name}")
        model = WhisperModel(model_name, compute_type="int8", device="cpu")  # Ensure compatibility
        print("Transcribing audio with Whisper...")

        segments, _ = model.transcribe(wav_file, language=language)
        transcription = ""
        for segment in segments:
            transcription += segment.text + " "
        return transcription.strip()
    except Exception as e:
        print(f"Error occurred during transcription with Whisper: {e}")
        return None

# ========== Preprocess Transcription ==========
def preprocess_transcription(text):
    text = re.sub(r"(\s+)", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1. \2", text)
    if not text.endswith("."):
        text += "."
    return text.strip()

# ========== Ask for Summary Preferences ==========
def ask_summary_preferences():
    word_options = {
        1: 50,
        2: 100,
        3: 150,
        4: 200,
        5: 250
    }

    print("Select a word limit for the summary:")
    for key, val in word_options.items():
        print(f"{key}. 0 - {val} words")
        
    while True:
        try:
            choice = int(input("Enter your choice (1-5): "))
            if choice in word_options:
                word_limit = word_options[choice]
                break
            else:
                print("Please choose a valid option.")
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 5.")

    format_choice = input("Choose summary format - paragraph / point_wise / both: ").strip().lower()
    while format_choice not in ["paragraph", "point_wise", "both"]:
        format_choice = input("Invalid format. Choose from: paragraph, point_wise, both: ").strip().lower()

    return word_limit, format_choice

# ========== Convert Summary to Bullet Points ==========
def convert_to_bullet_points(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    points = ["• " + s.strip() for s in sentences if len(s.strip()) > 5]
    return "\n".join(points)

# ========== Summarize ==========
def summarize_text_with_limit(text, max_words, format_type="paragraph"):
    try:
        print("Loading Pegasus summarization model...")
        model_name = "google/pegasus-cnn_dailymail"
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name)

        processed_text = preprocess_transcription(text)
        inputs = tokenizer(processed_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_words,
            min_length=max(10, max_words // 2),
            length_penalty=1.0,
            num_beams=4,
            no_repeat_ngram_size=2,
            early_stopping=False  # Let it finish meaningfully
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
        if not summary.endswith(('.', '!', '?')):
            summary += "."

        if format_type == "point_wise":
          summary = convert_to_bullet_points(summary)
        elif format_type == "both":
          summary = format_summary_paragraph_with_bullets(summary)


        return summary
    except Exception as e:
        print(f"Error in summarizing text: {e}")
        return None

# ========== Generate PDF ==========
from fpdf import FPDF
import os
import textwrap
import re

# 🔧 Load Unicode font
def generate_pdf(summary, output_filename="summary.pdf"):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("NotoSans", "", "NotoSans-Regular.ttf", uni=True)
        pdf.set_font("NotoSans", "", 12)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.cell(0, 10, txt="Video Summary", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, summary, border=0, align='L')


        pdf.output(output_filename)
        print(f"✅ Summary saved as PDF: {output_filename}")
    except Exception as e:
        print(f"Error generating PDF: {e}")

# ✅ Better Natural Bullet Formatter
def format_summary_paragraph_with_bullets(text):
    # Split text into sentences using punctuation
    sentences = re.split(r'(?<=[.!?]) +', text)
    summary = ""
    para = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Rule of thumb: Use bullet if sentence starts like a point
        if sentence.lower().startswith(("firstly", "secondly", "additionally", "moreover", "in conclusion", "•", "-")) or len(sentence.split()) < 10:
            if para:
                summary += para.strip() + "\n\n"
                para = ""
            summary += f"• {sentence}\n"
        else:
            para += sentence + " "

    if para:
        summary += para.strip()

    return summary.strip()


# ========== Chatbot ==========
def chatbot_interface():
    while True:
        user_query = input("Ask the chatbot a question (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        print("Chatbot:", chatbot_response(user_query))

def chatbot_response(query):
    try:
        openai.api_key = "your_openai_api_key"  # Replace with your OpenAI key
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": query}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error in chatbot response: {e}")
        return "Sorry, I couldn't process that."

# ========== Word Count ==========
def word_count(text):
    return len(text.strip().split())

# ========== MAIN ==========
def main():
    url = input("Enter the YouTube video URL: ")
    output_dir = input("Enter the directory to save the MP3 file: ")
    language = input("Enter language code (e.g., 'en' for English, 'hi' for Hindi): ").strip()

    download_youtube_mp3(url, output_dir)

    mp3_files = [f for f in os.listdir(output_dir) if f.endswith('.mp3')]
    if not mp3_files:
        print("No MP3 files found.")
        sys.exit()

    mp3_file = os.path.join(output_dir, mp3_files[0])
    wav_file = mp3_file.replace('.mp3', '.wav')
    convert_mp3_to_wav(mp3_file, wav_file)

    transcription = transcribe_audio_with_whisper(wav_file, language=language)
    if not transcription or transcription.strip() == "":
        print("Error: Transcription failed.")
        sys.exit()

    print("\n[Transcription Preview]:\n", transcription[:500])
    print("\nTranscription Word Count:", word_count(transcription))

    max_words, format_type = ask_summary_preferences()
    summary = summarize_text_with_limit(transcription, max_words, format_type)
    if not summary:
        print("Error: Summary generation failed.")
        sys.exit()

    print("\n[Formatted Summary]:\n", summary)
    print("\nSummary Word Count:", word_count(summary))
    

    # Use formatted output in summarization
    if format_type == "both":
      summary = format_summary_paragraph_with_bullets(summary)
    download_font()
    generate_pdf(summary)


    chatbot_interface()

if __name__ == "__main__":
    main()
