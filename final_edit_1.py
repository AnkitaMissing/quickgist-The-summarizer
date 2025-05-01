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
from keybert import KeyBERT
import spacy

# Set stable base directory
BASE_DIR = r"J:\final year project"
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
os.chdir(BASE_DIR)


def download_font():
    font_url = "https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf"
    font_path = os.path.join(BASE_DIR, "NotoSans-Regular.ttf")

    if not os.path.exists(font_path):
        print("Downloading NotoSans-Regular.ttf font...")
        urllib.request.urlretrieve(font_url, font_path)
        print("Font downloaded.")
    
    # ✅ Check file size
    if os.path.exists(font_path):
        size = os.path.getsize(font_path)
        if size < 10000:
            print(f"⚠️ Font file might be corrupted (size: {size} bytes). Try redownloading.")
        else:
            print(f"✅ Font file verified (size: {size} bytes).")

def test_font_rendering():
    try:
        test_pdf = FPDF()
        test_pdf.add_page()
        test_pdf.add_font("NotoSans", "", os.path.join(BASE_DIR, "NotoSans-Regular.ttf"), uni=True)
        test_pdf.set_font("NotoSans", "", 12)
        test_pdf.cell(0, 10, "Test passed: NotoSans loaded successfully", ln=True)
        test_pdf.output("test_font_output.pdf")
        print("✅ Font test PDF created.")
    except Exception as e:
        print(f"❌ Font loading failed: {e}")


# ========== Download YouTube MP3 ==========
def download_youtube_mp3(url, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    downloaded_file = {"filename": None}
    
    def hook(d):
        if d['status'] == 'finished':
            downloaded_file["filename"] = d['filename']
        elif d['status'] == 'downloading':
            print(f"Downloading: {d['_percent_str']}")

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'progress_hooks': [hook],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print("Downloading audio from the provided URL...")
            info_dict = ydl.extract_info(url, download=True)
            video_title = info_dict.get("title", "summary")
        print("Download complete!")
    except Exception as e:
        print(f"Error occurred during download: {e}")
        return None, None

    return video_title, downloaded_file["filename"]


# ========== Convert MP3 to WAV ==========
def convert_mp3_to_wav(mp3_file, wav_file):
    try:
        audio = AudioSegment.from_mp3(mp3_file)
        audio.export(wav_file, format="wav")
        print(f"Conversion to WAV complete: {wav_file}")
    except Exception as e:
        print(f"Error occurred during conversion to WAV: {e}")

# ========== Transcribe Audio with Whisper ==========
def transcribe_audio_with_whisper(wav_file, model_name_or_path="base", language="hi"):
    try:
        print(f"Loading Whisper model: {model_name_or_path}")
        model = WhisperModel(model_name_or_path, compute_type="int8", device="cpu")
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
            max_length=330,
            min_length=150,
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
        font_path = os.path.join(BASE_DIR, "NotoSans-Regular.ttf")  # Ensure it points to the correct file
        if "NotoSans" not in pdf.fonts:
            pdf.add_font("NotoSans", "", font_path, uni=True)

        pdf.set_font("NotoSans", "", 12)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.cell(0, 10, txt="Video Summary", ln=True, align='C')
        pdf.ln(10)
        print("PDF summary preview (first 200 chars):", summary[:200])
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

# ========== Named Entity Recognition ==========
def extract_named_entities(text):
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        entities = list(set([(ent.text.strip(), ent.label_) for ent in doc.ents if len(ent.text.strip()) > 2]))
        return entities
    except Exception as e:
        print(f"Error in NER: {e}")
        return []

# ========== Keyword Extraction ==========
def extract_keywords(text, num_keywords=10):
    try:
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(text, top_n=num_keywords, stop_words='english')
        return [kw[0] for kw in keywords]
    except Exception as e:
        print(f"Error in keyword extraction: {e}")
        return []


# ========== MAIN ==========
def main():
    url = input("Enter the YouTube video URL: ")
    output_dir = os.path.join(BASE_DIR)

    language = input("Enter language code (e.g., 'en' for English, 'hi' for Hindi): ").strip()

    video_title, _ = download_youtube_mp3(url, output_dir)



    mp3_files = [f for f in os.listdir(output_dir) if f.endswith('.mp3')]
    if not mp3_files:
        print("No MP3 files found.")
        sys.exit()

    mp3_file = os.path.join(output_dir, mp3_files[0])
    wav_file = mp3_file.replace('.mp3', '.wav')
    convert_mp3_to_wav(mp3_file, wav_file)

    model_name_or_path = input("Enter Faster-Whisper model name or checkpoint path (e.g., 'base', 'large-v3'): ").strip()
    transcription = transcribe_audio_with_whisper(wav_file, model_name_or_path=model_name_or_path, language=language)

    if not transcription or transcription.strip() == "":
        print("Error: Transcription failed.")
        sys.exit()

    print("\n[Transcription Preview]:\n", transcription[:500])
    print("\nTranscription Word Count:", word_count(transcription))

    print("\n[Extracting Named Entities and Keywords...]")
    named_entities = extract_named_entities(transcription)
    keywords = extract_keywords(transcription)

    print("\nNamed Entities:")
    for ent, label in named_entities:
        print(f"- {ent} ({label})")

    print("\nKeywords:")
    for kw in keywords:
        print(f"- {kw}")


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

    # Clean title
    safe_title = re.sub(r'[\\/*?:"<>|]', "", video_title)
    pdf_filename = f"{safe_title}.pdf"

# Clean summary
    summary = summary.replace("📌", "[Keywords]").replace("🏷️", "[Named Entities]")
    summary = re.sub(r'[^\x00-\x7F]+', '', summary)

# Ensure font is ready
    download_font()

# Generate the PDF
    generate_pdf(summary, pdf_filename)


    chatbot_interface()

if __name__ == "__main__":
    download_font()
    test_font_rendering()
    main()
