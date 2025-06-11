import streamlit as st
from time import sleep
import yt_dlp
from pydub import AudioSegment
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from fpdf import FPDF
import openai
import os
import re
import urllib.request
from faster_whisper import WhisperModel
import base64
import time


# ========== Set Background Image and Text Color ==========
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: black;
    }}
    h1, h2, h3, h4, h5, h6, p, div, span, label {{
        color: black !important;
    }}
    .transcription-box {{
        background-color: #f2f2f2;
        color: #000000;
        padding: 15px;
        border-radius: 10px;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ========== Download Unicode Fonts ==========
def download_font():
    font_urls = {
        "NotoSans-Regular.ttf": "https://github.com/googlefonts/noto-fonts/blob/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf?raw=true",
        "NotoSerif-Regular.ttf": "https://github.com/googlefonts/noto-fonts/blob/main/hinted/ttf/NotoSerif/NotoSerif-Regular.ttf?raw=true"
    }
    for font_name, font_url in font_urls.items():
        if not os.path.exists(font_name):
            urllib.request.urlretrieve(font_url, font_name)

# ========== Typing Effect ==========
def typing_effect(text, speed=0.05):
    container = st.empty()
    full_text = ''
    for char in text:
        full_text += char
        container.markdown(f"<h1 style='font-size: 36px; color: #4FC3F7;'>{full_text}</h1>", unsafe_allow_html=True)
        sleep(speed)

# ========== Download YouTube MP3 ==========
def download_youtube_mp3(url, output_dir=r"D:\final year project\downloads"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'quiet': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            video_title = info_dict.get("title", "summary")
            mp3_filename = os.path.join(output_dir, f"{video_title}.mp3")
            if not os.path.exists(mp3_filename):
                files = [f for f in os.listdir(output_dir) if f.endswith(".mp3")]
                if files:
                    mp3_filename = os.path.join(output_dir, files[0])
                else:
                    st.error("MP3 file was not found after download.")
                    return None, None
        return video_title, mp3_filename
    except Exception as e:
        st.error(f"Error occurred during download: {e}")
        return None, None

# ========== Convert MP3 to WAV ==========
import subprocess

def convert_mp3_to_wav(mp3_file, wav_file):
    try:
        subprocess.run(["ffmpeg", "-i", mp3_file, wav_file], check=True)
    except subprocess.CalledProcessError as e:
        st.error("Error during MP3 to WAV conversion: " + str(e))


# ========== Transcribe Audio with Whisper ==========
def transcribe_audio_with_whisper(wav_file, model_name_or_path="base", language="en"):
    model = WhisperModel(model_name_or_path, compute_type="int8", device="cpu")
    segments, _ = model.transcribe(wav_file, language=language)
    transcription = " ".join(segment.text for segment in segments)
    return transcription.strip()

# ========== Preprocess Transcription ==========
def preprocess_transcription(text):
    text = re.sub(r"(\s+)", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1. \2", text)
    if not text.endswith("."):
        text += "."
    return text.strip()

# ========== Summarizers ==========
def summarize_with_pegasus(text, format_type="paragraph"):
    model_name = "google/pegasus-cnn_dailymail"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    processed_text = preprocess_transcription(text)
    inputs = tokenizer(processed_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=330, min_length=150)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
    if not summary.endswith(('.', '!', '?')):
        summary += "."
    if format_type == "point_wise":
        summary = convert_to_bullet_points(summary)
    elif format_type == "both":
        summary = format_summary_paragraph_with_bullets(summary)
    return summary


def summarize_with_gpt(text, format_type="paragraph"):
    prompt = f"Summarize the following text into a {format_type.replace('_', ' ')} summary:\n\n{text}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert summarizer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.5
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"OpenAI Error: {e}")
        return ""

# ========== Formatters ==========
def convert_to_bullet_points(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    points = ["• " + s.strip() for s in sentences if len(s.strip()) > 5]
    return "\n".join(points)

def format_summary_paragraph_with_bullets(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    summary = ""
    para = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
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

# ========== PDF Generation ==========
def generate_pdf(summary, pdf_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("NotoSans", "", "NotoSans-Regular.ttf", uni=True)
    pdf.set_font("NotoSans", "", 12)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.cell(0, 10, txt="Video Summary", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, summary, border=0, align='L')
    pdf.output(pdf_path)

def generate_transcription_pdf(transcription_text, pdf_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("NotoSerif", "", "NotoSerif-Regular.ttf", uni=True)
    pdf.set_font("NotoSerif", "", 12)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.cell(0, 10, txt="Full Transcription", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, transcription_text, border=0, align='L')
    pdf.output(pdf_path)

# ========== Streamlit App ==========
def main():
    st.set_page_config(page_title="VidioMind - YouTube Summarizer", page_icon="🎮", layout="wide")
    set_background("image_3.jpg")
    typing_effect("🎥 Welcome to VidioMind!")
    st.markdown("---")

    st.write("VidioMind helps you summarize YouTube videos into quick notes and download full transcripts. 🚀")
    st.markdown("---")
    st.sidebar.header("1. Download Audio")
    youtube_url = st.sidebar.text_input("Enter YouTube Video URL")
    language = st.sidebar.selectbox("Choose Language", ["en"])
    whisper_model = st.sidebar.selectbox("Choose Whisper Model", ["base", "large-v3"])
    summary_format = st.sidebar.selectbox("Summary Format", ["paragraph", "point_wise", "both"])
    openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
    summarizer_model = st.sidebar.selectbox("Summarizer Model", ["Pegasus", "T5", "OpenAI GPT"])

    if st.sidebar.button("Start Summarization"):
        tmpdir = r"j:\final year project\downloads"
        try:
            
            st.info("📥 Downloading Audio...")
            
            video_title, mp3_path = download_youtube_mp3(youtube_url, tmpdir)
            
            if not video_title:
                return
            st.success(f"Audio Downloaded: {video_title}")

            wav_path = os.path.join(tmpdir, "audio.wav")
            convert_mp3_to_wav(mp3_path, wav_path)
            
            
            st.info("📝 Transcribing Audio...")
            transcription = transcribe_audio_with_whisper(wav_path, model_name_or_path=whisper_model, language=language)
           
            if not transcription:
                st.error("Transcription Failed.")
                return
            st.success("Transcription Done! ✍️")
            st.subheader("📄 Transcription Preview")
            st.markdown(f"<div class='transcription-box'>{transcription[:1500]}...</div>", unsafe_allow_html=True)

            st.info("🧠 Summarizing...")
            
            if summarizer_model == "OpenAI GPT":
                if not openai_key:
                    st.warning("OpenAI API Key is required for GPT summarization.")
                    return
                openai.api_key = openai_key
                summary = summarize_with_gpt(transcription, format_type=summary_format)
            elif summarizer_model == "T5":
                summary = summarize_with_t5(transcription, format_type=summary_format)
            else:
                summary = summarize_with_pegasus(transcription, format_type=summary_format)
                

            st.success("Summarization Complete!")
            st.subheader("🧾 Summary Output")
            st.text_area("Summary", summary, height=400)

            download_font()
            safe_title = re.sub(r'[\\/*?:"<>|]', "", video_title)

            summary_pdf_path = os.path.join(tmpdir, f"{safe_title}_summary.pdf")
            transcription_pdf_path = os.path.join(tmpdir, f"{safe_title}_transcription.pdf")

            generate_pdf(summary, summary_pdf_path)
            generate_transcription_pdf(transcription, transcription_pdf_path)
            

            with open(summary_pdf_path, "rb") as f:
                st.download_button("📄 Download Summary PDF", f, file_name=f"{safe_title}_summary.pdf")

        except Exception as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
