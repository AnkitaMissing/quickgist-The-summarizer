import streamlit as st
from time import sleep
import yt_dlp
from pydub import AudioSegment
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from fpdf import FPDF
import openai
import os
import re
import tempfile
import urllib.request
from faster_whisper import WhisperModel
import base64
import imageio_ffmpeg
import os

# Tell yt-dlp where ffmpeg is
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())


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
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ========== Download Unicode Font ==========
def download_font():
    font_url = "https://github.com/googlefonts/noto-fonts/blob/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf?raw=true"
    font_path = "NotoSans-Regular.ttf"
    if not os.path.exists(font_path):
        urllib.request.urlretrieve(font_url, font_path)

# ========== Typing Effect ==========
def typing_effect(text, speed=0.05):
    container = st.empty()
    full_text = ''
    for char in text:
        full_text += char
        container.markdown(f"<h1 style='font-size: 36px; color: #4FC3F7;'>{full_text}</h1>", unsafe_allow_html=True)
        sleep(speed)

# ========== Download YouTube MP3 ==========
import os
import yt_dlp

def download_youtube_mp3(url: str, output_dir: str = "downloads"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False
    }

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded_file = {"filename": None}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_title = info_dict.get("title", "summary")

    # Look for expected file
    expected_filename = os.path.join(output_dir, f"{video_title}.mp3")
    if os.path.exists(expected_filename):
        downloaded_file["filename"] = expected_filename
    else:
        for file in os.listdir(output_dir):
            if file.startswith(video_title) and file.endswith(".mp3"):
                downloaded_file["filename"] = os.path.join(output_dir, file)
                break

    if not downloaded_file["filename"] or not os.path.exists(downloaded_file["filename"]):
        raise FileNotFoundError("MP3 file was not found after download.")

    return video_title, downloaded_file["filename"]

# Example usage:
# video_title, filepath = download_youtube_mp3("https://www.youtube.com/watch?v=VIDEO_ID")
# print(f"Downloaded: {filepath}")

    

# ========== Convert MP3 to WAV ==========
def convert_mp3_to_wav(mp3_file, wav_file):
    if not os.path.exists(mp3_file):
        raise FileNotFoundError(f"MP3 file not found: {mp3_file}")
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format="wav")

# ========== Transcribe Audio with Whisper ==========
def transcribe_audio_with_whisper(wav_file, model_name_or_path="base", language="en"):
    model = WhisperModel(model_name_or_path, compute_type="int8", device="cpu")
    segments, _ = model.transcribe(wav_file, language=language)
    transcription = ""
    for segment in segments:
        transcription += segment.text + " "
    return transcription.strip()

# ========== Preprocess Transcription ==========
def preprocess_transcription(text):
    text = re.sub(r"(\s+)", " ", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1. \2", text)
    if not text.endswith("."):
        text += "."
    return text.strip()

# ========== Summarize Text ==========
def summarize_text_with_limit(text, format_type="paragraph"):
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
        early_stopping=False
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
    if not summary.endswith(('.', '!', '?')):
        summary += "."
    if format_type == "point_wise":
        summary = convert_to_bullet_points(summary)
    elif format_type == "both":
        summary = format_summary_paragraph_with_bullets(summary)
    return summary

# ========== Bullet Point Formatter ==========
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

# ========== Generate PDF ==========
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

# ========== Streamlit App ==========
def main():
    st.set_page_config(
        page_title="VidioMind - YouTube Summarizer",
        page_icon="🎬",
        layout="wide"
    )

    set_background("image_3.jpg")

    typing_effect("🎥 Welcome to VidioMind!")
    st.markdown("---")

    st.write(""" 
    VidioMind helps you summarize YouTube videos into quick notes 
    and even download the summary as a PDF. 
    Perfect for students, creators, and curious minds! 🚀
    """)
    st.markdown("---")

    st.sidebar.header("1. Download Audio")
    youtube_url = st.sidebar.text_input("Enter YouTube Video URL")
    language = st.sidebar.selectbox("Choose Language", ["en", "hi"], index=0)
    whisper_model = st.sidebar.selectbox("Choose Whisper Model", ["base", "large-v3"], index=0)
    summary_format = st.sidebar.selectbox("Summary Format", ["paragraph", "point_wise", "both"])

    if st.sidebar.button("Start Summarization"):
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                st.info("📥 Downloading Audio...")
                video_title, mp3_path = download_youtube_mp3(youtube_url, tmpdir)
                st.success(f"Audio Downloaded: {video_title}")

                wav_path = os.path.join(tmpdir, "audio.wav")
                convert_mp3_to_wav(mp3_path, wav_path)

                st.info("📝 Transcribing Audio...")
                transcription = transcribe_audio_with_whisper(wav_path, model_name_or_path=whisper_model, language=language)

                if not transcription:
                    st.error("Transcription Failed.")
                    return

                st.success("Transcription Done! ✍️")
                st.subheader("🔹 Transcription Preview")
                st.write(transcription[:1000] + "...")

                st.info("🧠 Summarizing...")
                summary = summarize_text_with_limit(transcription, format_type=summary_format)
                st.success("Summarization Complete!")

                st.subheader("🔹 Summary Output")
                st.text_area("Summary", summary, height=400)

                download_font()
                safe_title = re.sub(r'[\\/*?:"<>|]', "", video_title)
                pdf_path = os.path.join(tmpdir, f"{safe_title}.pdf")
                generate_pdf(summary, pdf_path)

                with open(pdf_path, "rb") as f:
                    st.download_button("📄 Download PDF Summary", f, file_name=f"{safe_title}.pdf")
            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
