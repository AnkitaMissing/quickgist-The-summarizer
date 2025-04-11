# quickgist-The-summarizer
# QuickGist - The Summarizer

QuickGist is an intelligent summarization tool that simplifies audio and video content into concise, easy-to-digest summaries. With powerful transcription, summarization, and AI-based features, QuickGist is the perfect companion for anyone looking to save time while consuming content. 

## Features

### 1. **YouTube Audio Downloader**
   - Downloads audio from YouTube videos in MP3 format using `yt_dlp`.
   - Converts MP3 files to WAV format for easier processing.

### 2. **Audio Transcription**
   - Transcribes audio files into text using OpenAI's Whisper model, ensuring accurate transcription even for complex audio.

### 3. **Text Summarization**
   - Uses the `Pegasus` transformer model to generate concise summaries of transcribed text.
   - Summaries are customizable with a word limit to fit your needs.

### 4. **PDF Summary Generation**
   - Automatically generates a PDF file containing the summarized content.
   - Ideal for saving and sharing insights from long-form content.

### 5. **AI Chatbot** (Upcoming)
   - Ask questions about the summarized content with an intelligent chatbot powered by OpenAI's GPT.
   - Get accurate, context-aware answers to deepen your understanding of the material.

### 6. **Flashcard Generator** (Upcoming)
   - Automatically extracts key points and generates flashcards from the summary.
   - Boost your learning and retention with this interactive feature.

### 7. **Streamlit Frontend** (Upcoming)
   - A simple and intuitive web interface for interacting with QuickGist.
   - Upload audio/video files, view summaries, chat with the AI, and generate flashcards—all from your browser.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AnkitaMissing/quickgist-The-summarizer.git
   cd quickgist-The-summarizer
2.Install the required dependencies:
    pip install -r requirements.txt
Run the main script:

bash
python main.py
Follow the prompts to:

Enter a YouTube video URL.
Download and transcribe the audio.
Summarize the text.
Generate a PDF of the summary.
(Upcoming) Use the Streamlit frontend:

Launch the Streamlit app:
bash
streamlit run app.py
Interact with the chatbot, view summaries, and generate flashcards.
Technologies Used
Python Libraries:

yt_dlp for YouTube audio downloading.
pydub for audio processing.
whisper for transcription.
transformers for text summarization.
openai for chatbot integration.
fpdf for PDF generation.
streamlit for the frontend interface.
AI Models:

OpenAI Whisper for audio transcription.
Pegasus for text summarization.
OpenAI GPT for the chatbot.
Future Enhancements
AI Chatbot: Enable users to ask context-aware questions about the summarized content.
Flashcard Generation: Automatically create interactive flashcards for learning and retention.
Streamlit Web App: Provide a user-friendly interface for all features.

### Next Steps
1. Add a `requirements.txt` file with all the dependencies.
2. Implement the chatbot, flashcards, and Streamlit frontend.
3. Update the repository with the `app.py` file for the Streamlit app.

