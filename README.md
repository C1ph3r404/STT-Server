# KION STT Server (Speech-to-Text)

This is the lightweight backend service to run `nvidia/parakeet-tdt-0.6b-v2` locally on a CPU for the KION voice assistant. It is designed to run efficiently on 4GB RAM devices.

## Installation

1. **Clone this repository** to your 4GB laptop.
2. **Install ffmpeg** (required by pydub for audio conversion):
   ```bash
   sudo apt-get install ffmpeg
   ```
3. **Create a virtual environment and install dependencies**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Running the Server

Start the FastAPI server using `uvicorn`:
```bash
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 9000
```

> **Note:** The first time you run this, it will download the 600M parameter model from the Hugging Face / NVIDIA registry. This may take several minutes depending on your internet connection.

## API Endpoint

**`POST /stt`**
- **Payload**: Multipart form-data with an `audio` file.
- **Response**: JSON containing the transcribed text `{"text": "your speech here"}`.
