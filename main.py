import os
import uuid
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
import asyncio
from pydantic import BaseModel
from faster_whisper import WhisperModel

app = FastAPI(title="KION STT Server")

# Load model onto CPU with INT8 quantization for maximum speed
print("Loading Whisper small.en model onto CPU with INT8... This might take a few seconds.")
# Using int8 for CPU quantization (optimized for CPU execution)
whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4)

class STTResponse(BaseModel):
    text: str

@app.websocket("/ws/stt")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WS Connected: Client joined STT")
    audio_buffer = bytearray()
    last_transcribed_len = 0
    
    try:
        while True:
            print("STT: Waiting for WS data...")
            data = await websocket.receive()
            print(f"STT: Received raw data: keys={data.keys()}")
            if data.get("bytes"):
                print(f"STT: Received {len(data['bytes'])} bytes")
                audio_buffer.extend(data["bytes"])
                
                # Only run partial transcription if we've accumulated at least 0.5 seconds (16000 bytes) 
                # of NEW audio since the last time we transcribed. This prevents CPU backlog.
                if len(audio_buffer) - last_transcribed_len >= 16000:
                    print(f"Running partial transcription on {len(audio_buffer)} bytes...")
                    text = await run_transcription(audio_buffer)
                    await websocket.send_json({"event": "partial", "text": text})
                    last_transcribed_len = len(audio_buffer)
            elif data.get("text"):
                msg = data["text"]
                print(f"STT: Received text message: {msg}")
                import json
                try:
                    msg_data = json.loads(msg)
                    if msg_data.get("text") == "stop":
                        print("Received stop signal. Running final transcription...")
                        # Final transcription
                        if len(audio_buffer) > 0:
                            text = await run_transcription(audio_buffer)
                            await websocket.send_json({"event": "final", "text": text})
                            print(f"Final STT Result: '{text}'")
                        break
                except Exception as e:
                    print(f"STT: JSON Parse error or not stop: {e}")
                    pass
            elif data.get("type") == "websocket.disconnect":
                print("STT: Client disconnected normally.")
                break
    except WebSocketDisconnect:
        print("WS Disconnected")
    except RuntimeError as e:
        if "websocket.close" in str(e) or "websocket.send" in str(e):
            # Client disconnected abruptly while server was trying to send data
            print("WS Client disconnected abruptly")
        else:
            print(f"WS RuntimeError: {e}")
    except Exception as e:
        print(f"WS Error: {e}")

async def run_transcription(audio_bytes: bytearray) -> str:
    temp_path = f"/tmp/{uuid.uuid4()}_stream.wav"
    try:
        # Create a valid WAV file from the raw PCM bytes
        # 16kHz Mono 16-bit PCM
        import wave
        with wave.open(temp_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(audio_bytes)
            
        # Inference
        # Run in executor to not block the async event loop!
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, transcribe_file, temp_path)
        return text
    except Exception as e:
        print(f"Transcription error: {e}")
        return ""
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def transcribe_file(file_path: str) -> str:
    # Transcribe using faster-whisper with VAD to filter out background noise/silence
    segments, info = whisper_model.transcribe(
        file_path,
        beam_size=3,             # Balanced beam size for speed and accuracy
        vad_filter=True,         # Enable Voice Activity Detection (filters silence/background noise)
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    segments_list = list(segments)
    return "".join(seg.text for seg in segments_list).strip()

@app.post("/stt", response_model=STTResponse)
async def transcribe(audio: UploadFile = File(...)):
    # Save uploaded file
    temp_path = f"/tmp/{uuid.uuid4()}_{audio.filename}"
    with open(temp_path, "wb") as f:
        f.write(await audio.read())
        
    try:
        from pydub import AudioSegment
        sound = AudioSegment.from_file(temp_path)
        sound = sound.set_frame_rate(16000).set_channels(1)
        clean_wav_path = f"{temp_path}_clean.wav"
        sound.export(clean_wav_path, format="wav")
        
        text = transcribe_file(clean_wav_path)
            
        os.remove(temp_path)
        os.remove(clean_wav_path)
        
        return STTResponse(text=text)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return STTResponse(text=f"Error transcribing: {str(e)}")
