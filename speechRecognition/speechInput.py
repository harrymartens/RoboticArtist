#!/usr/bin/env python3

import pyaudio

import io
from google.oauth2 import service_account
from google.cloud import speech

client_file = 'config/service-account.json'
credentials = service_account.Credentials.from_service_account_file(client_file)
client = speech.SpeechClient(credentials=credentials)

def recognizeSpeech():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16  # 16-bit resolution
    CHANNELS = 1              # mono
    RATE = 16000              # 16kHz sample rate
    RECORD_SECONDS = 5
        
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    print("* Recording...")
    
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* Done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_content = b"".join(frames)
        
    recognized_audio = speech.RecognitionAudio(content=audio_content)
    
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code='en-AU'
    )
    
    response = client.recognize(config=config, audio=recognized_audio)
    
    for result in response.results:
        print("Request: ", result.alternatives[0].transcript)
        return result.alternatives[0].transcript

if __name__ == "__main__":
    recognizeSpeech()