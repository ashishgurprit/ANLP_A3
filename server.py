import os
import asyncio
import websockets
from openai import OpenAI
from google.cloud import texttospeech
import numpy as np
import io
import wave
import json

import ffmpeg

import importlib
import userdatamy

# Reload the module
importlib.reload(userdatamy)
oai_key = userdatamy.OpenAI_A3_Key1
OpenAI_org_id = userdatamy.OpenAI_org_id
OpenAI_project_id = userdatamy.OpenAI_project_id

# Set up Google Cloud credentials

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/delb/Documents/2024S1/NLP/dauntless-gate-128103-f99ede4152be.json"
# Initialize Google Cloud TTS client
tts_client = texttospeech.TextToSpeechClient()

client = OpenAI(
    # organization=OpenAI_org_id,
    # project=OpenAI_project_id,
    api_key=oai_key,
)

async def transcribe_audio(client, audio_buffer):
    audio_buffer.seek(0)
    file_tuple = ("audio.webm", audio_buffer, "audio/webm")
    print("Sending audio to OpenAI for transcription...")
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=file_tuple,
        response_format="text",
        language="hi"  # Set language to Hindi for transcription
    )
    print("Transcription received:", transcription)
    return transcription

async def translate_text(client, text):
    print("Translating text from Hindi to English:", text)
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a translation assistant."},
            {"role": "user", "content": f"Translate the following Hindi text to English: {text}"}
        ]
    )
    full_text = completion.choices[0].message.content
    parts = full_text.split('\n')
    translation_text = parts[1].strip() if len(parts) > 1 else parts[0].strip()
    print("Translation received:", translation_text)
    return translation_text

async def synthesize_speech(text):
    print("Synthesizing speech for text:", text)
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    print('Audio content synthesized')
    return response.audio_content

connected_clients = set()

async def handler(websocket, path):
    try:
        async for message in websocket:
            # Create a BytesIO buffer from the received message
            audio_buffer = io.BytesIO(message)
            
            # Save the received audio data to an MP3 file for inspection
            with open("input_audio.webm", "wb") as out:
                out.write(audio_buffer.getvalue())
                print('Audio content written to file "input_audio.webm"')
            
            # Send the webm audio directly to OpenAI for transcription
            transcription = await transcribe_audio(client, audio_buffer)
            if transcription.strip():
                translation = await translate_text(client, transcription)
                if translation:
                    audio_response = await synthesize_speech(translation)
                    message = {
                        "transcription": transcription,
                        "translation": translation,
                        "audio_response": audio_response.hex()
                    }
                    await websocket.send(json.dumps(message))

    except websockets.exceptions.ConnectionClosed:
        pass

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
