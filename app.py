import streamlit as st
import asyncio
import wave
import os
from st_audiorec import st_audiorec
from temporalio.client import Client
from workflows import WavAudioClassificationWorkflow
import numpy as np

async def run_audio_workflow(file_path):
    client = await Client.connect("localhost:7233")

    result = await client.execute_workflow(
        WavAudioClassificationWorkflow.run,
        file_path,  # Pass file path, NOT raw audio data
        id="wav-workflow",
        task_queue="audio-classification-task-queue",
    )
    return result

def save_audio_as_wav(audio_chunks, sample_rate=16000, output_path="./uploads/realtime_audio.wav"):
    os.makedirs("./uploads", exist_ok=True)

    # Convert numpy array to int16 PCM format
    audio_data = np.concatenate(audio_chunks, axis=0).astype(np.int16)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)  # Mono audio
        wf.setsampwidth(2)  # 16-bit audio (2 bytes per sample)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    return output_path

def main():
    st.header("Audio Classification with Temporal")

    uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

    if uploaded_file is not None:
        os.makedirs("./uploads", exist_ok=True)
        file_path = f"./uploads/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File saved: {file_path}")

        if st.button("Process Audio"):
            with st.spinner("Processing..."):
                result = asyncio.run(run_audio_workflow(file_path))
                st.write("Classification Result:", result)
    
    st.header("Real-time Audio Classification")
    st.write("Click the button below to start recording audio for real-time classification.")

    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        st.audio(wav_audio_data, format='audio/wav')
        os.makedirs("./uploads", exist_ok=True)
        filename = "./uploads/recorded_audio.wav"
        with open(filename, "wb") as f:
            f.write(wav_audio_data)
        st.success(f"Recording saved as {filename}")

        if st.button("Process Real-time Audio"):
            wav_file_path = save_audio_as_wav(wav_audio_data)
            result = asyncio.run(run_audio_workflow(wav_file_path))
            st.write("Classification Result:", result)

if __name__ == "__main__":
    main()
