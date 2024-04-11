import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

# Function to extract Mel spectrograms from audio files
def extract_mel_spectrogram(audio_file, n_mels=128, duration=2):
    y, sr = librosa.load(audio_file, duration=duration)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram, sr

# Load the trained model
model = load_model('audio_classification_model5.h5')

def main():
    st.title("Audio Classification App")

    choice = st.radio("Choose an option:", ["Upload WAV file", "Record Audio"])

    # Initialize audio_data
    audio_data = None

    if choice == "Upload WAV file":
        uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav", start_time=0, sample_rate=44100)

            if st.button("Make Prediction"):
                mel_spectrogram, sr = extract_mel_spectrogram(uploaded_file)
                mel_spectrogram = mel_spectrogram[np.newaxis, ..., np.newaxis]
                predictions = model.predict(mel_spectrogram)
                class_labels = ['FAKE', 'REAL']
                max_label = class_labels[np.argmax(predictions)]
                st.success(f'The audio is predicted as: {max_label}')

                # Plot the Mel spectrogram
                #fig, ax = plt.subplots()
                #librosa.display.specshow(mel_spectrogram[0, :, :], sr=sr, x_axis='time', y_axis='mel', ax=ax)
                #ax.set(title='Mel Spectrogram')
                #st.pyplot(fig)

    elif choice == "Record Audio":
        st.info("Click the 'Record' button to start recording.")

        # Record audio
        recording = st.button("Record")
        if recording:
            duration = 10  # You can adjust the duration based on your needs
            fs = 44100  # Sampling frequency

            st.write("Recording... Speak now!")

            # Record audio for the specified duration
            audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
            sd.wait()

            st.write("Recording complete. Now predicting...")

            # Save the recorded audio to a file
            recorded_audio_path = "C:/Users/kmgee/OneDrive/Desktop/Real_Fake_Main"
            sf.write(recorded_audio_path, audio_data.flatten(), fs)

            # Extract and preprocess Mel spectrogram
            mel_spectrogram, _ = extract_mel_spectrogram(recorded_audio_path)
            mel_spectrogram = mel_spectrogram[np.newaxis, ..., np.newaxis]

            # Make predictions
            predictions = model.predict(mel_spectrogram)

            # Get the label with the maximum probability
            class_labels = ['FAKE', 'REAL']
            max_label = class_labels[np.argmax(predictions)]

            # Print the result
            st.write(f'The audio is predicted as: {max_label}')

if __name__ == "__main__":
    main()
