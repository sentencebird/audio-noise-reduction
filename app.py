import streamlit as st
import streamlit.components.v1 as stc
import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np
import plotly.graph_objects as go

def trim_audio(data, rate, start_sec=None, end_sec=None):
    start, end = int(start_sec * rate), int(end_sec * rate)
    if data.ndim == 1: # mono
        return data[start:end]
    elif data.ndim == 2: # stereo
        return data[:, start:end]
    
title = 'Audio noise reduction'
st.set_page_config(page_title=title, page_icon=":sound:", layout='wide')
st.title(title)

uploaded_file = st.file_uploader("Upload your audio file (.wav)") 

is_file_uploaded = uploaded_file is not None
if not is_file_uploaded:
    uploaded_file = 'sample.wav'    

wav, sr = librosa.load(uploaded_file, sr=None)
wav_seconds = int(len(wav)/sr)

st.subheader('Original audio')
st.audio(uploaded_file)

st.subheader('Noise part')
default_ranges = (0.0, float(wav_seconds)) if is_file_uploaded else (73.0, float(wav_seconds))
noise_part_ranges = st.slider("Select a part of the noise (sec)", 0.0, float(wav_seconds), default_ranges, step=0.1)

fig = go.Figure()
x_wav = np.arange(len(wav)) / sr
fig.add_trace(go.Scatter(y=wav[::1000]))
fig.add_vrect(x0=int(noise_part_ranges[0]*sr/1000), x1=int(noise_part_ranges[1]*sr/1000), fillcolor="Red", opacity=0.2)
fig.update_layout(width=700, margin=dict(l=0, r=0, t=0, b=0, pad=0))
fig.update_yaxes(visible=False, ticklabelposition='inside', tickwidth=0)
st.plotly_chart(fig, use_container_with=True)

st.text('Noise audio')
sf.write('noise_clip.wav', trim_audio(wav, sr, noise_part_ranges[0], noise_part_ranges[1]), sr)
noise_wav, sr = librosa.load('noise_clip.wav', sr=None)
st.audio('noise_clip.wav')

if st.button('Denoise the audio!'):
    with st.spinner('Please wait for completion'):
        nr_wav = nr.reduce_noise(audio_clip=wav, noise_clip=noise_wav, prop_decrease=1.0)

        st.subheader('Denoised audio')
        sf.write('nr_clip.wav', nr_wav, sr)
    st.success('Done!')
    st.text('Denoised audio')
    st.audio('nr_clip.wav')
