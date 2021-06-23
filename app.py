import streamlit as st
import streamlit.components.v1 as stc
import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np
import plotly.graph_objects as go
import pickle

from pyannote.audio.utils.signal import Binarize
import torch

@st.cache
def speech_activity_detection_model():
    # sad = torch.hub.load('pyannote-audio', 'sad_ami', source='local', device='cpu', batch_size=128)
    with open('speech_activity_detection_model.pkl', 'rb') as f:
        sad = pickle.load(f)
    return sad

@st.cache
def trim_noise_part_from_speech(sad, fname, speech_wav, sr):
    file_obj = {"uri": "filename", "audio": fname}
    sad_scores = sad(file_obj)
    binarize = Binarize(offset=0.52, onset=0.52, log_scale=True, min_duration_off=0.1, min_duration_on=0.1)
    speech = binarize.apply(sad_scores, dimension=1)
    
    noise_wav = np.zeros((speech_wav.shape[0], 0))
    append_axis = 1 if speech_wav.ndim == 2 else 0
    noise_ranges = []
    noise_start = 0
    for segmentation in speech.segmentation():
        noise_end, next_noise_start = int(segmentation.start*sr), int(segmentation.end*sr)
        noise_wav = np.append(noise_wav, speech_wav[:, noise_start:noise_end], axis=append_axis)
        noise_ranges.append((noise_start/sr, noise_end/sr))
        noise_start = next_noise_start
    return noise_wav.T, noise_ranges

@st.cache
def trim_audio(data, rate, start_sec=None, end_sec=None):
    start, end = int(start_sec * rate), int(end_sec * rate)
    if data.ndim == 1: # mono
        return data[start:end]
    elif data.ndim == 2: # stereo
        return data[:, start:end]
    
title = 'Audio noise reduction'
st.set_page_config(page_title=title, page_icon=":sound:")
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
noise_part_detection_method = st.radio('Noise source detection', ['Manually', 'Automatically (using speech activity detections)'])
if noise_part_detection_method == "Manually": # ノイズ区間は1箇所
    default_ranges = (0.0, float(wav_seconds)) if is_file_uploaded else (73.0, float(wav_seconds))
    noise_part_ranges = [st.slider("Select a part of the noise (sec)", 0.0, float(wav_seconds), default_ranges, step=0.1)]
    noise_wav = trim_audio(wav, sr, noise_part_ranges[0][0], noise_part_ranges[0][1])
    
elif noise_part_detection_method == "Automatically (using speech activity detections)": # ノイズ区間が複数
    with st.spinner('Please wait for Detecting the speech activities'):
        sad = speech_activity_detection_model()
        noise_wav, noise_part_ranges = trim_noise_part_from_speech(sad, uploaded_file, wav, sr)
   
fig = go.Figure()
x_wav = np.arange(len(wav)) / sr
fig.add_trace(go.Scatter(y=wav[::1000]))
for noise_part_range in noise_part_ranges:
    fig.add_vrect(x0=int(noise_part_range[0]*sr/1000), x1=int(noise_part_range[1]*sr/1000), fillcolor="Red", opacity=0.2)
fig.update_layout(width=700, margin=dict(l=0, r=0, t=0, b=0, pad=0))
fig.update_yaxes(visible=False, ticklabelposition='inside', tickwidth=0)
st.plotly_chart(fig, use_container_with=True)

st.text('Noise audio')
sf.write('noise_clip.wav', noise_wav, sr)
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
