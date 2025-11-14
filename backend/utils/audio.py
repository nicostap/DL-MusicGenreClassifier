import librosa
import numpy as np
from pydub import AudioSegment
import tempfile
import os


def load_audio_from_mp3(mp3_bytes, target_sr=22050):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.write(mp3_bytes)
    tmp.close()

    audio = AudioSegment.from_file(tmp.name)
    tmp_wav = tmp.name.replace(".mp3", ".wav")
    audio.export(tmp_wav, format="wav")

    y, sr = librosa.load(tmp_wav, sr=target_sr, mono=True)

    os.remove(tmp.name)
    os.remove(tmp_wav)
    return y, sr
