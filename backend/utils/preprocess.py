import librosa
import numpy as np


def preprocess_audio(y, sr, n_mfcc=40, duration=3):
    target_len = sr * duration
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T  # (timesteps, mfcc)
    return np.expand_dims(mfcc, axis=0)
