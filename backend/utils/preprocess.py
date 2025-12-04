import librosa
import numpy as np


def preprocess_audio_1(y, sr, n_mfcc=40, duration=3):
    target_len = sr * duration
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T  # (timesteps, mfcc)
    return np.expand_dims(mfcc, axis=0)

def preprocess_audio(y, sr, scaler, n_fft=2048, segment_duration=3, segment_overlap=0.5, max_pad_len=129):
    segment_samples = segment_duration * sr
    segments_mel = []
    
    for start in range(0, len(y) - int(segment_samples*(1-segment_overlap)), int(segment_samples*(1-segment_overlap))):
        segment = y[start:start + segment_samples]
        mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128,
                                                n_fft=n_fft, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        if mel_spec_db.shape[1] < max_pad_len:
            pad_width = max_pad_len - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :max_pad_len]
        
        segments_mel.append(mel_spec_db)
    
    X = np.array(segments_mel)
    X_reshaped = X.reshape(X.shape[0], -1)
    X_scaled = scaler.transform(X_reshaped)
    X = X_scaled.reshape(X.shape[0], X.shape[1], X.shape[2])
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    return X
    
