import librosa
import numpy as np
import soundfile as sf

def extract_segment(audio_path, segment_duration=30, output_path="segment.wav",
                    hop_length=512, weight_onset=1.0, weight_rms=1.0):
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    if len(y) <= int(segment_duration * sr):
        sf.write(output_path, y, sr)
        print(f"Audio shorter than {segment_duration}s — saved full file to {output_path}")
        return

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    L = min(len(onset_env), len(rms))
    onset_env = onset_env[:L]
    rms = rms[:L]

    def _norm(x):
        x = np.asarray(x, dtype=float)
        denom = x.max() - x.min()
        return x if denom == 0 else (x - x.min()) / denom

    combined = weight_onset * _norm(onset_env) + weight_rms * _norm(rms)

    segment_frames = int(round(segment_duration * sr / hop_length))
    if segment_frames <= 0:
        raise ValueError("segment_duration too small for given hop_length/sr")

    if segment_frames >= len(combined):
        start_frame = 0
    else:
        window = np.ones(segment_frames, dtype=float)
        scores = np.convolve(combined, window, mode='valid')
        start_frame = int(np.argmax(scores))

    start_sample = start_frame * hop_length
    end_sample = start_sample + int(segment_duration * sr)
    end_sample = min(end_sample, len(y))

    segment = y[start_sample:end_sample]
    sf.write(output_path, segment, sr)
    print(f"Segment saved to {output_path} (start {start_sample/sr:.2f}s, end {end_sample/sr:.2f}s)")