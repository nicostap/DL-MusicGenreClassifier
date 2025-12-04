# use WSL to install essentia
import essentia
import essentia.standard as es
import numpy as np
import soundfile as sf

def extract_segment(audio_path, segment_duration=30, output_path="segment.wav"):
    loader = es.MonoLoader(filename=audio_path, sampleRate=44100)
    y = loader()
    sr = 44100

    hop_length = 512
    frame_size = 1024

    if len(y) < frame_size:
        onset_env = np.array([np.sum(y**2)])
    else:
        frames = [y[i:i+frame_size] for i in range(0, len(y) - frame_size + 1, hop_length)]
        onset_env = np.array([np.sum(f**2) for f in frames])

    segment_frames = int((segment_duration * sr) / hop_length)

    if len(onset_env) <= segment_frames:
        best_start = 0
    else:
        window = np.ones(segment_frames)
        energy_sums = np.convolve(onset_env, window, mode='valid')
        best_start = int(np.argmax(energy_sums))

    start_sample = best_start * hop_length
    end_sample = start_sample + (segment_duration * sr)

    end_sample = min(int(end_sample), len(y))
    segment = y[int(start_sample):int(end_sample)]

    sf.write(output_path, segment, sr)
    print(f"Segment saved to {output_path}")