import tempfile
import yt_dlp
import os
import librosa
import numpy as np
from pyAudioAnalysis import audioSegmentation as aS


def resolve_segment(start1, end1, start2, end2, thumbnail_duration, audio_duration):
    if (end1 - start1) >= thumbnail_duration:
        return start1, end1
    elif (end2 - start2) >= thumbnail_duration:
        return start2, end2
    else:
        center1 = (start1 + end1) / 2
        center2 = (start2 + end2) / 2
        center_audio = audio_duration / 2
        if abs(center1 - center_audio) < abs(center2 - center_audio):
            center = center1
        else:
            center = center2
        half_duration = thumbnail_duration / 2
        new_start = max(0, center - half_duration)
        new_end = min(audio_duration, center + half_duration)
        if new_end - new_start < thumbnail_duration:
            if new_start == 0:
                new_end = thumbnail_duration
            elif new_end == audio_duration:
                new_start = audio_duration - thumbnail_duration
        return new_start, new_end


def load_audio_from_youtube(url, target_sr=22050):
    tmpdir = tempfile.mkdtemp()
    out_path = os.path.join(tmpdir, "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_path,
        "quiet": False,
        "no_warnings": False,
        "socket_timeout": 60,
        "retries": 3,
        "fragment_retries": 3,
        "noplaylist": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
                "skip": ["dash", "hls"]
            }
        },
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        # "max_filesize": 50 * 1024 * 1024,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    wav_path = out_path.replace("%(ext)s", "wav")
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    
    audio_duration = len(y) / sr
    thumbnail_duration = 30.0
    
    if audio_duration <= thumbnail_duration:
        return y, sr
    
    start1, end1, start2, end2, _ = aS.music_thumbnailing(
        y, sr, thumb_size=thumbnail_duration)
    
    start_time, end_time = resolve_segment(
        start1, end1,
        start2, end2,
        thumbnail_duration,
        audio_duration
    )
    
    start_frame = int(start_time * sr)
    end_frame = int(end_time * sr)
    thumbnail_signal = y[start_frame:end_frame]
    
    return thumbnail_signal, sr
