import tempfile
import yt_dlp
import os
import librosa


def load_audio_from_youtube(url, target_sr=22050):
    tmpdir = tempfile.mkdtemp()
    out_path = os.path.join(tmpdir, "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_path,
        "quiet": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    wav_path = out_path.replace("%(ext)s", "wav")
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    return y, sr
