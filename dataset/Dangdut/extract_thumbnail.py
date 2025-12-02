import os
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioBasicIO as aIO
import scipy.io.wavfile as wavf


def write_audio(path, sampling_rate, signal):
    wavf.write(path, sampling_rate, signal)


def resolve_segment(start1, end1, start2, end2, thumbnail_duration, audio_duration):
    if (end1 - start1) >= thumbnail_duration:
        return start1, end1
    elif (end2 - start2) >= thumbnail_duration:
        return start2, end2
    else:
        # Find segment that is closer to the center
        center1 = (start1 + end1) / 2
        center2 = (start2 + end2) / 2
        center_audio = audio_duration / 2
        if abs(center1 - center_audio) < abs(center2 - center_audio):
            center = center1
        else:
            center = center2
        # Extend the segment to meet the required duration
        # without exceeding audio bounds
        half_duration = thumbnail_duration / 2
        new_start = max(0, center - half_duration)
        new_end = min(audio_duration, center + half_duration)
        # Adjust if the segment is still shorter than required
        if new_end - new_start < thumbnail_duration:
            if new_start == 0:
                new_end = thumbnail_duration
            elif new_end == audio_duration:
                new_start = audio_duration - thumbnail_duration
        return new_start, new_end


def main(source_dir, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)

    # List all audio files in the source directory
    audio_files = [f for f in os.listdir(source_dir) if f.endswith('.wav')]

    print(f"Found {len(audio_files)} audio files.")
    print("Starting audio thumbnail extraction...")
    for audio_file in audio_files:
        print("-" * 40)
        input_path = os.path.join(source_dir, audio_file)
        print(f"Processing file: {input_path}")

        sampling_rate, signal = aIO.read_audio_file(input_path)
        audio_duration = len(signal) / sampling_rate

        thumbnail_duration = 30.0  # seconds
        start1, end1, start2, end2, _ = aS.music_thumbnailing(
            signal, sampling_rate, thumb_size=thumbnail_duration)

        start_time, end_time = resolve_segment(
            start1, end1,
            start2, end2,
            thumbnail_duration,
            audio_duration
        )

        print("Extracted audio thumbnail segment: "
              f"{start_time:.2f}s to {end_time:.2f}s")

        # Save the thumbnail segment
        start_frame = int(start_time * sampling_rate)
        end_frame = int(end_time * sampling_rate)
        thumbnail_signal = signal[start_frame:end_frame]
        output_path = os.path.join(destination_dir, audio_file)
        write_audio(output_path, sampling_rate, thumbnail_signal)

        print(f"Thumbnail saved to: {output_path}")

    print("-" * 40)
    print("Audio thumbnail extraction completed.")


if __name__ == "__main__":
    # Configuration
    source_dir = "formatted"
    destination_dir = "thumbnails"

    main(source_dir, destination_dir)
