import os
import csv
import yt_dlp


def read_csv(file_path):
    """Reads a CSV file and returns its content as a list of dictionaries.

    Args:
        file_path (str): The path to the CSV file.
    Returns:
        list: A list of dictionaries representing the rows in the CSV file.
    """
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        return [row for row in reader]


def download_audio(youtube_urls, output_path):
    """Downloads audio from a YouTube URL and saves it as an MP3 file.

    Args:
        youtube_url (str): The YouTube video URL.
        output_path (str): The path where the MP3 file will be saved.
    """
    ydl_opts = {
        'js_runtimes': {'deno': {'executable': 'deno'}},
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'noplaylist': True,
        # 'postprocessors': [{
        #     'key': 'FFmpegExtractAudio',
        #     'preferredcodec': 'wav',
        #     'preferredquality': '352',
        # }],
        # 'postprocessor_args': [
        #     '-acodec', 'pcm_s16le',
        #     '-ar', 20500,
        #     '-ac', '1'
        # ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(youtube_urls)


if __name__ == "__main__":
    file_path = 'song_links.csv'
    data = read_csv(file_path)
    links = [row['link'] for row in data]
    os.makedirs("downloads", exist_ok=True)
    download_audio(links, 'downloads/%(title)s.%(ext)s')
