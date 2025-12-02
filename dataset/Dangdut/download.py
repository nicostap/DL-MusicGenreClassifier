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
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(youtube_urls)


def extract_video_id(youtube_url):
    """Extracts the video ID from a YouTube URL.

    Args:
        youtube_url (str): The YouTube video URL.
    Returns:
        str: The extracted video ID.
    """
    return youtube_url.split("v=")[-1].split("&")[0]


def main(csv_path, save_dir):
    # Read CSV and download audio files
    data = read_csv(csv_path)
    os.makedirs(save_dir, exist_ok=True)
    for row in data:
        index = row["no"]
        link = row["link"]
        save_path = os.path.join(save_dir, f"dangdut.{index}.%(ext)s")
        download_audio([link], save_path)


if __name__ == "__main__":
    # Configuration
    csv_path = "song_links.csv"
    save_dir = "downloads"

    main(csv_path, save_dir)
