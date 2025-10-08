"""this script is for scraping channel videos """
import yt_dlp as ydlp

def download_videos( video_ids):
    for video_id in video_ids:
        video_url = f"https://www.youtube.com/shorts/{video_id}"
        ydl_opts = {
            'outtmpl' : 'agentic/Video/%(title)s.%(ext)s'
        }

        with ydlp.YoutubeDL(ydl_opts) as ydl :
            ydl.download(video_url)

def channel_videoids(url , num_videos):
    ydl_opts = {
        'extract_flat' : True , 
        'skip_download' : True,
        'playlist_items' : f'1-{num_videos}'
    }
    with ydlp.YoutubeDL(ydl_opts) as ydl : 
        info = ydl.extract_info(url , download=False)
        video_ids = [entry['id'] for entry in info['entries']]
    return video_ids

if __name__ == "__main__" : 
    url = 'https://www.youtube.com/@InfisDiary/shorts'
    num_videos = 4
    video_ids = channel_videoids(url)
    print(video_ids)
    download_videos(video_ids)