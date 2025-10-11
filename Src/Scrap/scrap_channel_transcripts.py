import os
import re
import time
import yt_dlp
import pandas as pd

def get_videoids(channel_url , numb_videos):
    ydl_opts = {
        'extract_flat': True,
        'skip_download' : True,
        'playlist_items': f"1-{numb_videos}"
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(channel_url , download = False)
        video_ids = [entry['id'] for entry in info['entries']]
        number_videos = len(info['entries'])
    return video_ids , number_videos
def get_transcript(video_ids ):
    channel_metadata = []
    for video_id in video_ids:
            url = f"https://www.youtube.com/shorts/{video_id}"

            # Configure yt-dlp options
            ydl_opts = {
                'skip_download': True,            
                'writeautomaticsub': True,      
                'writesubtitles': True,          
                'subtitlesformat': 'srt'       
            }

            # Extract and print subtitles
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Extract video info without downloading
                    info = ydl.extract_info(url, download=False)
                    title = info.get('title')
                    duration = info.get('duration')
                    upload_date = info.get('upload_date')
                    view_count = info.get('view_count')
                    
                    # Access subtitles from the info dictionary
                    subtitles = info.get('subtitles', {}).get("en") or info.get('automatic_captions', {}).get(f"en")
                    
                    if subtitles:
                        # Find the SRT subtitle entry
                        for sub in subtitles:
                            if sub.get('ext') == 'srt':
                                subtitle_url = sub.get('url')
                                if subtitle_url:
                                    # Download the subtitle content
                                    subtitle_content = ydl.urlopen(subtitle_url).read().decode('utf-8')
                                    lines = subtitle_content.splitlines()
                                    processed_lines = []

                                    # Remove line numbers, timestamps, and empty lines
                                    for line in lines:
                                        # Skip line numbers (digits only)
                                        if re.match(r'^\d+$', line.strip()):
                                            continue
                                        # Skip timestamps (format: 00:00:00,000 --> 00:00:00,000)
                                        if re.match(r'^\d{2}:\d{2}:\d{2},\d{3}\s-->\s\d{2}:\d{2}:\d{2},\d{3}$', line.strip()):
                                            continue
                                        # Skip empty lines
                                        if line.strip() == '':
                                            continue
                                        processed_lines.append(line.strip())

                                    # Combine into a single line
                                    transcription = ' '.join(processed_lines)
                                    channel_metadata.append({"video_id" : video_id, 
                                                            "title" : title,
                                                            "view_count" : view_count,
                                                            "duration" : duration,
                                                            "upload_date" : upload_date,
                                                            "transcript" : transcription})
                                    break
                        else:
                            print(f"No SRT subtitles found for language en.")
                    else:
                        print(f"No en subtitles (manual or automatic) available.")
            except Exception as e:
                print(f"An error occurred: {e}")
    return channel_metadata
def save_to_csv(channel_metadata):
    df = pd.DataFrame(channel_metadata , columns=['video_id' ,'title' , 'view_count', 'duration' , 'upload_date' ,'transcript'])
    df.to_csv("agentic/Data/channel_caption.csv" , index=False)


if __name__ == "__main__":
    channel_url = "https://www.youtube.com/@Mrpeach-i4t/shorts"
    video_ids = get_videoids(channel_url , 2)
    channel_metadata = get_transcript(video_ids)
    save_to_csv(channel_metadata)