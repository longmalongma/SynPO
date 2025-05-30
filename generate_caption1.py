from tqdm import tqdm
import cv2
import os, re, json, argparse
from tqdm import tqdm
import av
import numpy as np
from dataloader import record_video_length_packet,record_video_length_stream
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dir_path', type=str, required=True,
                    help='Directory path containing the video files')

parser.add_argument('--output_dir', type=str, required=True,
                    help='Directory path to save the output cache files')

# python generate_caption1.py --dir_path /your/path/to/video_dir --output_dir /your/path/to/output_dir

args = parser.parse_args()

dir_path = args.dir_path
output_dir = args.output_dir

def cutscene_detection(video_path):
    scene_list=[]
    
    container = av.open(video_path)
    if "webm" not in video_path and "mkv" not in video_path:

        try:
            stream = container.streams.video[0]
            total_duration = float(stream.duration * stream.time_base)
            total_frames = stream.frames
            
        except:
            container.streams.video[0].thread_type = "AUTO"
            frames,total_duration = record_video_length_packet(container)
            total_frames = len(frames)
    else:
        container.streams.video[0].thread_type = "AUTO"
        frames,total_duration = record_video_length_packet(container)
        total_frames = len(frames) 
    container.close()
            
    num_segments=int(total_duration//10) +1
    num_segments=max(2,num_segments)
    num_segments=min(num_segments,16)
    frames_per_segment = total_frames // num_segments
    
    for j in range(num_segments):
        segment_start = j * frames_per_segment
        segment_end = (j + 1) * frames_per_segment if j < num_segments - 1 else total_frames-1
        scene_list.append([segment_start,segment_end])
        
        
    return scene_list


import os
from datasets import Dataset
from dataloader import find_video_path
from tqdm import tqdm 
import datasets



def process_videos_in_directory(dir_path):

    video_data = []
    
    video_files = [file_name for file_name in os.listdir(dir_path) if file_name.endswith('.mp4')]
    
    for file_name in tqdm(video_files, desc="Processing Videos", unit="video"):
        video_name = os.path.splitext(file_name)[0] 
        video_path = os.path.join(dir_path, file_name)
        
        frame_range = cutscene_detection(video_path)
        
        video_data.append({
            'video_name': video_name,
            'frame_range': frame_range,
            'question':"Describe in detail what is happening in the video, including the subject matter, the setting, and possible character activities.",
            'answer':''
        })
    
    return video_data

video_data = process_videos_in_directory(dir_path)

dataset = Dataset.from_list(video_data)

data_list=[]

for i,(video_name,question,answer,frame_range) in enumerate(zip(dataset['video_name'],dataset['question'],dataset['answer'],dataset['frame_range'])):
    for j,(start_frame,end_frame) in enumerate(frame_range):
        data_dict={}
        
        data_dict['video_name']=video_name
        data_dict['question']=question
        data_dict['answer']=answer
        data_dict['frame_range']=[start_frame,end_frame]
        data_dict['idx']=i
        data_dict['idy']=j
        data_list.append(data_dict)
        
dataset=datasets.Dataset.from_list(data_list)
dataset.save_to_disk(output_dir)