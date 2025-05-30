import av
import numpy as np
import torch
from aurora.src.xtuner.xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, PROMPT_TEMPLATE
from pathlib import Path
from helper import TensorManager
import random
import cv2

def find_video_path(video_name, dir_path):
    search_path = Path(dir_path)
    
    for file_path in search_path.rglob('*'):
        if file_path.is_file() and file_path.stem == video_name:
            return str(file_path)
    
    raise ValueError(f'No suitable file for {video_name}')

def process_text(inputs, tokenizer):
    chunk_encode = []
    for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
        if idx == 0:
            cur_encode = tokenizer.encode(chunk)
        else:
            cur_encode = tokenizer.encode(chunk, add_special_tokens=False)
        chunk_encode.append(cur_encode)
    ids = []
    for idx, cur_chunk_encode in enumerate(chunk_encode):
        ids.extend(cur_chunk_encode)
        if idx != len(chunk_encode) - 1:
            ids.append(IMAGE_TOKEN_INDEX)
    ids = torch.tensor(ids)
    return ids


def _process_with_opencv(video_path, min_frm, max_frm, fps_frm, sampled_frm):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open: " + video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_duration = total_frames / float(fps)

    if sampled_frm is None:
        num_frm = int(total_duration * fps_frm)
        sampled_frm = min(max_frm, num_frm)
        sampled_frm = max(min_frm, sampled_frm)
        sampled_frm = min(total_frames, sampled_frm)

    indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
    frames = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
        else:
            for j in range(10):
                cap.set(cv2.CAP_PROP_POS_FRAMES, index+j+1)
                ret, frame = cap.read()
                if ret:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb_frame)
                    break
            else:
                raise ValueError('Cannot find specified frame')

    cap.release()
    return np.stack(frames), total_duration

def _process_with_av(video_path, min_frm, max_frm, fps_frm, sampled_frm):
    container = av.open(video_path)
    container.streams.video[0].thread_type = "AUTO"

    frames, total_duration = record_video_length_packet(container)
    total_frames = len(frames)
    num_frm = int(total_duration * fps_frm)

    if sampled_frm is None:
        sampled_frm = min(max_frm, num_frm)
        sampled_frm = max(min_frm, sampled_frm)
        sampled_frm = min(total_frames, sampled_frm)

    indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
    frames = [frames[i] for i in indices]
    frames = np.stack([x.to_ndarray(format="rgb24") for x in frames])
    container.close()

    return frames, total_duration

def video_process2(video_path, min_frm=8, max_frm=32, fps_frm=1, sampled_frm=None):
    if "webm" not in video_path and "mkv" not in video_path:
        try:
            return _process_with_opencv(video_path, min_frm, max_frm, fps_frm, sampled_frm)
        except Exception as e:
            return _process_with_av(video_path, min_frm, max_frm, fps_frm, sampled_frm)
    else:
        return _process_with_av(video_path, min_frm, max_frm, fps_frm, sampled_frm)

def record_video_length_stream(container, indices):
    frames = []
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return frames

def record_video_length_packet(container):
    start_time = None
    end_time = None
    frames = []

    for packet in container.demux(video=0):
        for frame in packet.decode():
            if start_time is None:
                start_time = frame.pts
            end_time = frame.pts
            frames.append(frame)
            
    if start_time is not None and end_time is not None:
        duration = (end_time - start_time) * container.streams.video[0].time_base
    else:
        duration = 0.0    
        
    return frames,float(duration)

def video_process(video_path, min_frm=8,max_frm=32,fps_frm=1,sampled_frm=None):
    container = av.open(video_path)
    container.streams.video[0].thread_type = "AUTO"

    if "webm" not in video_path and "mkv" not in video_path:
        try:
            stream = container.streams.video[0]
            total_duration = float(stream.duration * stream.time_base)
            total_frames = stream.frames
            num_frm = int(total_duration * fps_frm)
            
            if sampled_frm is None:
                sampled_frm = min(max_frm, num_frm)
                sampled_frm = max(min_frm, sampled_frm)
                sampled_frm = min(total_frames, sampled_frm)
            
            indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)

            frames = record_video_length_stream(container, indices)
        except:
            frames,total_duration = record_video_length_packet(container)
            total_frames = len(frames)
            num_frm = int(total_duration * fps_frm)
            
            if sampled_frm is None:
                sampled_frm = min(max_frm, num_frm)
                sampled_frm = max(min_frm, sampled_frm)
                sampled_frm = min(total_frames, sampled_frm)

            indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)

            frames = [frames[i] for i in indices]
    else:
        frames,total_duration = record_video_length_packet(container)
        total_frames = len(frames)
        num_frm = int(total_duration * fps_frm)
        
        if sampled_frm is None:
            sampled_frm = min(max_frm, num_frm)
            sampled_frm = max(min_frm, sampled_frm)
            sampled_frm = min(total_frames, sampled_frm)
        indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)

        frames = [frames[i] for i in indices]
        
    container.close()
        
    return np.stack([x.to_ndarray(format="rgb24") for x in frames]),total_duration


class DataCollatorVdd:
    def __init__(self, tokenizer, image_processor,dir_path,
                 video_process,process_text,
                 min_frm=8,max_frm=32,fps_frm=0.8,sampled_frm=None):
        
        self.tokenizer=tokenizer
        self.image_processor=image_processor
        self.dir_path=dir_path
        
        self.video_process=video_process
        self.process_text=process_text
        
        self.min_frm=min_frm
        self.max_frm=max_frm
        self.fps_frm=fps_frm
        self.sampled_frm=sampled_frm
        
    def __call__(self, batch):
        data= dict()
        data['question']=[]
        data['answer']=[]
        data['pixel_values']=[]
        data['input_ids']=[]
        data['answer_ids']=[]
        data['durations']=[]
        data['video_name']=[]
        
        for example in batch:

            video_name, question, answer=example['video_name'],example['question'],example['answer']
            vedio_path=find_video_path(video_name, self.dir_path)
            
            data['video_name'].append(video_name)
            data['question'].append(question.strip())
            data['answer'].append(answer.strip())
            
            video_frames,duration=self.video_process(vedio_path,self.min_frm,self.max_frm,self.fps_frm,self.sampled_frm)
            data['durations'].append(duration)
            
            image_tensor = self.image_processor(video_frames, return_tensors='pt')['pixel_values']
            image_tensor = [_image.to(dtype=torch.float16) for _image in image_tensor]
            data["pixel_values"].append(torch.stack(image_tensor))
            
            image_tokens = [DEFAULT_IMAGE_TOKEN] * len(video_frames)
            image_tokens = " ".join(image_tokens)
            
            text_input = image_tokens + "\n" + question+'\n'
            prompt_text = PROMPT_TEMPLATE.vicuna["INSTRUCTION"].format(input=text_input, round=1)
            data["input_ids"].append(self.process_text(prompt_text, self.tokenizer))
            
            answer_ids=self.tokenizer.encode(answer.strip(), add_special_tokens=False)+[self.tokenizer.eos_token_id]
            data['answer_ids'].append(answer_ids)
            
        tm=TensorManager()
        data["pixel_values"]=tm.concatenate_tensors(data["pixel_values"])
        data['tm']=tm
        
        return data
    
    
class DataCollatorVdc:
    def __init__(self, tokenizer, image_processor,dir_path,
                 video_process,process_text,prompts,
                 min_frm=8,max_frm=32,fps_frm=0.8,sampled_frm=None):
        
        self.tokenizer=tokenizer
        self.image_processor=image_processor
        self.dir_path=dir_path
        
        self.video_process=video_process
        self.process_text=process_text
        self.prompts=prompts
        
        self.min_frm=min_frm
        self.max_frm=max_frm
        self.fps_frm=fps_frm
        self.sampled_frm=sampled_frm
        
    def __call__(self, batch):
        data= dict()
        data['qa_list']=[]
        data['caption']=[]
        data['questions']=[]
        data['pixel_values']=[]
        data['input_ids']=[]
        data['durations']=[]
        
        for example in batch:

            video_name, qa_list, caption=example['video_name'],example['qa_list'],example['caption']
            vedio_path=find_video_path(video_name, self.dir_path)
            data['qa_list'].append(qa_list)
            data['caption'].append(caption.strip())
            
            video_frames,duration=self.video_process(vedio_path,self.min_frm,self.max_frm,self.fps_frm,self.sampled_frm)
            data['durations'].append(duration)
            
            image_tensor = self.image_processor(video_frames, return_tensors='pt')['pixel_values']
            image_tensor = [_image.to(dtype=torch.float16) for _image in image_tensor]
            data["pixel_values"].append(torch.stack(image_tensor))
            
            image_tokens = [DEFAULT_IMAGE_TOKEN] * len(video_frames)
            image_tokens = " ".join(image_tokens)
            
            question=random.choice(self.prompts)
            data['questions'].append(question)
            text_input = image_tokens + "\n" + question+'\n'
            prompt_text = PROMPT_TEMPLATE.vicuna["INSTRUCTION"].format(input=text_input, round=1)
            data["input_ids"].append(self.process_text(prompt_text, self.tokenizer))
            
        tm=TensorManager()
        data["pixel_values"]=tm.concatenate_tensors(data["pixel_values"])
        data['tm']=tm
        
        return data
    
class DataCollatorVdc2:
    def __init__(self, tokenizer, image_processor,dir_path,
                 video_process,process_text,
                 min_frm=8,max_frm=32,fps_frm=0.8,sampled_frm=None):
        
        self.tokenizer=tokenizer
        self.image_processor=image_processor
        self.dir_path=dir_path
        
        self.video_process=video_process
        self.process_text=process_text
        
        self.min_frm=min_frm
        self.max_frm=max_frm
        self.fps_frm=fps_frm
        self.sampled_frm=sampled_frm
        
    def __call__(self, batch):
        data= dict()
        data['question']=[]
        data['answer']=[]
        data['pixel_values']=[]
        data['input_ids']=[]
        data['answer_ids']=[]
        data['durations']=[]
        
        for example in batch:
            
            question='Please describe the video in detail, the more detail the better.'
            video_name, answer=example['video_name'],example['caption']
            vedio_path=find_video_path(video_name, self.dir_path)
            data['question'].append(question.strip())
            data['answer'].append(answer.strip())
            
            video_frames,duration=self.video_process(vedio_path,self.min_frm,self.max_frm,self.fps_frm,self.sampled_frm)
            data['durations'].append(duration)
            
            image_tensor = self.image_processor(video_frames, return_tensors='pt')['pixel_values']
            image_tensor = [_image.to(dtype=torch.float16) for _image in image_tensor]
            data["pixel_values"].append(torch.stack(image_tensor))
            
            image_tokens = [DEFAULT_IMAGE_TOKEN] * len(video_frames)
            image_tokens = " ".join(image_tokens)
            
            text_input = image_tokens + "\n" + question+'\n'
            prompt_text = PROMPT_TEMPLATE.vicuna["INSTRUCTION"].format(input=text_input, round=1)
            data["input_ids"].append(self.process_text(prompt_text, self.tokenizer))
            
            answer_ids=self.tokenizer.encode(answer.strip(), add_special_tokens=False)+[self.tokenizer.eos_token_id]
            data['answer_ids'].append(answer_ids)
            
        tm=TensorManager()
        data["pixel_values"]=tm.concatenate_tensors(data["pixel_values"])
        data['tm']=tm
        
        return data