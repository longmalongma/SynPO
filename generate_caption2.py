import argparse
import os

parser = argparse.ArgumentParser()
args = parser.parse_args()


parser = argparse.ArgumentParser(description="Process video dataset and generate captions")

parser.add_argument('--dir_path', type=str, required=True,
                    help='Directory path containing the video files')
parser.add_argument('--pretrained_pth', type=str, required=True,
                    help='Path to the pretrained model weights')
parser.add_argument('--image_processor_path', type=str, required=True,
                    help='Path to the image processor model')
parser.add_argument('--dataset_path', type=str, required=True,
                    help='Path to the preprocessed dataset directory')
parser.add_argument('--output_file', type=str, required=True,
                    help='Path to save the output JSONL file')

parser.add_argument('--token_kept_ratio', type=float, default=0.1,
                    help='Ratio of tokens to keep during token compression')
parser.add_argument('--max_frm', type=int, default=16,
                    help='Maximum number of frames to sample from a video')
parser.add_argument('--min_frm', type=int, default=8,
                    help='Minimum number of frames to sample from a video')
parser.add_argument('--fps_frm', type=float, default=0.5,
                    help='FPS ratio for frame sampling')
parser.add_argument('--sampled_frm', type=int, default=8,
                    help='Number of frames to be sampled uniformly')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size for processing')


# python generate_caption2.py \
#   --dir_path /your/path/to/video_dir \
#   --pretrained_pth /your/path/to/pretrained_model \
#   --image_processor_path /your/path/to/image_processor \
#   --dataset_path /your/path/to/dataset \
#   --output_file /your/path/to/short_generation.jsonl \
#   --batch_size 16

args = parser.parse_args()

dir_path = args.dir_path
pretrained_pth = args.pretrained_pth
image_processor_path = args.image_processor_path
dataset_path = args.dataset_path
output_file = args.output_file

token_kept_ratio = args.token_kept_ratio
max_frm = args.max_frm
min_frm = args.min_frm
fps_frm = args.fps_frm
sampled_frm = args.sampled_frm
batch_size = args.batch_size


import os.path as osp
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor 
import json
import os

from aurora.src.xtuner.xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, PROMPT_TEMPLATE
from aurora.src.xtuner.xtuner.model.aurora_v import AuroraEncoder, AuroraModel
from aurora.src.xtuner.xtuner.model.utils import prepare_inputs_labels_for_multimodal



pretrained_vit = osp.join(pretrained_pth, "visual_encoder")
projector_path = osp.join(pretrained_pth, "projector")

auroracap = AuroraModel(
    llm=AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=pretrained_pth,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    ),
    visual_encoder=AuroraEncoder.from_pretrained(
        pretrained_model_name_or_path=pretrained_vit,
        torch_dtype=torch.float16,
    ),
).cuda()

auroracap.llm.config.output_attentions=False
auroracap.config._attn_implementation = "sdpa"
auroracap.projector = AutoModel.from_pretrained(projector_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
auroracap.visual_encoder.reset_tome_r(token_kept_ratio)

image_processor = CLIPImageProcessor.from_pretrained(
    pretrained_model_name_or_path=image_processor_path, 
    trust_remote_code=True,
    size=378,
    crop_size=378,
)
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=pretrained_pth,
    trust_remote_code=True,
    padding_side='right',
)

from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
from dataloader import process_text,find_video_path,record_video_length_packet,record_video_length_stream
from tqdm import tqdm
from torch.nn.attention import SDPBackend, sdpa_kernel
from helper import TensorManager,call_api_vdd,prepare_data_for_multimodel,sampling,extract_score
from concurrent.futures import ThreadPoolExecutor, as_completed
import datasets
from transformers import DynamicCache
import time
from torch.nn.functional import softmax
from openai import OpenAI
import av


dataset=datasets.load_from_disk(dataset_path)
print(len(dataset))


def video_process(video_path,frame_range, min_frm=8,max_frm=32,fps_frm=1,sampled_frm=None):
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
            
            indices = np.linspace(frame_range[0], frame_range[1], sampled_frm, dtype=int)
            
            frames = []
            for frame in container.decode(video=0):
                frames.append(frame)

            frames=[frames[i] for i in indices]
        except:
            frames,total_duration = record_video_length_packet(container)
            total_frames = len(frames)
            num_frm = int(total_duration * fps_frm)
            
            if sampled_frm is None:
                sampled_frm = min(max_frm, num_frm)
                sampled_frm = max(min_frm, sampled_frm)
                sampled_frm = min(total_frames, sampled_frm)

            indices = np.linspace(frame_range[0], frame_range[1], sampled_frm, dtype=int)

            frames = [frames[i] for i in indices]
    else:
        frames,total_duration = record_video_length_packet(container)
        total_frames = len(frames)
        num_frm = int(total_duration * fps_frm)
        
        if sampled_frm is None:
            sampled_frm = min(max_frm, num_frm)
            sampled_frm = max(min_frm, sampled_frm)
            sampled_frm = min(total_frames, sampled_frm)

        indices = np.linspace(frame_range[0], frame_range[1], sampled_frm, dtype=int)

        frames = [frames[i] for i in indices]
        
    container.close()
        
    return np.stack([x.to_ndarray(format="rgb24") for x in frames]),total_duration


class DataCollator:
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
        data['durations']=[]
        data['video_name']=[]
        data['frame_range']=[]
        data['idx']=[]
        data['idy']=[]
        
        for example in batch:
            
            video_name, answer=example['video_name'],example['answer']
            question=("Describe in detail what is happening in the video, including the subject matter, "
                    "the setting, and possible character activities. "
                    "However, remain cautious and refrain from stating anything you are not certain about.")
            
            vedio_path=find_video_path(video_name, self.dir_path)
            data['question'].append(question.strip())
            data['answer'].append(answer.strip())
            data['video_name'].append(video_name)
            
            video_frames,duration=self.video_process(vedio_path,example['frame_range'],self.min_frm,self.max_frm,self.fps_frm,self.sampled_frm)
            data['durations'].append(duration)
            data['frame_range'].append(example['frame_range'])
            
            image_tensor = self.image_processor(video_frames, return_tensors='pt')['pixel_values']
            image_tensor = [_image.to(dtype=torch.float16) for _image in image_tensor]
            data["pixel_values"].append(torch.stack(image_tensor))
            
            image_tokens = [DEFAULT_IMAGE_TOKEN] * len(video_frames)
            image_tokens = " ".join(image_tokens)
            
            text_input = image_tokens + "\n" + question+'\n'
            prompt_text = PROMPT_TEMPLATE.vicuna["INSTRUCTION"].format(input=text_input, round=1)
            data["input_ids"].append(self.process_text(prompt_text, self.tokenizer))
            
            data['idx'].append(example['idx'])
            data['idy'].append(example['idy'])
            
        tm=TensorManager()
        data["pixel_values"]=tm.concatenate_tensors(data["pixel_values"])
        data['tm']=tm
        
        return data


        
datacollator=DataCollator(tokenizer,image_processor,dir_path,
                        video_process=video_process,process_text=process_text,
                        max_frm=max_frm,min_frm=min_frm,fps_frm=fps_frm,sampled_frm=sampled_frm)
dataloader=DataLoader(dataset,collate_fn=datacollator,num_workers=12,persistent_workers=True,batch_size=batch_size,shuffle=False,drop_last=False)

informations=[]

with torch.inference_mode():
    for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        
        pixel_values=batch["pixel_values"].cuda(non_blocking=True)
        
        pixel_values = auroracap(pixel_values, mode="inference")
        pixel_values=batch['tm'].split_tensor(pixel_values)
        
        input_ids=[x.cuda(non_blocking=True) for x in batch['input_ids']]
        data=prepare_data_for_multimodel(input_ids=input_ids,pixel_values=pixel_values,llm=auroracap.llm)
        
        outputs = auroracap.llm.generate(
            inputs_embeds=data['inputs_embeds'],
            attention_mask=data['attention_mask'],
            do_sample=False,
            max_new_tokens=800
        )
        text_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for idx,idy,pred,video_name,frame_range in zip(batch['idx'],batch['idy'],text_outputs,batch['video_name'],batch['frame_range']):
            informations.append(
                {'idx':idx,
                'idy':idy,
                'past_pred':pred,
                'video_name':video_name,
                'frame_range':frame_range
                }
            )
        
        if (i+1)%8==0:
            torch.cuda.empty_cache()
            

class DataCollatorRetrospective:
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
        data['pixel_values']=[]
        data['input_ids']=[]
        data['idx']=[]
        data['idy']=[]
        
        for example in batch:

            vedio_path=find_video_path(example['video_name'], self.dir_path)
            
            video_frames,duration=self.video_process(vedio_path,example['frame_range'],self.min_frm,self.max_frm,self.fps_frm,self.sampled_frm)
            
            image_tensor = self.image_processor(video_frames, return_tensors='pt')['pixel_values']
            image_tensor = [_image.to(dtype=torch.float16) for _image in image_tensor]
            data["pixel_values"].append(torch.stack(image_tensor))
            
            image_tokens = [DEFAULT_IMAGE_TOKEN] * len(video_frames)
            image_tokens = " ".join(image_tokens)
            retrospective_prompt=("Given the provided video and the previously predicted description of the video, "
            "your task is to generate an enhanced description of the video clip. "
            "The generated description should provide a comprehensive understanding of the video's content while forming a coherent story.\n"
            "Note that the previous description might include irrelevant or inappropriate words. "
            "Thus, you don't have to include all the contents in the previous description. "
            "The focus is on generating a new description with improved accuracy and richer details, "
            "while incorporating more temporal information, such as the activities of animals or humans, "
            "changes in the scene, and other sequential elements. "
            "However, remain cautious and refrain from stating anything you are not certain about.\n"
            f"Previous description:\n{example['past_pred'].strip()}\n"
            "Now, generate the improved description below.\nImproved description: "
            )
            
            text_input = image_tokens + "\n" + retrospective_prompt+'\n'
            prompt_text = PROMPT_TEMPLATE.vicuna["INSTRUCTION"].format(input=text_input, round=1)
            data["input_ids"].append(self.process_text(prompt_text, self.tokenizer))

            data['idx'].append(example['idx'])
            data['idy'].append(example['idy'])
            
        tm=TensorManager()
        data["pixel_values"]=tm.concatenate_tensors(data["pixel_values"])
        data['tm']=tm
        
        return data
    
    
retrospective_dataset=datasets.Dataset.from_list(informations)

datacollator=DataCollatorRetrospective(tokenizer,image_processor,dir_path,
                        video_process=video_process,process_text=process_text,
                        max_frm=max_frm,min_frm=min_frm,fps_frm=fps_frm,sampled_frm=sampled_frm)
dataloader=DataLoader(retrospective_dataset,collate_fn=datacollator,num_workers=12,persistent_workers=True,batch_size=batch_size,shuffle=False,drop_last=False)

informations=[]

with torch.inference_mode():
    for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        
        pixel_values=batch["pixel_values"].cuda(non_blocking=True)
        
        pixel_values = auroracap(pixel_values, mode="inference")
        pixel_values=batch['tm'].split_tensor(pixel_values)
        
        input_ids=[x.cuda(non_blocking=True) for x in batch['input_ids']]
        data=prepare_data_for_multimodel(input_ids=input_ids,pixel_values=pixel_values,llm=auroracap.llm)
        
        output = auroracap.llm.generate(
            inputs_embeds=data['inputs_embeds'],
            attention_mask=data['attention_mask'],
            do_sample=False,
            max_new_tokens=800
        )
        text_outputs = tokenizer.batch_decode(output, skip_special_tokens=True)
        
        for idx,idy,pred in zip(batch['idx'],batch['idy'],text_outputs):
            informations.append(
                {'idx':idx,
                'idy':idy,
                'pred':pred
                }
            )
        
        if (i+1)%16==0:
            torch.cuda.empty_cache()
            
            
combined_informations=[{'idx':idx,'content':[]} for idx in range(1000000)]

for example in informations:
    idx,idy,pred=example['idx'],example['idy'],example['pred']
    combined_informations[idx]['content'].append(
        {'idy':idy,'pred':pred}
    )
    
combined_informations=[x for x in combined_informations if len(x['content'])!=0]

for i in range(len(combined_informations)):
    information=combined_informations[i]
    content=sorted(information['content'],key= lambda x:x['idy'])
    combined_informations[i]['short_answers']=[]
    
    for x in content:
        combined_informations[i]['short_answers'].append(x['pred'])
    del combined_informations[i]['content']
    
    

with open(output_file, 'w', encoding='utf-8') as file:
    for item in combined_informations:
        json.dump(item, file, ensure_ascii=False)
        file.write('\n')