import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--dir_path', type=str, required=True,
                    help='Directory path containing the video files')
parser.add_argument('--pretrained_pth', type=str, required=True,
                    help='Path to the pretrained model weights')
parser.add_argument('--image_processor_path', type=str, required=True,
                    help='Path to the image processor model')
parser.add_argument('--output_file', type=str, required=True,
                    help='Path to save the output JSONL file')

parser.add_argument('--token_kept_ratio', type=float, default=0.1,
                    help='Ratio of tokens to keep during token compression')
parser.add_argument('--generate_nums', type=int, default=10,
                    help='Number of generation outputs per input')
parser.add_argument('--max_frm', type=int, default=16,
                    help='Maximum number of frames to sample from a video')
parser.add_argument('--min_frm', type=int, default=8,
                    help='Minimum number of frames to sample from a video')
parser.add_argument('--fps_frm', type=float, default=0.5,
                    help='FPS ratio for frame sampling')
parser.add_argument('--sampled_frm', type=int, nargs='?', const=None, default=None,
                    help='Number of frames to be sampled uniformly')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size for processing')

# python generate_caption3.py \
#   --dir_path /your/path/to/video_dir \
#   --pretrained_pth /your/path/to/pretrained_model \
#   --image_processor_path /your/path/to/image_processor \
#   --output_file /your/path/to/long_generation.jsonl \
#   --batch_size 16

args = parser.parse_args()

dir_path = args.dir_path
pretrained_pth = args.pretrained_pth
image_processor_path = args.image_processor_path
output_file = args.output_file

token_kept_ratio = args.token_kept_ratio
generate_nums=args.generate_nums
max_frm = args.max_frm
min_frm = args.min_frm
fps_frm = args.fps_frm
sampled_frm = args.sampled_frm
batch_size = args.batch_size

import os.path as osp
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor 
import json

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
from dataloader import DataCollatorVdd,video_process,process_text,find_video_path,record_video_length_packet,record_video_length_stream
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

video_data = []

video_files = [file_name for file_name in os.listdir(dir_path) if file_name.endswith('.mp4') or file_name.endswith('.mov')]

for file_name in video_files:
    video_name = os.path.splitext(file_name)[0] 

    video_data.append({
        'video_name': video_name,
        'question':"Describe in detail what is happening in the video, including the subject matter, the setting, and possible character activities.",
        'answer':''
    })
dataset= datasets.Dataset.from_list(video_data)

def add_idx(example, idx):
    example['idx'] = idx  
    return example

dataset = dataset.map(add_idx, with_indices=True)
print(len(dataset))



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
        data['durations']=[]
        data['video_name']=[]
        data['idx']=[]
        
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
            
            data['idx'].append(example['idx'])
            
        tm=TensorManager()
        data["pixel_values"]=tm.concatenate_tensors(data["pixel_values"])
        data['tm']=tm
        
        return data
        
datacollator=DataCollatorVdd(tokenizer,image_processor,dir_path,
                        video_process=video_process,process_text=process_text,
                        max_frm=max_frm,min_frm=min_frm,fps_frm=fps_frm,sampled_frm=sampled_frm)
dataloader=DataLoader(dataset,collate_fn=datacollator,num_workers=12,persistent_workers=True,batch_size=batch_size,shuffle=False,drop_last=False)

informations=[]


def generate_num(model,inputs_embeds,attention_mask,
                max_new_tokens=1024,do_sample=False,
                generate_nums=8,top_k=32,top_p=0.9,temperature=0.8):
    
    with torch.inference_mode():
        model.eval()
        past_key_values = DynamicCache()
        
        outputs=model(inputs_embeds=inputs_embeds,attention_mask=attention_mask,
                    past_key_values=past_key_values,use_cache=True)
        
        logits = outputs.logits[:, -1, :]
        first_logits=logits
        generated_sequences_num=[]
        past_key_values_len=past_key_values.get_seq_length()
        
        for _ in range(generate_nums):
        
            if do_sample:
                next_tokens = sampling(first_logits,top_k,top_p,temperature).unsqueeze(-1)
            else:
                next_tokens=first_logits.argmax(-1).unsqueeze(-1)
                
            generated_sequences=next_tokens
            check_end=[0]*attention_mask.shape[0]
            
            for i in range(1,max_new_tokens):
                new_attention_mask=torch.full((attention_mask.shape[0],1),True,dtype=torch.bool,device=attention_mask.device)
                attention_mask=torch.cat([attention_mask,new_attention_mask],dim=-1)

                outputs=model(input_ids=next_tokens,attention_mask=attention_mask,
                        past_key_values=past_key_values,use_cache=True)
                
                logits = outputs.logits[:, -1, :]
                
                if do_sample:
                    next_tokens = sampling(logits,top_k,top_p,temperature)
                else:
                    next_tokens=logits.argmax(-1)
                
                for j in range(next_tokens.shape[0]):
                    if next_tokens[j].item()==model.config.eos_token_id:
                        if check_end[j]==0:
                            check_end[j]=i
                    if check_end[j]!=0:
                        next_tokens[j].fill_(model.config.eos_token_id)
                            
                next_tokens=next_tokens.unsqueeze(-1)
                generated_sequences=torch.cat([generated_sequences,next_tokens],dim=-1)
                            
                if 0 not in check_end:
                    break
                
            generated_sequences_num.append(generated_sequences)
            past_key_values.crop(past_key_values_len)
            
        return generated_sequences_num

with torch.inference_mode():
    for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        
        pixel_values=batch["pixel_values"].cuda(non_blocking=True)
        
        pixel_values = auroracap(pixel_values, mode="inference")
        pixel_values=batch['tm'].split_tensor(pixel_values)
        
        input_ids=[x.cuda(non_blocking=True) for x in batch['input_ids']]
        data=prepare_data_for_multimodel(input_ids=input_ids,pixel_values=pixel_values,llm=auroracap.llm)
        
        outputs = generate_num(
            model=auroracap.llm,
            inputs_embeds=data['inputs_embeds'],
            attention_mask=data['attention_mask'],
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            max_new_tokens=800,
            generate_nums=generate_nums
        )
        
        for k in range(generate_nums):
            text_outputs = tokenizer.batch_decode(outputs[k], skip_special_tokens=True)
            
            for j in range(len(batch['video_name'])):
                informations.append({
                    'video_name':batch['video_name'][j],
                    'question':batch['question'][j],
                    'answer':batch['answer'][j],
                    'idx':batch['idx'][j],
                    'past_pred':text_outputs[j]
                })
        
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
        data['question']=[]
        data['answer']=[]
        data['pixel_values']=[]
        data['input_ids']=[]
        data['durations']=[]
        data['video_name']=[]
        data['past_pred']=[]
        data['idx']=[]
        
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
            retrospective_prompt=("Given the provided video and the previously predicted description of the video, "
            "your task is to generate an enhanced description of the video clip. "
            "The generated description should provide a comprehensive understanding of the video's content while forming a coherent story.\n"
            "Note that the previous description might include irrelevant or inappropriate words. "
            "Thus, you don't have to include all the contents in the previous description. "
            "The focus is on generating a new description with improved accuracy and richer details, "
            "while incorporating more temporal information, such as the activities of animals or humans, "
            "changes in the scene, and other sequential elements.\n"
            f"Previous description:\n{example['past_pred'].strip()}\n"
            "Now, generate the improved description below.\nImproved description:"
            )
            
            text_input = image_tokens + "\n" + retrospective_prompt+'\n'
            prompt_text = PROMPT_TEMPLATE.vicuna["INSTRUCTION"].format(input=text_input, round=1)
            data["input_ids"].append(self.process_text(prompt_text, self.tokenizer))

            data['past_pred'].append(example['past_pred'])
            data['idx'].append(example['idx'])
            
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

def generate(model,inputs_embeds,attention_mask,do_sample=False,max_new_tokens=1024,return_perplexity=False,
                temperature=0.9,top_p=0.9,top_k=32):
    with torch.inference_mode():
        model.eval()
        past_key_values = DynamicCache()
        
        outputs=model(inputs_embeds=inputs_embeds,attention_mask=attention_mask,
                    past_key_values=past_key_values,use_cache=True)
        
        if do_sample:
            next_tokens = sampling(outputs.logits[:, -1, :],top_k=top_k,
                                   temperature=temperature,top_p=top_p,eos_token_id=model.config.eos_token_id).unsqueeze(-1)
        else:
            next_tokens = outputs.logits[:, -1, :].argmax(-1).unsqueeze(-1)
            
        if return_perplexity:
            log_probs_sum = torch.zeros(attention_mask.shape[0], dtype=torch.float32, device=attention_mask.device)
            total_tokens = 0
            
            probs = torch.softmax(outputs.logits[:, -1, :], dim=-1) 
            selected_probs = torch.gather(probs, dim=-1, index=next_tokens)  
            log_probs_sum += torch.log(selected_probs.squeeze(-1))
            total_tokens += 1
            
        generated_sequences=next_tokens
        check_end=[0]*attention_mask.shape[0]
        
        for i in range(1,max_new_tokens):
            new_attention_mask=torch.full((attention_mask.shape[0],1),True,dtype=torch.bool,device=attention_mask.device)
            attention_mask=torch.cat([attention_mask,new_attention_mask],dim=-1)

            outputs=model(input_ids=next_tokens,attention_mask=attention_mask,
                    past_key_values=past_key_values,use_cache=True)
            
            if do_sample:
                next_tokens = sampling(outputs.logits[:, -1, :],top_k=top_k,
                                       temperature=temperature,top_p=top_p,eos_token_id=model.config.eos_token_id)
            else:
                next_tokens = outputs.logits[:, -1, :].argmax(-1)
                
            if return_perplexity:
                probs = torch.softmax(outputs.logits[:, -1, :], dim=-1)  
                selected_probs = torch.gather(probs, dim=-1, index=next_tokens.unsqueeze(-1)) 
                log_probs_sum += torch.log(selected_probs.squeeze(-1)) 
                total_tokens += 1 
            
            for j in range(next_tokens.shape[0]):
                if next_tokens[j].item()==model.config.eos_token_id:
                    if check_end[j]==0:
                        check_end[j]=i
                if check_end[j]!=0:
                    next_tokens[j].fill_(model.config.eos_token_id)
                        
            next_tokens=next_tokens.unsqueeze(-1)
            generated_sequences=torch.cat([generated_sequences,next_tokens],dim=-1)
                        
            if 0 not in check_end:
                break
            
    if return_perplexity:
        return generated_sequences,torch.exp(-log_probs_sum / total_tokens)
    else:
        return generated_sequences

with torch.inference_mode():
    for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        
        pixel_values=batch["pixel_values"].cuda(non_blocking=True)
        
        pixel_values = auroracap(pixel_values, mode="inference")
        pixel_values=batch['tm'].split_tensor(pixel_values)
        
        input_ids=[x.cuda(non_blocking=True) for x in batch['input_ids']]
        data=prepare_data_for_multimodel(input_ids=input_ids,pixel_values=pixel_values,llm=auroracap.llm)
        
        output = generate(model=auroracap.llm,
            inputs_embeds=data['inputs_embeds'],
            attention_mask=data['attention_mask'],
            do_sample=True,
            max_new_tokens=800,
            temperature=0.9,
            top_p=0.95
        )
        text_outputs = tokenizer.batch_decode(output, skip_special_tokens=True)
        
        for j,(question,answer,pred) in enumerate(zip(batch['question'],batch['answer'],text_outputs)):

            informations.append({
                'video_name':batch['video_name'][j],
                'question':question,
                'answer':answer,
                'idx':batch['idx'][j],
                'duration':batch['durations'][j],
                'pred':pred
                })
        
        if (i+1)%16==0:
            torch.cuda.empty_cache()
            
def process_informations(informations):
    informations.sort(key=lambda x: x['idx'])
    
    processed_informations = []
    current_idx = None
    temp_entry = None
    
    for entry in informations:
        idx = entry['idx']
        if idx != current_idx:
            if temp_entry is not None:
                processed_informations.append(temp_entry)
            temp_entry = {
                'idx': idx,
                'video_name':entry['video_name'],
                'question': entry['question'],
                'answer': entry['answer'],
                'duration': entry['duration'],
                'long_answers': [entry['pred']]
            }
            current_idx = idx
        else:
            temp_entry['long_answers'].append(entry['pred'])
    
    if temp_entry is not None:
        processed_informations.append(temp_entry)
    
    return processed_informations

informations=process_informations(informations)
    
import json

with open(output_file, 'w', encoding='utf-8') as file:
    for item in informations:
        json.dump(item, file, ensure_ascii=False)
        file.write('\n')