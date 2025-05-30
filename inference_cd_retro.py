import os
import os.path as osp
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor 
from peft import PeftModel
import json

from aurora.src.xtuner.xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, PROMPT_TEMPLATE
from aurora.src.xtuner.xtuner.model.aurora_v import AuroraEncoder, AuroraModel
from aurora.src.xtuner.xtuner.model.utils import prepare_inputs_labels_for_multimodal
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--dir_path', type=str, required=True,
                    help='Directory containing the video files')
parser.add_argument('--output_file', type=str, required=True,
                    help='Path to save the output JSONL file')
parser.add_argument('--pretrained_pth', type=str, required=True,
                    help='Path to the pretrained model weights')
parser.add_argument('--image_processor_path', type=str, required=True,
                    help='Path to the image processor model')
parser.add_argument('--dataset_dir', type=str, required=True,
                    help='Directory containing the dataset annotations')

parser.add_argument('--token_kept_ratio', type=float, default=0.1,
                    help='Ratio of tokens to keep during token compression')
parser.add_argument('--max_frm', type=int, default=16,
                    help='Maximum number of frames to sample from a video')
parser.add_argument('--min_frm', type=int, default=8,
                    help='Minimum number of frames to sample from a video')
parser.add_argument('--fps_frm', type=float, default=0.5,
                    help='FPS ratio for frame sampling')
parser.add_argument('--sampled_frm', type=int, nargs='?', const=None, default=None,
                    help='Number of frames to be sampled uniformly')
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size for processing')

args = parser.parse_args()

# python inference_cd_retro.py \
#   --dir_path /your/path/to/video \
#   --output_file /your/path/to/output.jsonl \
#   --pretrained_pth /your/path/to/pretrained_model \
#   --image_processor_path /your/path/to/image_processor \
#   --dataset_dir /your/path/to/dataset \
#   --batch_size 8

dir_path=args.dir_path
output_file=args.output_file
pretrained_pth=args.pretrained_pth
image_processor_path=args.image_processor_path
dataset_dir=args.dataset_dir

batch_size=args.batch_size
token_kept_ratio=args.token_kept_ratio
max_frm=args.max_frm
min_frm=args.min_frm
fps_frm=args.fps_frm
sampled_frm=args.sampled_frm

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
)

auroracap.llm.config.output_attentions=False
auroracap.config._attn_implementation = "sdpa"
auroracap.projector = AutoModel.from_pretrained(projector_path, torch_dtype=torch.float16, trust_remote_code=True)
auroracap.visual_encoder.reset_tome_r(token_kept_ratio)

auroracap=auroracap.cuda()
auroracap.visual_encoder.eval()
auroracap.projector.eval()
auroracap.llm.eval()

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
import av
import numpy as np
from dataloader import DataCollatorVdd,video_process,process_text,find_video_path,record_video_length_packet,record_video_length_stream
from tqdm import tqdm
from torch.nn.attention import SDPBackend, sdpa_kernel
from helper import TensorManager,call_api_vdd,prepare_data_for_multimodel,extract_score
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import datasets
from transformers import DynamicCache
from helper import sampling

def video_process_cd(video_path, min_frm=8,max_frm=32,fps_frm=1,sampled_frm=None,frm_cd=4):
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
            
            frm_cd=(sampled_frm+2)//3
            indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
            indices_cd = np.linspace(0, total_frames - 1, frm_cd, dtype=int)
            
            frames = []
            for frame in container.decode(video=0):
                frames.append(frame)

            frames_cd=[frames[i] for i in indices_cd]
            frames=[frames[i] for i in indices]
        except:
            frames,total_duration = record_video_length_packet(container)
            total_frames = len(frames)
            num_frm = int(total_duration * fps_frm)
            
            if sampled_frm is None:
                sampled_frm = min(max_frm, num_frm)
                sampled_frm = max(min_frm, sampled_frm)
                sampled_frm = min(total_frames, sampled_frm)

            frm_cd=(sampled_frm+2)//3
            indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
            indices_cd = np.linspace(0, total_frames - 1, frm_cd, dtype=int)

            frames_cd=[frames[i] for i in indices_cd]
            frames = [frames[i] for i in indices]
    else:
        frames,total_duration = record_video_length_packet(container)
        total_frames = len(frames)
        num_frm = int(total_duration * fps_frm)
        
        if sampled_frm is None:
            sampled_frm = min(max_frm, num_frm)
            sampled_frm = max(min_frm, sampled_frm)
            sampled_frm = min(total_frames, sampled_frm)

        frm_cd=(sampled_frm+2)//3
        indices = np.linspace(0, total_frames - 1, sampled_frm, dtype=int)
        indices_cd = np.linspace(0, total_frames - 1, frm_cd, dtype=int)

        frames_cd=[frames[i] for i in indices_cd]
        frames = [frames[i] for i in indices]
        
    container.close()
        
    return np.stack([x.to_ndarray(format="rgb24") for x in frames]),np.stack([x.to_ndarray(format="rgb24") for x in frames_cd]),total_duration

class DataCollator_cd:
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
        data['pixel_values_cd']=[]
        data['input_ids_cd']=[]
        data['video_name']=[]
        
        for example in batch:
            
            video_name, answer=example['video_name'],example['answer']
            question=example['question']
            vedio_path=find_video_path(video_name, self.dir_path)
            
            data['video_name'].append(video_name)
            data['question'].append(question.strip())
            data['answer'].append(answer.strip())
            
            video_frames,video_frames_cd,duration=self.video_process(vedio_path,self.min_frm,self.max_frm,self.fps_frm,self.sampled_frm)
            data['durations'].append(duration)
            
            image_tensor = self.image_processor(video_frames, return_tensors='pt')['pixel_values']
            image_tensor = [_image.to(dtype=torch.float16) for _image in image_tensor]
            data["pixel_values"].append(torch.stack(image_tensor))
            
            image_tokens = [DEFAULT_IMAGE_TOKEN] * len(video_frames)
            image_tokens = " ".join(image_tokens)
            
            text_input = image_tokens + "\n" + question+'\n'
            prompt_text = PROMPT_TEMPLATE.vicuna["INSTRUCTION"].format(input=text_input, round=1)
            data["input_ids"].append(self.process_text(prompt_text, self.tokenizer))
            
            image_tensor_cd = self.image_processor(video_frames_cd, return_tensors='pt')['pixel_values']
            image_tensor_cd = [_image.to(dtype=torch.float16) for _image in image_tensor_cd]
            data["pixel_values_cd"].append(torch.stack(image_tensor_cd))
            
            image_tokens_cd = [DEFAULT_IMAGE_TOKEN] * len(video_frames_cd)
            image_tokens_cd = " ".join(image_tokens_cd)
            
            text_input_cd = image_tokens_cd + "\n" + question+'\n'
            prompt_text_cd = PROMPT_TEMPLATE.vicuna["INSTRUCTION"].format(input=text_input_cd, round=1)
            data["input_ids_cd"].append(self.process_text(prompt_text_cd, self.tokenizer))
            
        data["pixel_values"]=data["pixel_values"]+data["pixel_values_cd"]
        tm=TensorManager()
        data["pixel_values"]=tm.concatenate_tensors(data["pixel_values"])
        data['tm']=tm
        
        return data

dataset=datasets.load_dataset(dataset_dir)

        
datacollator=DataCollator_cd(tokenizer,image_processor,dir_path,
                          video_process=video_process_cd,process_text=process_text,
                          max_frm=max_frm,min_frm=min_frm,fps_frm=fps_frm,sampled_frm=sampled_frm)
dataloader=DataLoader(dataset,collate_fn=datacollator,num_workers=12,persistent_workers=True,batch_size=batch_size,shuffle=False,drop_last=False)

informations=[]

def generate_cd(model,inputs_embeds,attention_mask,
                inputs_embeds_cd,attention_mask_cd,
                max_new_tokens=1024,cd_alpha=0.5,do_sample=False,
                top_k=32,top_p=0.9,temperature=0.8):
    
    with torch.inference_mode():
        model.eval()
        past_key_values = DynamicCache()
        past_key_values_cd = DynamicCache()
        
        outputs=model(inputs_embeds=inputs_embeds,attention_mask=attention_mask,
                    past_key_values=past_key_values,use_cache=True)
        outputs_cd=model(inputs_embeds=inputs_embeds_cd,attention_mask=attention_mask_cd,
                    past_key_values=past_key_values_cd,use_cache=True)
        
        cd_logits = (1+cd_alpha)*outputs.logits[:, -1, :] - cd_alpha*outputs_cd.logits[:, -1, :]
        
        if do_sample:
            next_tokens = sampling(cd_logits,top_k,top_p,temperature).unsqueeze(-1)
        else:
            next_tokens=cd_logits.argmax(-1).unsqueeze(-1)
            
        generated_sequences=next_tokens
        check_end=[0]*attention_mask.shape[0]
        
        for i in range(1,max_new_tokens):
            new_attention_mask=torch.full((attention_mask.shape[0],1),True,dtype=torch.bool,device=attention_mask.device)
            attention_mask=torch.cat([attention_mask,new_attention_mask],dim=-1)
            attention_mask_cd=torch.cat([attention_mask_cd,new_attention_mask],dim=-1)

            outputs=model(input_ids=next_tokens,attention_mask=attention_mask,
                    past_key_values=past_key_values,use_cache=True)
            outputs_cd=model(input_ids=next_tokens,attention_mask=attention_mask_cd,
                             past_key_values=past_key_values_cd,use_cache=True)
            
            cd_logits = (1+cd_alpha)*outputs.logits[:, -1, :] - cd_alpha*outputs_cd.logits[:, -1, :]
            
            if do_sample:
                next_tokens = sampling(cd_logits,top_k,top_p,temperature)
            else:
                next_tokens=cd_logits.argmax(-1)
            
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
        return generated_sequences


with torch.inference_mode():
    for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        
        pixel_values=batch["pixel_values"].cuda(non_blocking=True)
        
        pixel_values = auroracap(pixel_values, mode="inference")
        pixel_values=batch['tm'].split_tensor(pixel_values)
        
        half_len=len(pixel_values)//2
        pixel_values,pixel_values_cd=pixel_values[:half_len],pixel_values[half_len:]
        
        input_ids=[x.cuda(non_blocking=True) for x in batch['input_ids']]
        data=prepare_data_for_multimodel(input_ids=input_ids,pixel_values=pixel_values,llm=auroracap.llm)
        
        input_ids_cd=[x.cuda(non_blocking=True) for x in batch['input_ids_cd']]
        data_cd=prepare_data_for_multimodel(input_ids=input_ids_cd,pixel_values=pixel_values_cd,llm=auroracap.llm)
        
        outputs=generate_cd(auroracap.llm,data['inputs_embeds'],data['attention_mask'],
                            data_cd['inputs_embeds'],data_cd['attention_mask'],
                            max_new_tokens=1024,cd_alpha=0.5,do_sample=True,
                            top_p=0.9,temperature=0.5)

        text_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for j,(question,answer,pred) in enumerate(zip(batch['question'],batch['answer'],text_outputs)):

            informations.append({
                'question':question,
                'answer':answer,
                'past_pred':pred,
                'video_name':batch['video_name'][j]
                })
        
        if (i+1)%16==0:
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
        data['answer_ids']=[]
        data['durations']=[]
        data['video_name']=[]
        data['past_pred']=[]
        data['pixel_values_cd']=[]
        data['input_ids_cd']=[]
        
        for example in batch:

            video_name, question, answer=example['video_name'],example['question'],example['answer']
            vedio_path=find_video_path(video_name, self.dir_path)
            
            data['video_name'].append(video_name)
            data['question'].append(question.strip())
            data['answer'].append(answer.strip())
            
            video_frames,video_frames_cd,duration=self.video_process(vedio_path,self.min_frm,self.max_frm,self.fps_frm,self.sampled_frm)
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
            
            image_tensor_cd = self.image_processor(video_frames_cd, return_tensors='pt')['pixel_values']
            image_tensor_cd = [_image.to(dtype=torch.float16) for _image in image_tensor_cd]
            data["pixel_values_cd"].append(torch.stack(image_tensor_cd))
            
            image_tokens_cd = [DEFAULT_IMAGE_TOKEN] * len(video_frames_cd)
            image_tokens_cd = " ".join(image_tokens_cd)
            
            text_input_cd = image_tokens_cd + "\n" + retrospective_prompt+'\n'
            prompt_text_cd = PROMPT_TEMPLATE.vicuna["INSTRUCTION"].format(input=text_input_cd, round=1)
            data["input_ids_cd"].append(self.process_text(prompt_text_cd, self.tokenizer))
            
            answer_ids=self.tokenizer.encode(answer.strip(), add_special_tokens=False)+[self.tokenizer.eos_token_id]
            data['answer_ids'].append(answer_ids)
            data['past_pred'].append(example['past_pred'])
            
        data["pixel_values"]=data["pixel_values"]+data["pixel_values_cd"]
        tm=TensorManager()
        data["pixel_values"]=tm.concatenate_tensors(data["pixel_values"])
        data['tm']=tm
        
        return data


retrospective_dataset=datasets.Dataset.from_list(informations)

datacollator=DataCollatorRetrospective(tokenizer,image_processor,dir_path,
                          video_process=video_process_cd,process_text=process_text,
                          max_frm=max_frm,min_frm=min_frm,fps_frm=fps_frm,sampled_frm=sampled_frm)
dataloader=DataLoader(retrospective_dataset,collate_fn=datacollator,num_workers=12,persistent_workers=True,batch_size=batch_size,shuffle=False,drop_last=False)

informations=[]

with torch.inference_mode():
    for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        
        pixel_values=batch["pixel_values"].cuda(non_blocking=True)
        
        pixel_values = auroracap(pixel_values, mode="inference")
        pixel_values=batch['tm'].split_tensor(pixel_values)
        
        half_len=len(pixel_values)//2
        pixel_values,pixel_values_cd=pixel_values[:half_len],pixel_values[half_len:]
        
        input_ids=[x.cuda(non_blocking=True) for x in batch['input_ids']]
        data=prepare_data_for_multimodel(input_ids=input_ids,pixel_values=pixel_values,llm=auroracap.llm)
        
        input_ids_cd=[x.cuda(non_blocking=True) for x in batch['input_ids_cd']]
        data_cd=prepare_data_for_multimodel(input_ids=input_ids_cd,pixel_values=pixel_values_cd,llm=auroracap.llm)
        
        outputs=generate_cd(auroracap.llm,data['inputs_embeds'],data['attention_mask'],
                            data_cd['inputs_embeds'],data_cd['attention_mask'],
                            max_new_tokens=1024,cd_alpha=0.5,do_sample=True,
                            top_p=0.9,temperature=0.5)
        text_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for j,(question,answer,pred) in enumerate(zip(batch['question'],batch['answer'],text_outputs)):

            informations.append({
                'question':question,
                'answer':answer,
                'past_pred':batch['past_pred'][j],
                'duration':batch['durations'][j],
                'pred':pred
                })
        
        if (i+1)%16==0:
            torch.cuda.empty_cache()

with open(output_file,'w',encoding='utf-8') as f:
    for x in informations:
        f.write(json.dumps({
            "question": x['question'],
            'answer':x['answer'],
            'past_pred':x['past_pred'],
            'pred':x['pred'],
            'duration':x['duration']
        })+'\n')