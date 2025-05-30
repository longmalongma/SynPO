import os
import os.path as osp
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, CLIPImageProcessor 

from aurora.src.xtuner.xtuner.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, PROMPT_TEMPLATE
from aurora.src.xtuner.xtuner.model.aurora_v import AuroraEncoder, AuroraModel
from aurora.src.xtuner.xtuner.model.utils import prepare_inputs_labels_for_multimodal

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import time

from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
from dataloader import DataCollatorVdd,video_process,process_text,find_video_path
from tqdm import tqdm
from torch.nn.attention import SDPBackend, sdpa_kernel
from helper import TensorManager,prepare_data_for_training_multimodel
import datasets
from transformers import get_linear_schedule_with_warmup
import json
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftType
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--pretrained_pth', type=str, required=True,
                    help='Path to the pretrained model weights')
parser.add_argument('--image_processor_path', type=str, required=True,
                    help='Path to the image processor model')
parser.add_argument('--dataset_path', type=str, required=True,
                    help='Path to the dataset JSONL file')
parser.add_argument('--dir_path', type=str, required=True,
                    help='Directory path containing the video files')
parser.add_argument('--log_file', type=str, required=True,
                    help='Path to save the training log file')
parser.add_argument('--model_dir', type=str, required=True,
                    help='Directory to save the trained model')


parser.add_argument('--token_kept_ratio', type=float, default=0.1,
                    help='Ratio of tokens to keep during token compression')
parser.add_argument('--max_frm', type=int, default=16,
                    help='Maximum number of frames to sample from a video')
parser.add_argument('--min_frm', type=int, default=8,
                    help='Minimum number of frames to sample from a video')
parser.add_argument('--fps_frm', type=float, default=0.5,
                    help='FPS ratio for frame sampling (default: 0.5)')
parser.add_argument('--sampled_frm', type=int, nargs='?', const=None, default=None,
                    help='Number of frames to be sampled uniformly')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=5,
                    help='Number of training epochs')
parser.add_argument('--lr', type=float, default=5e-6,
                    help='Learning rate for optimizer')
parser.add_argument('--accumulation_steps', type=int, default=32,
                    help='Gradient accumulation steps')
parser.add_argument('--warmup_ratio', type=float, default=0.1,
                    help='Warmup ratio for learning rate scheduler')
parser.add_argument('--alpha', type=float, default=30,
                    help='Implicitly control the gap between positive rewards and negative rewards')
parser.add_argument('--beta', type=float, default=0.2,
                    help='Control the trade-off between preference ranking and retainment of both semantics and syntax')

args = parser.parse_args()

token_kept_ratio = args.token_kept_ratio
max_frm = args.max_frm
min_frm = args.min_frm
fps_frm = args.fps_frm
sampled_frm = args.sampled_frm
batch_size = args.batch_size
num_epochs = args.num_epochs
lr = args.lr
accumulation_steps = args.accumulation_steps
warmup_ratio = args.warmup_ratio


pretrained_pth = args.pretrained_pth
image_processor_path = args.image_processor_path
dataset_path = args.dataset_path
dir_path = args.dir_path
log_file = args.log_file
model_dir = args.model_dir
alpha = args.alpha
beta = args.beta

# python train_synpo_check.py \
#   --pretrained_pth /your/path/to/pretrained_model \
#   --image_processor_path /your/path/to/image_processor \
#   --dataset_path /your/path/to/dataset.jsonl \
#   --dir_path /your/path/to/video \
#   --log_file /your/path/to/output.log \
#   --model_dir /your/path/to/model \
#   --batch_size 1 \
#   --num_epochs 5 \
#   --lr 5e-6 \
#   --accumulation_steps 32 \
#   --alpha 30 \
#   --beta 0.2

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

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
auroracap.llm.train()

image_processor = CLIPImageProcessor.from_pretrained(
    pretrained_model_name_or_path=image_processor_path,
    trust_remote_code=True,
    size=378,
    crop_size=378,
)
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=pretrained_pth,
    trust_remote_code=True,
    padding_side='left',
)


class DataCollator_sharegpt4:
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
        data['pixel_values']=[]
        data['input_ids']=[]
        
        data['idx']=[]
        data['video_name']=[]
        
        data['pos_pre']=[]
        data['neg_pre']=[]
        data['pos_score']=[]
        data['neg_score']=[]
        data['pos_pre_ids']=[]
        data['neg_pre_ids']=[]
        
        for example in batch:
            
            video_name,pos_pre, neg_pre=example['video_name'],example['pos_pre'], example['neg_pre']
            
            question="Describe in detail what is happening in the video, including the subject matter, the setting, and possible character activities."
            data['question'].append(question.strip())
            
            vedio_path=find_video_path(video_name, self.dir_path)
            video_frames,duration=self.video_process(vedio_path,self.min_frm,self.max_frm,self.fps_frm,self.sampled_frm)
            
            image_tensor = self.image_processor(video_frames, return_tensors='pt')['pixel_values']
            image_tensor = [_image.to(dtype=torch.float16) for _image in image_tensor]
            data["pixel_values"].append(torch.stack(image_tensor))
            
            image_tokens = [DEFAULT_IMAGE_TOKEN] * len(video_frames)
            image_tokens = " ".join(image_tokens)
            
            text_input = image_tokens + "\n" + question+'\n'
            prompt_text = PROMPT_TEMPLATE.vicuna["INSTRUCTION"].format(input=text_input, round=1)
            data["input_ids"].append(self.process_text(prompt_text, self.tokenizer))
            
            pos_pre_ids=self.tokenizer.encode(pos_pre, add_special_tokens=False)+[self.tokenizer.eos_token_id]
            data['pos_pre_ids'].append(pos_pre_ids)
            
            neg_pre_ids=self.tokenizer.encode(neg_pre, add_special_tokens=False)+[self.tokenizer.eos_token_id]
            data['neg_pre_ids'].append(neg_pre_ids)
            
            data['idx'].append(example['idx'])
            data['video_name'].append(video_name)
            
            data['pos_pre'].append(example['pos_pre'])
            data['neg_pre'].append(neg_pre)
            
        tm=TensorManager()
        data["pixel_values"]=tm.concatenate_tensors(data["pixel_values"])
        data['tm']=tm
        
        return data
    
    
import json

dataset=[]
with open(dataset_path,'r',encoding='utf-8') as f:
    for line in f:
        data=json.loads(line)
        if (data['pos_score']-data['neg_score'])>=4:
            dataset.append(data)

print(len(dataset))
dataset=datasets.Dataset.from_list(dataset)


datacollator=DataCollator_sharegpt4(tokenizer,image_processor,dir_path,
                        video_process=video_process,process_text=process_text,
                        max_frm=max_frm,min_frm=min_frm,fps_frm=fps_frm,sampled_frm=sampled_frm)
dataloader=DataLoader(dataset,collate_fn=datacollator,num_workers=12,persistent_workers=True,batch_size=batch_size,shuffle=True,drop_last=False)


from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftType

def get_prob(logits, labels):
    logits = logits.float()
    labels = labels.to(logits.device)
    
    labels = F.pad(labels, (0, 1), value=-100)
    labels = labels[..., 1:].contiguous()
    
    mask=(labels!=-100)
    valid_labels = labels.clone()
    valid_labels[labels == -100] = 0
    
    logits = torch.gather(
        logits.log_softmax(-1), dim=2,
        index=valid_labels.unsqueeze(2)).squeeze(2)
    logits= logits * mask
    
    prob=logits.sum(-1)
    prob/=mask.sum(-1)
    
    return prob

def get_prob2(logits, labels):
    logits = logits.float()
    labels = labels.to(logits.device)
    
    labels = F.pad(labels, (0, 1), value=-100)
    labels = labels[..., 1:].contiguous()
    
    mask=(labels!=-100)
    valid_labels = labels.clone()
    valid_labels[labels == -100] = 0
    
    logits = torch.gather(
        logits.softmax(-1), dim=2,
        index=valid_labels.unsqueeze(2)).squeeze(2)
    logits= logits * mask
    
    prob=logits.sum(-1)
    prob/=mask.sum(-1)
    
    return prob

def loss_fn(pos_prob,neg_prob,pos_prob2):

    gap=torch.exp(pos_prob)-torch.exp(neg_prob)
    loss=-F.sigmoid(gap*alpha)-pos_prob2*beta

    return loss.mean()

def compute_sft_loss(logits,labels):
    logits = logits.float()
    labels = labels.to(logits.device)
    batch,seq_len,vocab_size=logits.shape
    
    labels = F.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:].contiguous()
    
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    
    loss=F.cross_entropy(logits, shift_labels,ignore_index=-100,reduction='mean')
    return loss

def compute_normalized_gradient_l2_norm(model):
    gradient_l2_norm = torch.norm(
        torch.cat([param.grad.view(-1) for param in model.parameters() if param.grad is not None])
    )
    num_grad_params = sum(
        param.grad.numel() for param in model.parameters() if param.grad is not None
    )
    normalized_gradient_l2_norm = gradient_l2_norm / num_grad_params
    
    return normalized_gradient_l2_norm


for name,param in auroracap.visual_encoder.named_parameters():
    param.requires_grad=False
for name,param in auroracap.projector.named_parameters():
    param.requires_grad=False

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,          
    r=128,                           
    lora_alpha=64,                
    lora_dropout=0.05,              
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)

auroracap.llm = get_peft_model(auroracap.llm, lora_config)
auroracap.llm.print_trainable_parameters()

optimizer = torch.optim.AdamW(auroracap.llm.parameters(), lr=lr)
num_training_steps = num_epochs * (len(dataloader)//accumulation_steps)
num_warmup_steps = int(warmup_ratio * num_training_steps)
lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

with open(log_file,'w',encoding='utf-8') as f:
    pass
batch_logs = []
step=0

for epoch in range(num_epochs):
    
    optimizer.zero_grad(set_to_none=True)
    data_per_steps=[]
    
    for i, batch in enumerate(dataloader):
        
        with torch.no_grad():
            pixel_values=batch["pixel_values"].cuda(non_blocking=True)
            pixel_values = auroracap(pixel_values, mode="inference")
            pixel_values=batch['tm'].split_tensor(pixel_values)
            
            the_batch_size=len(batch['input_ids'])
            
            input_ids=[x.cuda() for x in batch['input_ids']]
            input_ids_copy=[x.clone() for x in input_ids]
            input_ids=input_ids+input_ids_copy
            
            pixel_values_copy=[x.clone() for x in pixel_values]
            pixel_values=pixel_values+pixel_values_copy
            
            answer_ids=batch['pos_pre_ids']+batch['neg_pre_ids']
            
            data=prepare_data_for_training_multimodel(input_ids=input_ids,pixel_values=pixel_values,
                                                        answer_ids=answer_ids,llm=auroracap.llm)
            data_per_steps.append(data)
            
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):

            optimizer.zero_grad(set_to_none=True)
            count=0
            for data in data_per_steps:
                data['inputs_embeds'].requires_grad=True
                the_batch_size=len(data['attention_mask'])//2
                
                with torch.amp.autocast(dtype=torch.float16, device_type=str(input_ids[0].device)):
                    output = auroracap.llm(
                        inputs_embeds=data['inputs_embeds'][:the_batch_size],
                        attention_mask=data['attention_mask'][:the_batch_size],
                        use_cache=False
                    )
            
                pos_prob=get_prob(output.logits,data['labels'][:the_batch_size])
                pos_rewards=torch.exp(pos_prob)
                loss=-pos_rewards.mean() 
                
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    if data['inputs_embeds'].grad is not None:
                        data['inputs_embeds'].grad.zero_()
                    
                    loss = loss.detach()
                    del loss
                    del output, pos_rewards
                    torch.cuda.empty_cache()
                else:
                    count+=1
                    loss.backward()
                
            pos_l2_norm=compute_normalized_gradient_l2_norm(auroracap.llm)
            pos_l2_norm=pos_l2_norm/count if count!=0 else torch.tensor(0)
            torch.cuda.empty_cache()
            
            optimizer.zero_grad(set_to_none=True)
            count=0
            for data in data_per_steps:
                the_batch_size=len(data['attention_mask'])//2
                
                with torch.amp.autocast(dtype=torch.float16, device_type=str(input_ids[0].device)):
                    output = auroracap.llm(
                        inputs_embeds=data['inputs_embeds'][the_batch_size:],
                        attention_mask=data['attention_mask'][the_batch_size:],
                        use_cache=False
                    )
            
                neg_prob=get_prob(output.logits,data['labels'][the_batch_size:])
                neg_rewards=torch.exp(neg_prob)
                loss=neg_rewards.mean()   
                
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    if data['inputs_embeds'].grad is not None:
                        data['inputs_embeds'].grad.zero_()
                    
                    loss = loss.detach()
                    del loss
                    del output,neg_rewards
                    torch.cuda.empty_cache()
                else:
                    count+=1
                    loss.backward()
                
            neg_l2_norm=compute_normalized_gradient_l2_norm(auroracap.llm)
            neg_l2_norm=neg_l2_norm/count if count!=0 else torch.tensor(0)
            torch.cuda.empty_cache()
            
            optimizer.zero_grad(set_to_none=True)
            for data in data_per_steps:
                the_batch_size=len(data['attention_mask'])//2
                
                with torch.amp.autocast(dtype=torch.float16, device_type=str(input_ids[0].device)):
                    output = auroracap.llm(
                        inputs_embeds=data['inputs_embeds'],
                        attention_mask=data['attention_mask'],
                        use_cache=False
                    )
            
                prob_batch=get_prob(output.logits,data['labels'])
                pos_prob,neg_prob=prob_batch[:the_batch_size],prob_batch[the_batch_size:]
                
                prob_batch2=get_prob2(output.logits,data['labels'])
                pos_prob2=prob_batch2[:the_batch_size]
                
                loss=loss_fn(pos_prob,neg_prob,pos_prob2)
                sft_loss=compute_sft_loss(output.logits[:the_batch_size],data['labels'][:the_batch_size])
                
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    if data['inputs_embeds'].grad is not None:
                        data['inputs_embeds'].grad.zero_()
                    
                    loss = loss.detach()
                    del loss
                    del output, prob_batch, prob_batch2, pos_prob, neg_prob, pos_prob2,sft_loss
                    torch.cuda.empty_cache()
                else:
                
                    batch_logs.append({
                        "loss": loss.item(),
                        "sft_loss": sft_loss.item(),
                        "pos_rewards": pos_prob.item(),
                        "neg_rewards": neg_prob.item()
                    })
                    
                    loss=loss/accumulation_steps
                    loss.backward()
                    
                if len(batch_logs)%8==0:
                    torch.cuda.empty_cache()
                
            normalized_gradient_l2_norm=compute_normalized_gradient_l2_norm(auroracap.llm)
        
            step+=1
            avg_logs = {
                "step": step,
                "loss": round(sum(log["loss"] for log in batch_logs) / len(batch_logs) ,8),
                "sft_loss": round(sum(log["sft_loss"] for log in batch_logs) / len(batch_logs) ,8),
                "pos_rewards": round(sum(log["pos_rewards"] for log in batch_logs) / len(batch_logs) ,8),
                "neg_rewards": round(sum(log["neg_rewards"] for log in batch_logs) / len(batch_logs) ,8),
                "normalized_gradient_l2_norm": normalized_gradient_l2_norm.item(),
                "batch_size": len(batch_logs),
                "pos_l2_norm": pos_l2_norm.item(),
                "neg_l2_norm": neg_l2_norm.item()
            }

            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(avg_logs) + '\n')

            batch_logs.clear()
            data_per_steps.clear()
                          
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            
        if step%100==0 and step>0:
            auroracap.llm.save_pretrained(f'{model_dir}/step{step}')