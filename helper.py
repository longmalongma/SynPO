import torch
from openai import OpenAI
import re
from aurora.src.xtuner.xtuner.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from transformers import DynamicCache
import time
from torch.nn.functional import softmax
import warnings
import os

os.environ['API_URL']=''
os.environ['API_KEY']=''
os.environ['API_MODEL']=''

class TensorManager:
    def __init__(self):
        self.tensor_shapes = []
    
    def concatenate_tensors(self, tensors):

        start_idx = 0
        self.tensor_shapes.clear()
        for tensor in tensors:
            self.tensor_shapes.append((start_idx, tensor.size(0)))
            start_idx += tensor.size(0)
        concatenated_tensor = torch.cat(tensors, dim=0)
        return concatenated_tensor.contiguous()
    
    def split_tensor(self, tensor):

        tensors = []
        for start_idx, size_0 in self.tensor_shapes:
            end_idx = start_idx + size_0
            original_tensor = tensor[start_idx:end_idx]
            tensors.append(original_tensor)
        return tensors
    
    
def extract_score(text):
    try:
        match = re.search('[\'\"]?score[\'\"]?:\s*(-?\d+\.?\d*)', text)
        if match:
            if '.' in match.group(1):
                return float(match.group(1))
            else:
                return int(match.group(1))
        else:
            return None
    except:
        return None
    
def extract_acc(text):
    try:
        match = re.search("[\'\"]?pred[\'\"]?:\s*'([^']*)'", text)
        if match:
            return 1 if match.group(1) == 'yes' else 0
        else:
            return 0
    except:
        return 0

def call_api_vdd(idx,question,answer,pred):
    
    data=dict()
    data['idx']=idx
    data['question']=question
    data['answer']=answer
    data['pred']=pred
    
    try:
        messages = [
            {
                "role": "system",
                "content": "You are an intelligent chatbot designed for evaluating the detail orientation of generative outputs for video-based question-answer pairs. "
                "Your task is to compare the predicted answer with the correct answer and determine its level of detail, considering both completeness and specificity. Here's how you can accomplish the task:"
                "------"
                "##INSTRUCTIONS: "
                "- Check if the predicted answer covers all major points from the video. The response should not leave out any key aspects.\n"
                "- Evaluate whether the predicted answer includes specific details rather than just generic points. It should provide comprehensive information that is tied to specific elements of the video.\n"
                "- Consider synonyms or paraphrases as valid matches.\n"
                "- Provide a single evaluation score that reflects the level of detail orientation of the prediction, considering both completeness and specificity.",
            },
            {
                "role": "user",
                "content": "Provide your evaluation only as a detail orientation score where the detail orientation score is an integer value between 0 and 5, with 5 indicating the highest level of detail orientation. "
                "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the detail orientation score in INTEGER, not STRING."
                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                "For example, your response should look like this: {'score': 4}\n"
                "Please evaluate the following video-based question-answer pair:\n\n"
                f"Question: {question}\n"
                f"Correct Answer: {answer}\n"
                f"Predicted Answer: {pred}\n\n",
            },
        ]
        
        client = OpenAI(api_key=os.getenv('API_KEY'), base_url=os.getenv('API_URL'))

        response = client.chat.completions.create(
            model=os.getenv('API_MODEL'),
            messages=messages,
            temperature=0,
            max_tokens=64,
            stream=False
        )
        
        data['api_content']=response.choices[0].message.content
        data['score']=extract_score(response.choices[0].message.content)
    except:
        data['api_content']=None
        data['score']=None
    
    return data

def prepare_data_for_multimodel(input_ids,pixel_values,llm):
    new_inputs_embeds = []
    
    for batch_idx, cur_input_ids in enumerate(input_ids):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            raise ValueError('num_images == 0')
        
        image_token_indices = [-1] + torch.where(
        cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
            cur_input_ids.shape[0]
        ]
        cur_input_ids_noim = []

        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] +
                                                    1:image_token_indices[i +
                                                                            1]])
        split_sizes = [x.shape[0] for x in cur_input_ids_noim]
        cur_inputs_embeds = llm.get_input_embeddings()(
            torch.cat(cur_input_ids_noim))
        
        cur_inputs_embeds_no_im = torch.split(
            cur_inputs_embeds, split_sizes, dim=0)
        
        cur_new_inputs_embeds = []

        for i in range(num_images + 1):
            cur_new_inputs_embeds.append(cur_inputs_embeds_no_im[i])
            if i < num_images:
                cur_pixel_values = pixel_values[batch_idx][i]
                cur_new_inputs_embeds.append(cur_pixel_values)

        cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds)

        new_inputs_embeds.append(cur_new_inputs_embeds)
        
    max_len = max(x.shape[0] for x in new_inputs_embeds)
    batch_size = len(new_inputs_embeds)

    new_inputs_embeds_padded = []
    attention_mask = torch.zeros((batch_size, max_len),
                                dtype=torch.bool,
                                device=input_ids[0].device)

    for i, cur_new_embed in enumerate(new_inputs_embeds):
        cur_len = cur_new_embed.shape[0]

        new_inputs_embeds_padded.append(
            torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device),
                    cur_new_embed),
            dim=0))
        
        attention_mask[i, max_len - cur_len:] = True
        
    new_inputs_embeds = torch.stack(new_inputs_embeds_padded, dim=0).contiguous()
    
    return {
        'inputs_embeds':new_inputs_embeds,
        'attention_mask':attention_mask
    }
    
    
def prepare_data_for_training_multimodel(input_ids,pixel_values,answer_ids,llm,max_length=2560):
    new_inputs_embeds = []
    new_labels = []

    for batch_idx, (cur_input_ids,cur_answer_ids) in enumerate(zip(input_ids,answer_ids)):
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            raise ValueError('num_images == 0')
        
        image_token_indices = [-1] + torch.where(
        cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [
            cur_input_ids.shape[0]
        ]
        cur_input_ids_noim = []

        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] +
                                                    1:image_token_indices[i +
                                                                            1]])
        split_sizes = [x.shape[0] for x in cur_input_ids_noim]
        cur_inputs_embeds = llm.get_input_embeddings()(
            torch.cat(cur_input_ids_noim))
        
        cur_inputs_embeds_no_im = torch.split(
            cur_inputs_embeds, split_sizes, dim=0)
        
        cur_new_inputs_embeds = []

        for i in range(num_images + 1):
            cur_new_inputs_embeds.append(cur_inputs_embeds_no_im[i])
            if i < num_images:
                cur_pixel_values = pixel_values[batch_idx][i]
                cur_new_inputs_embeds.append(cur_pixel_values)

        cur_new_inputs_embeds = torch.cat(cur_new_inputs_embeds)
        input_ids_len=cur_new_inputs_embeds.shape[0]
        cur_answer_ids=torch.tensor(cur_answer_ids,dtype=torch.long,device=cur_new_inputs_embeds.device)
        
        cur_new_answer_embeds=llm.get_input_embeddings()(cur_answer_ids)
        cur_new_inputs_embeds=torch.cat((cur_new_inputs_embeds,cur_new_answer_embeds),dim=0)
        
        cur_new_labels=torch.full((input_ids_len,),IGNORE_INDEX,
                                dtype=torch.long,device=cur_new_inputs_embeds.device)
        cur_new_labels=torch.cat((cur_new_labels,cur_answer_ids),dim=0)

        new_inputs_embeds.append(cur_new_inputs_embeds)
        new_labels.append(cur_new_labels)
        

    batch_size = len(new_inputs_embeds)
    
    lengths = [min(x.shape[0], max_length) for x in new_inputs_embeds]
    max_len = min(max(lengths), max_length)

    new_inputs_embeds_padded = []
    new_labels_padded = []

    attention_mask = torch.zeros((batch_size, max_len),
                                 dtype=torch.bool,
                                 device=new_inputs_embeds[0].device)

    for i, (cur_new_embed, cur_new_label) in enumerate(zip(new_inputs_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]

        if cur_len > max_length:
            cur_new_embed = cur_new_embed[:max_length]
            cur_new_label = cur_new_label[:max_length]
            cur_len = max_length

        padding_len = max_len - cur_len
        new_inputs_embeds_padded.append(
            torch.cat((
                torch.zeros((padding_len, cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device),
                cur_new_embed
            ), dim=0)
        )

        new_labels_padded.append(
            torch.cat((
                torch.full((padding_len,), IGNORE_INDEX,
                           dtype=torch.long,
                           device=cur_new_embed.device),
                cur_new_label
            ), dim=0)
        )

        attention_mask[i, padding_len:] = True

    new_inputs_embeds = torch.stack(new_inputs_embeds_padded, dim=0).contiguous()
    new_labels = torch.stack(new_labels_padded, dim=0)
    
    return {
        'inputs_embeds':new_inputs_embeds,
        'attention_mask':attention_mask,
        'labels':new_labels
    }
        
        
def sampling(logits, top_k=32, top_p=0.9, temperature=0.6, eos_token_id=2):

    batch_size, vocab_size = logits.shape
    
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logits = torch.where(torch.isnan(logits) | torch.isinf(logits), torch.tensor(float('-inf'), dtype=logits.dtype, device=logits.device), logits)

    valid_mask = ~torch.isinf(logits).all(dim=-1) 
    if not valid_mask.any():
        warnings.warn("All sequences in the batch are invalid. Returning EOS token IDs as fallback.")
        return torch.full(
            (batch_size,), 
            fill_value=eos_token_id, 
            dtype=torch.long, 
            device=logits.device
        )

    valid_indices = torch.where(valid_mask)[0]
    invalid_indices = torch.where(~valid_mask)[0]

    sampled_tokens = torch.full(
        (batch_size,), 
        fill_value=eos_token_id, 
        dtype=torch.long, 
        device=logits.device
    )

    if valid_indices.numel() > 0:
        valid_logits = logits[valid_indices]  
        valid_logits /= temperature
        
        probs = softmax(valid_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
        top_k_probs /= top_k_probs.sum(dim=-1, keepdim=True)
        
        sorted_probs, sorted_indices = torch.sort(top_k_probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        mask = cumulative_probs > top_p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False
        sorted_probs[mask] = 0.0
        
        sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
        
        sampled_indices = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)

        valid_sampled_tokens = top_k_indices.gather(-1, sorted_indices.gather(-1, sampled_indices.unsqueeze(-1))).squeeze(-1)
        
        sampled_tokens[valid_indices] = valid_sampled_tokens

    return sampled_tokens

def generate(model,inputs_embeds,attention_mask,max_new_tokens=1024,do_sample=False):
    with torch.inference_mode():
        model.eval()
        past_key_values = DynamicCache()
        
        outputs=model(inputs_embeds=inputs_embeds,attention_mask=attention_mask,
                    past_key_values=past_key_values,use_cache=True)
        
        if do_sample:
            next_tokens = sampling(outputs.logits[:, -1, :]).unsqueeze(-1)
        else:
            next_tokens = outputs.logits[:, -1, :].argmax(-1).unsqueeze(-1)
        next_tokens = sampling(outputs.logits[:, -1, :]).unsqueeze(-1)
        generated_sequences=next_tokens
        check_end=[0]*attention_mask.shape[0]
        
        for i in range(1,max_new_tokens):
            new_attention_mask=torch.full((attention_mask.shape[0],1),True,dtype=torch.bool,device=attention_mask.device)
            attention_mask=torch.cat([attention_mask,new_attention_mask],dim=-1)

            outputs=model(input_ids=next_tokens,attention_mask=attention_mask,
                    past_key_values=past_key_values,use_cache=True)
            
            if do_sample:
                next_tokens = sampling(outputs.logits[:, -1, :])
            else:
                next_tokens = outputs.logits[:, -1, :].argmax(-1)
            
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


def generate_cd(model,inputs_embeds,attention_mask,
                inputs_embeds_cd,attention_mask_cd,
                max_new_tokens=1024,cd_alpha=0.5,do_sample=False):
    
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
            next_tokens = sampling(cd_logits).unsqueeze(-1)
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
                next_tokens = sampling(cd_logits)
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
 