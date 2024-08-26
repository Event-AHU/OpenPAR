import math
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
from config import argument_parser
parser = argument_parser()
args = parser.parse_args()
from minigpt4.models.blip2 import Blip2Base
from minigpt4.models.eva_vit import Block
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from instruction.template import instruction
from transformers import StoppingCriteria, StoppingCriteriaList
from models.cbam import CBAM
import torch.nn.init as init
from local import minigpt4_path, vicuna_7b_path, blip2_path, eva_vit_g_path

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):].to('cpu'))).item():
                return True

        return False
  
class SeqPAR2(Blip2Base):
    def __init__(self, 
        dataset,
        all_sentence,
        device,
        attr_num, 
        attributes,
        cross_layer_num=3,
        num_query=128,
        limit_words=[],
        #eva vit-g Config
        vit_model="eva_clip_g", 
        vit_model_path=eva_vit_g_path,
        img_size=224,
        drop_path_rate=0.4,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=False,
        #Q-former Config
        has_qformer=True,
        freeze_qformer=False,
        num_query_token=32,
        #LLM Config
        llama_model=vicuna_7b_path,
        max_txt_len=100, 
        end_sym='</s>', 
        #LoRA Confing 
        lora_r=32,
        lora_target_modules=['model.lm_head']+[f"model.layers.{31-k}.self_attn.q_proj" for k in range(3)] + [f"model.layers.{31-k}.self_attn.v_proj" for k in range(3)],
        lora_alpha=16,
        lora_dropout=0.05):
        super(SeqPAR2, self).__init__()
        #init eva vit-g
        self.custom_insturction = instruction[dataset]['instruction']
        self.group_details = instruction[dataset]['group_details']
        self.insert_name = instruction[dataset]['insert_name']
        self.attr_group_num = len(self.group_details)
        self.num_query_token = num_query_token
        self.attr_num = attr_num
        self.has_qformer = has_qformer
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym
        self.all_sentence=all_sentence
        
        print('Loading VIT')
        self.visual_encoder, self.cross_layers, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, cross_layer_num, vit_model_path
        )           
        
        print('Loading VIT Done')
        self.norm = nn.LayerNorm(1408)
        for name, param in self.cross_layers.named_parameters():
            param.requires_grad = True
        a = torch.sqrt(torch.tensor(0.05, dtype=torch.float32))
        self.attr_group_num_query = num_query
        self.group_embedding = nn.Parameter(torch.empty((self.attr_group_num, num_query, 1408)).uniform_(-a, a))   
        
        if self.has_qformer:
            print('Loading Q-Former')
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features
            )
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.load_from_pretrained(blip2_path)

            img_f_dim = self.Qformer.config.hidden_size
            print('Loading Q-Former Done')
        # init LLaMa    
        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False, padding_side='right')
        self.llama_tokenizer.pad_token = "$$"
        self.llama_model = LlamaForCausalLM.from_pretrained(
            llama_model,
            torch_dtype=torch.float16,
        )
        
        if lora_r>0:
            print("Using LLama With LoRA!!")
            print(lora_target_modules)
            self.llama_model = prepare_model_for_kbit_training(self.llama_model)
            loraconfig = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llama_model = get_peft_model(self.llama_model, loraconfig)
            self.llama_model.print_trainable_parameters()
            
        else:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
                
        print('Loading LLAMA Done')
        self.llama_proj = nn.Linear(
            img_f_dim, self.llama_model.config.hidden_size
        )

        print(self.llama_model.config.hidden_size)
        ckpt = torch.load(minigpt4_path, map_location="cpu")
        msg = self.load_state_dict(ckpt['model'], strict=False)
       

        self.cbam_class = nn.ModuleList([CBAM(1408, output_dim=1) for _ in range(attr_num)])
        self.cbam_llm = nn.ModuleList([CBAM(self.llama_model.config.hidden_size, output_dim=1) for _ in range(attr_num)])
        self.cbam_instance = nn.ModuleList([CBAM(1408, output_dim=self.group_details[i]) for i in range(self.attr_group_num)])
        
        self.instance_proj = nn.Linear(1408, 1408)
        self.llm_proj = nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size)
        # 使用 Kaiming 初始化
        init.kaiming_normal_(self.instance_proj.weight)
        init.constant_(self.instance_proj.bias, 0)
        init.kaiming_normal_(self.llm_proj.weight)
        init.constant_(self.llm_proj.bias, 0)
        self.instance_relu = nn.ReLU()
        self.llm_relu = nn.ReLU()
        self.bn_class = nn.BatchNorm1d(self.attr_num)
        self.bn_instance = nn.BatchNorm1d(self.attr_num)
        self.bn_llm = nn.BatchNorm1d(self.attr_num)
        print('Prompt Example \n{}'.format(self.custom_insturction))
        print("Visual Encoder dtype:", next(self.visual_encoder.parameters()).dtype)
        print("Qformer dtype:", next(self.Qformer.parameters()).dtype)
        print("Llama Model dtype:", next(self.llama_model.parameters()).dtype)
        print(f"max text length: {max_txt_len}")

        
    def encode_img(self, im):
        with self.maybe_autocast():
            device = im.device
            im_embeds = self.ln_vision(self.visual_encoder(im)).to(device)#[8, 257, 1408]
            B,L,D = im_embeds.size()
            im_embeds = im_embeds.unsqueeze(1).expand(B, self.attr_group_num, L, D)
            im_embeds = im_embeds.contiguous().view(-1, L, D)
            group_embedding = self.group_embedding.expand(B, -1, -1, -1)
            group_embedding = group_embedding.contiguous().view(-1, self.attr_group_num_query, D)
            for block in self.cross_layers:
                group_embedding = block(x=group_embedding, enc=im_embeds, is_cross=True)
            group_embedding= self.norm(group_embedding).view(B, self.attr_group_num, self.attr_group_num_query, D)

            #im atts : 对于视觉embed的掩码 在Qformer中
            im_atts = torch.ones((B,self.attr_group_num_query), dtype=torch.long).to(device)
            query_tokens = self.query_tokens.expand(B, -1, -1)
            query_outputs = []
            for gidx in range(self.attr_group_num):
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    attention_mask = None,
                    encoder_hidden_states=group_embedding[:,gidx].detach(),
                    encoder_attention_mask=im_atts,
                    return_dict=True,
                )
                query_outputs.append(query_output.last_hidden_state)
                    
            query_outputs = torch.stack(query_outputs, dim=1).view(B, self.attr_group_num*self.num_query_token, self.Qformer.config.hidden_size)
            inputs_llama = self.llama_proj(query_outputs)
            _,_,D2 = inputs_llama.size()
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(im.device)

        return group_embedding, inputs_llama.view(B, self.attr_group_num, self.num_query_token, D2), atts_llama    

    def prompt_wrap(self, img_embeds, atts_img, prompts):
        if prompts:
            emb_lists = []
            if isinstance(prompts, str):
                prompts = [prompts] * len(img_embeds)
            for each_img_embed, each_prompt in zip(img_embeds, prompts):
                cur_prompt = each_prompt
                p_list = []
                for insert_name in self.insert_name:
                    p_before, p_after = cur_prompt.split(insert_name)
                    p_list.append(p_before)
                    cur_prompt = p_after
                p_list.append(cur_prompt)
                p_embeds_list=[]
                for p in p_list :
                    p_tokens= self.llama_tokenizer(
                        p, return_tensors="pt", add_special_tokens=False).to(img_embeds.device).input_ids
                    p_embeds_list.append(self.embed_tokens(p_tokens))
                wrapped_emb = p_embeds_list[0]
                for idx in range(0, len(p_embeds_list)-1):
                    wrapped_emb = torch.cat([wrapped_emb, each_img_embed[None][:,idx], p_embeds_list[idx+1]], dim=1)
                emb_lists.append(wrapped_emb)
            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=img_embeds.device))
            wrapped_embs = pad_emb.expand(len(emb_lens), max(emb_lens), -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max(emb_lens)], dtype=torch.int, device=img_embeds.device)
            for i, emb in enumerate(emb_lists):
                wrapped_embs[i, :emb_lens[i]] = emb
                wrapped_atts[i, :emb_lens[i]] = 1
            return wrapped_embs, wrapped_atts
        else:
            return img_embeds, atts_img

    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        input_lens = []
        cat_embs = []
        cat_atts = []
        for i in range(input_embs.size(0)):
            input_len = input_atts[i].sum()
            input_lens.append(input_len)
            cat_embs.append(
                torch.cat([
                    input_embs[i][:input_len],
                    output_embs[i],
                    input_embs[i][input_len:]
                ])
            )
            cat_atts.append(
                torch.cat([
                    input_atts[i][:input_len],
                    output_atts[i],
                    input_atts[i][input_len:]
                ])
            )
        cat_embs = torch.stack(cat_embs)
        cat_atts = torch.stack(cat_atts)
        return cat_embs, cat_atts, input_lens
    
    def embed_tokens(self, token_ids):
        if hasattr(self.llama_model.base_model, 'model'): ## lora wrapped model
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds
    
    #for training
    def forward(self, im, simple_answer, random_sentences, gt_label=None):
        b_s=im.shape[0]
        # 获取im embed & ims_mask  from eva vit-g&Qformer
        vit_img_embeds, inputs_llama, atts_img = self.encode_img(im)
        #随机选取指令
        instruction = self.custom_insturction
        #将im embed 填入指令 更新后的im embed & imgae mask after llm_token_embedding
        img_embeds, atts_img = self.prompt_wrap(inputs_llama, atts_img, instruction)
        #处理真值
        self.llama_tokenizer.padding_side = "right"
        # text = [t + self.end_sym for t in answer]
        text = [t + self.end_sym for t in simple_answer]
        #真值token
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(im.device)
        
        random_maxlength=to_regress_tokens.input_ids.size(1)
        random_text = [t + self.end_sym for t in random_sentences]
        random_tokens = self.llama_tokenizer(
            random_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=random_maxlength,
            add_special_tokens=False
        ).to(im.device)     
        to_regress_embeds = self.embed_tokens(random_tokens.input_ids)
        
        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]
        IB,IL,ID=img_embeds.size()
        LB,LL,LD = to_regress_embeds.size()
        #回答这里的问题
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(img_embeds, atts_img, to_regress_embeds, to_regress_tokens.attention_mask)#这里应该使用to_regress_token还是random_tokens的attention_mask

        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, attention_mask], dim=1)
        
        #填充真值
        part_targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        labels = (
            torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                       dtype=torch.long).to(im.device).fill_(-100)
        )
        #input_lens : instruction长度
        for i, target in enumerate(part_targets):
            labels[i, input_lens[i] + 1:input_lens[i] + len(target) + 1] = target  # plus 1 for bos
        
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
                labels=labels,
            )
        
        #添加BLEU CIDEr指标
        hidden_states = outputs.hidden_states[-1]
        hidden_states = self.llm_relu(self.llm_proj(hidden_states[:,-LL:]))
        llm_logits = torch.cat([self.cbam_llm[i](hidden_states) for i in range(self.attr_num)], dim=1)
        
        B,K,L,D = vit_img_embeds.size()
        
        logits_class = torch.cat([self.cbam_class[j + sum(self.group_details[:i])](vit_img_embeds[:,i]) for i in range(self.attr_group_num) for j in range(0, self.group_details[i])], dim=1)
        
        instance_vit_embeds = self.instance_relu(self.instance_proj(vit_img_embeds))
        logits_instance = torch.cat([self.cbam_instance[i](instance_vit_embeds[:,i]) for i in range(self.attr_group_num)], dim=1)
        return outputs.loss, self.bn_class(logits_class), self.bn_instance(logits_instance),  self.bn_llm(llm_logits)
    
        # =return loss, {"class": self.bn(logits_class), "instance": self.bn(logits_instance), "llm":  self.bn(llm_logits)}
    def infer(self, im, imgname=None, gt_label=None, max_new_tokens=60, num_beams=1, min_length=10, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=0.01, max_length=1000):
        b_s=im.shape[0]
        stop_words_ids = [torch.tensor([835]).to('cpu'),
                          torch.tensor([2277, 29937]).to('cpu')]  # '###' can be encoded in two different ways.
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        vit_img_embeds, inputs_llama, atts_img = self.encode_img(im, imgname=imgname)
        B,K,L,D = vit_img_embeds.size()

        instruction = self.custom_insturction
        img_embeds, atts_img = self.prompt_wrap(inputs_llama, atts_img, instruction)
        dtype = img_embeds.dtype
        device = img_embeds.device
        
        #起始token
        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=torch.long,
                         device=img_embeds.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]
        
        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)
        def prefix_allowed_tokens_fn(batch_id, input_ids):
            return [token_id.item() for token_id in self.limit_token_ids]
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=100,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_attentions = True
            # prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
        ) 
        answers = []
        breakpoint()
        for output_token in outputs['sequences']:
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_texts = self.llama_tokenizer.decode(output_token, skip_special_tokens=True)
            output_texts = output_texts.split('</s>')[0]  # remove the stop sign </s>
            output_texts = output_texts.replace("<s>", "")
            output_texts = output_texts.split(r'[/INST]')[-1].strip()
            answers.append(output_texts)        
        hidden_states=torch.empty(len(outputs['hidden_states'][1:]),outputs['sequences'].shape[0],1,4096)
        for idx, hiddens in enumerate(outputs['hidden_states'][1:]):
            hidden_states[idx] = hiddens[-1]
        hidden_states = hidden_states.squeeze(2).to(self.device).permute(1,0,2).to(dtype) 
        hidden_states = self.llm_relu(self.llm_proj(hidden_states))
        llm_logits = torch.cat([self.cbam_llm[i](hidden_states) for i in range(self.attr_num)], dim=1)
        
        B,K,L,D = vit_img_embeds.size()
        logits_class = torch.cat([self.cbam_class[j + sum(self.group_details[:i])](vit_img_embeds[:,i]) for i in range(self.attr_group_num) for j in range(0, self.group_details[i])], dim=1)
        instance_vit_embeds = self.instance_relu(self.instance_proj(vit_img_embeds))
        logits_instance = torch.cat([self.cbam_instance[i](instance_vit_embeds[:,i]) for i in range(self.attr_group_num)], dim=1)
        return answers, self.bn_class(logits_class), self.bn_instance(logits_instance),  self.bn_llm(llm_logits)
    
        # return {"mean": self.bn(logits_class), "instance": self.bn(logits_instance), "llm":  self.bn(llm_logits)}