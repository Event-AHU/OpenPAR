import math
from functools import reduce
from operator import mul
import os
import sys
from torchvision import transforms
import torch.nn.functional as F
import torch
import torch.nn as nn
from CLIP import clip
import numpy
import numpy as np
from PIL import Image
from models.vit import *
from decoder.decoders import Decoder
import utils
from decoder.containers import Module
import numpy as np

class TransformerClassifier(Module):
    def __init__(self, attr_num, attributes,vocab_attr,args, dim=768, device='cuda'):
        super().__init__()
        super().__init__()
        self.clip_model, _ = clip.load("ViT-L/14", device=device,download_root=args.model_path) 
        self.clip_model=self.clip_model.float()
        self.device = device
        self.attr_num = attr_num
        self.attributes=attributes
        self.max_length=len(attributes)+1
        self.vocab_size=len(attributes)+3
        print(f'max_length {self.max_length}, vocab_size {self.vocab_size}, decoder depth {args.depth}')
        self.decoder=Decoder(self.vocab_size,self.max_length,args.depth,padding_idx=vocab_attr['<pad>'])

        self.text = clip.tokenize(attributes).to(self.device)
        self.vocab_attr=vocab_attr

        self.bos_embedding = torch.ones(1, dim)
        self.pad_embedding = torch.zeros(1, dim)
        self.eos_embedding = torch.full((1, dim), -1)
        self.bos_embedding = nn.Parameter(self.bos_embedding, requires_grad=False)
        self.pad_embedding = nn.Parameter(self.pad_embedding, requires_grad=False)
        self.eos_embedding = nn.Parameter(self.eos_embedding, requires_grad=False)
        self.special_index2vector={'<pad>':self.pad_embedding,'<bos>':self.bos_embedding,'<eos>':self.eos_embedding}
        self.args=args

    def forward(self,imgs,input):
        b_s=imgs.shape[0]
        ViT_image_features,all_class,attenmap=self.clip_model.visual(imgs.type(self.clip_model.dtype))
        word2vector=self.forward_text()
        out,logits=self.decoder(input.to(self.device),ViT_image_features.to(self.device),word2vector,self.vocab_attr)

        return out,logits

    def forward_text(self):  
        text_features = self.clip_model.encode_text(clip.tokenize(self.attributes).to(self.device)).to(self.device).float()     
        word2vector=dict(zip(self.attributes,text_features))#生成一个word2vector的dict
        word2vector.update(self.special_index2vector)
        return word2vector

    def beam_search_generate(self,imgs,max_length=51+3,num_beams=5,vocab_size=102+3,do_sample=False,top_k=0,top_p=1.0,length_penalty=0.0):
        """
            准备初始输入
            在当前生成的序列长度未达到max_length时扩展生成序列
            准备最终输出的序列
        """
        max_length=self.max_length
        vocab_size=self.vocab_size       
        bos_token_id=self.vocab_attr['<bos>']
        pad_token_id=self.vocab_attr['<pad>']
        eos_token_id=self.vocab_attr['<eos>']
        batch_size=imgs.shape[0]
        ViT_image_features,all_class,attenmap=self.clip_model.visual(imgs.type(self.clip_model.dtype))
        word2vector=self.forward_text()
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty=length_penalty)
            for _ in range(batch_size)
        ]
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=self.device)
        beam_scores = beam_scores.view(-1)
        done = [False for _ in range(batch_size)]
        input_ids = torch.full(
            (batch_size*num_beams, 1),  
            bos_token_id,
            dtype=torch.long,
            device=self.device,
        )
        cur_len = 1
        expand_featurs=[]
        for bs in range(batch_size):
            for _ in range(num_beams) :
                expand_featurs.append(ViT_image_features[bs])
        ViT_image_features=torch.stack(expand_featurs).to(self.device).float()     
        while cur_len < max_length:
            output,_ = self.decoder(input_ids.to(self.device),ViT_image_features,word2vector,self.vocab_attr)
            scores = output[:, -1, :] 
            next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
            if cur_len==1 :
                next_score_resize=torch.zeros((batch_size, vocab_size), dtype=torch.float, device=self.device)
                for batchs in range(batch_size):
                    next_score_resize[batchs]=next_scores[batchs*num_beams]  
                next_scores, next_tokens = torch.topk(next_score_resize, 2 * num_beams, dim=1, largest=True, sorted=True)
            else : 
                # re-organize to group the beam together (we are keeping top hypothesis accross beams)                 
                next_scores = next_scores.view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)
                next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
            next_batch_beam = []
            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                    continue
                next_sent_beam = []
                for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(zip(next_tokens[batch_idx], next_scores[batch_idx])):
                    # get beam and word IDs

                    if cur_len==1 and do_sample==False:
                        token_id=beam_token_id
                        effective_beam_id=token_id
                    else :
                        beam_id = torch.div(beam_token_id, vocab_size, rounding_mode='floor')
                        #beam_id = beam_token_id // vocab_size 
                        token_id = beam_token_id % vocab_size
                        effective_beam_id = batch_idx * num_beams + beam_id                  
                    if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(
                            input_ids[effective_beam_id].clone(), beam_token_score.item(),
                        )
                    else:
                        # add next predicted word if it is not eos_token
                        next_sent_beam.append((beam_token_score, token_id,effective_beam_id))
                    if len(next_sent_beam) == num_beams:
                        break
                done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len=cur_len
                )
                next_batch_beam.extend(next_sent_beam)
            if all(done):
                break
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
            beam_idx = input_ids.new([x[2] for x in next_batch_beam])
            
            if cur_len==1 and do_sample==False:
                input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            else :
                input_ids = input_ids[beam_idx, :]
                input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1
            # end of length while
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
        output_batch_size=batch_size
        output_num_return_sequences_per_batch = 1
        sent_lengths = input_ids.new(output_batch_size)
        best = []
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                #best_hyp_scores=sorted_hyps.pop()[0]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                #best_scores.append(torch.tensor(best_hyp_scores,dtype=torch.float32))
                best.append(best_hyp)
        if sent_lengths.min().item() != sent_lengths.max().item():
            sent_max_len = min(sent_lengths.max().item() + 1, max_length)
            decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_length:
                    decoded[i, sent_lengths[i]] = eos_token_id
        else:
            decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)
        #decoded_scores=torch.stack(best_scores).to(next(self.parameters()).device)
        return decoded
class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty=0.0):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9
        self.length_penalty=length_penalty
    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        if len(self) < self.num_beams:
            return False
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret
