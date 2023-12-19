import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import pdb
from decoder.attention import MultiHeadAttention
from decoder.utils import sinusoid_encoding_table, PositionWiseFeedForward
from decoder.containers import Module, ModuleList
from decoder.beam_search.beam_search import *

class DecoderLayer(Module):
    def __init__(self, d_model=768, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    #input out, enc_output encoder_output, mask_pad mask_queries, mask_self_att mask_self_attention,mask_enc_att mask_encoder
    #mask_enc_att的作用?  先删除
    def forward(self, input, enc_output, mask_pad=None, mask_self_att=None):
        if mask_self_att is not None :
            self_att = self.self_att(input, input, input, mask_self_att)
        else :
            self_att = self.self_att(input, input, input)
        if mask_pad is not None :
            self_att = self_att * mask_pad
        #(self_att, enc_output[:, 0], enc_output[:, 0],mask_enc_att) q,k,v
        
        enc_att=self.enc_att(self_att, enc_output, enc_output) 
        if mask_pad is not None :
            enc_att = enc_att * mask_pad

        ff = self.pwff(enc_att)
        if mask_pad is not None :
            ff = ff * mask_pad
        
        return ff


class Decoder(Module):
    # vocab_size=len(text_field.vocab), max_len 54, N_dec 3, padding_idx text_field.vocab.stoi['<pad>']
    def __init__(self, vocab_size, max_len, N_dec, padding_idx=-1, d_model=768, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(Decoder, self).__init__()
        self.d_model = d_model
        #self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)
        self.layers = ModuleList(
            [DecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])
        self.fc = nn.Linear(d_model, vocab_size,bias=False)
        self.max_len = max_len
        self.padding_idx = padding_idx
        self.N = N_dec
        
    def forward(self, input, encoder_output,word2vector,vocab_attr):
        # input (b_s, seq_len)=captionstorch.Size([30, 16])
        # encoder_output.torch.Size([30, 3, 50, 512])]
        #mask_encoder torch.Size([30, 1, 1, 50])
        b_s, seq_len = input.shape[:2]
        input_embedding=my_Embedding(input,word2vector,vocab_attr,self.d_model)
        
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),
                                         diagonal=1)#返回一个张量，包含输入矩阵 (2D 张量) 的上三角部分，其余部分被设为 0。上三角部分为矩阵指定对角线 diagonal 之上的元素。
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)

                    
        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)

        out = input_embedding + self.pos_emb(seq)#替换为 自建词库 mmas queries是什么 输出seq 
        for i, l in enumerate(self.layers):
            out = l(out, encoder_output, mask_queries, mask_self_attention)
        #out [bs,102,768]
        out = self.fc(out)#bs,51,102
        return F.log_softmax(out, dim=-1),out
        
def my_Embedding(input,word2vector,vocab_attr,d_model):
    input_embedding=torch.empty((input.shape[0],input.shape[1],d_model),dtype=torch.float,device='cuda')
    for b_c,batchs in enumerate(input) :
        for e_c,elem in enumerate(batchs) :
            #print(b_c,vocab_attr.lookup_token(int(elem)))
            input_embedding[b_c,e_c]=word2vector[vocab_attr.lookup_token(int(elem))].cuda()
    
    return input_embedding.cuda()
