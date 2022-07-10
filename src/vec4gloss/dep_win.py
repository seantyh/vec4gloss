from dataclasses import dataclass
from typing import List, Tuple

import torch
import numpy as np

from .model_utils import extract_encoder_vector

SeqMask = Tuple[int, int, torch.LongTensor]

@dataclass
class DepWinOutput:
    mask_win: Tuple[int,int]
    lratio: float
    token_prob: float
    token_text: str
    mask_text: str    
    dep_text: str 
    
    def __repr__(self):
        return "({:>2d}-{:>2d}) [{:5.2f}/{:.2f}] {:\u3000>2s} / {}".format(
            self.mask_win[0], self.mask_win[1], 
            self.lratio, self.token_prob, 
            self.token_text, self.dep_text)
    
def compute_mean_vec(examples, tokenizer, model): 
    enc_vecs = []
    for ex in examples:
        try:
            with torch.no_grad():
                enc_vecs.append(extract_encoder_vector(ex, tokenizer, model))
        except AssertionError:
            return None
    if len(enc_vecs) == 0:
        return None
    return torch.cat(enc_vecs).mean(0, keepdim=True).to(model.device)

def generate_seq_masks(
        inseq, 
        tokenizer, 
        model, max_win=4, 
        dbg=False) -> List[SeqMask]:
    
    def dbg_print(*x):
        if dbg: print(*x)
    seq_masks = []
    input_ids = torch.tensor([inseq]).to(model.device)
    period_fst_idx = inseq.index(306)  # the position of the first period "。"    
    T = len(inseq)-period_fst_idx-1   # the token count of being masked    
    
    ## backward  
    for cursor in range(min(max_win, T+1)):
        seq_mask = torch.ones(1, len(inseq), dtype=torch.long)
        mask_start = len(inseq)-cursor
        mask_end = len(inseq)        
        seq_mask[0, mask_start:mask_end] = 0
        # seq_masks.append(seq_mask.to(model.device))        
        seq_masks.append((mask_start, mask_end, seq_mask))
    
    ## forward 
    for cursor in range(len(inseq)-1, mask_start, -1):
        seq_mask = torch.ones(1, len(inseq), dtype=torch.long)        
        mask_end = cursor
        seq_mask[0, mask_start:mask_end] = 0         
        seq_masks.append((mask_start, mask_end, seq_mask))
        
    for s, e, mask_x in seq_masks:
        mask_text = tokenizer.decode(torch.mul(input_ids, mask_x)[0, 1:])
        mask_text = mask_text.replace("<pad>", "Ｏ")
        dbg_print("({:>2d}:{:>2d}) {}".format(s, e, mask_text))
    
    return seq_masks

def compute_conditional_prob(
        tgt_token: int,
        seq_mask: torch.LongTensor,        
        mean_vec: torch.FloatTensor, 
        input_ids: torch.LongTensor,          
        tokenizer, model) -> float:
    
    with torch.no_grad():        
        out = model(decoder_encoder_vector=mean_vec, 
                    decoder_input_ids=input_ids,
                    decoder_attention_mask=seq_mask)                
        token_prob = out.logits.softmax(-1)[0, -1, tgt_token].item()                   
            
    return token_prob

def compute_dependents(tgt_seq, tgt_idx, mean_vec, tokenizer, model, dbg=False):
    assert tgt_idx < len(tgt_seq)
    token_probs = []    
    generated = [0] + tgt_seq[:tgt_idx]
    tgt_token = tgt_seq[tgt_idx]
    ret_lr = 0    
    lrs = []
    seq_masks = generate_seq_masks(generated, tokenizer, model, max_win=8, dbg=dbg)
    input_ids = torch.tensor([generated]).to(model.device)
    
    def dbg_print(*x):
        if dbg: print(*x)
        
    dbg_print("target: ", tokenizer.decode([tgt_token]))
    depwins = []
    for s, e, seq_mask_x in seq_masks:
             
        token_prob = compute_conditional_prob(tgt_token, seq_mask_x, mean_vec, input_ids, tokenizer, model)
        token_probs.append(token_prob)        
        mask_text = tokenizer.decode(torch.mul(input_ids, seq_mask_x)[0, 1:])
        mask_text = mask_text.replace("<pad>", "Ｏ")        
                
        if len(token_probs) > 1:             
            lr = token_probs[0]/token_probs[-1]
        else:
            lr = 0
        lrs.append(lr)
        
        dep_text = tokenizer.decode(input_ids[0, s:e])
        token_text = tokenizer.decode([tgt_token])
        depwin_x = DepWinOutput((s,e), lr, token_prob, token_text, mask_text, dep_text)
        depwins.append(depwin_x)
        dbg_print(depwin_x)
                  
    return depwins

def select_highest_lr(depwins: List[DepWinOutput]) -> DepWinOutput:
    idx = np.argmax([x.lratio for x in depwins])
    return idx