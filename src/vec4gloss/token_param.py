from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Literal

import torch
import numpy as np

from .model_utils import extract_encoder_vector

@dataclass
class SeqMask:
    start: int
    end: int
    mask: torch.Tensor
SeqMaskDict = Dict[Literal["no", "full", "def"], SeqMask]

@dataclass
class TokenParam:
    token_idx: int
    token_text: str
    examples: List[str]
    def_token_ids: List[int]
    def_token_texts: List[str]    
    full_prob: float = -1.
    allmasked_prob: float = -1.
    defmasked_prob: float = -1.
    replaced_prob: float = -1.
    random_prob: float = -1.
    replaced_vec: Optional[torch.Tensor] = None
    original_vec: Optional[torch.Tensor] = None
    random_vec: Optional[torch.Tensor] = None

    def __repr__(self):
        return "<TokenParam {:\u3000>2}: F/A/D/R/X {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}>".format(
            self.token_text,
            self.full_prob, self.allmasked_prob, self.defmasked_prob,
            self.replaced_prob, self.random_prob)

class TokenParamFactory:
    def __init__(self, examples, definition, tokenizer, model):                
        self.def_token_ids = [0] + tokenizer(definition)["input_ids"]
        self.def_token_texts = tokenizer.convert_ids_to_tokens(self.def_token_ids)
        self.period_fst_idx = self.def_token_ids.index(306)  # the position of the first period "。"
        self.tokenizer = tokenizer
        self.model = model
        self.mean_vec = self.compute_mean_vec(examples)
        self.examples = examples
        self.replaced_vec: Optional[torch.Tensor] = None

        if self.mean_vec is None:
            raise ValueError("invalid examples")

    def set_replaced_vec(self, replaced_vec: torch.Tensor):
        self.replaced_vec = replaced_vec

    def build_all_sequences(self, dbg=False) -> List[TokenParam]:
        assert self.replaced_vec is not None
        token_param_list = []
        ## there are a period and a EOS token at the end, therefore minus 2.
        for i in range(self.period_fst_idx+1, len(self.def_token_ids)-2):
            token_param_x = self.build_token_param(i, dbg)
            token_param_list.append(token_param_x)
        return token_param_list

    def build_token_param(self, pos: int,
            dbg: bool=False) -> Optional[TokenParam]:
        assert self.mean_vec is not None
        assert self.replaced_vec is not None
        
        token_text = self.def_token_texts[pos]
        tok_idx = self.def_token_ids[pos]
        
        if pos <= self.period_fst_idx:
            raise ValueError("`pos` is smaller or equal than the first index of `period`.")

        input_ids = torch.tensor([self.def_token_ids[:pos]])\
                         .to(self.model.device)
        
        random_enc_vec = torch.randn(*self.mean_vec.shape)
        seq_masks = self.generate_seq_masks(input_ids, dbg=dbg)

        compute_args = (self.mean_vec, input_ids)
        full_prob = self.compute_conditional_prob(tok_idx, seq_masks["no"].mask, *compute_args)
        allmasked_prob = self.compute_conditional_prob(tok_idx, seq_masks["full"].mask, *compute_args)
        defmasked_prob = self.compute_conditional_prob(tok_idx, seq_masks["def"].mask, *compute_args)
        replaced_prob = self.compute_conditional_prob(tok_idx, 
                            seq_masks["no"].mask, self.replaced_vec, input_ids)
        random_prob = self.compute_conditional_prob(tok_idx, 
                            seq_masks["full"].mask, random_enc_vec, input_ids)
        return TokenParam(
                tok_idx, 
                token_text, 
                self.examples, 
                self.def_token_ids, 
                self.def_token_texts,
                full_prob=full_prob,
                allmasked_prob=allmasked_prob,
                defmasked_prob=defmasked_prob,                
                replaced_prob=replaced_prob,
                random_prob=random_prob                
                )

    def compute_prob_with_replaced(self, 
            token_param: TokenParam, 
            replaced_vec: torch.Tensor
            ) -> TokenParam:         
        tok_idx = token_param.token_idx
        input_ids = self.def_token_ids[:tok_idx]        
        seq_masks = self.generate_seq_masks(input_ids)
        replaced_prob = self.compute_conditional_prob(
            tok_idx, seq_masks["no"].mask, replaced_vec, input_ids
        )
        token_param.replaced_prob = replaced_prob
        token_param.replaced_vec = replaced_vec
        return token_param

    def compute_mean_vec(self, examples: List[str]) -> Optional[torch.Tensor]:
        enc_vecs = []
        for ex in examples:
            try:
                with torch.no_grad():
                    enc_vecs.append(extract_encoder_vector(ex, self.tokenizer, self.model))
            except AssertionError:
                return None
        if len(enc_vecs) == 0:
            return None
        return torch.cat(enc_vecs).mean(0, keepdim=True).to(self.model.device)

    def compute_conditional_prob(self,
        tgt_token: int,
        seq_mask: torch.Tensor,
        mean_vec: torch.Tensor,
        input_ids: torch.Tensor,
        ) -> float:

        model = self.model
        with torch.no_grad():
            out = model(decoder_encoder_vector=mean_vec,
                        decoder_input_ids=input_ids,
                        decoder_attention_mask=seq_mask)
            token_prob = out.logits.softmax(-1)[0, -1, tgt_token].item()

        return token_prob

    def generate_seq_masks(self,
            input_ids: torch.Tensor,            
            dbg=False) -> SeqMaskDict:

        model = self.model
        tokenizer = self.tokenizer

        def dbg_print(*x):
            if dbg: print(*x)
        seq_masks = {}        
        seq_len = input_ids.shape[1]

        ## no-mask
        seq_mask = torch.ones(*input_ids.shape, dtype=torch.long).to(model.device)
        seq_masks["no"] = SeqMask(0, 0, seq_mask)

        ## full-mask
        seq_mask = torch.zeros(*input_ids.shape, dtype=torch.long).to(model.device)
        seq_masks["full"] = SeqMask(0, seq_len, seq_mask)

        ## definition-mask                
        seq_mask = torch.zeros(1, seq_len, dtype=torch.long).to(model.device)
        seq_mask[0, :self.period_fst_idx+1] = 1
        seq_masks["def"] = SeqMask(self.period_fst_idx+1, seq_len, seq_mask)

        if dbg:
            for mask_x in seq_masks.values():
                s = mask_x.start
                e = mask_x.end
                mask_seq = mask_x.mask
                mask_text = tokenizer.decode(torch.mul(input_ids, mask_seq)[0, 1:])
                mask_text = mask_text.replace("<pad>", "Ｏ")
                dbg_print("({:>2d}:{:>2d}) {}".format(s, e, mask_text))

        return seq_masks

