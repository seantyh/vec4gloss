from typing import List
from dataclasses import dataclass
from .dep_win import DepWinOutput

@dataclass
class Scheme:
    type: str; start: int; end: int
@dataclass
class AnnotFrame:
    sense_id: str;    POS: str; head_word: str
    definition: str;  event_role: str
    schemas: List[Scheme]
    
    def __post_init__(self):
        self.schemas = self.preprocess()
        
    def get_frame(self, frame_idx):
        frame_x = self.schemas[frame_idx]
        s, e = frame_x.start, frame_x.end
        return (frame_x.type, self.definition[s:e])
    
    def preprocess(self):        
        new_frames = []
        idx = 0
        for frame_x in self.schemas:
            if idx < frame_x.start:
                new_frames.append(Scheme('--', idx, frame_x.start))                
            elif idx > frame_x.start:
                assert("shouldn't happen")
            new_frames.append(frame_x)
            idx = frame_x.end
        return new_frames
    
    def show(self):
        print(" ".join([
            "<{}>{}".format(
                x.type, self.definition[x.start:x.end]) 
            for x in self.schemas
        ]))

@dataclass
class AnnotDepInfo:
    tgt_idx: int
    dep_wins: List[DepWinOutput]
    sel_dep_idx: int
    
    def __repr__(self):
        return "<AnnotDepInfo ({}) {:.2f}: {}>".format(
            self.tgt_idx, self.token_prob(),
            self.dep_win())
    
    def dep_win(self):
        return self.dep_wins[self.sel_dep_idx]
    
    def token_prob(self):
        return self.dep_wins[0].token_prob
    
    def dep_lratio(self):
        return self.dep_wins[self.sel_dep_idx].lratio
    
@dataclass
class AnnotFrameInfo:
    annot_frame: AnnotFrame
    dep_info: AnnotDepInfo