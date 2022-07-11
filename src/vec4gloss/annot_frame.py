from typing import List, Dict, Any
from dataclasses import dataclass
from .token_param import TokenParam

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
class AnnotFrameInfo:
    annot_frame: AnnotFrame
    dep_info: List[TokenParam]        

@dataclass
class RatingFrameInfo:
    noun_entry: Dict[str, Any]
    dep_info: List[TokenParam]