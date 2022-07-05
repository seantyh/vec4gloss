import torch

def get_marked_pos(text):
    assert text.count("<") == text.count(">") == 1
    s, e = text.index("<")+1, text.index(">")    
    assert s != e
    return s, e

def extract_encoder_vector(intext, tokenizer, model):    
    max_length = 256
    vbatch = tokenizer(intext, return_tensors="pt", 
                       max_length=max_length, truncation=True).to(model.device)
    s,e = get_marked_pos(intext)   
    s = vbatch.char_to_token(s)
    e = vbatch.char_to_token(e)
    vbatch["decoder_start_markers"] = torch.tensor([s]).to(model.device)
    vbatch["decoder_end_markers"] = torch.tensor([e]).to(model.device)
    encoder = model.get_encoder()
    enc_out = encoder(
            input_ids=vbatch["input_ids"], 
            attention_mask=vbatch["attention_mask"])
    enc_vec = enc_out.last_hidden_state[[0],s:e,:] \
                     .mean(1, keepdim=True)
    return enc_vec

def decode_vector(vec, tokenizer, model, max_length=50):
    vgenout = model.generate(decoder_encoder_vector=vec, bos_token_id=0, max_length=max_length)
    return tokenizer.batch_decode(vgenout[:, 1:-1])[0]

def gen_func(tokenizer, model):
    def _gen_func(text):
        enc_vec = extract_encoder_vector(text, tokenizer, model)
        return decode_vector(enc_vec, tokenizer, model)
    return _gen_func
