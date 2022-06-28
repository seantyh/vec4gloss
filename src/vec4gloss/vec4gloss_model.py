
import torch
from transformers import MT5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput
from typing import Optional, Union, Tuple
from torch.nn import CrossEntropyLoss

class Vec4GlossModel(MT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,        
        input_ids: Optional[torch.LongTensor] = None,        
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_start_markers: torch.LongTensor = None,
        decoder_end_markers: torch.LongTensor = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,        
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        Following https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/models/t5/modeling_t5.py#L1456
        But only implement model training and inference code (no parallelism)
        
        adding two parameters: `decoder_start_markers` and `decoder_end_markers`.
        These parameters are prefixed with `decoder_` to be compatible with hgf's generate mechanism, 
        where it pass all arguments into Encoder alone, and only `decoder_` prefixed parameters are kept in
        generation process`
        """

        # Encode
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
        
        
        # an ugly implementation. a better way?
        enc_last_hiddens = torch.zeros(
            (decoder_start_markers.shape[0], # batch size
             1, # sequence length
             self.model_dim # model dimension
            ), dtype=encoder_outputs.last_hidden_state.dtype
        ).to(self.device)
        
        enc_attention_mask = torch.ones(
            (decoder_start_markers.shape[0], 1)
        ).to(self.device)
        
        for i in torch.arange(decoder_start_markers.shape[0]):
            s, e = decoder_start_markers[i], decoder_end_markers[i]
            enc_last_hiddens[i, 0, :] = encoder_outputs.last_hidden_state[i, s:e, :].mean(axis=0)        
        
        

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=enc_last_hiddens,
            encoder_attention_mask=enc_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,            
        )

        sequence_output = decoder_outputs[0]


        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))            

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=enc_last_hiddens,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
    def prepare_inputs_for_generation(
        self,
        input_ids,
        decoder_start_markers=None,
        decoder_end_markers=None,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,        
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]
                
        return {
            "decoder_input_ids": input_ids,
            "decoder_start_markers": decoder_start_markers,
            "decoder_end_markers": decoder_end_markers,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }