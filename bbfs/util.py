
from dec.bfs import assemble_pad, assemble_pad_plus, right_side_padding
from typing import List 

import torch

def step_model_decode(model, device, expansion_list:List, doc_input_token, tokenizer):
    
    dec_prefixes = [x.get_token_idx() for x in expansion_list]
    concat_prefixes = [doc_input_token + x for x in dec_prefixes]

    (
        concat_input_tensor,
        padding_pos,
        attention_mask_concat_input_tensor,
    ) = assemble_pad_plus(
        concat_prefixes, device=device, pad_token_idx=tokenizer.bos_token_id
    )  # batch x seq      store [PAD] * ? + enc + dec
    batch_size, full_length = concat_input_tensor.size()
    # enc + dec length, exclude padding
    enc_dec_length = [full_length - x for x in padding_pos]

    dec_input_tensor = assemble_pad(
        dec_prefixes, device, tokenizer.bos_token_id
    )  # only store the decoded tokens after enc, batch x dec_seq_len

    output = model(
        input_ids=concat_input_tensor, 
        attention_mask=attention_mask_concat_input_tensor        # shold use encoder_attention_mask instead of attention_mask!!!!! 
        # TODO double check which attention mask to use
        # encoder_attention_mask
    )
    logits = output.logits[:, -1, :]  # batch x seq_len x vocab =>  batch x vocab
    return logits, concat_input_tensor, dec_input_tensor, batch_size, padding_pos, enc_dec_length


def step_model_seq2seq_decode(model, device, expansion_list:List, doc_input_token, tokenizer):
    batch_size = len(expansion_list)
    dec_prefixes = [x.get_token_idx() for x in expansion_list]
    # concat_prefixes = [doc_input_token + x for x in dec_prefixes]
    
    # prepare encoder input
    enc_input_tensor = assemble_pad([doc_input_token]*batch_size, device, tokenizer.pad_token_id)
    
    (
        dec_input_tensor,
        padding_pos,
        decoder_attention_mask,
    ) = assemble_pad_plus(
        dec_prefixes, device=device, pad_token_idx=tokenizer.pad_token_id
    )  # batch x seq      store [PAD] * ? + dec
    
    # batch_size, full_length = concat_input_tensor.size()
    full_length = enc_input_tensor.size()[1] + dec_input_tensor.size()[1]
    # enc + dec length, exclude padding
    enc_dec_length = [full_length - x for x in padding_pos]

    # dec_input_tensor = assemble_pad(
    #     dec_prefixes, device, tokenizer.bos_token_id
    # )  # only store the decoded tokens after enc, batch x dec_seq_len

    output = model(
        input_ids=enc_input_tensor, 
        decoder_input_ids = dec_input_tensor,
        decoder_attention_mask=decoder_attention_mask
        # attention_mask = decoder_attention_mask
        # shold use encoder_attention_mask instead of attention_mask!!!!! 
        # TODO double check which attention mask to use
        # encoder_attention_mask
    )
    logits = output.logits[:, -1, :]  # batch x seq_len x vocab =>  batch x vocab
    return logits, enc_input_tensor, dec_input_tensor, batch_size, padding_pos, enc_dec_length


def step_model_bart_decode(model, device, expansion_list:List, doc_input_token, tokenizer, padding_token=1):
    batch_size = len(expansion_list)
    dec_prefixes = [x.get_token_idx() for x in expansion_list]
    # concat_prefixes = [doc_input_token + x for x in dec_prefixes]
    
    # prepare encoder input
    enc_input_tensor = assemble_pad([doc_input_token]*batch_size, device, padding_token)
    enc_attn_mask = torch.ones(enc_input_tensor.size(), device=device)

    dec_input_tensor, extraction_points , decoder_attention_mask=right_side_padding(dec_prefixes, device=device, pad_token_idx=padding_token)

    output = model(
        input_ids=enc_input_tensor, 
        decoder_input_ids = dec_input_tensor,
        decoder_attention_mask=decoder_attention_mask
        # attention_mask = all_attention_mask
        # shold use encoder_attention_mask instead of attention_mask!!!!! 
        # TODO double check which attention mask to use
        # encoder_attention_mask
    )
    outputs_logits =  output.logits
    b, seq_len, vocab = outputs_logits.size()
    reshaped_logits = outputs_logits.view(-1, vocab)
    
    bias = torch.arange(0, b, device=device) * seq_len
    biased_extraction_points = extraction_points + bias
    # extraction_points: batch size, => [batch, ]
    
    logits = torch.index_select(reshaped_logits, 0, biased_extraction_points)
    # logits = output.logits[:, -1, :]  # batch x seq_len x vocab =>  batch x vocab

    return logits,  dec_input_tensor, batch_size