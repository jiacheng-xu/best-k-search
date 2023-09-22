from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Salesforce/mixqg-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def run_qg(input_text, **generator_args):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    generated_ids = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(run_qg("Robert Boyle \\n In the late 17th century, Robert Boyle proved that air is necessary for combustion."))
