#  itertools = importlib.import_module('itertools')

COMMON_FIELD = ['input','output','ref','uid']

TASKS = {
    'cg':{
        'model': "mrm8488/t5-base-finetuned-common_gen",
        'type': 'AutoModelForSeq2SeqLM',
        'field':['concepts'] + COMMON_FIELD,
        'end': ['.', '</s>'], # ending condition. 
        # 'end': ['.'], # ending condition. 
        'start': '<pad>',    # decoding prefix
        'min_len':8
    },
    'squad':{
        'model': "Salesforce/mixqg-large",
                'type': 'AutoModelForSeq2SeqLM',
        'field':['context','answer'],
        'end': ['?', '</s>'],  
        'start': '<pad>',
        'min_len':4
    },
    'quoref':{
        'model': "Salesforce/mixqg-large",
                'type': 'AutoModelForSeq2SeqLM',
        'field':['context','answer'],
        'end': ['?', '</s>'],  
        'start': '<pad>',
        'min_len':4
    },
    'drop':{
        'model': "Salesforce/mixqg-large",
                'type': 'AutoModelForSeq2SeqLM',
        'field':['context','answer'],
        'end': ['?', '</s>'],  
        'start': '<pad>',
        'min_len':4
    },
    'gec':{
        'model':'Unbabel/gec-t5_small',
        'type': 'AutoModelForSeq2SeqLM',
        'field': COMMON_FIELD,
        'end': ['.', '</s>'], # ending condition. 
        'start': '<pad> ',    # decoding prefix
        'min_len':4
    },
    'opus':{
        'model':'Helsinki-NLP/opus-mt-en-de',
        'type':'AutoModelForSeq2SeqLM',
        'field': COMMON_FIELD,
        'end': ['</s>'],  
        'start': '<pad>',
        
        'min_len':4
    },
    'mbart':{
        'model':'facebook/mbart-large-50',
        'type':'MBartForConditionalGeneration',
        'field': COMMON_FIELD,
        'end': ['</s>'],  
        'start': '<pad>',
        'min_len':4
    },
    
    'enfr':{
        'model':'facebook/mbart-large-50-many-to-many-mmt',
        'type':'MBartForConditionalGeneration',
        'field': COMMON_FIELD,
        'end': ['</s>'],  
        'start': '</s>fr_XX',
        'min_len':4
    },
    'ende':{
        'model':'facebook/mbart-large-50-many-to-many-mmt',
        'type':'MBartForConditionalGeneration',
        'field': COMMON_FIELD,
        'end': ['</s>'],  
        'start': '</s>de_DE',
        'min_len':4
    },
            
        'xsum-t5':{
        'model':'t5-small',
        'type':'AutoModelForSeq2SeqLM',
        'field': COMMON_FIELD,
        'end': ['</s>', '.'],  
        'start': '<pad> ',
        'min_len':8
    },        
    'xsum-peg':{
        'model':'google/pegasus-xsum',
        'type':'AutoModelForSeq2SeqLM',
        'field': COMMON_FIELD,
        'end': ['</s>', '.'],  
        'start': '<pad> ',
        'min_len':8
    },
        'xsum-bart':{
        'model':'facebook/bart-large-xsum',
        'type':'AutoModelForSeq2SeqLM',
        'field': COMMON_FIELD,
        'end': ['</s>'],  
        'start': '<s>',
        'min_len':8
    },
    'samsum-bart':{
        'model':'philschmid/bart-large-cnn-samsum',
        'type':'AutoModelForSeq2SeqLM',
        'field': COMMON_FIELD,
        'end': ['</s>'],  
        'start': '<s>',
        'min_len':8
    },
}