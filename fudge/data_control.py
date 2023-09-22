import os

def load_topic_data(args, dataset_info):

    input_texts, conditions, categories = [], [], []

    if args.condition_file is not None:
        with open(args.condition_file, 'r') as rf:
            for line in rf:
                input_texts.append(line.strip().split('\t')[0])
                conditions.append(line.strip().split('\t')[1])
                categories.append(None)
                for cw in conditions[-1].split():
                    assert cw in dataset_info.word2index
    else:
        prefixes = []
        with open(args.prefix_file, 'r') as rf:
            for line in rf:
                prefixes.append(line.strip())
        condition_wordlists = []
        for root, _, files in os.walk(args.wordlist_dir):
            for fname in files:
                words = []
                with open(os.path.join(root, fname), 'r') as rf:
                    for line in rf:
                        word = line.strip()
                        if word in dataset_info.word2index:
                            words.append(word)
                        else:
                            if args.verbose:
                                print('word not found:', word)
                condition_wordlists.append((' '.join(words), fname.split('.')[0]))
        for p in prefixes:
            for c, category in condition_wordlists:
                input_texts.append(p)
                conditions.append(c)
                categories.append(category)
    return {
        "input_texts":input_texts, 
        "conditions":conditions, 
        "categories":categories
    }