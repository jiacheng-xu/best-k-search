import os
from comet import download_model, load_from_checkpoint

# model_path = download_model("wmt21-comet-qe-mqm")
model_path = download_model("wmt21-comet-qe-da")
model = load_from_checkpoint(model_path)

def run_file(raw_data):
    feed = []
    result = {}
    for piece in raw_data:
        ref = piece['ref']
        # assert len(ref) == 1
        # ref_txt = ref[0]
        uid = piece['uid']
        src = piece['input']
        outputs = piece['output']
        for idx, out in enumerate(outputs):
            feed.append(
                {
                    'src': src,
                    'uid':uid,
                    'offset':idx,
                    'mt': out
                }
            )
        result[uid] = [0 for _ in range(len(outputs))]
    seg_scores, sys_score = model.predict(feed, batch_size=8, gpus=1)
    for idx, fee in enumerate(feed):
        result[fee['uid']][fee['offset']] = seg_scores[idx]
    # merge raw_data and result
    new_data = []
    for rd in raw_data:
        uid = rd['uid']
        tmp = rd.copy()
        tmp['rank'] = result[uid]
        new_data.append(tmp)
    return new_data

def main():
    # read a json file, rerank all the outputs according to COMET score. 
    prefix = 'enfr'
    # prefix = 'ende'
    dir = '/export/home/cond-text-gen/outputs'
    dir  = os.path.join(dir, prefix)
    files = os.listdir(dir)

    files = [ f for f in files if f.endswith('.json') and f.startswith(prefix)]

    import json
    for file in files:
        target_fname = os.path.join('/export/home/cond-text-gen/outputs/rank_'+prefix, 'rank_'+file)
        file_exists = os.path.exists(target_fname)
        # check existing
        if  file_exists:
            print(f"{target_fname} exists")
            continue
        with open(os.path.join(dir, file), 'r') as fd:
            raw_data = json.load(fd)
        print('running',file)

        updated_data = run_file(raw_data)
        with open(target_fname, 'w') as fd:
            json.dump(updated_data, fd)

if __name__ == '__main__':
    main()