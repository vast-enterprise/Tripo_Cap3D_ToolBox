import json

def load_jsonl(file_path):
    data = []
    sourceid2modelid = {}
    with open(file_path, 'r') as f:
        for line in f:
            data_item = json.loads(line)
            data.append(data_item)
            sourceid2modelid[data_item['source_id']] = data_item['model_id']
    return data, sourceid2modelid

datas, sourceid2modelid = load_jsonl('/mnt/pfs/users/yuzhipeng/10M3D/434w_sketchfab.txt')


train_list = []
for data in open('data/train_list.txt'):
    source_id, geo_score, tex_score = data.split()
    model_id = sourceid2modelid.get(source_id, None)
    if model_id is not None:
        train_list.append((model_id, geo_score, tex_score))

test_list = []
for data in open('data/val_list.txt'):
    source_id, geo_score, tex_score = data.split()
    model_id = sourceid2modelid.get(source_id, None)
    if model_id is not None:
        test_list.append((model_id, geo_score, tex_score))

with open('data/train_list.txt.model_id', 'w') as f:
    for item in train_list:
        f.write(' '.join(item) + '\n')


with open('data/val_list.txt.model_id', 'w') as f:
    for item in test_list:
        f.write(' '.join(item) + '\n')