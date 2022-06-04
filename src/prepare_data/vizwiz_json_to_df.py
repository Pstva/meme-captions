import pandas as pd
import json


val_file = 'data/raw/vizwiz/val.json'
train_file = 'data/raw/vizwiz/train.json'


def make_df(file):
    with open(file, 'r') as f:
        data = json.load(f)

    image_data = pd.DataFrame(columns=['file_name', 'id'])
    ann_data = pd.DataFrame(columns=['caption', 'id'])

    files = [x['file_name'] for x in data['images']]
    ids1 = [x['id'] for x in data['images']]
    image_data['file_name'] = files
    image_data['id'] = ids1

    captions = [x['caption'] for x in data['annotations']]
    ids2 = [x['image_id'] for x in data['annotations']]
    ann_data['caption'] = captions
    ann_data['id'] = ids2

    data = ann_data.merge(image_data, how='inner', on='id')[['caption', 'file_name']]
    data['file_name'] = data['file_name'].apply(lambda x: x[:-4])

    data = data[data['caption'] != "Quality issues are too severe to recognize visual content."]
    return data

data = make_df(val_file)
data.to_csv('data/raw/vizwiz_val.csv', index=False)
data = make_df(train_file)
data.to_csv('data/raw/vizwiz_train.csv', index=False)
