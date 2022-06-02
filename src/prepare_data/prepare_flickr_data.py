import pandas as pd
import numpy as np
from load_memes import load_save


if __name__ == "__main__":
    # preparing translated flickr data
    data = pd.read_csv('data/raw/flickr_data.csv')
    data = data.sample(frac=0.5, random_state=100)
    train_size = int(len(data) * 0.7)
    data = data.rename({'image_name': 'uid'}, axis=1)
    data['uid'] = data['uid'].apply(lambda x: x[:-4])
    data.iloc[:train_size, :].to_csv('data/flickr/train.csv', index=False, sep='\t')
    data.iloc[train_size:, :].to_csv('data/flickr/val.csv', index=False, sep='\t')

    # preparing translated conceptual captions data
    data = pd.read_csv('data/raw/CC_data.csv')
    data = data.sample(frac=1, random_state=100)
    data = data.rename({'caption': 'text'}, axis=1)
    data = data.reset_index()
    for ind, x in data.iterrows():
        url, text = x.url, x.text
        if not load_save(url, f'{ind}', 'data/conceptual_captions/images'):
            data['text'] = 0
    data = data[data['text'] != 0]
    train_size = int(len(data) * 0.7)
    data.loc[:train_size, ['uid', 'text']].to_csv('data/conceptual_captions/train.csv', index=False, sep='\t')
    data.loc[train_size:, ['uid', 'text']].to_csv('data/conceptual_captions/val.csv', index=False, sep='\t')
