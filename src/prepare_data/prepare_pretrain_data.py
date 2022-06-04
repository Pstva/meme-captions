import pandas as pd
import numpy as np
from load_memes import load_save


if __name__ == "__main__":
    # # preparing translated flickr data
    # data = pd.read_csv('data/raw/flickr_data.csv')
    # data = data.sample(frac=0.5, random_state=100)
    # train_size = int(len(data) * 0.7)
    # data = data.rename({'image_name': 'uid'}, axis=1)
    # data['uid'] = data['uid'].apply(lambda x: x[:-4])
    # data.iloc[:train_size, :].to_csv('data/flickr/train.csv', index=False, sep='\t')
    # data.iloc[train_size:, :].to_csv('data/flickr/val.csv', index=False, sep='\t')

    # # preparing full translated flickr data
    # data = pd.read_csv('data/raw/flickr_data.csv')
    # data = data.sample(frac=1, random_state=100)
    # train_size = int(len(data) * 0.7)
    # data = data.rename({'image_name': 'uid'}, axis=1)
    # data['uid'] = data['uid'].apply(lambda x: x[:-4])
    # data.iloc[:train_size, :].to_csv('data/flickr/train_full.csv', index=False, sep='\t')
    # data.iloc[train_size:, :].to_csv('data/flickr/val_full.csv', index=False, sep='\t')

    # preparing val vizwiz data
    data = pd.read_csv('data/raw/vizwiz_val_translated.csv')
    data = data.rename({'file_name': 'uid'}, axis=1)
    data[['uid', 'text']].to_csv('data/vizwiz/val.csv', index=False, sep='\t')

    # preparing train vizwiz data
    data = pd.read_csv('data/raw/vizwiz_train_translated.csv')
    data = data.rename({'file_name': 'uid'}, axis=1)
    data[['uid', 'text']].to_csv('data/vizwiz/train.csv', index=False, sep='\t')
