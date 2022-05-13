import pandas as pd
import argparse
import numpy as np
import os


def check_for_image(uid, folder):
    return os.path.exists(f'{folder}/{uid}.jpg')


def main(args):
    np.random.seed(10)
    # переводит тексты описаний в токены
    texts_data = pd.read_csv(args.texts_df, usecols=['uid', 'group_domain', 'text'])
    texts_data = texts_data[texts_data['group_domain'] != 'poiskmemow']
    texts_data['text'] = texts_data['text'].apply(lambda x: " ".join(str(x).split()))
    texts_data['has_image'] = texts_data['uid'].apply(lambda x: check_for_image(x, args.image_folder))
    texts_data = texts_data[texts_data['has_image'] == True]
    # делим на train, test, dev
    texts_data = texts_data.sample(frac=1).reset_index()
    train_size = int(args.train_size * len(texts_data))
    val_size = int(args.val_size * len(texts_data))

    texts_data.loc[range(0, train_size), ['uid', 'text']].to_csv(f'{args.output_folder}/train.csv',
                                                                 index=False, sep='\t', quotechar='&')
    texts_data.loc[range(train_size, train_size+val_size), ['uid', 'text']].to_csv(f'{args.output_folder}/val.csv',
                                                                                   index=False,  sep='\t', quotechar='&')
    texts_data.loc[range(train_size+val_size, len(texts_data)), ['uid', 'text']].to_csv(f'{args.output_folder}/test.csv',
                                                                                        index=False,  sep='\t', quotechar='&')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--texts_df", default='data/raw/memes_data.csv')
    parser.add_argument('-i', "--image_folder", default='data/images')
    parser.add_argument('-o', "--output-folder", default='data/for_training')
    parser.add_argument('-train', "--train_size", default=0.7)
    parser.add_argument('-val', "--val_size", default=0.15)
    args = parser.parse_args()
    main(args)
