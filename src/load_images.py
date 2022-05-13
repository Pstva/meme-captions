from PIL import Image
import requests
import pandas as pd
import argparse
"""
скачивает мемы по ссылкам вк и сохраняет в папку
"""


def load_save(url, name, folder_to_save):
    try:
        im = Image.open(requests.get(url, stream=True).raw)
        im.save(f"{folder_to_save}/{name}.jpg")
    except:
        return


# пока только textmeme  и badtextmeme - до 3641 индекса включительно
def main(args):
    data = pd.read_csv(args.input_data)
    #for i in range(len(data)):
    for i in range(3642):
        if data.loc[i, 'photo_url'] is not None:
            load_save(data.loc[i, 'photo_url'], i, args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input-data", required=True)
    parser.add_argument('-o', "--output-folder", required=True)
    args = parser.parse_args()
    main(args)
