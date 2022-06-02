import pandas as pd
import requests
import argparse


# отправляем запрос к VK API
def send_request(domain, offset, key_access, count):
    r = requests.get("https://api.vk.com/method/wall.get",
                     params={"domain": domain,
                             "count": count,
                             "offset": offset,
                             "access_token": key_access,
                             "v": 5.131})

    if r.status_code == 200:
        return r
    return None


# достаем ссылку на фото-приложение к посту
def get_photo_url(item):
    # если тип приложения к посту - фото
    if 'attachments' not in item:
        return None
    if 'photo' in item['attachments'][0]:
        # перебираем разные размеры - берем самый большой
        cur_max_height = 0
        cur_url = None
        for size in item['attachments'][0]['photo']['sizes']:
            if size['height'] > cur_max_height:
                cur_max_height = size['height']
                cur_url = size['url']
        return cur_url
    return None


# распаковка ответа от сервера в датафрейм
def unpack_response(response, group_domain):
    data = {"group_domain":[], 'id': [],  'text': [], 'photo_url': []}
    if 'items' not in response.json()['response']:
        return None

    for item in response.json()['response']['items']:
        if int(item['marked_as_ads']) != 1:
            data['id'].append(item['id'])
            data['group_domain'].append(group_domain)
            data['text'].append(item['text'])
            data['photo_url'].append(get_photo_url(item))
    return pd.DataFrame(data)


def load_all_posts(key_access, group_domain):
    # потом подкорректируем реальное кол-во постов
    posts_count = 1000
    count = 100
    posts_loaded = 0
    responses = []
    offset = 0

    # пока не скачали все посты
    while offset < posts_count:
        response = send_request(group_domain, offset, key_access, count)
        posts_count = int(response.json()['response']['count'])
        responses.append(response)
        offset += count
    return responses


def main(args):
    # в файле лежит мой код доступа приложения
    with open('key.txt', 'r') as f:
        key_access = f.read().strip()
    data = pd.DataFrame(columns=['group_domain', 'id', 'text', 'photo_url'])

    for group_domain in args.group_domains:
        responses = load_all_posts(key_access, group_domain)
        for response in responses:
            new_data = unpack_response(response, group_domain)
            if new_data is not None:
                data = pd.concat([data, new_data], axis=0)

    data = data.drop_duplicates()
    data.dropna(inplace=True, subset=['photo_url'])
    data = data.reset_index()
    data['uid'] = data.index
    data[['uid', 'group_domain', 'id', 'text', 'photo_url']].to_csv(args.output_file)
    print(f'Скачано {len(data)} постов')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', "--group_domains", nargs='+', required=True)
    parser.add_argument('-o', "--output-file", required=True)
    args = parser.parse_args()
    main(args)
