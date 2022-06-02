#  DL проект: Описание мемов по картинке

Структура репозитория:

## Данные

[Данные](data/)

[Код для загрузки и подготовки данных](src/prepare_data.py)

## Бейзлайновая модель Encoder(ResNet-50) + Decoder (LSTM)

[Код для обучения](src/baseline/train_baseline.py)

[Код для оценки моделей](src/baseline/eval_baseline.py)

[Логи обучения и оценка на тестовых данных](output/baseline.log)

## TODO: ruCLIP + ruGPT

[Код для обучения](src/clip_model/train.py)


TODO: код для использования моделей генерации описания
