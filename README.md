#  DL проект: Описание мемов по картинке


Milestone лежит [тут](Milestone.md)

Структура репозитория:


## Данные

[Данные](data)

[Скрипты для загрузки и подготовки данных](src/prepare_data)

## Бейзлайновая модель Encoder(ResNet-50) + Decoder (LSTM)

[Код для обучения](src/baseline/train_baseline.py)

[Код для оценки моделей](src/baseline/eval_baseline.py)


## Бейзлайновая модель 2 Encoder(ResNet-101) + Decoder (LSTM)

[Код для обучения](src/baseline/train_baseline2.py)

[Код для оценки моделей](src/baseline/eval_baseline2.py)

## ruCLIP + MLP + ruGPT

[Код для обучения](src/clip_model/train.py)

[Код для оценки моделей](src/clip_model/evaluate.py)

## RuDOLPH

[Блокнот с дообучениeм и оценкой](src/finetune_eval_rudolph.ipynb)
