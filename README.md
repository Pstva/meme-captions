#  DL проект: Описание мемов по картинке


Milestone лежит [тут](Milestone.md)

Структура репозитория:


## Данные

[Данные](data)

[Скрипты для загрузки и подготовки данных](src/prepare_data)

## Бейзлайновая модель Encoder(ResNet-50) + Decoder (LSTM)

[Код для обучения](src/baseline/train_baseline.py)

[Код для оценки моделей](src/baseline/eval_baseline.py)

[Логи обучения и оценка на тестовых данных](output/baseline.log)


## Бейзлайновая модель 2 Encoder(ResNet-101) + Decoder (LSTM)

[Код для обучения](src/baseline/train_baseline2.py)

[Код для оценки моделей](src/baseline/eval_baseline2.py)

[Логи обучения и оценка на тестовых данных](output/baseline2.log)

## TODO: ruCLIP + ruGPT

[Код для обучения](src/clip_model/train.py)


TODO: код для использования моделей генерации описания
