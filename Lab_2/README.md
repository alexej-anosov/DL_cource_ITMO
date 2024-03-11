# Лабораторная работа №2

## Цель работы
Научиться работать с предобученными моделями и на основе предобученных эмбеддингов строить новые модели.

## Данные

Был взят [Youtube Videos Dataset](https://www.kaggle.com/datasets/rajatrc1705/youtube-videos-dataset). Рассматривалась задача предсказания категории видео эмбеддингам его заголовка.

## Эмбеддинги

Ддя получения эмбеддингов была взята предобученная модель [Bert](https://huggingface.co/google-bert/bert-base-uncased). В качестве возможных фичей рассматривались эмбеддинги CLS токена с 12 Encoder-слоев модели.


## Модель

Так как задача довольно простая, и слишком легко решалаись при использовании исходных эмбеддингов размера 768, в начале модели было решено добавить 1-d свертку (kernel=3, stride=3), чтобы снизить размерность.
Основная часть модели состояла из двух полносвязных слоев с функцией акативации ReLU. Размер скрытого слоя 512.

В случае исполоьзования выходов из нескольких слоев, сначала применялась point-wise свертка c out_channels=1.


## Эксперименты

В ходе выполнения работы в качестве фичей использовались выходы последних 4 слоев по отдельности, а также объединенные последние 2 слоя и последние 4 слоя.


## Результаты

Лучше всего себя показали выходы с последнего слоя. ОБъединенные выходы с последних двух слоев имеют сопоставимые результаты. Возможно, объединение с другими слоями не дало прироста в связи с тем, что задача довольно простая и не требует дополнительной информации, которая может содержаться на боеле глубоких слоях, или способ объединения эмбеддингов оказался недостаточно эффективным. Однако, в рамках данной задачи более сложные конструкции не имеют большого смысла.

| Используемые слои      | F1-macro последней эпохи |
|:----------------------:|:------------------------:| 
| 12                     | 0.9565                   |
| 11                     | 0.9032                   | 
| 10                     | 0.9034                   |
| 9                      | 0.9023                   | 
| 12, 11                 | 0.9546                   |
| 12, 11, 10, 9          | 0.9114                   | 