# Mmm — Football Over Predictor

Простой проект: Telegram-бот для прогнозов тотала (Over 1.5 / 2.5).

Что будет:
- Парсеры для sstats.net, soccer365.ru, 4score.ru
- Агрегация коэффициентов, анализ комментариев
- Feature engineering и модель для P(total > X)
- Telegram-бот с командами /today и /predict

Как запустить (локально):
1. Скопировать `.env.example` -> `.env` и вставить TELEGRAM_TOKEN
2. Установить зависимости: pip install -r requirements.txt
3. Поместить исторические данные в data/matches.csv
4. Обучить модель: python train.py
5. Запустить бота: python bot.py

Примечание: после добавления README дай знать — я создам ветку feature/football-over и открою PR с кодом.
