import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
from aiogram import Router, types
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardRemove, FSInputFile
from keyboards.reply import get_days_keyboard
from models.models import predict_combined, calculate_metrics
import joblib
from datetime import datetime, timedelta

# Настройка логирования
logger = logging.getLogger(__name__)

# Создаем роутер для обработчиков
router = Router()

# Загрузка данных и scaler
data = pd.read_csv('latest_data.csv', index_col='data', parse_dates=True)
scaler = joblib.load('scaler.joblib')

# Колонки для предсказания
FEATURES = ['gold', 'silver', 'palladium', 'key interest rate', 'inflation', 
            'curs eur', 'brent (in usd)', 'curs usd_lag1', 'curs usd_lag2', 
            'curs usd_lag3', 'curs usd_lag4', 'curs usd_lag5', 'curs usd_lag6', 
            'curs usd_lag7']

# Основные признаки для корреляции (без лагов)
CORRELATION_FEATURES = ['curs usd', 'gold', 'silver', 'palladium', 
                        'key interest rate', 'inflation', 'curs eur', 'brent (in usd)']

# Функция проверки торгового дня
def is_trading_day(date):
    # Получаем день недели (0 = понедельник, 6 = воскресенье)
    weekday = date.weekday()
    # Считаем воскресенье (6) и понедельник (0) неторговыми днями
    if weekday in [0, 6]:
        logger.info(f"Дата {date} неторговая (день недели: {weekday})")
        return False
    # Если дата в прошлом и присутствует в данных, проверяем её
    trading_dates = pd.to_datetime(data.index).date
    if date in trading_dates:
        logger.info(f"Дата {date} найдена в данных, считается торговой")
        return True
    # Для будущих дат предполагаем, что день торговый, если не выходной
    logger.info(f"Дата {date} будущая, считается торговой")
    return True

# Обработчик команды /start
@router.message(Command('start'))
async def send_welcome(message: Message):
    logger.info(f"Получена команда /start от пользователя {message.from_user.id}")
    await message.reply(
        "Привет! Я бот для прогнозирования курса доллара США к рублю. "
        "Мои команды:\n"
        "/predict - прогноз курса доллара на выбранное количество дней (или просто введите число от 1 до 7)\n"
        "/metrics - посмотреть метрики качества\n"
        "/last - последний курс\n"
        "/correlation - матрица корреляции признаков\n"
        "/help - список команд и информация о боте",
        reply_markup=ReplyKeyboardRemove()
    )

# Обработчик команды /help
@router.message(Command('help'))
async def send_help(message: Message):
    logger.info(f"Получена команда /help от пользователя {message.from_user.id}")
    await message.reply(
        "Доступные команды:\n"
        "/start - запустить/перезапустить бота\n"
        "/predict - прогноз курса доллара на выбранное количество дней (или просто введите число от 1 до 7)\n"
        "/metrics - метрики качества модели\n"
        "/last - последний известный курс\n"
        "/correlation - матрица корреляции признаков\n"
        "/help - этот список\n\n"
        "Я бот, который строит прогнозы курса USD к RUB на 1–7 дней вперед. Использую комбинированную модель машинного обучения: линейную регрессию, случайный лес, SVR и нейронную сеть. Точность оцениваю через метрики R², MAE, MAPE, MSE, RMSE. Для прогноза учитываю макроэкономические показатели (цену на золото, серебро, палладий, цену на нефть марки Brent, ключевую ставку, инфляцию, курс EUR) и исключаю неторговые дни (воскресенье, понедельник)",
        reply_markup=ReplyKeyboardRemove()
    )

# Обработчик команды /correlation
@router.message(Command('correlation'))
async def send_correlation_matrix(message: Message):
    logger.info(f"Получена команда /correlation от пользователя {message.from_user.id}")
    try:
        # Загружаем полные данные
        df = pd.read_csv('df_d2.csv', index_col='data', parse_dates=True)
        
        # Выбираем только основные признаки для корреляции
        corr_matrix = df[CORRELATION_FEATURES].corr()

        # Создаем тепловую карту
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                    square=True, cbar=True, annot_kws={'size': 8})
        plt.title('Correlation Matrix')
        plt.tight_layout()
        
        # Сохраняем изображение
        image_path = 'correlation_matrix.png'
        plt.savefig(image_path)
        plt.close()
        
        # Проверяем, существует ли файл
        if not os.path.exists(image_path):
            logger.error(f"Файл {image_path} не найден")
            await message.reply(
                "Ошибка: не удалось создать изображение матрицы корреляции. Попробуйте позже",
                reply_markup=ReplyKeyboardRemove()
            )
            return
        
        # Отправляем изображение
        await message.reply_photo(
            photo=FSInputFile(image_path),
            caption="Матрица корреляции признаков",
            reply_markup=ReplyKeyboardRemove()
        )

        # Описание признаков
        description = (
            "Описание признаков:\n"
            "- curs usd: Курс доллара США к рублю\n"
            "- gold: Цена золота (в рублях за грамм)\n"
            "- silver: Цена серебра (в рублях за грамм)\n"
            "- palladium: Цена палладия (в рублях за грамм)\n"
            "- key interest rate: Ключевая ставка\n"
            "- inflation: Уровень инфляции\n"
            "- curs eur: Курс евро к рублю\n"
            "- brent (in usd): Цена на нефть Brent (в USD)"
        )
        await message.answer(description, reply_markup=ReplyKeyboardRemove())

        # Удаляем временный файл
        os.remove(image_path)
        logger.info(f"Матрица корреляции успешно отправлена пользователю {message.from_user.id}")
        
    except Exception as e:
        logger.error(f"Ошибка при генерации матрицы корреляции: {str(e)}")
        await message.reply(
            "Произошла ошибка при генерации матрицы корреляции. Попробуйте позже.",
            reply_markup=ReplyKeyboardRemove()
        )

# Обработчик команды /last
@router.message(Command('last'))
async def send_last(message: Message):
    logger.info(f"Получена команда /last от пользователя {message.from_user.id}")
    last_date = data.index[-1].strftime('%Y-%m-%d')
    last_curs = data['curs usd'].iloc[-1]
    await message.reply(
        f"Последний курс доллара на {last_date}: {last_curs:.2f} руб.",
        reply_markup=ReplyKeyboardRemove()
    )

# Обработчик команды /metrics
@router.message(Command('metrics'))
async def send_metrics(message: Message):
    logger.info(f"Получена команда /metrics от пользователя {message.from_user.id}")
    # Для примера берем последние тестовые данные
    X_test = data.iloc[-100:][FEATURES]
    y_true = data.iloc[-100:]['curs usd']
    y_pred = predict_combined(X_test)
    metrics = calculate_metrics(y_true, y_pred)
    
    response = (
        "Метрики качества комбинированной модели:\n"
        f"R²: {metrics['R²']:.4f}\n"
        f"MAE: {metrics['MAE']:.4f}\n"
        f"MAPE: {metrics['MAPE']:.2f}%\n"
        f"MSE: {metrics['MSE']:.4f}\n"
        f"RMSE: {metrics['RMSE']:.4f}"
    )
    await message.reply(response, reply_markup=ReplyKeyboardRemove())

# Обработчик команды /predict
@router.message(Command('predict'))
async def predict_curs(message: Message):
    logger.info(f"Получена команда /predict от пользователя {message.from_user.id}")
    await message.reply(
        "Выберите день для прогноза (1-7 дней). Для этого нажмите кнпку на интерактивной клавиатуре, или отправьте цифру от 1 до 7",
        reply_markup=get_days_keyboard()
    )

# Обработчик выбора дня для прогноза
@router.message(lambda message: message.text in [str(i) for i in range(1, 8)])
async def process_days(message: Message):
    logger.info(f"Пользователь {message.from_user.id} выбрал прогноз на {message.text} дней")
    days = int(message.text)
    last_data = data.iloc[-1][FEATURES].copy()
    predictions = []
    
    for day in range(1, days + 1):
        pred_date = data.index[-1] + timedelta(days=day)
        pred_date_str = pred_date.strftime('%Y-%m-%d')
        
        # Проверяем, является ли день торговым
        if not is_trading_day(pred_date.date()):
            predictions.append(f"На {pred_date_str} не было торгов.")
            continue
        
        # Подготовка данных для предсказания
        data_scaled = scaler.transform(last_data.values.reshape(1, -1))
        pred = predict_combined(last_data.values.reshape(1, -1))[0]
        predictions.append(f"Прогноз на {pred_date_str}: {pred:.2f} руб.")
        
        # Обновляем лаги для следующего дня
        last_data['curs usd_lag7'] = last_data['curs usd_lag6']
        last_data['curs usd_lag6'] = last_data['curs usd_lag5']
        last_data['curs usd_lag5'] = last_data['curs usd_lag4']
        last_data['curs usd_lag4'] = last_data['curs usd_lag3']
        last_data['curs usd_lag3'] = last_data['curs usd_lag2']
        last_data['curs usd_lag2'] = last_data['curs usd_lag1']
        last_data['curs usd_lag1'] = pred
        
        # Предполагаем, что остальные признаки остаются неизменными

    await message.reply("\n".join(predictions), reply_markup=get_days_keyboard())

# Обработчик случайных сообщений
@router.message()
async def handle_unknown(message: Message):
    logger.warning(f"Получено неизвестное сообщение '{message.text}' от пользователя {message.from_user.id}")
    await message.answer(
        "Я знаю только команды и числа от 1 до 7 \nИспользуйте /help для открытия списка команд",
        reply_markup=ReplyKeyboardRemove()
    )