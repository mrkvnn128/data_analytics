from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

def get_days_keyboard():
    buttons = [[KeyboardButton(text=str(i))] for i in range(1, 8)]
    keyboard = ReplyKeyboardMarkup(keyboard=buttons, resize_keyboard=True)
    return keyboard