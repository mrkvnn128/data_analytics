import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand
from dotenv import load_dotenv
import os
from handlers import commands
# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения из .env
load_dotenv()
API_TOKEN = os.getenv('BOT_TOKEN')

# Инициализация бота и диспетчера
bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Регистрация обработчиков
dp.include_router(commands.router)

async def set_commands(bot: Bot):
    commands = [
        BotCommand(command="/start", description="Запустить/перезапустить бота"),
        BotCommand(command="/predict", description="Прогноз курса доллара"),
        BotCommand(command="/metrics", description="Метрики качества модели"),
        BotCommand(command="/last", description="Последний известный курс"),
        BotCommand(command="/help", description="Список команд"),
        BotCommand(command="/correlation", description="Матрица корреляции признаков")
    ]
    await bot.set_my_commands(commands)
    logger.info("Меню команд настроено")

async def main():
    logger.info("Бот запущен")
    await set_commands(bot)
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())