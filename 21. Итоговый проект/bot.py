import os
import logging

import asyncio
import contextlib

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from aioutils import set_commands
from handlers import base_router, user_router


logging.basicConfig(format="%(asctime)s - [%(levelname)s] -  %(funcName)s - %(message)s",
                    datefmt="%d/%m/%Y %H:%M:%S",
                    level=logging.INFO)

logger = logging.getLogger(__name__)

load_dotenv()
TOKEN = os.getenv('TOKENBOT')
MAIN_ADMIN = int(os.getenv("MAIN_ADMIN_ID"))

async def start_bot(bot: Bot):
    await set_commands(bot)
    try:
        await bot.send_message(chat_id=MAIN_ADMIN, text='The VideoDubBot запущен!')
        logger.info(msg='Бот VideoDubBot запущен!')
    except Exception as e:
        logger.error(f'[Exception] - {e}', exc_info=True)


async def stop_bot(bot: Bot):
    try:
        await bot.send_message(chat_id=MAIN_ADMIN, text='Бот VideoDubBot остановлен!')
        logger.info(msg='Бот VideoDubBot остановлен!')
    except Exception as e:
        logger.error(msg=f'[Exception] - {e}', exc_info=True)


async def main():
    logger.info("Запуск бота VideoDubBot!")
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await bot.delete_webhook()
    dp_storage = MemoryStorage()
    dp = Dispatcher(storage=dp_storage)
    dp.startup.register(start_bot)
    dp.shutdown.register(stop_bot)
    dp.include_routers(user_router, base_router)

    try:
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(msg=f'[Exception] - {e}', exc_info=True)
    finally:
        await bot.session.close()
        await dp.storage.close()


if __name__ == '__main__':
    with contextlib.suppress(KeyboardInterrupt, SystemExit):
        asyncio.run(main())
        