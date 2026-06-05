import logging

from aiogram import Router, F
from aiogram.filters import CommandStart, or_f

from aiogram.types import Message, CallbackQuery
from aiogram.fsm.context import FSMContext

from keyboards import kb_main


base_router = Router()
logger = logging.getLogger(__name__)


# --- Старт телеграм бота ---
@base_router.message(or_f(CommandStart(), F.text == '/start'))
async def cmd_start(message: Message, state: FSMContext):
    tg_user_id = message.from_user.id
    await state.clear()
    logger.debug(msg=f'Пользователь {tg_user_id} нажал /start')
    need_text = ("Добро пожаловать! Я бот-ассистент, созданный в рамках финального "
                 "проекта Deep Learning School (семестр 2, 2025) - 'Video Dubbing'" 
                 "\n\nЯ могу помочь перевести тебе видео с англиского на русский, получить "
                 "транскрипцию видео, его перевод, а также сгенерировать аудио с помощью: "
                 "\n\n1) <b>Дообученой</b> модели XTTS_v2 на голосе В.В. Путина (Few-Short learning); "
                 "\n2)<b>Оригинальной</b> XTTS_v2 модели твоего любого голоса (Zero-Short "
                 "learning)"
                 "\n\nОбщение со мной происходит посредством клавиатуры, кнопок и команд! "
                 "\U00002B07")
    
    need_kb = kb_main

    await message.answer(
        text=need_text,
        reply_markup=need_kb)
    

# --- Handler на главное меню и команду cancel ---
@base_router.message(F.text.in_({'На главное меню', '/cancel'}))
async def go_home(message: Message, state: FSMContext):
    tg_user_id = message.from_user.id
    await state.clear()
    logger.debug(msg=f'Пользователь {tg_user_id} нажал /cancel')
    need_text = ("Вы вернулись на главное меню!"
                 "\n\nПожалуйста, выберите пункт в меню на <b>клавиатуре</b> \U00002B07")
    
    need_kb = kb_main

    await message.reply(text=need_text,
                        reply_markup=need_kb)


# --- Callback на главное меню ---
@base_router.callback_query(F.data == 'go_home')
async def go_home(callback: CallbackQuery, state: FSMContext):
    tg_user_id = callback.from_user.id
    await state.clear()
    logger.debug(msg=f'Пользователь {tg_user_id} нажал /go_home')
    callback_text = "Вы нажали 'На главную'"
    need_text = ("Вы вернулись на главное меню!"
                 "\n\nПожалуйста, выберите пункт в меню на <b>клавиатуре</b> \U00002B07")
    
    need_kb = kb_main

    await callback.answer(text=callback_text)
    await callback.message.reply(text=need_text,
                                 reply_markup=need_kb)
    

@base_router.message()
async def unknow_handler(message: Message, state: FSMContext):
    tg_user_id = message.from_user.id
    await state.clear()
    logger.debug(msg=f'Пользователь {tg_user_id} отправил неизвестный handler')
    need_text = ("Я вас не понимаю \U0001F614"
                 "\n\nОбщение со мной происходит посредством клавиатуры, кнопок и команд! "
                 "\U00002B07")
    need_kb = kb_main

    await message.answer(text=need_text,
                         reply_markup=need_kb)


@base_router.callback_query()
async def unknow_callback(callback: CallbackQuery, state: FSMContext):
    tg_user_id = callback.from_user.id
    await state.clear()
    logger.debug(msg=f'Пользователь {tg_user_id} нажал неизвестный callback')
    callback_text = ("Вы нажимали 'На главную'"
                     "\n\nПожалуйста, начните с главного меню \U00002B07")

    await callback.answer(text=callback_text,
                          show_alert=True)
