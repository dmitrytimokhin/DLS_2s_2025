import os
import logging

from dotenv import load_dotenv
from aiogram import Router, F
from aiogram.types import Message, CallbackQuery, FSInputFile
from aiogram.fsm.context import FSMContext

import torch
import whisper
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from TTS.tts.layers.xtts.trainer.gpt_trainer import XttsConfig
from TTS.tts.models.xtts import Xtts

from pydub import AudioSegment

from src import seed_torch, create_path, download_and_unzip
from src import download_missing_model_files

from handlers import FSMVoice
from keyboards import (kb_main, kb_voice_video, kb_voice_audio, kb_transcription_type, 
                       kb_translation_type, kb_reply_go_home, kb_go_home, kb_again)
from src import manage_directory
from src import full_pipeline, run_transcription, run_translation, gen_audio



user_router = Router()
logger = logging.getLogger(__name__)

load_dotenv()
FINETUNED_TTS_MODEL = os.getenv('FINETUNED_TTS_MODEL')

SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

INPUT_PATH = f"./data/"
OUTPUT_PATH = f"./data/"

WHISPER_NAME = "small"
MODEL_MT_NAME = "facebook/nllb-200-distilled-600M"
ORIGINAL_MODEL_TTS_DIR = "./original_tts_model/"
manage_directory(ORIGINAL_MODEL_TTS_DIR)

FINETUNED_MODEL_TTS_DIR = "./finetuned_tts_model/"
FINETUNED_CONFIG = create_path(FINETUNED_MODEL_TTS_DIR, "config.json")
FINETUNED_CHECKPOINT = create_path(FINETUNED_MODEL_TTS_DIR, "best_model.pth")
FINETUNED_REF = create_path(FINETUNED_MODEL_TTS_DIR, "putin_ref.wav")
manage_directory(FINETUNED_MODEL_TTS_DIR)

MAX_DURATION_MS = 20 * 1000

seed_torch(SEED)

LOAD_MODELS = True
if LOAD_MODELS:
    try:
        logger.info(f"Загрузка модели WHISPER: {WHISPER_NAME}")
        MODEL_ASR = whisper.load_model(name=WHISPER_NAME).to(DEVICE)
        logger.info("Модель ASR успешно загружена!")

        logger.info(f"Загрузка модели перевода: {MODEL_MT_NAME}")
        TOKENIZATOR_MT = AutoTokenizer.from_pretrained(MODEL_MT_NAME)
        MODEL_MT = AutoModelForSeq2SeqLM.from_pretrained(MODEL_MT_NAME).to(DEVICE)
        PIPELINE_MT = pipeline(
            task="translation",
            model=MODEL_MT,
            tokenizer=TOKENIZATOR_MT,
            src_lang="eng_Latn",
            tgt_lang="rus_Cyrl",
            max_length=1024,
            device=DEVICE)
        logger.info("Модель MT успешно загружена!")

        logger.info("Инициализируем конфиг и оригинальную модель XTTS")
        XTTS_FILES = download_missing_model_files(ORIGINAL_MODEL_TTS_DIR)
        orig_config = XttsConfig()
        orig_config.load_json(XTTS_FILES['config.json'])

        ORIG_MODEL_TTS = Xtts.init_from_config(orig_config)
        ORIG_MODEL_TTS.load_checkpoint(
            orig_config,
            checkpoint_path=XTTS_FILES['model.pth'],
            vocab_path=XTTS_FILES['vocab.json'],
            speaker_file_path=XTTS_FILES['speakers_xtts.pth'],
            eval=True
            )
        ORIG_MODEL_TTS.to(DEVICE)
        logger.info("Оригинальная модель XTTS_v2 успешно инициализированна!")

        logger.info("Инициализируем конфиг и дообученную модель XTTS")
        fine_config = XttsConfig()
        fine_config.load_json(FINETUNED_CONFIG)
        
        download_and_unzip(file_id=FINETUNED_TTS_MODEL)

        FINE_MODEL_TTS = Xtts.init_from_config(fine_config)
        FINE_MODEL_TTS.load_checkpoint(
            fine_config,
            checkpoint_dir=FINETUNED_MODEL_TTS_DIR,
            checkpoint_path=FINETUNED_CHECKPOINT,
            eval=True
        )
        FINE_MODEL_TTS.to(DEVICE)
        logger.info("Дообученная модель XTTS успешно инициализированна")

    except Exception as e:
        logger.error(f"Ошибка при загрузке одной из моделей: {e}")


# --- Handler Видео дубляж ---
@user_router.message(F.text=='Видео дубляж')
async def get_generation_video(message: Message, state: FSMContext):
    tg_user_id = message.from_user.id
    await state.clear()
    logger.debug(msg=f'Пользователь {tg_user_id} нажал "Видео дубляж"')
    # Устанавливаем состояние на tg_user_id
    await state.set_state(FSMVoice.tg_user_id)
    await state.update_data(tg_user_id=tg_user_id)
    # Устанавливаем состояние на media_type
    await state.set_state(FSMVoice.media_type)
    await state.update_data(media_type='video')
    # Утанавливаем состояние на флаг транскрипции
    await state.set_state(FSMVoice.transcription)
    await state.update_data(transcription=True)
    # Устанавливаем состояние на флаг перевода с английского на русский
    await state.set_state(FSMVoice.translate)
    await state.update_data(translate=True)
    # Устанавливаем состояние на флаг генерации видео
    await state.set_state(FSMVoice.generation)
    await state.update_data(generation=True)
    # Устанавливаем состояние на выбор голос видео
    await state.set_state(FSMVoice.voice_generation)
    need_text = ("Вы перешли в панель 'Видео дубляж'."
                 "\n\nПожалуйста, выберите голос для дубляжа \U00002B07")
    need_kb = kb_voice_video

    await message.reply(text=need_text,
                        reply_markup=need_kb)


# --- Callback FSMVoice.voice
@user_router.callback_query(F.data.startswith('voice_'), FSMVoice.voice_generation)
async def get_voice(callback: CallbackQuery, state: FSMContext):
    tg_user_id = callback.from_user.id
    logger.debug(msg=f'Пользователь {tg_user_id} нажал voice_')
    voice_name = callback.data.split('_')[1]
    await state.update_data(voice_generation=voice_name)
    data = await state.get_data()

    if voice_name == 'yours':
        await state.set_state(FSMVoice.voice_path)
        need_text = ("Вы выбрали озвучку с вашего аудио при помощи Zero-Short клонирования!"
                    "\n\nПожалуйста, запишите голосовое сообщение в ответном сообщении, "
                    "выразительно прочитав следующий отрывок из стихотворения или вернитесь на "
                    "главное меню:"
                    "\n\n<i>'У лукоморья дуб зелёный;"
                    "\nЗлатая цепь на дубе том:"
                    "\nИ днём и ночью кот учёный"
                    "\nВсё ходит по цепи кругом;"
                    "\nИдёт направо — песнь заводит,"
                    "\nНалево — сказку говорит."
                    "\nТам чудеса: там леший бродит,"
                    "\nРусалка на ветвях сидит'</i>")
    else:
        if data['media_type']=='video':
            await state.set_state(FSMVoice.video_path)
            need_text = ("Пожалуйста, отправьте видео (размером менее 20мб), для которого "
                         "необходимо сделать дубляж в ответном сообщении или вернитесь на главное "
                         "меню.")
        elif data['media_type']=='text':
            await state.set_state(FSMVoice.text_geneneration)
            need_text = need_text = ("Пожалуйста, отправьте текст, который необходимо озвучить!")
        else:
            await state.clear()
            need_text = ("Данный тип не поддерживается, возможен либо 'Видео дубляж', либо "
                         "'Озвучка текста'")

    need_kb = kb_reply_go_home
    
    await callback.answer(text="Далее...")
    await callback.message.reply(text=need_text,
                                 reply_markup=need_kb)


# --- Handler на голосовое сообщение ---
@user_router.message(F.voice, FSMVoice.voice_path)
async def get_voice_yours(message: Message, state: FSMContext):
    tg_user_id = message.from_user.id
    logger.debug(msg=f'Пользователь {tg_user_id} отправил голосове сообщение для ZS клонирования.')

    voice_file_id = message.voice.file_id

    output_dir = create_path(INPUT_PATH, f"tg_user_id_{tg_user_id}")
    manage_directory(output_dir, action='create')

    ogg_path = create_path(output_dir, f"voice.ogg")
    wav_path = create_path(output_dir, f"yours_ref.wav")

    file = await message.bot.get_file(voice_file_id)
    await message.bot.download_file(file.file_path, ogg_path)

    # Конвертируем в .wav
    audio = AudioSegment.from_ogg(ogg_path)
    os.remove(ogg_path)

    # Проверяем длительность
    if len(audio) > MAX_DURATION_MS:
        audio = audio[:MAX_DURATION_MS]  # обрезаем до 20 секунд
    audio.export(wav_path, format="wav")

    await state.update_data(voice_path=wav_path)

    data = await state.get_data()
    if data['media_type']=='video':
        await state.set_state(FSMVoice.video_path)
        need_text = ("Пожалуйста, отправьте видео (размером менее 20мб), для которого необходимо "
                     "сделать дубляж в ответном сообщении или вернитесь на главное меню.")
    elif data['media_type']=='text':
        await state.set_state(FSMVoice.text_geneneration)
        need_text = ("Пожалуйста, отправьте текст, который необходимо озвучить!")
    else:
        await state.clear()
        need_text = ("Данный тип не поддерживается, возможен либо 'Видео дубляж', либо "
                     "'Озвучка текста'")

    need_kb = kb_reply_go_home
    await message.reply(text=need_text,
                        reply_markup=need_kb)


# --- Handler на FSMVoice.video_path ---
@user_router.message(FSMVoice.video_path)
async def handle_video(message: Message, state: FSMContext):
    tg_user_id = message.from_user.id
    logger.debug(f'Пользователь {tg_user_id} отправил mdeia_type для дубляжа.')

    if message.video:
        media_obj = message.video
    else:
        await state.clear()
        need_text = ("Вы отправили не видеофайл. Пожалуйста начните сначала и строго следуйте "
                     "инструкциям!")
        await message.reply(text=need_text, reply_markup=kb_main)
        return

    # --- Создаем директорию для пользователя ---
    output_dir = create_path(INPUT_PATH, f"tg_user_id_{tg_user_id}")
    manage_directory(output_dir, action='create')

    # --- Обработка видео ---
    data = await state.get_data()

    if data['voice_generation'] == 'original':
        model_tts = ORIG_MODEL_TTS
        mode = 'original'
    elif data['voice_generation'] == 'putin':
        model_tts = FINE_MODEL_TTS
        mode = 'putin'
    else:
        model_tts = ORIG_MODEL_TTS
        mode = 'yours'

    video_file_id = media_obj.file_id
    video_path = create_path(output_dir, "video.mp4")

    file = await message.bot.get_file(video_file_id)
    await message.bot.download_file(file.file_path, video_path)
    await state.update_data(video_path=video_path)

    need_text = ("Ваша задача в очереди!"
                 "\n\nЯ скоро приступлю к анализу, и <b>обязательно</b> "
                 "вернусь к Вам с ответом, ожидайте пожалуйста! \U0001F60A")
#    need_kb = kb_go_home

    # --- Ответ пользователю ---
    await message.reply(text=need_text,
#                        reply_markup=need_kb
                        )

    data = await state.get_data()
    RESULT = await full_pipeline(
        model_asr=MODEL_ASR,
        pipeline_mt=PIPELINE_MT,
        model_tts=model_tts,
        tg_user_id=tg_user_id,
        mode=mode,
        device=DEVICE
    )

    # --- Отправляем результат ---
    final_video = FSInputFile(path=RESULT, filename="output.mp4")
    await message.answer_document(
        document=final_video,
        caption="Ваше видео с синхронным переводом!",
        reply_markup=kb_main
    )

    # --- Чистим временные файлы ---
    manage_directory(output_dir, action='delete')


# --- Handler Видео / аудио транскрипция ---
@user_router.message(F.text=='Видео / аудио транскрипция')
async def get_transcription(message: Message, state: FSMContext):
    tg_user_id = message.from_user.id
    await state.clear()
    logger.debug(msg=f'Пользователь {tg_user_id} нажал "Видео / аудио транскрипция"')
    # Устанавливаем состояние на tg_user_id
    await state.set_state(FSMVoice.tg_user_id)
    await state.update_data(tg_user_id=tg_user_id)
    # Устанавливаем состояние на выбор голос
    await state.set_state(FSMVoice.media_type)
    need_text = ("Вы перешли в панель 'Видео / аудио транскрипция'."
                 "\n\nПожалуйста, выберите тип медиа для которого необходимо получить "
                 "транскрипцию.")
    need_kb = kb_transcription_type

    await message.reply(text=need_text,
                        reply_markup=need_kb)


# --- Callback FSMVoice.voice
@user_router.callback_query(F.data.startswith('transcriprion_type_'), FSMVoice.media_type)
async def get_voice(callback: CallbackQuery, state: FSMContext):
    tg_user_id = callback.from_user.id
    logger.debug(msg=f'Пользователь {tg_user_id} нажал transcriprion_type_')
    transcriprion_type = callback.data.split('_')[2]
    await state.update_data(media_type=transcriprion_type)
    await state.set_state(FSMVoice.transcription)
    if transcriprion_type == 'video':
        need_text = ("Вы выбрали получить транскрипцию для видео. "
                     "\n\nПожалуйста, отправьте видео (размером менее 20мб) в ответном сообщении или "
                     "вернитесь на главное меню.")
        need_callback_text = 'Видео'
    else:
        need_text = ("Вы выбрали получить транскрипцию для голосового сообщения. "
                     "\n\nПожалуйста, отправьте голосовое сообщение в ответном сообщении или "
                     "вернитесь на главное меню")
        need_callback_text = 'Голосовое'
        
    need_kb = kb_reply_go_home
    await callback.answer(text=need_callback_text)
    await callback.message.reply(text=need_text,
                                 reply_markup=need_kb)


# --- Handler  ---
@user_router.message(FSMVoice.transcription)
async def handle_video(message: Message, state: FSMContext):
    tg_user_id = message.from_user.id
    logger.debug(f'Пользователь {tg_user_id} отправил mdeia_type для дубляжа.')
    await state.set_state(FSMVoice.process)
    data = await state.get_data()

    if message.video:
        media_obj = message.video
    elif message.voice:
        media_obj = message.voice
    else:
        return
    
    # --- Создаем директорию для пользователя ---
    output_dir = create_path(INPUT_PATH, f"tg_user_id_{tg_user_id}")
    manage_directory(output_dir, action='create')

    need_text = ("Ваша задача в обработке!"
                 "\n\nЯ скоро приступлю к анализу, и <b>обязательно</b> "
                 "вернусь к Вам с ответом, ожидайте пожалуйста! \U0001F60A")
#    need_kb = kb_go_home

    await message.reply(text=need_text,
#                        reply_markup=need_kb
                        )

    if data['media_type'] == 'video':

        video_file_id = media_obj.file_id
        video_path = create_path(output_dir, "video.mp4")

        file = await message.bot.get_file(video_file_id)
        await message.bot.download_file(file.file_path, video_path)

        need_text = await run_transcription(
            model_asr=MODEL_ASR,
            video=True,
            tg_user_id=tg_user_id,
            device=DEVICE)

    else:
        voice_file_id = message.voice.file_id

        ogg_path = create_path(output_dir, f"voice.ogg")
        wav_path = create_path(output_dir, f"yours_ref.wav")

        file = await message.bot.get_file(voice_file_id)
        await message.bot.download_file(file.file_path, ogg_path)

        # Конвертируем в .wav
        audio = AudioSegment.from_ogg(ogg_path)
        os.remove(ogg_path)
        audio.export(wav_path, format="wav")

        need_text = await run_transcription(
            model_asr=MODEL_ASR,
            video=False,
            tg_user_id=tg_user_id,
            device=DEVICE)
        
    need_kb = kb_main

    await message.reply(text=need_text,
                        reply_markup=need_kb)
    
    manage_directory(output_dir, action='delete')


# --- Handler Видео / текст перевод ---
@user_router.message(F.text=='Видео / текст перевод')
async def get_transcription_video(message: Message, state: FSMContext):
    tg_user_id = message.from_user.id
    await state.clear()
    logger.debug(msg=f'Пользователь {tg_user_id} нажал "Видео / текст перевод"')
    # Устанавливаем состояние на tg_user_id
    await state.set_state(FSMVoice.tg_user_id)
    await state.update_data(tg_user_id=tg_user_id)
    # Устанавливаем состояние на выбор голос
    await state.set_state(FSMVoice.media_type)
    need_text = ("Вы перешли в панель 'Видео / текст перевод'."
                 "\n\nПожалуйста, выберите тип медиа для которого необходимо получить "
                 "перевод.")
    need_kb = kb_translation_type

    await message.reply(text=need_text,
                        reply_markup=need_kb)


# --- Callback FSMVoice.voice
@user_router.callback_query(F.data.startswith('translation_type_'), FSMVoice.media_type)
async def get_voice(callback: CallbackQuery, state: FSMContext):
    tg_user_id = callback.from_user.id
    logger.debug(msg=f'Пользователь {tg_user_id} нажал translation_type_')
    transcriprion_type = callback.data.split('_')[2]
    await state.update_data(media_type=transcriprion_type)
    await state.set_state(FSMVoice.translate)
    if transcriprion_type == 'video':
        need_text = ("Вы выбрали получить перевод для видео. "
                     "\n\nПожалуйста, отправьте видео (размером менее 20мб) в ответном сообщении или "
                     "вернитесь на главное меню.")
        need_callback_text = 'Видео'
    else:
        need_text = ("Вы выбрали получить перевод для текста. "
                     "\n\nПожалуйста, отправьте текст на английском в ответном сообщении или "
                     "вернитесь на главное меню")
        need_callback_text = 'Текст'
        
    need_kb = kb_reply_go_home
    await callback.answer(text=need_callback_text)
    await callback.message.reply(text=need_text,
                                 reply_markup=need_kb)


# --- Handler  ---
@user_router.message(FSMVoice.translate)
async def handle_video(message: Message, state: FSMContext):
    tg_user_id = message.from_user.id
    logger.debug(f'Пользователь {tg_user_id} отправил mdeia_type для дубляжа.')
    await state.set_state(FSMVoice.process)
    data = await state.get_data()

    if message.video:
        media_obj = message.video
    elif message.text:
        media_obj = message.text
    else:
        return
    
    # --- Создаем директорию для пользователя ---
    output_dir = create_path(INPUT_PATH, f"tg_user_id_{tg_user_id}")
    manage_directory(output_dir, action='create')

    need_text = ("Ваша задача в обработке!"
                 "\n\nЯ скоро приступлю к анализу, и <b>обязательно</b> "
                 "вернусь к Вам с ответом, ожидайте пожалуйста! \U0001F60A")
#    need_kb = kb_go_home

    # --- Ответ пользователю ---
    await message.reply(text=need_text,
#                        reply_markup=need_kb
                        )

    if data['media_type'] == 'video':

        video_file_id = media_obj.file_id
        video_path = create_path(output_dir, "video.mp4")

        file = await message.bot.get_file(video_file_id)
        await message.bot.download_file(file.file_path, video_path)

        need_text = await run_translation(
            model_asr=MODEL_ASR,
            pipeline_mt=PIPELINE_MT,
            video=True,
            tg_user_id=tg_user_id,
            device=DEVICE)
    else:
        text = media_obj

        need_text = await run_translation(
            model_asr=MODEL_ASR,
            pipeline_mt=PIPELINE_MT,
            video=False,
            text=text,
            tg_user_id=tg_user_id,
            device=DEVICE)
        
    need_kb = kb_main

    await message.reply(text=need_text,
                        reply_markup=need_kb)
    
    manage_directory(output_dir, action='delete')


# --- Handler озвучить текст ---
@user_router.message(F.text=='Озвучка текста')
async def generation_audio(message: Message, state: FSMContext):
    tg_user_id = message.from_user.id
    await state.clear()
    logger.debug(msg=f'Пользователь {tg_user_id} нажал "Озвучить текст"')
    # Устанавливаем состояние на tg_user_id
    await state.set_state(FSMVoice.tg_user_id)
    await state.update_data(tg_user_id=tg_user_id)
    # Устанавливаем состояние на media_type
    await state.set_state(FSMVoice.media_type)
    await state.update_data(media_type='text')
    # Устанавливаем состояние на выбор голос видео
    await state.set_state(FSMVoice.voice_generation)
    need_text = ("Вы перешли в панель 'Озвучить текст'."
                 "\n\nПожалуйста, выберите голос спикера. \U00002B07")
    need_kb = kb_voice_audio

    await message.reply(text=need_text,
                        reply_markup=need_kb)


# --- Handler на получение текста для генерации
@user_router.message(FSMVoice.text_geneneration)
async def handle_text(message: Message,  state: FSMContext):
    tg_user_id = message.from_user.id
    logger.debug(msg=f'Пользователь {tg_user_id} отправил текст для генерации.')

    output_dir = create_path(INPUT_PATH, f"tg_user_id_{tg_user_id}")
    manage_directory(output_dir, action='create')

    ogg_path = os.path.join(output_dir, "final_voice.ogg")

    text = message.text
    await state.update_data(text_geneneration=text)
    await state.set_state(FSMVoice.process)

    data = await state.get_data()
    if data['voice_generation'] == 'putin':
        model_tts = FINE_MODEL_TTS
        mode = 'putin'
    else:
        model_tts = ORIG_MODEL_TTS
        mode = 'yours'
    

    need_text = ("Ваша задача в обработке!"
                 "\n\nЯ скоро приступлю к анализу, и <b>обязательно</b> "
                 "вернусь к Вам с ответом, ожидайте пожалуйста! \U0001F60A")
#    need_kb = kb_go_home

    await message.reply(text=need_text,
#                        reply_markup=need_kb
                        )

    RESULT = await gen_audio(
        model_tts=model_tts,
        text=text,
        tg_user_id=tg_user_id,
        mode=mode,
        device=DEVICE)


    audio = AudioSegment.from_wav(RESULT)

    audio.export(ogg_path, format="ogg", codec="libopus")

    voice = FSInputFile(ogg_path)

    await message.answer_voice(voice=voice)
    await message.reply(text='Повторить с этим же голосом?', reply_markup=kb_again)


# --- Callback FSMVoice.voice
@user_router.callback_query(F.data.in_({'yes', 'go_home'}), FSMVoice.process)
async def handle_text_again(callback: CallbackQuery, state: FSMContext):
    tg_user_id = callback.from_user.id
    callback_data = callback.data
    logger.debug(msg=f'Пользователь {tg_user_id} нажал {callback_data}')

    output_dir = create_path(INPUT_PATH, f"tg_user_id_{tg_user_id}")

    if callback_data == 'yes':
        await state.set_state(FSMVoice.text_geneneration)
        need_text = ("Пожалуйста, отправьте текст, который необходимо озвучить!")
        need_kb = kb_reply_go_home
    else:
        need_text = ("Вы вернулись на главное меню!"
                     "\n\nПожалуйста, выберите пункт в меню на <b>клавиатуре</b> \U00002B07")
        need_kb = kb_main
        
        manage_directory(output_dir, action='delete')

    await callback.answer(text=callback_data)
    await callback.message.answer(text=need_text, reply_markup=need_kb)
