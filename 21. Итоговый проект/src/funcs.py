import os
import logging
from typing import NoReturn, Optional, Tuple
import shutil
import random
import numpy as np
import subprocess
import torch
from pydub import AudioSegment

import gdown
import zipfile

logger = logging.getLogger(__name__)


def seed_torch(seed: int=42) -> NoReturn:
    """
    Фиксирует все seed по библиотекам random, os, numpy и torch для воспроизводимости результатов
    
    Параметры:
        seed (int): значение фиксации.
    """
    try:
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.info(f"Все библиотеки зафиксированы с seed= {seed}")

    except Exception as e:
        logger.error(f"Фиксирование seed с ошибкой: {e}")


def create_path(dir_name: str='./', file_name: str='./') -> str:
    """
    Создает наименование пути

    Параметры:
        dir_name (str): наименование директории
        file_name (str): наименование файла
    Вовзращает:
        str: наименование созданного пути
    """
    try:
        path_name = os.path.join(dir_name, file_name)
        logger.info(F"Путь {path_name} создан.")
        return path_name
    except Exception as e:
        logger.error("Ошибка при создании наименования пути")
        return ""


def manage_directory(directory_path: str='./',
                     action: str="create") -> NoReturn:
    """
    Создает или удаляет директорию по указанному пути.

    Параметры:
        directory_path (str): путь к директории.
        action (str): действие — 'create' для создания, 'delete' для удаления.

    Исключения:
        ValueError: если указано неверное действие.
    """
    if action == "create":
        os.makedirs(directory_path, exist_ok=True)
        logger.info(f"Директория '{directory_path}' успешно создана или уже существует.")
    
    elif action == "delete":
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
            logger.info(f"Директория '{directory_path}' успешно удалена.")
        else:
            logger.info(f"Директория '{directory_path}' не существует.")
    
    else:
        raise ValueError("Неверное действие. Используйте 'create' или 'delete'.")


def extract_audio_from_video(video_path: str,
                             audio_path: str) -> NoReturn:
    """
    Извлекает аудиодорожку из видео с помощью FFmpeg.

    Параметры:
        video_path (str): путь к исходному видеофайлу
        audio_path (str): путь для сохранения аудиофайла (например, .wav или .mp3)

    Исключения:
        RuntimeError: если не удалось извлечь аудио
    """
    logger.info(f"Извлечение аудио из {video_path} → {audio_path}")

    if os.path.exists(audio_path):
        logger.info("Аудио уже существует, пропуск.")
        return

    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-q:a", "0",
        "-map", "a",
        "-y",
        audio_path
    ]

    try:
        logger.debug(f"Выполняется команда: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True
        )
        logger.info("Аудио успешно извлечено из видео.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка при извлечении аудио:\n{e.stdout}")
        raise RuntimeError(f"Не удалось извлечь аудио из видео: {e}")


def mix_audio_tracks(
    voice_over_path: str,
    background_path: str,
    output_path: str,
    original_audio_path: Optional[str] = None,
    voice_gain: float = 0.0,
    background_gain: float = -10.0,
    original_gain: float = -16.0
) -> NoReturn:
    """
    Смешивает дубляж, фоновую музыку и оригинал с помощью pydub.

    Параметры:
        voice_over_path: путь к русскоязычному дубляжу
        background_path: путь к фоновой музыке/шуму
        original_audio_path: путь к оригинальному аудио (опционально)
        voice_gain: громкость дубляжа
        background_gain: громкость фона
        original_gain: громкость оригинала
    """
    logger.info("Загрузка аудиодорожек...")

    # Загружаем дорожки
    voice_audio = AudioSegment.from_wav(voice_over_path).apply_gain(voice_gain)
    background_audio = AudioSegment.from_wav(background_path).apply_gain(background_gain)

    # Создаём фон нужной длины
    max_duration = len(voice_audio)
    background_audio = (background_audio * (max_duration // len(background_audio) + 2))[:max_duration]

    # Накладываем дубляж на фон
    full_audio = background_audio.overlay(voice_audio)

    # Если есть оригинал — добавляем его как фон
    if original_audio_path and os.path.exists(original_audio_path):
        original_audio = AudioSegment.from_wav(original_audio_path).apply_gain(original_gain)
        full_audio = full_audio.overlay(original_audio)


    # Сохраняем результат
    full_audio.export(output_path, format="wav")
#    os.remove(voice_over_path)
    logger.info(f"Аудио смикшировано и сохранено: {output_path}")


def add_audio_to_video(
    video_path: str,
    audio_path: str,
    output_video_path: str
) -> NoReturn:
    """
    Добавляет аудиодорожку к видео через FFmpeg.

    Параметры:
        video_path: путь к исходному видео
        audio_path: путь к новому аудиофайлу
        output_video_path: путь к выходному видео
    """
    logger.info("Добавление аудио к видео...")

    # Проверяем наличие файлов
    if not os.path.exists(video_path):
        logger.error(f"Видео не найдено: {video_path}")
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    if not os.path.exists(audio_path):
        logger.error(f"Аудиофайл не найден: {audio_path}")
        raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")

    # Подготовка директории для выходного файла
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Команда FFmpeg
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-y",  # перезапись без вопросов
        output_video_path
    ]

    try:
        logger.info(f"Выполняется команда: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#        os.remove(audio_path)
        logger.info(f"Видео сохранено: {output_video_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка при обработке видео: {e.stdout.decode()}")
        raise RuntimeError("Не удалось добавить аудио к видео")


def download_and_unzip(file_id: str,
                       output_zip: str='finetuned_tts_model.zip',
                       extract_to: str='./',
                       folder_name: str='finetuned_tts_model') -> Tuple[bool, str]:
    """
    Скачивает и распаковывает файлы дообученной модели на голосе В.В. Путина
    
    Параметры:
        file_id (str): id файла в google drive.
        output_zip (str): наименование .zip файла для сохранения.
        extract_to (str): путь для распаковки .zip файла.
        folder_name (str): наименование директории для проверки на наличие перед скачиванием
    """
    
    try:
        target_folder = create_path(extract_to, folder_name)

        # Проверяем, существует ли уже нужная папка
        if os.path.exists(target_folder):
            logger.info(f"Папка {target_folder} уже существует. Пропускаем скачивание.")
            return

        url = f" https://drive.google.com/uc?id={file_id}"
        logger.info(f"Скачивание 'finetuned_tts_model' по {url}...")
        gdown.download(url, output_zip, quiet=False)
        
        logger.info(f"Распаковка 'finetuned_tts_model' {output_zip} в {extract_to}...")
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        logger.debug(f"Очистка: удаление {output_zip}")
        os.remove(output_zip)
    except Exception as e:
        logger.error(f'Ошибка при загрузке файлов дообученной модели {e}')
        return False, target_folder
