import os
import logging
from typing import Tuple
import numpy as np
import subprocess

import noisereduce as nr
from pydub import AudioSegment
from scipy.io.wavfile import read, write


logger = logging.getLogger(__name__)


def separate_audio_sources(
    input_audio_path: str,
    temp_dir: str,
    model_name: str = "htdemucs",
    output_format: str = "wav",
    device: str = "cpu"
    ) -> Tuple[str]:
    """
    Разделяет аудиофайл на голос и фоновые звуки с использованием Demucs.
    
    Параметры:
        input_audio_path (str): путь к исходному аудиофайлу.
        temp_dir (str): путь к временным файлам.
        model_name (str): модель Demucs для использования. По умолчанию 'htdemucs'.
        output_format (str): формат выходных файлов (например, 'wav', 'mp3').
        device (str): выбор ГПУ / ЦПУ

    Возвращает:
        Tuple[str]: кортеж из пути к голосу и фоновым звукам из аудио.
    """
    if not os.path.exists(input_audio_path):
        logger.error(f"Файл {input_audio_path} не найден.")

    output_dir = os.path.dirname(input_audio_path)

    try:

        cmd = [
            "demucs",
            "-n", model_name,
            "--out", output_dir,
            input_audio_path
        ]

        
        if device == 'cuda':
            cmd.append("-d")
            cmd.append(device)
            

        # Добавляем опции для формата
        if output_format == "mp3":
            cmd.append("--mp3")
        elif output_format == "flac":
            cmd.append("--flac")
        elif output_format != "wav":
            logger.warning(f"Неподдерживаемый формат: {output_format}. Используется WAV.")
            
        logger.info(f"Запуск команды: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        # После выполнения перемещаем нужные дорожки
        base_name = os.path.splitext(os.path.basename(input_audio_path))[0]
        stems_dir = os.path.join(output_dir, model_name, base_name)

        # Перемещаем вокал и инструментал
        voice_source = os.path.join(stems_dir, "vocals.wav")
        background_source = os.path.join(stems_dir, "other.wav")

        if not os.path.exists(voice_source) or not os.path.exists(background_source):
            logger.error("Demucs не создал ожидаемые файлы (vocals.wav или other.wav).")

        output_voice_path = os.path.join(temp_dir, f"vocals.wav")
        output_background_path = os.path.join(temp_dir, f"background.wav")
        
        # Копируем или перемещаем файлы в целевые пути
        os.rename(voice_source, output_voice_path)
        os.rename(background_source, output_background_path)

        logger.info(f"Голос сохранён в: {output_voice_path}")
        logger.info(f"Фоновые звуки сохранены в: {output_background_path}")

        return (output_voice_path, output_background_path)
        
    except Exception as e: 
        logger.error(f"Ошибка: {e}")
        return ()
    

def preprocess_audio_for_asr(
    input_path: str,
    target_sample_rate: int = 16000,
    noise_reduce: bool = True,
    gain_increase: float = 10.0,
    prop_decrease: float = 0.75,       # ← доля подавления шума (меньше — осторожнее)
    n_fft: int = 512,
    hop_length: int = 512
) -> str:
    """
    Предобрабатывает аудиофайл для ASR: изменяет частоту, делает моно, применяет денойзинг и усиливает громкость.

    Параметры:
        input_path (str): путь к исходному аудиофайлу.
        target_sample_rate (int): целевая частота дискретизации.
        noise_reduce (bool): флаг необходимости удаления шума.
        gain_increase (float): усиление громкости в дБ.
        prop_decrease (float): степень подавления шума (от 0 до 1).
        n_fft (int): размер FFT для анализа спектра.
        hop_length (int): шаг между фреймами.

    Возвращает:
        str - путь к обработанному файлу.
    """
    if not os.path.exists(input_path):
        logger.error(f"Файл {input_path} не найден.")
        return ''

    output_dir = os.path.dirname(input_path)
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Конвертируем аудио в нужный формат
        audio = AudioSegment.from_file(input_path)

        # Повышаем громкость
        audio = audio.apply_gain(gain_increase)

        # Преобразуем в моно
        audio = audio.set_channels(1)

        # Изменяем частоту дискретизации
        audio = audio.set_frame_rate(target_sample_rate)

        # Сохраняем временный WAV
        temp_wav_path = os.path.join(output_dir, "temp_processed.wav")
        audio.export(temp_wav_path, format="wav")

        output_path = os.path.join(output_dir, f"vocals_processed.wav")

        # Удаление шума
        if noise_reduce:
            rate, data = read(temp_wav_path)

            reduced_noise = nr.reduce_noise(
                y=data,
                sr=rate,
                prop_decrease=prop_decrease,
                n_fft=n_fft,
                hop_length=hop_length
            )

            # Сохраняем результат
            write(output_path, rate, reduced_noise.astype(np.int16))
        else:
            # Сохраняем без удаления шума
            audio.export(output_path, format="wav")

        os.remove(temp_wav_path)
        logger.info(f"Аудио успешно обработано и сохранено в: {output_path}")

        return output_path

    except Exception as e:
        logger.exception(f"Ошибка при предобработке аудио: {e}")
        return ''
    
