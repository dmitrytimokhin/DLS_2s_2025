import os
import logging
from typing import List, Dict, Tuple, NoReturn
import shutil
from tqdm.auto import tqdm
import re

from TTS.utils.manage import ModelManager
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
from pydub.effects import speedup
import soundfile as sf
from pydub.effects import compress_dynamic_range


logger = logging.getLogger(__name__)


def download_missing_model_files(model_dir: str='./') -> Dict[str, str]:
    """
    Проверяет наличие необходимых файлов модели в model_dir.
    Если какого-то файла нет — загружает его.

    Параметры:
        model_dit (str): путь к модели

    Возвращает:
        Dict[str, str]: словарь с необходимыми файлами и путем к ним.
    """

    logger.info("Проверяем наличие файлов модели...")

    required_files = {
        "config.json": "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/v2.0.2/config.json",
        "model.pth": "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/v2.0.2/model.pth",
        "vocab.json": "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/v2.0.2/vocab.json",
        "mel_stats.pth": "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth",
        "dvae.pth": "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth",
        "speakers_xtts.pth": "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/speakers_xtts.pth"
    }

    target_files = {}
    
    for fname, link in required_files.items():
        target_path = os.path.join(model_dir, fname)
        target_files[fname] = target_path
        if not os.path.isfile(target_path):
            logger.info(f"Файл {fname} отсутствует. Загружаем его...")
            ModelManager._download_model_files([link], model_dir, progress_bar=True)

            # Ищем загруженный файл
            found = False
            for f in os.listdir(model_dir):
                src = os.path.join(model_dir, f)
                if fname in f:
                    if src != target_path:
                        shutil.copy(src, target_path)
                    else:
                        logger.info(f"Файл {fname} уже находится в нужной папке")
                    found = True
                    break
                    
            if not found:
                raise FileNotFoundError(f"Не найден файл после загрузки: {fname}")
                return Dict[str, str]
                
        else:
            logger.info(f"Файл {fname} уже существует")

    return target_files


def generate_audio_segment(
    model_tts,
    segment: Dict[str, any],
    output_dir: str,
    speaker_wav: str,
    language: str
) -> Tuple[str, float]:
    """
    Генерирует аудио по сегменту.

    Параметры:
        model_tts: модель tts для синтеза речи и генерации сегментов.
        segment Dict[str, any]: сегмент типа {'text': str, 'start': float, 'end': float}.
        output_dir (str): путь для сохранения сегмента.
        speaker_wav (str): сегмент для референса при дальнейшей генерации и ZS клонировании.
        language (str): язык генерации.

    Возвращает:
        Tuple[str, float]: кортеж (путь к файлу, фактическая длительность в секундах)
    """
    logger.info(f"Генерация сегмента: '{segment['text']}'")

    with NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        temp_path = tmpfile.name

    gpt_cond_latent, speaker_embedding = model_tts.get_conditioning_latents(audio_path=speaker_wav)
    
    output = model_tts.inference(
        text=segment["text"],
        language=language,
        speaker_embedding=speaker_embedding,
        gpt_cond_latent=gpt_cond_latent
    )

    sf.write(temp_path, output["wav"], 24000)
    
    # Сохраняем как AudioSegment для дальнейшей обработки
    audio = AudioSegment.from_wav(temp_path)

    if segment['start'] == -1:
        segment_name = 'final_voice.wav'
    else:
        segment_name = f"seg_{int(segment['start'] * 1000)}.wav"
        
    output_path = os.path.join(output_dir, segment_name)
    audio.export(output_path, format="wav")

    os.remove(temp_path)

    return output_path, len(audio) / 1000


def synthesize_segments_with_timing(
    model_tts,
    segments: List[Dict[str, any]],
    output_audio_path: str,
    speaker_wav: str,
    language: str,
    max_speedup_factor: float = 2.0,
    min_pause_between_segments: float = 0.5,
    fade_in_out_ms: int = 50,
    crossfade_ms: int = 30,
    max_shift_left_seconds: float = 1.0,
    threshold_compression: float = -15.0,
    ratio_compression: float = 2.5,
    attack_compression: int = 10,
    release_compression: int = 70,
    target_dBFS: float = -16.0
) -> NoReturn:
    """
    Синтезирует дубляж с учётом временных меток оригинала.
    
    Логика:
        1. Каждый сегмент начинается не ранее original_start - max_shift_left_seconds.
        2. Полностью сохраняется текст — НЕ обрезается никогда.
        3. Если не вписываемся → ускоряем динамически, но не более чем max_speedup_factor.
        4. Сдвигаем следующий сегмент только вправо.
        5. Применяем кроссфейды между соседними сегментами для естественности.
        6. Все спецсимволы удаляются перед генерацией (.,!? и т.д.) - помогает избежать странных звуков при генерации.
        7. Финальная компрессия и нормализация всего аудио — для равномерной громкости.

    Параметры:
        model_tts: модель tts для синтеза речи и генерации сегментов.
        segments (List[Dict[str, any]]): список словарей с ключами 'text', 'start', 'end'
        output_audio_path (str): путь к выходному файлу
        speaker_wav (str): путь к образцу голоса для XTTS
        max_speedup_factor (float): максимальное ускорение, чтобы речь оставалась естественной
        min_pause_between_segments (float): минимальная пауза между фразами (в секундах)
        fade_in_out_ms (int): длительность fade-in/out (в мс)
        crossfade_ms (int): длительность кроссфеда между сегментами (в мс)
        max_shift_left_seconds (float): максимальный допустимый сдвиг влево (в секундах)
        threshold_compression (float): уровень сигнала в децибелах относительно максимума (dBFS), выше которого начинает работать компрессор (компрессор работает мягче , реже активируется.)
        ratio_compression (float): отношение "вход/выход" для сигналов выше порога (1.5:1 ... 3:1 — мягкое сжатие; 4:1 и выше — уже ближе к радио или подкастам)
        attack_compression (int): время в миллисекундах, за которое компрессор начинает реагировать на превышение порога (чем больше значение (например, 30 мс) → компрессор реагирует медленнее , оставляя начальный импульс звука нетронутым.)
        release_compression (int): время в миллисекундах, за которое компрессор возвращается к нормальному состоянию после того, как сигнал опускается ниже порога (Чем меньше значение (например, 20 мс) → компрессор быстрее «отпускает» сигнал → возможны слышимые изменения громкости.)
        target_dBFS (foat): целевой уровень громкости , к которому приводится аудиосигнал при нормализации
    """
    try:
        tts = model_tts

        output_segment_dir = os.path.join(os.path.dirname(output_audio_path),
                                          'temp/audio_segments')

        os.makedirs(output_segment_dir, exist_ok=True)

        full_duration_sec = max(seg["end"] for seg in segments) + 5
        full_audio = AudioSegment.silent(duration=int(full_duration_sec * 1000))

        # === Сохраняем оригинальный старт для каждого сегмента ===
        for i, segment in enumerate(segments):
            if "original_start" not in segment:
                segment["original_start"] = segment.get("start", 0.0)

        prev_end_time_sec = 0.0  # будем обновлять после каждого сегмента

        # === Цикл по всем сегментам ===
        for i, segment in tqdm(enumerate(segments)):
            original_start_sec = segment["original_start"]
            current_start_sec = segment["start"]
            current_start_ms = int(current_start_sec * 1000)

            # === Первый сегмент: проверяем, есть ли свободное время перед ним ===
            if i == 0:
                available_shift_left_for_first = max(0, original_start_sec - prev_end_time_sec)
                possible_shift_sec = min(available_shift_left_for_first, max_shift_left_seconds)

                if possible_shift_sec > 0.05:
                    logger.info(f"[{i}] Есть {available_shift_left_for_first:.2f} сек до первого сегмента → "
                                f"двигаем его влево на {possible_shift_sec:.2f} сек")
                    segment["start"] -= possible_shift_sec
                    segment["start"] = max(original_start_sec - max_shift_left_seconds, segment["start"])
                    current_start_sec = segment["start"]
                    current_start_ms = int(current_start_sec * 1000)

            # === Защита от слишком большого сдвига влево ===
            shift_seconds = current_start_sec - original_start_sec

            if shift_seconds < -max_shift_left_seconds:
                logger.warning(f"[{i}] Сегмент сдвинут слишком рано ({shift_seconds:.2f}s), "
                               f"восстанавливаем до {original_start_sec:.2f}s")
                segment["start"] = original_start_sec
                current_start_sec = original_start_sec
                current_start_ms = int(original_start_sec * 1000)
            elif shift_seconds < 0:
                logger.warning(f"[{i}] Сегмент сдвинут влево: оригинал {original_start_sec:.2f} → текущий {current_start_sec:.2f} сек")
            else:
                logger.info(f"[{i}] Сегмент начинается как и планировалось: {original_start_sec:.2f} → {current_start_sec:.2f} сек")

            # === Очистка текста перед генерацией ===
            raw_text = segment["text"]

            clean_text = raw_text.lower()
            clean_text = re.sub(r'[^\w\s]', '', clean_text)  # удаляем знаки препинания
            clean_text = re.sub(r'[\U00010000-\U0010ffff]', '', clean_text)  # удаляем эмодзи
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # убираем лишние пробелы

            segment["cleaned_text"] = clean_text

            if not clean_text:
                logger.warning(f"[{i}] Сегмент после очистки стал пустым → пропускаем")
                continue

            # === Генерируем аудио ===
            seg_path, _ = generate_audio_segment(
                model_tts=tts,
                segment={**segment, "text": clean_text},
                output_dir=output_segment_dir,
                speaker_wav=speaker_wav,
                language=language
            )

            seg_audio = AudioSegment.from_wav(seg_path)
            generated_duration_ms = len(seg_audio)

            # === Получаем оригинальную длительность с защитой от start > end ===
            original_start_val = segment["start"]
            original_end_val = segment.get("end", segment["start"] + 1.0)

            if original_start_val > original_end_val:
                logger.warning(f"[{i}] start > end. Исправляем.")
                original_start_val, original_end_val = original_end_val, original_start_val
                segment["start"], segment["end"] = segment["end"], segment["start"]

            original_duration_ms = int((original_end_val - original_start_val) * 1000)
            logger.info(f"[{i}] Оригинальная длительность: {original_duration_ms / 1000:.2f} сек")
            logger.info(f"[{i}] Сгенерированная длительность: {generated_duration_ms / 1000:.2f} сек")

            # === Определяем доступное время до следующего сегмента ===
            if i < len(segments) - 1:
                next_start_sec = segments[i + 1]["start"]
                time_before_next_ms = max(0, int((next_start_sec - original_end_val) * 1000))
            else:
                time_before_next_ms = 0

            # === Вычисляем, сколько можно добавить к текущему сегменту ===
            available_extension_ms = max(
                0,
                time_before_next_ms - int(min_pause_between_segments * 1000)
            )
            extended_available_time_ms = max(100, original_duration_ms + available_extension_ms)

            logger.info(f"[{i}] Доступное время до следующего сегмента: {time_before_next_ms / 1000:.2f} сек")
            logger.info(f"[{i}] Можно увеличить на: {available_extension_ms / 1000:.2f} сек → "
                        f"теперь доступно: {extended_available_time_ms / 1000:.2f} сек")

            # === Проверяем, нужно ли ускорять ===
            corrected_audio = seg_audio
            applied_speedup = False

            if generated_duration_ms <= 0 or extended_available_time_ms <= 0:
                logger.warning(f"[{i}] Невозможно ускорить — нулевая длительность")
                corrected_duration_ms = generated_duration_ms
            elif generated_duration_ms > extended_available_time_ms:
                required_speedup = generated_duration_ms / extended_available_time_ms

                if required_speedup <= max_speedup_factor:
                    logger.info(f"[{i}] Применено ускорение {required_speedup:.2f}x чтобы уложиться в {extended_available_time_ms / 1000:.2f} сек")
                    corrected_audio = speedup(seg_audio, playback_speed=required_speedup)
                    corrected_duration_ms = int(generated_duration_ms / required_speedup)
                    applied_speedup = True
                else:
                    logger.warning(f"[{i}] Сегмент слишком длинный ({required_speedup:.2f}x), ограничиваем до {max_speedup_factor}x")
                    corrected_audio = speedup(seg_audio, playback_speed=max_speedup_factor)
                    corrected_duration_ms = int(generated_duration_ms / max_speedup_factor)
                    applied_speedup = True
            else:
                corrected_duration_ms = generated_duration_ms

            # === Применяем fade-in/out, но не нормализуем сейчас ===
            corrected_audio = corrected_audio.fade_in(fade_in_out_ms).fade_out(fade_in_out_ms)

            # === Сохраняем скорректированный аудио для кроссфейдов ===
            segment["corrected_audio"] = corrected_audio

            # === Вставляем по точному времени ===
            full_audio = full_audio[:current_start_ms] + corrected_audio + full_audio[current_start_ms + corrected_duration_ms:]

            actual_end_sec = current_start_sec + corrected_duration_ms / 1000
            segment["corrected_duration_sec"] = corrected_duration_ms / 1000
            logger.info(f"[{i}] Сегмент закончится на {actual_end_sec:.2f} сек")

            # === Обновляем начало следующего сегмента (только вправо!) ===
            if i < len(segments) - 1:
                next_segment = segments[i + 1]
                next_segment["original_start"] = next_segment.get("original_start", next_segment["start"])

                new_start_sec = actual_end_sec
                if new_start_sec < next_segment["original_start"]:
                    possible_shift_sec = next_segment["original_start"] - new_start_sec
                    if possible_shift_sec >= max_shift_left_seconds:
                        logger.warning(f"[{i+1}] Слишком большой сдвиг влево → используем оригинальный старт")
                        next_segment["start"] = next_segment["original_start"]
                    else:
                        logger.info(f"[{i+1}] Новый старт с допустимым сдвигом влево: {new_start_sec:.2f} сек")
                        next_segment["start"] = new_start_sec
                else:
                    next_segment["start"] = new_start_sec
                    logger.info(f"[{i+1}] Новый старт следующего сегмента: {next_segment['start']:.2f} сек")

        # === После всех сегментов применяем кроссфейды между соседними ===
        logger.info("Применяем кроссфейд между соседними сегментами...")
        for i in range(len(segments) - 1):
            curr_seg = segments[i]
            next_seg = segments[i + 1]

            curr_audio = curr_seg["corrected_audio"]
            next_audio = next_seg["corrected_audio"]

            curr_start_ms = int(curr_seg["start"] * 1000)
            next_start_ms = int(next_seg["start"] * 1000)

            overlap = curr_start_ms + len(curr_audio) - next_start_ms

            if 0 < overlap < crossfade_ms:
                logger.info(f"[{i}–{i+1}] Перекрытие {overlap} мс → применён кроссфейд на {overlap} мс")
                crossfaded = curr_audio.append(next_audio, crossfade=overlap)
                full_audio = full_audio[:curr_start_ms] + crossfaded + full_audio[curr_start_ms + len(crossfaded):]
            elif overlap >= crossfade_ms:
                logger.info(f"[{i}–{i+1}] Кроссфейд не применяется: перекрытие больше порога")
                pass
            else:
                full_audio = full_audio.overlay(next_audio, position=next_start_ms)

        # === Финальная обработка для равномерной громкости ===
        logger.info("Финальная обработка: компрессия + RMS-нормализация...")
        logger.info("Применяю компрессию динамического диапазона...")
        full_audio = compress_dynamic_range(
            full_audio,
            threshold=threshold_compression,
            ratio=ratio_compression,
            attack=attack_compression,
            release=release_compression
        )
        
        logger.info(f"Применяю RMS-нормализацию до {target_dBFS} dBFS...")
        change_in_dBFS = target_dBFS - full_audio.dBFS
        full_audio = full_audio.apply_gain(change_in_dBFS)

        # === Сохраняем финал ===
        full_audio.export(output_audio_path, format="wav")
        logger.info(f"Финальное аудио сохранено в: {output_audio_path}")

    except Exception as e:
        logger.error(f"Ошибка при синтезе аудио: {e}", exc_info=True)


def concatenate_audio_segments(
    model_tts,
    segments: List[Dict[str, any]],
    output_dir: str,
    speaker_wav: str,
    language: str
) -> str:
    """
    Последовательно генерирует аудиосегменты и склеивает их в один файл.

    Параметры:
        model_tts:
        segments (List[Dict[str, any]]): список сегментов текста
        output_dir (str): директория для сохранения результата
        speaker_wav (str): путь к аудиофайлу диктора
        language (str): язык синтеза

    Возвращает:
        str: путь к объединённому аудиофайлу
    """
    combined_audio = AudioSegment.silent(duration=0)

    for _, segment in enumerate(segments):
        # Генерируем аудиосегмент
        segment_path, _ = generate_audio_segment(
            model_tts=model_tts,
            segment=segment,
            output_dir=output_dir,
            speaker_wav=speaker_wav,
            language=language
        )

        # Загружаем сгенерированный сегмент
        audio_segment = AudioSegment.from_wav(segment_path)

        # Добавляем его к общему аудио
        combined_audio += audio_segment

        # Опционально: удаляем временный файл сегмента, если не нужен
        os.remove(segment_path)

    # Сохраняем финальный файл
    final_output_path = os.path.join(output_dir, "final_voice.wav")
    combined_audio.export(final_output_path, format="wav")

    return final_output_path
