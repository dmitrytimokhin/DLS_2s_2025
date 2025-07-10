import logging
from typing import List, Dict, Any
import soundfile as sf

from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class WordSegment:
    text: str
    start: float
    end: float


def transcribe_and_segment(
    model_asr,
    audio_path: str,
    max_pause_between_sentences: float = 1.5,
    max_audio_length_for_ref: float = 15.0,
    output_ref_path: str = "speaker_ref.wav"
) -> List[Dict[str, Any]]:
    """
    Транскрибирует аудио в текст с временными метками и сегментирует его на предложения.

    Параметры:
        model_asr: ASR-модель (например Whisper).
        audio_path (str): Путь к WAV-файлу.
        max_pause_between_sentences (float): Максимальная пауза между словами, после которой начинается новое предложение.
        max_audio_length_for_ref (float): Максимальная длина сегмента для референса (в секундах).
        output_ref_path (str): Путь, куда сохранить speaker_ref.wav.

    Возвращает:
        List[Dict[str, any]]: Список сегментов с текстом и временными метками.
    """
    try:
        # Копируем модель ASR
        model = model_asr

        # Загружаем аудиофайл целиком
        audio_data, sample_rate = sf.read(audio_path)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)  # моно

        # Транскрибируем с временными метками слов
        result = model.transcribe(
            audio_path,
            word_timestamps=True,
            task="transcribe"
        )

        # Извлекаем слова с временными метками
        words: List[WordSegment] = []
        for segment in result["segments"]:
            for word_info in segment.get("words", []):
                words.append(WordSegment(
                    text=word_info["word"],
                    start=word_info["start"],
                    end=word_info["end"]
                ))

        if not words:
            logger.error("Не найдено слов с временными метками.")
            return []

        # Сегментируем на предложения
        segments = []
        current_segment = {
            "text": "",
            "start": words[0].start,
            "end": words[0].end
        }

        for i in range(1, len(words)):
            word = words[i]
            prev_word = words[i - 1]

            # Если пауза больше порога — закрываем текущий сегмент
            if word.start - prev_word.end > max_pause_between_sentences:
                current_segment["text"] = current_segment["text"].strip()
                segments.append(current_segment)

                current_segment = {
                    "text": word.text,
                    "start": word.start,
                    "end": word.end
                }
            else:
                # Продолжаем текущий сегмент
                current_segment["text"] += " " + word.text
                current_segment["end"] = word.end

        # Добавляем последний сегмент
        current_segment["text"] = current_segment["text"].strip()
        segments.append(current_segment)

        logger.info(f"Создано {len(segments)} сегментов.")
        
        # Найдём самый длинный подходящий сегмент
        valid_segments = [seg for seg in segments if (seg["end"] - seg["start"]) <= max_audio_length_for_ref]
        if valid_segments:
            longest_segment = max(valid_segments, key=lambda x: x["end"] - x["start"])
            start_sample = int(longest_segment["start"] * sample_rate)
            end_sample = int(longest_segment["end"] * sample_rate)
            ref_audio = audio_data[start_sample:end_sample]

            sf.write(output_ref_path, ref_audio, sample_rate)
            logger.info(f"Референсное аудио сохранено как {output_ref_path}, длина: {end_sample / sample_rate:.2f} сек.")

        else:
            logger.warning("Не найдено подходящих сегментов для референса.")

        return segments

    except Exception as e:
        logger.error(f"Ошибка при транскрибации: {e}")
        return []
