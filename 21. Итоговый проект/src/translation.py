import logging
from typing import List, Dict


logger = logging.getLogger(__name__)


def translate_segments(
    pipeline_mt,
    segments: List[Dict[str, any]],
    batch_size: int = 8
    ) -> List[Dict[str, any]]:
    """
    Переводит текстовые сегменты из одного языка в другой, сохраняя временные метки.

    Логика:
        1. Загружает модель перевода
        2. Обрабатывает пустые строки
        3. Применяет перевод к каждому сегменту
        4. Возвращает список сегментов вида {"text", "original_text", "start", "end"}

    Параметры:
        pipeline_mt: pipeline с предобученной моделью перевода с английского на русский
        segments (List[Dict]): список сегментов с ключами 'text', 'start', 'end'
        batch_size (int): размер батча сегментов для перевода

    Возвращает:
        List[Dict]: список сегментов с переводом
    """
    try:

        translator = pipeline_mt

        translated_segments = []

        texts_to_translate = [seg["text"].strip() for seg in segments]
        valid_indices = [i for i, text in enumerate(texts_to_translate) if text]

        if not valid_indices:
            logger.error("Нет текста для перевода.")
            return []

        logger.info(f"Переводим {len(valid_indices)} сегментов...")

        translated_texts = translator(
            [texts_to_translate[i] for i in valid_indices],
            batch_size=batch_size
        )

        # === Собираем результаты обратно в сегменты ===
        result_index = 0
        for i, segment in enumerate(segments):
            original_text = segment["text"].strip()
            if not original_text:
                continue

            translated_text = translated_texts[result_index]["translation_text"]
            result_index += 1

            translated_segments.append({
                "text": translated_text,
                "original_text": original_text,
                "start": segment["start"],
                "end": segment["end"]
            })

        logger.info(f"Переведено {len(translated_segments)} сегментов.")
        return translated_segments

    except Exception as e:
        logger.exception(f"Ошибка при переводе: {e}")
        return []
    
