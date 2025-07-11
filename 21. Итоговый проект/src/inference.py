
import os
import logging
from typing import Optional
from dotenv import load_dotenv

from src import (create_path, manage_directory, extract_audio_from_video, mix_audio_tracks,
                 add_audio_to_video, split_text_by_chars)
from src import separate_audio_sources, preprocess_audio_for_asr
from src import transcribe_and_segment
from src import translate_segments
from src import generate_audio_segment, synthesize_segments_with_timing, concatenate_audio_segments


logger = logging.getLogger(__name__)

load_dotenv()
MAIN_ADMIN = int(os.getenv("MAIN_ADMIN_ID"))


async def full_pipeline(
        model_asr, pipeline_mt, model_tts, tg_user_id: int=MAIN_ADMIN, mode: str='original', 
        finetuned_dir: str = './finetuned_tts_model/', device: str='cpu') -> str:
    """
    Функция генерации видео:
        1) Извлекает аудиодорожку из видео
        2) Предобрабатывает аудиодорожку
        3) Транскрибирует и разбивает на сегменты, создает файл референса из оригинального голоса
        4) Переводит сегменты с английского на русский язык
        5) Генерирует аудиодорожку с переведенным текстом
        6) Склеивает сгенерированную аудиодорожку с оригинальным видео

    Параметры:
        model_asr: модель ASR для транскрибации аудио в текст.
        pipeline_mt: пайплайн для перевода сегментов с английского на русский язык.
        model_tts: модель tts для синтеза речи.
        tg_user_id (int): id пользователя телеграм для создания индивидуальной директории.
        mode (str): режим генерации ('original', 'putin', 'yours').
        finetuned_dir (str): путь к дообученной модели при mode='putin'.
        device (str): выбор ГПУ / ЦПУ.
    Возвращает:
        str: путь к финальному видео с дубляжом.
    """

    CLIENT_PATH = f"./data/tg_user_id_{tg_user_id}/"

    ORIGINAL_VIDEO_PATH = create_path(CLIENT_PATH, "video.mp4")
    TEMP_DIR = create_path(CLIENT_PATH, "temp")
    ORIGINAL_AUDIO_PATH = create_path(TEMP_DIR, f"extracted_audio.wav")

    SPEAKER_REF_PATH = create_path(TEMP_DIR, f"speaker_ref.wav")

    FINAL_VOICE_PATH = create_path(CLIENT_PATH, "final_dubbing.wav")
    FINAL_MIX_AUDIO_PATH = create_path(CLIENT_PATH, "final_mix.wav")
    FINAL_VIDEO_PATH = create_path(CLIENT_PATH, "final_video.mp4")

    LANGUAGE = "ru"

    manage_directory(TEMP_DIR, action="delete")
    extract_audio_from_video(ORIGINAL_VIDEO_PATH, ORIGINAL_AUDIO_PATH)

    output_voice_path, output_background_path = separate_audio_sources(
        input_audio_path=ORIGINAL_AUDIO_PATH,
        temp_dir=TEMP_DIR,
        device=device)

    output_voice_processed_path = preprocess_audio_for_asr(
        input_path=output_voice_path,
        target_sample_rate=16000,
        gain_increase=0.0,
        noise_reduce=True,
        prop_decrease=0.3,
        n_fft=512,
        hop_length=256
    )

    segments = transcribe_and_segment(
        model_asr=model_asr,
        audio_path=output_voice_processed_path,
        max_pause_between_sentences=0.3,
        max_audio_length_for_ref=15.0,
        output_ref_path=SPEAKER_REF_PATH
        )
    
    translated_segments = translate_segments(
        pipeline_mt=pipeline_mt,
        segments=segments, 
        batch_size=8)
    
    if mode == 'original':
        speaker_wav = SPEAKER_REF_PATH
    elif mode == 'putin':
        speaker_wav = create_path(finetuned_dir, 'putin_ref.wav')
    else:
        speaker_wav = create_path(CLIENT_PATH, "yours_ref.wav")

    synthesize_segments_with_timing(
        model_tts=model_tts,
        segments=translated_segments,
        output_audio_path=FINAL_VOICE_PATH,
        speaker_wav=speaker_wav,
        language=LANGUAGE,
        max_speedup_factor=1.5,
        min_pause_between_segments=0.2,
        fade_in_out_ms=50,
        crossfade_ms=25,
        max_shift_left_seconds=0.5,
        threshold_compression=-15.0,
        ratio_compression=2.0,
        attack_compression=25,
        release_compression=50,
        target_dBFS=-15.0
    )

    mix_audio_tracks(
        voice_over_path=FINAL_VOICE_PATH,
        background_path=output_background_path,
        output_path=FINAL_MIX_AUDIO_PATH,
        original_audio_path=ORIGINAL_AUDIO_PATH,
        voice_gain=-3.0,          # чуть ниже
        background_gain=-5.0,    # фон ниже
        original_gain=-10.0      # оригинал совсем тихий
    )

    add_audio_to_video(
        video_path=ORIGINAL_VIDEO_PATH,
        audio_path=FINAL_MIX_AUDIO_PATH,
        output_video_path=FINAL_VIDEO_PATH
    )

    return FINAL_VIDEO_PATH


async def run_transcription(model_asr, video: bool=True, tg_user_id: int=MAIN_ADMIN,
                            device: str='cpu') -> str:
    """
    Функция транскрибации видео / аудио:
        1) Извлекает аудиодорожку из видео - в случае видео
        2) Предобрабатывает аудиодорожку
        3) Транскрибирует

    Параметры:
        model_asr: модель ASR для транскрибации аудио в текст.
        video (bool): флаг типа меди (True - video, False - audio).
        tg_user_id (int): id пользователя телеграм для создания индивидуальной директории.
        device (str): выбор ГПУ / ЦПУ.
    Возвращает:
        str: транскрибированный текст по видео / аудио
    """

    CLIENT_PATH = f"./data/tg_user_id_{tg_user_id}/"
    ORIGINAL_VIDEO_PATH = create_path(CLIENT_PATH, "video.mp4")
    SPEAKER_REF_PATH = create_path(CLIENT_PATH, f"speaker_ref.wav")

    if video:
        ORIGINAL_AUDIO_PATH = create_path(CLIENT_PATH, f"extracted_audio.wav")
        extract_audio_from_video(ORIGINAL_VIDEO_PATH, ORIGINAL_AUDIO_PATH)
    else:
        ORIGINAL_AUDIO_PATH = create_path(CLIENT_PATH, f"yours_ref.wav")


    output_voice_path, _ = separate_audio_sources(
        input_audio_path=ORIGINAL_AUDIO_PATH,
        temp_dir=CLIENT_PATH,
        device=device)

    output_voice_processed_path = preprocess_audio_for_asr(
        input_path=output_voice_path,
        target_sample_rate=16000,
        gain_increase=0.0,
        noise_reduce=True,
        prop_decrease=0.3,
        n_fft=512,
        hop_length=256
    )

    segments = transcribe_and_segment(
        model_asr=model_asr,
        audio_path=output_voice_processed_path,
        max_pause_between_sentences=0.3,
        max_audio_length_for_ref=15.0,
        output_ref_path=SPEAKER_REF_PATH
        )
    
    RESULT = ' '.join(item['text'] for item in segments if item['text']).strip()
    
    return RESULT


async def run_translation(model_asr, pipeline_mt, video: bool=True, text: Optional[str]=None, 
                          tg_user_id: int=MAIN_ADMIN, device: str='cpu') -> str:
    """
    Функция перевода видео / текста:
        1) Извлекает аудиодорожку из видео - в режиме видео
        2) Предобрабатывает аудиодорожку - в режиме видео
        3) Транскрибирует аудиодорожку - в режиме видео
        4) Переводит текст с английского на русский язык

    Параметры:
        model_asr: модель ASR для транскрибации аудио в текст.
        pipeline_mt: пайплайн для перевода сегментов с английского на русский язык.
        video (bool): флаг типа меди (True - video, False - text).
        text (str): текст на английском языке для перевода на русский.
        tg_user_id (int): id пользователя телеграм для создания индивидуальной директории.
        device (str): выбор ГПУ / ЦПУ.
    Возвращает:
        str: транскрибированный текст по видео / аудио
    """


    if video:
        CLIENT_PATH = f"./data/tg_user_id_{tg_user_id}/"
        ORIGINAL_VIDEO_PATH = create_path(CLIENT_PATH, "video.mp4")
        SPEAKER_REF_PATH = create_path(CLIENT_PATH, f"speaker_ref.wav")
        ORIGINAL_AUDIO_PATH = create_path(CLIENT_PATH, f"extracted_audio.wav")
        extract_audio_from_video(ORIGINAL_VIDEO_PATH, ORIGINAL_AUDIO_PATH)


        output_voice_path, _ = separate_audio_sources(
            input_audio_path=ORIGINAL_AUDIO_PATH,
            temp_dir=CLIENT_PATH,
            device=device)

        output_voice_processed_path = preprocess_audio_for_asr(
            input_path=output_voice_path,
            target_sample_rate=16000,
            gain_increase=0.0,
            noise_reduce=True,
            prop_decrease=0.3,
            n_fft=512,
            hop_length=256
        )

        segments = transcribe_and_segment(
            model_asr=model_asr,
            audio_path=output_voice_processed_path,
            max_pause_between_sentences=0.3,
            max_audio_length_for_ref=15.0,
            output_ref_path=SPEAKER_REF_PATH
            )
        
        text = ' '.join(item['text'] for item in segments if item['text']).strip()
    else:
        text = text

    list_texts = split_text_by_chars(text)
    
    text_format = [{'text': text, 'start': -1, 'end': -1} for text in list_texts]

    translated_segments = translate_segments(
        pipeline_mt=pipeline_mt,
        segments=text_format,
        batch_size=8
        )
    
    RESULT = ' '.join(item['text'] for item in translated_segments if item['text']).strip()
    
    return RESULT


async def gen_audio(model_tts,
                    text: str,
                    tg_user_id: int=MAIN_ADMIN,
                    mode: str = 'yours',
                    finetuned_dir: str = './finetuned_tts_model/',
                    device: str='cpu') -> str:
    """
    Функция озвучки текста.

    Параметры:
        text (str): текст на русском языке для озвучки.
        tg_user_id (int): id пользователя телеграм для создания индивидуальной директории.
        mode (str): режим генерации ('putin', 'yours').
        finetuned_dir (str): путь к дообученной модели при mode='putin'.
        device (str): выбор ГПУ / ЦПУ.
    Возвращает:
        str: транскрибированный текст по видео / аудио
    """

    CLIENT_PATH = f"./data/tg_user_id_{tg_user_id}/"

    if mode == 'putin':
        speaker_wav = create_path(finetuned_dir, 'putin_ref.wav')
    else:
        speaker_wav = create_path(CLIENT_PATH, "yours_ref.wav")

    LANGUAGE = "ru"

    if len(text)>=182:
        parts = [t.strip() for t in text.split(".") if t.strip()]
        segments = [{"text": part, "start": i} for i, part in enumerate(parts)]

        RESULT = concatenate_audio_segments(
            model_tts=model_tts,
            segments=segments,
            output_dir=CLIENT_PATH,
            speaker_wav=speaker_wav,
            language=LANGUAGE
        )

    else:
        segment = {'text': text,
                   'start': -1,
                   'end': -1}

        RESULT, _ = generate_audio_segment(
            model_tts=model_tts,
            segment=segment,
            output_dir=CLIENT_PATH,
            speaker_wav=speaker_wav,
            language=LANGUAGE
            )
    
    return RESULT
