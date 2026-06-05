from aiogram.fsm.state import StatesGroup, State


class FSMVoice(StatesGroup):
    tg_user_id = State()
    media_type = State()

    voice_generation = State()

    voice_path = State()
    video_path = State()
    
    text_geneneration = State()

    transcription = State()
    translate = State()
    generation = State()

    process = State()
