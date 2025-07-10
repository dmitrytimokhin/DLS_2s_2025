from aiogram.types import (ReplyKeyboardMarkup, KeyboardButton, 
                           InlineKeyboardMarkup, InlineKeyboardButton)


kb_main = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text='Видео дубляж')],
                                        [KeyboardButton(text='Видео / аудио транскрипция')],
                                        [KeyboardButton(text='Видео / текст перевод')],
                                        [KeyboardButton(text='Озвучка текста')]],
                              resize_keyboard=True,
                              one_time_keyboard=True,
                              input_field_placeholder='')


kb_voice_video = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='(ZS) Голос оригинального спикера',
                          callback_data='voice_original')],
    [InlineKeyboardButton(text='(ZS) Голос с вашего аудио',
                          callback_data='voice_yours')],
    [InlineKeyboardButton(text='(FS) Голос В.В. Путина',
                          callback_data='voice_putin')],
    [InlineKeyboardButton(text='На главную',
                          callback_data='go_home')]]
                          )


kb_voice_audio = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='(ZS) Голос с вашего аудио',
                          callback_data='voice_yours')],
    [InlineKeyboardButton(text='(FS) Голос В.В. Путина',
                          callback_data='voice_putin')],
    [InlineKeyboardButton(text='На главную',
                          callback_data='go_home')]]
                          )


kb_transcription_type = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='Видео',
                          callback_data='transcriprion_type_video'),
    InlineKeyboardButton(text='Голосовое',
                          callback_data='transcriprion_type_voice')],
    [InlineKeyboardButton(text='На главную',
                          callback_data='go_home')]]
                          )


kb_translation_type = InlineKeyboardMarkup(inline_keyboard=[
    [InlineKeyboardButton(text='Видео',
                          callback_data='translation_type_video'),
    InlineKeyboardButton(text='Текст',
                          callback_data='translation_type_text')],
    [InlineKeyboardButton(text='На главную',
                          callback_data='go_home')]]
                          )


kb_reply_go_home = ReplyKeyboardMarkup(keyboard=[[KeyboardButton(text='На главное меню')]],
                                       resize_keyboard=True,
                                       one_time_keyboard=True,
                                       input_field_placeholder='')


kb_go_home = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text='На главную',
                                                                         callback_data='go_home')]])


kb_again = InlineKeyboardMarkup(inline_keyboard=[[InlineKeyboardButton(text='Да',
                                                                         callback_data='yes'),
                                                  InlineKeyboardButton(text='Нет',
                                                                       callback_data='go_home')]])
