"""
Дополнительные утилиты для работы с F5-TTS.
"""

import logging
from pathlib import Path
from typing import Optional


def get_default_reference_audio() -> Optional[str]:
    """
    Возвращает путь к дефолтному референсному аудио для F5-TTS.
    
    Returns:
        Путь к файлу или None если не найден
    """
    # Ищем дефолтный референс в папке models
    default_refs = [
        "models/f5_tts_reference.wav",
        "models/reference_audio.wav",
        "reference_audio.wav",
    ]
    
    for ref_path in default_refs:
        if Path(ref_path).exists():
            return ref_path
    
    return None


def create_reference_from_text(text: str, output_path: str, logger: Optional[logging.Logger] = None) -> bool:
    """
    Создает референсное аудио из текста используя F5-TTS.
    
    Это временное решение для случаев, когда F5-TTS требует референсное аудио,
    но его нет. Используем F5-TTS для генерации короткого референса.
    
    Args:
        text: Текст для генерации референса (первые 50-100 символов)
        output_path: Путь для сохранения референсного аудио
        logger: Logger
        
    Returns:
        True если успешно создано, False иначе
    """
    if logger:
        logger.debug(f"Creating reference audio from text: {text[:50]}...")
    
    try:
        # Используем F5-TTS для генерации референса
        from src.tts_gateway.f5_tts_engine import F5TTSEngine
        
        # Инициализируем F5-TTS с дефолтными настройками
        # Используем локальный путь к модели
        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / "models" / "F5-tts"
        
        f5_tts = F5TTSEngine(
            model_path=str(model_path),
            device="cuda",
            sample_rate=24000,
            use_stress_marks=True,
            logger=logger,
        )
        
        # Генерируем аудио
        audio = f5_tts.synthesize(text)
        
        # Сохраняем в файл
        import soundfile as sf
        sf.write(output_path, audio, 24000)
        
        if logger:
            logger.info(f"Reference audio created: {output_path}")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Failed to create reference audio: {e}", exc_info=True)
        return False

