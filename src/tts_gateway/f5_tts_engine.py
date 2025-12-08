"""
F5-TTS Engine для синтеза русской речи.
"""

import numpy as np
from typing import Optional, Generator
import logging
import torch
from pathlib import Path
import tempfile
import soundfile as sf

try:
    from f5_tts.api import F5TTS
except ImportError:
    F5TTS = None

from src.tts_gateway.text_preprocessing import TextPreprocessor
from src.tts_gateway.f5_tts_utils import get_default_reference_audio, create_reference_from_text


class F5TTSEngine:
    """
    TTS engine на базе F5-TTS для русского языка.
    
    Поддерживает загрузку модели из локальной директории или HuggingFace.
    Поддерживает автоматическую расстановку ударений через ruaccent.
    """

    def __init__(
        self,
        model_path: str = "models/F5-tts",
        device: str = "cuda",
        sample_rate: int = 24000,
        use_stress_marks: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Инициализация F5-TTS engine.
        
        Args:
            model_path: Путь к модели (локальный путь к директории или HuggingFace ID с префиксом "hf://")
            device: Устройство ('cuda' или 'cpu')
            sample_rate: Частота дискретизации (F5-TTS обычно 24 kHz)
            use_stress_marks: Использовать автоматическую расстановку ударений
            logger: Logger
        """
        self.model_path = model_path
        self.device = device
        self.sample_rate = sample_rate
        self.use_stress_marks = use_stress_marks
        self.logger = logger or logging.getLogger(__name__)
        
        if F5TTS is None:
            raise ImportError(
                "f5-tts not installed. Install with: pip install f5-tts"
            )
        
        # Проверяем CUDA если нужно
        if device == "cuda":
            if not torch.cuda.is_available():
                self.logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
            else:
                # Для RTX 5090 (sm_120) PyTorch может выдавать предупреждение,
                # но модель все равно может работать через совместимость с более старыми архитектурами
                # Проверяем реальную работоспособность GPU
                try:
                    # Пробуем создать простой тензор на GPU для проверки совместимости
                    test_tensor = torch.zeros(1, device="cuda")
                    # Пробуем выполнить простую операцию
                    result = test_tensor + 1
                    del test_tensor, result
                    torch.cuda.synchronize()
                    # Если дошли сюда - GPU работает, даже если есть предупреждение о sm_120
                    self.logger.info("GPU is functional, will use CUDA despite sm_120 warning")
                    self.device = "cuda"  # Принудительно оставляем CUDA
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if "no kernel image" in error_msg or "cuda" in error_msg:
                        self.logger.error(
                            f"CUDA kernel error: {e}. "
                            f"PyTorch {torch.__version__} may not fully support RTX 5090 (sm_120). "
                            f"Consider updating to PyTorch nightly build with sm_120 support."
                        )
                        # НЕ переключаемся на CPU автоматически - выдаем ошибку
                        raise RuntimeError(
                            f"GPU not functional: {e}. "
                            f"Please update PyTorch to a version with sm_120 support or use CPU explicitly."
                        ) from e
                    else:
                        raise
        
        # Инициализация предобработчика текста
        self.text_preprocessor = TextPreprocessor(
            use_stress_marks=use_stress_marks,
            logger=self.logger,
        )
        
        # Инициализация F5-TTS
        try:
            self.logger.info(f"Initializing F5-TTS with model: {model_path}")
            
            # Определяем путь к модели
            if model_path.startswith("hf://"):
                # HuggingFace модель
                model_name = model_path.replace("hf://", "")
                self.model_name = model_name
                self.logger.info(f"Loading model from HuggingFace: {model_name}")
                self.logger.info("This may take 10-30 seconds on first run (downloading model)...")
                # F5TTS() по умолчанию загружает стандартную модель
                # Для HuggingFace моделей может потребоваться специальная инициализация
                self.f5tts = F5TTS(device=self.device)
            else:
                # Локальный путь к модели
                model_dir = Path(model_path)
                if not model_dir.is_absolute():
                    # Относительный путь - делаем абсолютным от корня проекта
                    project_root = Path(__file__).parent.parent.parent
                    model_dir = project_root / model_path
                
                if not model_dir.exists():
                    raise FileNotFoundError(f"F5-TTS model directory not found: {model_dir}")
                
                self.model_name = str(model_dir)
                self.logger.info(f"Loading model from local path: {model_dir}")
                
                # Проверяем наличие файлов модели
                model_files = list(model_dir.glob("*.pt")) + list(model_dir.glob("*.safetensors"))
                if not model_files:
                    raise FileNotFoundError(
                        f"No model files found in {model_dir}. "
                        f"Expected files: model_last.pt or model_last_inference.safetensors"
                    )
                
                self.logger.info(f"Found model files: {[f.name for f in model_files]}")
                
                # Инициализируем F5-TTS с локальной моделью
                # F5TTS принимает device и ckpt_file для указания пути к чекпоинту
                model_file = None
                for pattern in ["model_last_inference.safetensors", "model_last.pt"]:
                    candidate = model_dir / pattern
                    if candidate.exists():
                        model_file = candidate
                        break
                
                if model_file:
                    self.logger.info(f"Using model checkpoint: {model_file}")
                    self.f5tts = F5TTS(device=self.device, ckpt_file=str(model_file))
                else:
                    self.logger.warning("Model checkpoint not found, using default F5-TTS model")
                    self.f5tts = F5TTS(device=self.device)
                
                # Сохраняем путь для использования при inference
                self._model_dir = model_dir
            
            self._initialized = True
            
            self.logger.info(
                "F5-TTS initialized successfully",
                extra={
                    "context": {
                        "device": self.device,
                        "model": self.model_name,
                        "sample_rate": self.sample_rate,
                        "use_stress_marks": self.use_stress_marks,
                    }
                },
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize F5-TTS: {e}", exc_info=True)
            self._initialized = False
            raise

    def synthesize(self, text: str, ref_audio: Optional[str] = None, ref_text: Optional[str] = None) -> np.ndarray:
        """
        Синтезирует речь из текста.
        
        Args:
            text: Текст для синтеза
            ref_audio: Путь к референсному аудио (опционально)
            ref_text: Текст референсного аудио (опционально)
            
        Returns:
            Аудио (sample_rate Hz, mono, float32)
        """
        if not self._initialized:
            raise RuntimeError("F5-TTS not initialized")
        
        try:
            # Предобработка текста (расстановка ударений)
            processed_text = self.text_preprocessor.preprocess(text, language="ru")
            
            # Создаем временный файл для вывода
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                output_path = tmp_file.name
            
            # Флаг для отслеживания временного референса
            temp_ref_created = False
            ref_audio_path = None
            
            try:
                # F5-TTS требует референсное аудио и текст для синтеза
                # Если референс не указан, пытаемся найти дефолтный или создать из текста
                if ref_audio is None:
                    # Пробуем найти дефолтный референс
                    default_ref = get_default_reference_audio()
                    if default_ref:
                        ref_audio_for_infer = default_ref
                        ref_text_for_infer = ref_text if ref_text else processed_text[:100]
                        self.logger.debug(f"Using default reference audio: {default_ref}")
                    else:
                        # Создаем временный референс из начала текста через F5-TTS
                        ref_text_short = processed_text[:100]  # Первые 100 символов
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_ref:
                            ref_audio_path = tmp_ref.name
                        
                        if create_reference_from_text(ref_text_short, ref_audio_path, self.logger):
                            ref_audio_for_infer = ref_audio_path
                            ref_text_for_infer = ref_text_short
                            temp_ref_created = True
                            self.logger.debug(f"Created temporary reference audio from text: {ref_text_short[:50]}...")
                        else:
                            # Если не удалось создать референс, пробуем без него
                            ref_audio_for_infer = None
                            ref_text_for_infer = ref_text if ref_text else processed_text[:100]
                            self.logger.warning(
                                "No reference audio available. F5-TTS may require it. "
                                "Consider providing ref_audio or using fallback TTS."
                            )
                else:
                    ref_audio_for_infer = ref_audio
                    ref_text_for_infer = ref_text if ref_text else processed_text[:100]
                    self.logger.debug(f"Using provided audio reference: {ref_audio}")
                
                # Вызываем F5-TTS inference
                # Примечание: для русской модели может потребоваться специальная конфигурация
                # при инициализации F5TTS, но стандартный API должен работать
                # Если ref_file=None, F5-TTS может использовать только ref_text (требует ASR модели)
                try:
                    wav, sr, spec = self.f5tts.infer(
                        ref_file=ref_audio_for_infer,
                        ref_text=ref_text_for_infer,
                        gen_text=processed_text,
                        file_wave=output_path,
                        file_spec=None,  # Не сохраняем спектрограмму
                        seed=None,
                    )
                except TypeError as e:
                    # Если F5-TTS не принимает None для ref_file, пробуем другой подход
                    if "ref_file" in str(e).lower() or ref_audio_for_infer is None:
                        self.logger.warning(
                            "F5-TTS requires reference audio. Generating with text-only reference."
                            " This may require additional GPU memory for ASR."
                        )
                        # Пробуем использовать пустую строку для ref_text, чтобы F5-TTS транскрибировал сам
                        # Но это требует ref_audio, поэтому если его нет - используем fallback
                        raise RuntimeError(
                            "F5-TTS requires reference audio file. "
                            "Please provide ref_audio parameter or use fallback TTS."
                        ) from e
                    raise
                
                # Загружаем аудио из файла
                audio, sample_rate = sf.read(output_path)
                
                # Конвертируем в mono если стерео
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                
                # Нормализуем в диапазон [-1, 1]
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                    if audio.max() > 1.0:
                        audio = audio / np.max(np.abs(audio))
                
                # Ресемплируем если нужно
                if sample_rate != self.sample_rate:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=self.sample_rate)
                
                return audio.astype(np.float32)
                
            finally:
                # Удаляем временные файлы
                try:
                    Path(output_path).unlink(missing_ok=True)
                except Exception:
                    pass
                
                # Удаляем временный референс если был создан
                if temp_ref_created and ref_audio_path:
                    try:
                        Path(ref_audio_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                    
        except Exception as e:
            self.logger.error(f"F5-TTS synthesis error: {e}", exc_info=True)
            raise

    def synthesize_streaming(
        self, text: str, chunk_size_samples: int = 4800  # 200ms @ 24kHz
    ) -> Generator[np.ndarray, None, None]:
        """
        Стриминговый синтез (эмуляция через чанки с оптимизацией для меньшей задержки).
        
        F5-TTS не поддерживает true streaming, поэтому генерируем весь аудио
        и разбиваем на чанки. Для оптимизации используем меньшие чанки для
        более быстрого первого аудио.
        
        Args:
            text: Текст
            chunk_size_samples: Размер чанка в сэмплах (оптимизирован для меньшей задержки)
            
        Yields:
            Аудио чанки
        """
        if not self._initialized:
            raise RuntimeError("F5-TTS not initialized")
        
        try:
            # Генерируем весь аудио
            audio = self.synthesize(text)
            
            # Используем меньшие чанки для более быстрого первого аудио
            # Первые несколько чанков делаем меньше для снижения TTFA
            initial_chunk_size = min(chunk_size_samples // 2, len(audio) // 4)
            remaining_chunk_size = chunk_size_samples
            
            i = 0
            is_initial = True
            
            while i < len(audio):
                if is_initial and i < len(audio) * 0.25:  # Первые 25% аудио - меньшие чанки
                    current_chunk_size = initial_chunk_size
                else:
                    current_chunk_size = remaining_chunk_size
                    is_initial = False
                
                chunk = audio[i : i + current_chunk_size]
                if len(chunk) > 0:
                    yield chunk
                
                i += current_chunk_size
                
        except Exception as e:
            self.logger.error(f"F5-TTS streaming error: {e}", exc_info=True)
            raise

