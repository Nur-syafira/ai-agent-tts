"""
Пререндер частых фраз для мгновенного TTS.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import logging
import redis
import os


class PrerenderCache:
    """Кэш для пререндеренных TTS фраз."""

    def __init__(
        self,
        cache_dir: str = "cache/tts",
        use_redis: bool = True,
        redis_url: Optional[str] = None,
        ttl_seconds: int = 3600,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Инициализация кэша.
        
        Args:
            cache_dir: Папка для файлового кэша
            use_redis: Использовать Redis
            redis_url: URL Redis сервера
            ttl_seconds: TTL для кэша
            logger: Logger
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.logger = logger or logging.getLogger(__name__)
        
        # Redis клиент
        self.redis_client = None
        if use_redis:
            try:
                redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379")
                self.redis_client = redis.from_url(redis_url, decode_responses=False)
                self.redis_client.ping()
                self.logger.info(f"Redis connected: {redis_url}")
            except Exception as e:
                self.logger.warning(f"Redis unavailable, using file cache only: {e}")
                self.redis_client = None

    def _get_cache_key(self, text: str) -> str:
        """Генерирует ключ кэша из текста."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Получает аудио из кэша.
        
        Args:
            text: Текст фразы
            
        Returns:
            Аудио или None если не в кэше
        """
        key = self._get_cache_key(text)
        
        # Пробуем Redis
        if self.redis_client:
            try:
                data = self.redis_client.get(f"tts:{key}")
                if data:
                    self.logger.debug(f"Cache HIT (Redis): {text[:30]}...")
                    return pickle.loads(data)
            except Exception as e:
                self.logger.error(f"Redis get error: {e}")
        
        # Пробуем файловый кэш
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    self.logger.debug(f"Cache HIT (file): {text[:30]}...")
                    return pickle.load(f)
            except Exception as e:
                self.logger.error(f"File cache read error: {e}")
        
        self.logger.debug(f"Cache MISS: {text[:30]}...")
        return None

    def set(self, text: str, audio: np.ndarray):
        """
        Сохраняет аудио в кэш.
        
        Args:
            text: Текст фразы
            audio: Аудио данные
        """
        key = self._get_cache_key(text)
        data = pickle.dumps(audio)
        
        # Сохраняем в Redis
        if self.redis_client:
            try:
                self.redis_client.setex(f"tts:{key}", self.ttl_seconds, data)
                self.logger.debug(f"Cached in Redis: {text[:30]}...")
            except Exception as e:
                self.logger.error(f"Redis set error: {e}")
        
        # Сохраняем в файл
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(audio, f)
            self.logger.debug(f"Cached in file: {text[:30]}...")
        except Exception as e:
            self.logger.error(f"File cache write error: {e}")

    def prerender(self, phrases: list[str], tts_engine):
        """
        Пререндерит список фраз.
        
        Args:
            phrases: Список фраз
            tts_engine: TTS engine для синтеза
        """
        self.logger.info(f"Prerendering {len(phrases)} phrases...")
        
        for i, phrase in enumerate(phrases, 1):
            # Проверяем, есть ли уже в кэше
            if self.get(phrase) is not None:
                continue
            
            # Синтезируем и кэшируем
            try:
                audio = tts_engine.synthesize(phrase, use_fallback=True)
                self.set(phrase, audio)
                self.logger.info(f"Prerendered ({i}/{len(phrases)}): {phrase}")
            except Exception as e:
                self.logger.error(f"Failed to prerender '{phrase}': {e}")
        
        self.logger.info("Prerendering completed")

