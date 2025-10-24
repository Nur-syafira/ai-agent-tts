"""
Async обёртка над gspread для работы с Google Sheets.
"""

import gspread
from gspread_asyncio import AsyncioGspreadClientManager
from google.oauth2.service_account import Credentials
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


class GoogleSheetsClient:
    """Async клиент для Google Sheets."""

    def __init__(
        self,
        credentials_path: str,
        spreadsheet_id: str,
        worksheet_name: str,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Инициализация Google Sheets клиента.
        
        Args:
            credentials_path: Путь к credentials.json
            spreadsheet_id: ID Google Sheets документа
            worksheet_name: Название листа (например, "Лист4")
            logger: Logger
        """
        self.credentials_path = credentials_path
        self.spreadsheet_id = spreadsheet_id
        self.worksheet_name = worksheet_name
        self.logger = logger or logging.getLogger(__name__)
        
        # Инициализация gspread_asyncio
        self.agcm = AsyncioGspreadClientManager(self._get_creds)
        
    def _get_creds(self):
        """
        Возвращает credentials для Google Sheets API.
        
        Returns:
            Google OAuth2 credentials
        """
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        
        creds = Credentials.from_service_account_file(
            self.credentials_path,
            scopes=scopes,
        )
        
        return creds

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((gspread.exceptions.APIError,)),
    )
    async def append_row(self, row_data: Dict[str, Any]):
        """
        Добавляет строку в Google Sheets (append-only).
        
        Args:
            row_data: Словарь с данными строки
            
        Raises:
            gspread.exceptions.APIError: При ошибке API
        """
        try:
            # Получаем async клиент
            agc = await self.agcm.authorize()
            
            # Открываем spreadsheet
            spreadsheet = await agc.open_by_key(self.spreadsheet_id)
            
            # Открываем worksheet
            try:
                worksheet = await spreadsheet.worksheet(self.worksheet_name)
            except gspread.exceptions.WorksheetNotFound:
                # Создаём лист если не существует
                worksheet = await spreadsheet.add_worksheet(
                    title=self.worksheet_name,
                    rows=1000,
                    cols=20,
                )
                
                # Добавляем заголовки
                headers = [
                    "Timestamp",
                    "Имя клиента",
                    "Телефон",
                    "Возраст",
                    "Вес (кг)",
                    "Симптомы",
                    "Тип исследования",
                    "Дата записи",
                    "Время записи",
                    "Стоимость исследования",
                    "Формат заключения",
                    "Стоимость носителя",
                    "Итоговая стоимость",
                    "Статус",
                ]
                await worksheet.append_row(headers)
            
            # Формируем строку данных
            timestamp = datetime.utcnow().isoformat() + "Z"
            row = [
                timestamp,
                row_data.get("client_name", ""),
                row_data.get("client_phone", ""),
                row_data.get("client_age", ""),
                row_data.get("client_weight", ""),
                row_data.get("symptoms", ""),
                row_data.get("study_type", ""),
                row_data.get("appointment_date", ""),
                row_data.get("appointment_time", ""),
                row_data.get("study_price", ""),
                row_data.get("media_type", ""),
                row_data.get("media_price", ""),
                row_data.get("total_price", ""),
                row_data.get("status", "записан"),
            ]
            
            # Добавляем строку (append-only)
            await worksheet.append_row(row, value_input_option="USER_ENTERED")
            
            self.logger.info(
                f"Row appended to '{self.worksheet_name}'",
                extra={"context": {"client_name": row_data.get("client_name")}},
            )
            
        except gspread.exceptions.APIError as e:
            self.logger.error(f"Google Sheets API error: {e}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error appending row: {e}", exc_info=True)
            raise

    async def get_all_records(self) -> List[Dict[str, Any]]:
        """
        Получает все записи из листа.
        
        Returns:
            Список словарей с данными
        """
        try:
            agc = await self.agcm.authorize()
            spreadsheet = await agc.open_by_key(self.spreadsheet_id)
            worksheet = await spreadsheet.worksheet(self.worksheet_name)
            
            records = await worksheet.get_all_records()
            return records
            
        except Exception as e:
            self.logger.error(f"Error getting records: {e}", exc_info=True)
            raise

    async def clear_sheet(self):
        """
        Очищает лист (ИСПОЛЬЗУЕТСЯ ТОЛЬКО ДЛЯ ТЕСТОВ, не в проде!).
        
        Warning:
            В проде используется append-only режим.
        """
        self.logger.warning(f"Clearing worksheet '{self.worksheet_name}' - USE WITH CAUTION!")
        
        try:
            agc = await self.agcm.authorize()
            spreadsheet = await agc.open_by_key(self.spreadsheet_id)
            worksheet = await spreadsheet.worksheet(self.worksheet_name)
            
            await worksheet.clear()
            
            self.logger.info(f"Worksheet '{self.worksheet_name}' cleared")
            
        except Exception as e:
            self.logger.error(f"Error clearing sheet: {e}", exc_info=True)
            raise

