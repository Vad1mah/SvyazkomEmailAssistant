import sqlite3
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import html
from datetime import datetime
import traceback

class EmailProcessor:
    def __init__(self, db_path: str, emails_dir: str, config_path: str = "config.json"):
        self.db_path = db_path
        self.emails_dir = emails_dir
        self.config = self.load_config(config_path)
        self.conn = None
        self.cursor = None
        
        # Загружаем роли
        self.roles = self.load_roles()
        
        # Компилируем regex паттерны
        self.compile_patterns()
    
    def load_config(self, config_path: str) -> Dict:
        """Загружает конфигурацию"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
                # Преобразуем старый формат в новый
                if "filter_patterns" in config and isinstance(config["filter_patterns"], list):
                    # Старый формат - список паттернов
                    old_patterns = config["filter_patterns"]
                    config["filter_patterns"] = {
                        "signatures": [
                            r"С уважением.*",
                            r"Best regards.*",
                            r"С наилучшими пожеланиями.*",
                            r"Спасибо.*",
                            r"Thanks.*",
                            r"________________.*",
                            r"IP: \d+.*",
                            r"mob: \+\d+.*",
                            r"e-mail: .*@.*",
                            r"г\. .*",
                            r"тел: .*",
                            r"Тел: .*"
                        ],
                        "auto_responses": [
                            r"Это автоматическое уведомление.*",
                            r"Automatic reply.*",
                            r"Out of office.*",
                            r"Не в офисе.*",
                            r"Отсутствую.*",
                            r"В отпуске.*",
                            r"On vacation.*"
                        ],
                        "greetings": [
                            r"^Доброе утро[,\s]*[А-Яа-я]*[!.]?\s*",
                            r"^Добрый день[,\s]*[А-Яа-я]*[!.]?\s*",
                            r"^Добрый вечер[,\s]*[А-Яа-я]*[!.]?\s*",
                            r"^День добрый[!.]?\s*",
                            r"^Коллеги[,\s]*добрый день[!.]?\s*",
                            r"^Коллеги[,\s]*доброе утро[!.]?\s*",
                            r"^Коллеги[,\s]*добрый вечер[!.]?\s*",
                            r"^Уважаемые коллеги[!.]?\s*",
                            r"^Уважаемый[,\s]*[А-Яа-я]*[!.]?\s*",
                            r"^Уважаемая[,\s]*[А-Яа-я]*[!.]?\s*",
                            r"^Здравствуйте[!.]?\s*",
                            r"^Здравствуй[!.]?\s*",
                            r"^Привет[!.]?\s*"
                        ],
                        "quotes": [
                            r"^>.*$",
                            r"^On .* wrote:$",
                            r"^От .* писал\(а\):$",
                            r"^From: .*$",
                            r"^To: .*$",
                            r"^Subject: .*$",
                            r"^Date: .*$"
                        ],
                        "custom": old_patterns  # Добавляем старые паттерны как custom
                    }
                
                # Добавляем недостающие поля
                if "max_tokens" not in config:
                    config["max_tokens"] = 4000
                if "min_chain_length" not in config:
                    config["min_chain_length"] = 2
                if "max_chain_length" not in config:
                    config["max_chain_length"] = 20
                if "specialists" not in config:
                    config["specialists"] = [
                        "staff1@svyazcom.ru",
                        "staff1@svyazcom.ru", 
                        "staff1@svyazcom.ru",
                        "staff1@svyazcom.ru",
                        "staff1@svyazcom.ru",
                        "staff1@svyazcom.ru"
                    ]
                if "exclude_domains" not in config:
                    config["exclude_domains"] = [
                        "staff1@svyazcom.ru",
                        "staff1@svyazcom.ru",
                        "staff1@svyazcom.ru",
                        "staff1@svyazcom.ru",
                        "staff1@svyazcom.ru"
                    ]
                
                return config
        else:
            # Конфигурация по умолчанию
            return {
                "max_tokens": 4000,
                "min_chain_length": 2,
                "max_chain_length": 20,
                "filter_patterns": {
                    "signatures": [
                        r"С уважением.*",
                        r"Best regards.*",
                        r"С наилучшими пожеланиями.*",
                        r"Спасибо.*",
                        r"Thanks.*",
                        r"________________.*",
                        r"IP: \d+.*",
                        r"mob: \+\d+.*",
                        r"e-mail: .*@.*",
                        r"г\. .*",
                        r"тел: .*",
                        r"Тел: .*"
                    ],
                    "auto_responses": [
                        r"Это автоматическое уведомление.*",
                        r"Automatic reply.*",
                        r"Out of office.*",
                        r"Не в офисе.*",
                        r"Отсутствую.*",
                        r"В отпуске.*",
                        r"On vacation.*"
                    ],
                    "greetings": [
                        r"^Доброе утро[,\s]*[А-Яа-я]*[!.]?\s*",
                        r"^Добрый день[,\s]*[А-Яа-я]*[!.]?\s*",
                        r"^Добрый вечер[,\s]*[А-Яа-я]*[!.]?\s*",
                        r"^День добрый[!.]?\s*",
                        r"^Коллеги[,\s]*добрый день[!.]?\s*",
                        r"^Коллеги[,\s]*доброе утро[!.]?\s*",
                        r"^Коллеги[,\s]*добрый вечер[!.]?\s*",
                        r"^Уважаемые коллеги[!.]?\s*",
                        r"^Уважаемый[,\s]*[А-Яа-я]*[!.]?\s*",
                        r"^Уважаемая[,\s]*[А-Яа-я]*[!.]?\s*",
                        r"^Здравствуйте[!.]?\s*",
                        r"^Здравствуй[!.]?\s*",
                        r"^Привет[!.]?\s*"
                    ],
                    "quotes": [
                        r"^>.*$",
                        r"^On .* wrote:$",
                        r"^От .* писал\(а\):$",
                        r"^From: .*$",
                        r"^To: .*$",
                        r"^Subject: .*$",
                        r"^Date: .*$"
                    ]
                },
                "specialists": [
                    "stenzer@svyazcom.ru",
                    "burnes@svyazcom.ru", 
                    "bulgakov@svyazcom.ru",
                    "karnaushenko@svyazcom.ru",
                    "javorsky@svyazcom.ru",
                    "maksimenko@svyazcom.ru"
                ],
                "exclude_domains": [
                    "support@svyazcom.ru",
                    "otrs@svyazcom.ru",
                    "git@svyazcom.ru",
                    "redmine@svyazcom.ru",
                    "bugz@svyazcom.ru"
                ]
            }
    
    def load_roles(self) -> Dict[str, str]:
        """Загружает роли из файла roles.txt"""
        roles = {}
        if os.path.exists("roles.txt"):
            with open("roles.txt", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and ":" in line:
                        email, role = line.split(":", 1)
                        roles[email.strip()] = role.strip()
        return roles
    
    def compile_patterns(self):
        """Компилирует regex паттерны"""
        self.patterns = {}
        for category, patterns in self.config["filter_patterns"].items():
            self.patterns[category] = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns]
    
    def connect_db(self):
        """Подключается к базе данных"""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
    
    def close_db(self):
        """Закрывает соединение с базой данных"""
        if self.conn:
            self.conn.close()
    
    def get_email_chains(self, test_mode=False, limit=10) -> List[Dict]:
        """Получает цепочки писем из базы данных"""
        if self.cursor is None:
            raise RuntimeError("Database cursor is not initialized. Call connect_db() before using this method.")
        query = """
        WITH root_emails AS (
            SELECT 
                e.id,
                e.message_id, 
                e.from_field, 
                e.to_field, 
                e.subject, 
                e.date,
                ec.left_val,
                ec.right_val
            FROM emails e
            JOIN email_chain ec ON e.id = ec.mail_id
            WHERE (
                e.from_field LIKE '%nora@mts.ru%' OR 
                e.from_field LIKE '%ryv@mts.ru%'
                )
                AND (e.reply_id IS NULL OR e.reply_id = '')
        ),
        specialist_responses AS (
            SELECT 
                e.id,
                e.message_id, 
                e.from_field, 
                e.to_field, 
                e.subject, 
                e.date,
                e.reply_id,
                ec.left_val,
                ec.right_val
            FROM emails e
            JOIN email_chain ec ON e.id = ec.mail_id
            WHERE (
                e.from_field LIKE '%sstaff1@svyazcom.ru%' OR 
                e.from_field LIKE '%sstaff1@svyazcom.ru%' OR 
                e.from_field LIKE '%sstaff1@svyazcom.ru%' OR 
                e.from_field LIKE '%sstaff1@svyazcom.ru%' OR 
                e.from_field LIKE '%sstaff1@svyazcom.ru%' OR 
                e.from_field LIKE '%sstaff1@svyazcom.ru%')
        ),
        other_svyazcom_responses AS (
            SELECT 
                e.id,
                e.message_id, 
                e.from_field, 
                e.to_field, 
                e.subject, 
                e.date,
                e.reply_id,
                ec.left_val,
                ec.right_val
            FROM emails e
            JOIN email_chain ec ON e.id = ec.mail_id
            WHERE e.from_field LIKE '%@svyazcom.ru%'
                AND e.from_field NOT LIKE '%sstaff1@svyazcom.ru%'
                AND e.from_field NOT LIKE '%sstaff1@svyazcom.ru%'
                AND e.from_field NOT LIKE '%sstaff1@svyazcom.ru%'
                AND e.from_field NOT LIKE '%sstaff1@svyazcom.ru%'
                AND e.from_field NOT LIKE '%sstaff1@svyazcom.ru%'
                AND e.from_field NOT LIKE '%sstaff1@svyazcom.ru%'
                AND e.from_field NOT LIKE '%sstaff1@svyazcom.ru%'
                AND e.from_field NOT LIKE '%sstaff1@svyazcom.ru%'
                AND e.from_field NOT LIKE '%sstaff1@svyazcom.ru%'
                AND e.from_field NOT LIKE '%sstaff1@svyazcom.ru%'
                AND e.from_field NOT LIKE '%sstaff1@svyazcom.ru%'
        ),
        emails_with_specialist_responses AS (
            SELECT DISTINCT r.message_id
            FROM root_emails r
            JOIN specialist_responses s ON (
                s.left_val > r.left_val 
                AND s.right_val < r.right_val
            )
        ),
        specialist_response_data AS (
            SELECT * FROM (
                SELECT 
                    r.message_id as original_message_id, 
                    r.from_field as original_from, 
                    r.to_field as original_to, 
                    r.subject as original_subject, 
                    r.date as original_date,
                    s.message_id as response_message_id, 
                    s.from_field as response_from, 
                    s.to_field as response_to, 
                    s.subject as response_subject, 
                    s.date as response_date, 
                    s.reply_id as response_reply_id,
                    'specialist' as response_type,
                    ROW_NUMBER() OVER (PARTITION BY r.message_id ORDER BY s.date ASC) as rn
                FROM root_emails r
                JOIN specialist_responses s ON (
                    s.left_val > r.left_val 
                    AND s.right_val < r.right_val
                )
            ) WHERE rn = 1
        ),
        other_response_data AS (
            SELECT * FROM (
                SELECT 
                    r.message_id as original_message_id, 
                    r.from_field as original_from, 
                    r.to_field as original_to, 
                    r.subject as original_subject, 
                    r.date as original_date,
                    s.message_id as response_message_id, 
                    s.from_field as response_from, 
                    s.to_field as response_to, 
                    s.subject as response_subject, 
                    s.date as response_date, 
                    s.reply_id as response_reply_id,
                    'other_svyazcom' as response_type,
                    ROW_NUMBER() OVER (PARTITION BY r.message_id ORDER BY s.date ASC) as rn
                FROM root_emails r
                JOIN other_svyazcom_responses s ON (
                    s.left_val > r.left_val 
                    AND s.right_val < r.right_val
                )
                WHERE r.message_id NOT IN (SELECT message_id FROM emails_with_specialist_responses)
            ) WHERE rn = 1
        )
        SELECT 
            original_message_id,
            original_from,
            original_subject,
            response_message_id,
            response_from,
            response_subject,
            response_type
        FROM specialist_response_data
        UNION ALL
        SELECT 
            original_message_id,
            original_from,
            original_subject,
            response_message_id,
            response_from,
            response_subject,
            response_type
        FROM other_response_data
        ORDER BY original_message_id
        """
        if test_mode:
            query += f" LIMIT {limit}"
        self.cursor.execute(query)
        results = self.cursor.fetchall()
        chains = []
        for row in results:
            chain = {
                "root_email": {
                    "message_id": row[0],
                    "from_field": row[1],
                    "subject": row[2]
                },
                "first_response": {
                    "message_id": row[3],
                    "from_field": row[4],
                    "subject": row[5],
                    "response_type": row[6]
                }
            }
            chains.append(chain)
        return chains
    
    def get_chain_emails(self, left_val: int, right_val: int) -> List[Dict]:
        """Получает все письма в цепочке"""
        if self.cursor is None:
            raise RuntimeError("Database cursor is not initialized. Call connect_db() before using this method.")
        query = """
        SELECT e.id, e.message_id, e.from_field, e.to_field, e.subject, e.date, e.reply_id,
               e.file_path, e.start_pos, e.end_pos
        FROM emails e
        JOIN email_chain ec ON e.id = ec.mail_id
        WHERE ec.left_val >= ? AND ec.right_val <= ?
        ORDER BY e.date
        """
        self.cursor.execute(query, (left_val, right_val))
        results = self.cursor.fetchall()
        emails = []
        for row in results:
            email = {
                "id": row[0],
                "message_id": row[1],
                "from_field": row[2],
                "to_field": row[3],
                "subject": row[4],
                "date": row[5],
                "reply_id": row[6],
                "file_path": row[7],
                "start_pos": row[8],
                "end_pos": row[9]
            }
            emails.append(email)
        return emails
    
    def clean_message_id(self, message_id):
        """Очищает message_id от HTML-кодов"""
        if not message_id:
            return ""
        
        # Убираем HTML-коды
        cleaned = message_id.replace('&lt;', '<').replace('&gt;', '>')
        cleaned = cleaned.replace('<br', '').replace('<br/>', '').replace('<br />', '')
        
        # Убираем лишние символы
        cleaned = cleaned.strip()
        
        return cleaned
    
    def load_email_content(self, message_id):
        """Загружает содержимое письма по message_id"""
        # Очищаем message_id от HTML-кодов
        cleaned_message_id = self.clean_message_id(message_id)
        
        # Ищем во всех папках с датами
        for date_dir in Path(self.emails_dir).iterdir():
            if date_dir.is_dir():
                # Ищем папку, которая содержит message_id (частичное совпадение)
                for email_dir in date_dir.iterdir():
                    if email_dir.is_dir() and cleaned_message_id in email_dir.name:
                        email_file = email_dir / "email.json"
                        if email_file.exists():
                            try:
                                with open(email_file, 'r', encoding='utf-8') as f:
                                    return json.load(f)
                            except Exception as e:
                                return None
        return None
    
    def filter_text(self, text):
        """Удаляет html/css/msword артефакты, типовые приветствия и служебные хвосты. Не трогает смысловую часть письма."""
        if not text:
            return ''
        import re
        # MS Word/html артефакты и служебные конструкции (ищем 1-4 экранированных слэша)
        patterns = [
            r'v\\:\* ?\{behavior:url\(#default#VML\);\}',
            r'v\\\\:\* ?\{behavior:url\(#default#VML\);\}',
            r'v\\\\\\:\* ?\{behavior:url\(#default#VML\);\}',
            r'v\\\\\\\\:\* ?\{behavior:url\(#default#VML\);\}',
            r'o\\:\* ?\{behavior:url\(#default#VML\);\}',
            r'o\\\\:\* ?\{behavior:url\(#default#VML\);\}',
            r'o\\\\\\:\* ?\{behavior:url\(#default#VML\);\}',
            r'o\\\\\\\\:\* ?\{behavior:url\(#default#VML\);\}',
            r'w\\:\* ?\{behavior:url\(#default#VML\);\}',
            r'w\\\\:\* ?\{behavior:url\(#default#VML\);\}',
            r'w\\\\\\:\* ?\{behavior:url\(#default#VML\);\}',
            r'w\\\\\\\\:\* ?\{behavior:url\(#default#VML\);\}',
            r'\.shape ?\{behavior:url\(#default#VML\);\}',
            r'\\?p\\?\\.?MsoNormal',
            r'\\?span\\.?EmailStyle\\d+',
            r'\\?.MsoChpDefault',
            r'WordSection\\d+',
            r'<[^>]+>',  # html-теги
            r'@font-face ?\{[^}]+\}',
            r'@media screen ?\{[^}]+\}',
            r'\b\+7 ?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b',
            r'\b\d{10,12}\b',
            r'\b\w+@\w+\.\w+\b',
            r'\bskype: ?[\w\.-]+\b',
            r'\bip:? ?\d{1,3}(?:\.\d{1,3}){3}\b',
            r'\bтел\.?[: ]?\+?\d+',
            r'\bмоб\.?[: ]?\+?\d+',
            r'\bsite:? ?[\w\.-]+',
            r'\bwww\.[\w\.-]+',
            r'\bemail:? ?[\w\.-]+@\w+\.\w+',
            r'\bС уважением[.,!\s\S]*$',
            r'\bBest regards[.,!\s\S]*$',
            r'\bСлужба технической поддержки[.,!\s\S]*$',
            r'\bTechnical Support Team[.,!\s\S]*$',
            r'\bООО "Связьком"[.,!\s\S]*$',
            r'\bSvyazcom LLC[.,!\s\S]*$',
            r'\bSent: [A-Za-z, ]+\d{1,2},? \d{4}',
            r'\bFrom: [^\n]+',
            r'\bTo: [^\n]+',
            r'\bSubject: [^\n]+',
            r'\bCc: [^\n]+',
            r'\b--+\s*$',
        ]
        greetings = [
            r'^(?:[а-яА-ЯёЁa-zA-Z]+,?\s*)?(уважаемые коллеги|коллеги|алексей|уважаемый(а)?[\w\s]*|добр(ый|ое|ого) (день|утро|вечер)|здравствуйте|привет)[!,.\s\\-]*',
        ]
        filtered = text
        prev = None
        # Цикл: удаляем артефакты и приветствия до полной очистки
        while prev != filtered:
            prev = filtered
            for pat in patterns:
                filtered = re.sub(pat, '', filtered, flags=re.IGNORECASE | re.MULTILINE)
            for greet_pat in greetings:
                filtered = re.sub(greet_pat, '', filtered, flags=re.IGNORECASE)
            filtered = re.sub(r'\s+', ' ', filtered).strip()
        if not filtered or not re.search(r'[а-яА-Яa-zA-Z]{4,}', filtered):
            return ''
        return filtered

    def count_tokens(self, text: str) -> int:
        """Примерный подсчет токенов"""
        return len(text.split())
    
    def find_email_by_message_id(self, message_id):
        """Ищет письмо по message_id (приводит к нижнему регистру, убирает < >)"""
        if not message_id:
            return None
        clean_id = message_id.strip().lower().replace('<', '').replace('>', '')
        for date_dir in Path(self.emails_dir).iterdir():
            if date_dir.is_dir():
                for email_dir in date_dir.iterdir():
                    if email_dir.is_dir():
                        email_file = email_dir / "email.json"
                        if email_file.exists():
                            try:
                                with open(email_file, 'r', encoding='utf-8') as f:
                                    email_data = json.load(f)
                                file_id = None
                                if 'headers' in email_data and 'message-id' in email_data['headers']:
                                    file_id_raw = email_data['headers']['message-id'][0] if isinstance(email_data['headers']['message-id'], list) else email_data['headers']['message-id']
                                    file_id = file_id_raw.strip().lower().replace('<', '').replace('>', '')
                                if file_id and file_id == clean_id:
                                    return email_data
                            except Exception:
                                continue
        return None

    def process_chain(self, chain):
        """Обрабатывает цепочку писем"""
        root_email_id = chain.get('root_email', {}).get('message_id') if isinstance(chain.get('root_email'), dict) else None
        response_email_id = chain.get('first_response', {}).get('message_id') if isinstance(chain.get('first_response'), dict) else None
        if not root_email_id or not response_email_id:
            return None
        root_email = self.find_email_by_message_id(root_email_id)
        response_email = self.find_email_by_message_id(response_email_id)
        if not root_email or not response_email:
            return None
        root_content = self.filter_text(root_email.get('text', ''))
        response_content = self.filter_text(response_email.get('text', ''))
        if not root_content or not response_content:
            return None
        return {
            'emails': [
                {
                    'email': {
                        'from_field': root_email.get('from', ''),
                        'subject': root_email.get('subject', ''),
                        'content': root_content
                    }
                },
                {
                    'email': {
                        'from_field': response_email.get('from', ''),
                        'subject': response_email.get('subject', ''),
                        'content': response_content,
                        'from_email': response_email.get('from', ''),
                        'attachments': response_email.get('attachments', [])
                    }
                }
            ]
        }

    def extract_instruction_response(self, processed_chain):
        """Извлекает пару instruction-response из обработанной цепочки"""
        emails = processed_chain.get('emails')
        if not isinstance(emails, list):
            return None
        client_msg = None
        specialist_msg = None
        for email_data in emails:
            email = email_data['email']
            from_email = email['from_field']
            is_specialist = any(spec in from_email for spec in self.config['specialists'])
            if is_specialist:
                if not specialist_msg:
                    specialist_msg = {
                        'content': email['content'],
                        'from_email': email.get('from_email', from_email),
                        'attachments': email.get('attachments', [])
                    }
            else:
                if not client_msg:
                    client_msg = {
                        'content': email['content']
                    }
        if client_msg and specialist_msg:
            return (client_msg, specialist_msg)
        return None

    def extract_email(self, s):
        """Извлекает email из строки вида 'Имя <email@domain.ru>' или просто 'email@domain.ru'"""
        if not s:
            return ''
        s = s.replace('\\u003c', '<').replace('\\u003e', '>')
        match = re.search(r'[\w\.-]+@[\w\.-]+', s)
        return match.group(0).lower() if match else ''

    def generate_jsonl(self, output_file, test_mode=False, limit=1000000):
        """Генерирует JSONL файл с парами instruction-response, применяя фильтрацию к каждому письму"""
        chains = self.get_email_chains(test_mode=test_mode, limit=limit)
        processed_chains = []
        seen_pairs = set()  # Для умной дедупликации
        for chain in chains:
            root_email_id = chain.get('root_email', {}).get('message_id') if isinstance(chain.get('root_email'), dict) else None
            response_email_id = chain.get('first_response', {}).get('message_id') if isinstance(chain.get('first_response'), dict) else None
            if not root_email_id or not response_email_id:
                continue
            root_email = self.find_email_by_message_id(root_email_id)
            response_email = self.find_email_by_message_id(response_email_id)
            if not root_email or not response_email:
                continue
            # Применяем фильтрацию к тексту
            instruction = self.filter_text(root_email.get('text', ''))
            response = self.filter_text(response_email.get('text', ''))
            if not instruction or not response or len(instruction) < 10 or len(response) < 10:
                continue
            # Извлекаем email адрес для route_to из headers.from
            from_field = ''
            if 'headers' in response_email and 'from' in response_email['headers']:
                from_list = response_email['headers']['from']
                if isinstance(from_list, list) and len(from_list) > 0:
                    from_field = from_list[0]
            route_to = self.extract_email(from_field)
            # Извлекаем attachments из поля files
            attachments = response_email.get('files', [])
            record = {
                'instruction': instruction.strip(),
                'response': response.strip(),
                'route_to': route_to,
                'attachments': attachments
            }
            # Дедупликация
            pair_key = (record['instruction'][:100], record['route_to'], record['response'][:100])
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                processed_chains.append(record)
        with open(output_file, 'w', encoding='utf-8') as f:
            for record in processed_chains:
                json.dump(record, f, ensure_ascii=False)
                f.write('\n')
        print(f"Сохранено {len(processed_chains)} цепочек в {output_file}")

    def iter_valid_chains(self, test_mode=False, limit=10):
        """
        Итератор по валидным цепочкам писем (после process_chain)
        """
        chains = self.get_email_chains(test_mode=test_mode, limit=limit)
        if not isinstance(chains, list):
            return
        for chain in chains:
            processed = self.process_chain(chain)
            if processed:
                yield processed

def main():
    processor = EmailProcessor(
        db_path="C:/Users/Admin/Desktop/SvyazkomProjectCopy/EmailConverter/emails_new/email_index.db",
        emails_dir="C:/Users/Admin/Desktop/SvyazkomProjectCopy/EmailConverter/emails_new"
    )
    processor.connect_db()
    processor.generate_jsonl(output_file="test_emails.jsonl", test_mode=False)
    processor.close_db()
    print("Готово. Проверьте test_emails.jsonl.")

if __name__ == "__main__":
    main() 