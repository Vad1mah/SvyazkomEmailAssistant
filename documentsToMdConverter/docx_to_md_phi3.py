#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Конвертер .docx в .md с анализом изображений через Phi-3.
Восстановленная версия с продвинутой структурой, логами и обработкой контекста.
"""

import os
import sys
import logging
import re
from pathlib import Path
from typing import List, Tuple
import io
from datetime import datetime
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from docx import Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import Table
from docx.text.paragraph import Paragraph
import zipfile
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conversion_phi3.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("docx_to_md_phi3")

class DocxConverterPhi3:
    UNWANTED_SECTION_HEADERS = [
        r'^Security Classification',
        r'^Restricted Information',
        r'^Copyright\s+Notice',
        r'^Document History',
        r'^Other Information',
        r'^Keywords',
        r'^Revision control',
        r'^Table of Contents'
    ]
    
    def __init__(self, input_dir="docx", output_dir="md_phi3", dry_run=False, workers=1):
        self._init_phi3()
        self.md_dir = Path(output_dir)
        if not dry_run:
            self.md_dir.mkdir(exist_ok=True)
        self.input_dir = Path(input_dir)
        self.dry_run = dry_run
        self.workers = workers
        self.stats = {
            'processed_files': 0,
            'extracted_images': 0,
            'ai_analyses': 0,
            'errors': 0
        }

    def _init_phi3(self):
        try:
            logger.info("Инициализация Phi-3...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model_id = "microsoft/phi-3-mini-4k-instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16 if self.device=="cuda" else torch.float32
            )
            logger.info("Phi-3 инициализирована успешно")
        except Exception as e:
            logger.error(f"Не удалось инициализировать Phi-3: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None

    def _is_unwanted_paragraph(self, text: str) -> bool:
        txt = text.strip()
        return any(re.match(pat, txt, re.IGNORECASE)
                   for pat in self.UNWANTED_SECTION_HEADERS)

    def _extract_all_images_from_docx(self, docx_path: Path) -> List[dict]:
        images = []
        hash_counts = {}
        try:
            with zipfile.ZipFile(docx_path, 'r') as zip_file:
                for file_info in zip_file.filelist:
                    if file_info.filename.startswith('word/media/'):
                        data = zip_file.read(file_info.filename)
                        h = hashlib.md5(data).hexdigest()
                        hash_counts[h] = hash_counts.get(h, 0) + 1
                        images.append({'data': data, 'size': len(data), 'hash': h})
        except Exception as e:
            logger.error(f"Ошибка чтения media в {docx_path.name}: {e}")
        for img in images:
            img['count'] = hash_counts[img['hash']]
        return images

    def extract_text_and_images_from_docx(
        self,
        docx_path: Path
    ) -> Tuple[str, List[Tuple[int, bytes, int, str, str]], dict]:
        """
        Возвращает:
         - склеенный текст без «мусорных» блоков,
         - список кортежей (номер, data, позиция, контекст, хэш),
         - словарь hash->(номер, data, позиция, контекст)
        """
        skip_block = False
        try:
            doc = Document(str(docx_path))
            image_files = self._extract_all_images_from_docx(docx_path)
            text_parts: List[str] = []
            image_hash_to_info = {}
            last_text_block = ""
            img_idx = 0
            image_counter = 0
            logo_hash = image_files[0]['hash'] if image_files else None

            for element in doc.element.body:

                # --- ПАРАГРАФЫ ---
                if isinstance(element, CT_P):
                    paragraph = Paragraph(element, doc)
                    text = paragraph.text.strip()

                    # старт/стоп пропуска текста
                    if self._is_unwanted_paragraph(text):
                        logger.info(f"Skip text section: {text}")
                        skip_block = True
                    elif skip_block and (
                        re.match(r'^\d+(\.\d+)*\s+', text) or
                        re.match(r'^[A-Z][A-Z\s]{8,}$', text)
                    ):
                        skip_block = False

                    # добавляем текст, только если не в skip_block
                    if not skip_block and text:
                        text_parts.append(text)
                        last_text_block = text

                    # но в любом случае обрабатываем картинки в этом параграфе
                    for run in paragraph.runs:
                        for child in run._element:
                            if child.tag.endswith('drawing') or child.tag.endswith('pict'):
                                if img_idx < len(image_files):
                                    info = image_files[img_idx]
                                    h = info['hash']
                                    # логотип пропускаем полностью
                                    if not (logo_hash and h == logo_hash):
                                        if h not in image_hash_to_info:
                                            image_counter += 1
                                            image_hash_to_info[h] = (
                                                image_counter,
                                                info['data'],
                                                len(text_parts),
                                                last_text_block
                                            )
                                            text_parts.append(f"[IMAGE_{image_counter}]")
                                        else:
                                            num = image_hash_to_info[h][0]
                                            text_parts.append(f"[IMAGE_{num}]")
                                    img_idx += 1
                                break
                        # если нашли картинку — переходим к следующему run
                    continue

                # --- ТАБЛИЦЫ ---
                if isinstance(element, CT_Tbl):
                    table = Table(element, doc)
                    first_cell = table.rows[0].cells[0].text.strip() if table.rows else ""
                    # если это «Document History» или подобное — пропускаем всю таблицу
                    if any(re.match(pat, first_cell, re.IGNORECASE)
                           for pat in self.UNWANTED_SECTION_HEADERS):
                        logger.info(f"Skip unwanted table: {first_cell}")
                        continue

                    # обычная таблица
                    table_text = self._extract_table_text(table)
                    if table_text:
                        text_parts.append(table_text)
                        last_text_block = table_text

                    # в таблицах тоже проверяем картинки
                    for row in table.rows:
                        for cell in row.cells:
                            for para in cell.paragraphs:
                                for run in para.runs:
                                    for child in run._element:
                                        if child.tag.endswith('drawing') or child.tag.endswith('pict'):
                                            if img_idx < len(image_files):
                                                info = image_files[img_idx]
                                                h = info['hash']
                                                if not (logo_hash and h == logo_hash):
                                                    if h not in image_hash_to_info:
                                                        image_counter += 1
                                                        image_hash_to_info[h] = (
                                                            image_counter,
                                                            info['data'],
                                                            len(text_parts),
                                                            last_text_block
                                                        )
                                                        text_parts.append(f"[IMAGE_{image_counter}]")
                                                    else:
                                                        num = image_hash_to_info[h][0]
                                                        text_parts.append(f"[IMAGE_{num}]")
                                                img_idx += 1
                                            break

            # формируем список из map
            images = [
                (num, data, pos, ctx, h)
                for h, (num, data, pos, ctx) in image_hash_to_info.items()
            ]
            return '\n'.join(text_parts), images, image_hash_to_info

        except Exception as e:
            logger.error(f"Ошибка извлечения из {docx_path.name}: {e}", exc_info=True)
            return "", [], {}
 
    def _extract_table_text(self, table: Table) -> str:
        """Генерирует markdown-таблицу, пропуская «историю версий» и т.п."""
        if not table.rows:
            return ""
        # Если первая ячейка показывает ненужный блок — сразу выходим
        first_cell = table.rows[0].cells[0].text.strip()
        if any(re.match(pattern, first_cell, re.IGNORECASE)
               for pattern in self.UNWANTED_SECTION_HEADERS):
            return ""

        # В противном случае строим markdown-таблицу
        table_lines = []
        for i, row in enumerate(table.rows):
            cells = [cell.text.strip() for cell in row.cells]
            if cells:
                line = '| ' + ' | '.join(cells) + ' |'
                table_lines.append(line)
                if i == 0:
                    sep = '| ' + ' | '.join(['---'] * len(cells)) + ' |'
                    table_lines.append(sep)
        return '\n'.join(table_lines)

    def postprocess_image_description(self, desc: str) -> str:
        # Удаляем неанглийский текст (если встречается)
        # Оставляем только строки, где >60% символов — латиница/цифры/знаки препинания
        lines = desc.split('\n')
        eng_lines = []
        for l in lines:
            l_strip = l.strip()
            if not l_strip:
                continue
            latin = sum(c.isalpha() and c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' for c in l_strip)
            total = sum(c.isalpha() for c in l_strip)
            if total == 0 or (latin / total > 0.6):
                eng_lines.append(l_strip)
        desc = '\n'.join(eng_lines)
        # Удаляем повторяющиеся строки
        seen = set()
        uniq_lines = []
        for l in desc.split('\n'):
            if l not in seen:
                uniq_lines.append(l)
                seen.add(l)
        desc = '\n'.join(uniq_lines)
        # Удаляем шаблонные фразы (case-insensitive)
        desc = re.sub(r'(?i)^(this is|the image shows|figure|diagram|picture|photo|image|document)[\s:：-]*', '', desc)
        # Удаляем лишние пробелы и пустые строки
        desc = re.sub(r'\n{2,}', '\n', desc)
        desc = desc.strip()
        # Ограничиваем длину описания (например, 600 символов)
        if len(desc) > 600:
            desc = desc[:600].rsplit(' ', 1)[0] + '...'
        return desc

    def _is_meaningless_image(self, image_data: bytes) -> bool:
        # Проверка: слишком маленькое изображение или почти полностью белое/чёрное
        try:
            img = Image.open(io.BytesIO(image_data))
            if img.width < 100 or img.height < 100:
                return True
            # Проверка на почти однотонность
            grayscale = img.convert('L')
            hist = grayscale.histogram()
            total = sum(hist)
            # Если 95%+ пикселей одного цвета (почти белое/чёрное)
            if max(hist) / total > 0.95:
                return True
        except Exception:
            return False
        return False

    def analyze_image_with_phi3(self, image_data: bytes, context_text: str = "") -> str:
        try:
            temp_path = "temp_image.jpg"
            with open(temp_path, 'wb') as f:
                f.write(image_data)
            prompt = (
                "Provide a concise, structured, and informative description of the diagram or image for a technical knowledge base. "
                "If the image is a diagram, flowchart, schema, table, or technical illustration, describe its structure in detail. "
                "If the image is a regular photo, background, logo, or does not contain technical content, reply: 'This image is not a technical diagram, chart, table, or illustration and will be skipped.' "
                "- List all main components and their relationships (hierarchy, connections, branches). "
                "- Enumerate all key fields, nodes, or blocks, specifying their names and any labels such as 'Mandatory (M)', 'Optional (O)', 'Conditional (C)', or 'Recommended (R)'. "
                "- Clearly explain the meaning of each field or label if possible. "
                "- Do not include introductions, repeated questions, or template phrases. "
                "Write in clear, professional English, focusing on technical accuracy and completeness. "
                f"Context: {context_text[:200]}\n<file://{temp_path}>\nDescription:"
            )
            if self.tokenizer is None or self.model is None:
                logger.error("Phi-3 model or tokenizer is not initialized.")
                os.remove(temp_path)
                return "[Image description unavailable]"
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                output = self.model.generate(**inputs, max_new_tokens=120)
                response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            except Exception as e:
                logger.error(f"Ошибка генерации описания изображения: {e}")
                os.remove(temp_path)
                return "[Image description unavailable]"
            os.remove(temp_path)
            self.stats['ai_analyses'] += 1
            # Обрезаем prompt из ответа
            if 'Description:' in response:
                response = response.split('Description:')[-1].strip()
            # Если модель явно сказала, что это не техническое изображение — пропускаем
            if 'not a technical diagram' in response.lower() or 'will be skipped' in response.lower():
                return '[Image skipped: not a technical diagram]'
            # Удаляем шаблонные слова и пустые строки
            response = re.sub(r'(?i)^(document|title|introduction|description|image|figure)[\s:：-]*', '', response)
            response = re.sub(r'(?i)^(document|title|introduction|description|image|figure)[\s:：-]*', '', response)
            response = re.sub(r'^[\s>\-*]+', '', response, flags=re.MULTILINE)
            response = re.sub(r'\n{2,}', '\n', response)
            # Постобработка описания
            response = self.postprocess_image_description(response)
            return response.strip()
        except Exception as e:
            logger.error(f"Phi-3 анализ изображения не удался: {e}")
            return "[Image description unavailable]"

    def clean_and_structure_text(self, text: str) -> str:
        """
        Очищает текст от оставшихся ненужных секций и структурирует в Markdown:
        заголовки, списки, таблицы и т.д.
        """
        # 1) Вырезаем блоки по UNWANTED_SECTION_HEADERS
        lines = text.split('\n')
        pruned: List[str] = []
        skip = False
        for line in lines:
            stripped = line.strip()
            # Начало ненужной секции
            if any(re.match(pat, stripped, re.IGNORECASE)
                   for pat in self.UNWANTED_SECTION_HEADERS):
                skip = True
                continue
            # Конец: заголовок в Markdown (#, ## и т.п.)
            if skip and re.match(r'^#{1,6}\s+\w+', stripped):
                skip = False
            if not skip:
                pruned.append(line)

        # 2) Основная логика преобразования в Markdown
        cleaned_lines: List[str] = []
        in_table = False
        for idx, raw in enumerate(pruned):
            line = raw.rstrip()

            # Таблица Markdown
            if line.strip().startswith('|') and line.strip().endswith('|'):
                cleaned_lines.append(line)
                in_table = True
                continue
            if in_table and not (line.strip().startswith('|') and line.strip().endswith('|')):
                in_table = False

            # Нумерованные заголовки вида 1. 1.1. A. I.
            m = re.match(r'^(\d+(?:\.\d+)*|[A-Z]|[IVX]+)\.?\s+(.+)', line)
            if m:
                prefix = m.group(1)
                header = m.group(2).strip()
                level = prefix.count('.') + 1 if '.' in prefix else 2
                level = min(level, 6)
                cleaned_lines.append(f"{'#' * level} {header}")
                continue

            # Явные заголовки: все CAPS >8 символов
            if re.match(r'^[A-Z][A-Z\s]{8,}$', line):
                cleaned_lines.append(f"# {line.title()}")
                continue

            # Маркированный список
            if re.match(r'^\s*[-*•]\s+', line):
                cleaned_lines.append(re.sub(r'^\s*[-*•]\s+', '- ', line))
                continue

            # Нумерованный список
            if re.match(r'^\s*\d+\.', line):
                cleaned_lines.append(re.sub(r'^\s*(\d+)\.', r'\1.', line))
                continue

            # Вложенные списки
            m_indent = re.match(r'^(\s+)[-*•]\s+', line)
            if m_indent:
                indent = len(m_indent.group(1)) // 2
                cleaned_lines.append('  ' * indent + '- ' + line.lstrip(' -*•'))
                continue

            # Обычный текст
            cleaned_lines.append(line)

        # 3) Убираем лишние пустые строки
        result: List[str] = []
        prev_blank = False
        for l in cleaned_lines:
            if l.strip() == '':
                if not prev_blank:
                    result.append('')
                prev_blank = True
            else:
                result.append(l)
                prev_blank = False

        return '\n'.join(result).strip()
 
    def _process_document_worker(self, docx_file):
        # Для multiprocessing: инициализация модели в каждом процессе не требуется, только обработка файла
        try:
            text, images, image_hash_to_info = self.extract_text_and_images_from_docx(docx_file)
            if not text:
                return (docx_file.name, None, 0, 0, 1)  # errors=1
            image_descriptions = {}
            extracted_images = 0
            ai_analyses = 0
            # Анализируем только уникальные изображения
            for num, img_data, position, context, h in images:
                if self._is_meaningless_image(img_data):
                    image_descriptions[num] = "[Image skipped: meaningless photo]"
                    continue
                try:
                    description = self.analyze_image_with_phi3(img_data, context)
                    image_descriptions[num] = description
                    extracted_images += 1
                    ai_analyses += 1
                except Exception:
                    image_descriptions[num] = "[Image description unavailable]"
            # Подставляем описания по номерам в текст
            def replace_marker(match):
                try:
                    image_num = int(match.group(1))
                    desc = image_descriptions.get(image_num, "")
                    if desc.startswith('[Image skipped:'):
                        return ""
                    if image_num in image_descriptions:
                        return str(desc).strip()
                    else:
                        return ""
                except ValueError:
                    return match.group(0)
            import re
            final_text = re.sub(r'\[IMAGE_(\d+)\]', replace_marker, text)
            # Удаляем строки, где остались '[Image skipped:' или пустые маркеры
            final_text = '\n'.join([line for line in final_text.splitlines() if '[Image skipped:' not in line and not re.match(r'^\s*\[IMAGE_\d+\]\s*$', line)])
            cleaned_text = self.clean_and_structure_text(final_text)
            md_content = self._create_markdown_content(docx_file, cleaned_text, extracted_images)
            return (docx_file.name, md_content, extracted_images, ai_analyses, 0)
        except Exception:
            return (docx_file.name, None, 0, 0, 1)

    def _create_markdown_content(self, docx_path: Path, text: str, image_count: int) -> str:
        title = docx_path.stem.replace('_', ' ').title()
        metadata = f"# {title}\n\n"
        return metadata + text

    def convert_all_documents(self, target_file: str = "", file_filter: str = "*.docx"):
        docx_dir = self.input_dir
        if not docx_dir.exists():
            logger.error(f"Папка {docx_dir}/ не найдена")
            return
        if target_file:
            docx_files = [docx_dir / target_file]
            if not docx_files[0].exists():
                logger.error(f"Файл {target_file} не найден в папке {docx_dir}")
                return
        else:
            docx_files = list(docx_dir.glob(file_filter))
        docx_files = [f for f in docx_files if not f.name.startswith('~$')]
        if not docx_files:
            logger.warning("Документы .docx не найдены")
            return
        logger.info(f"Найдено {len(docx_files)} документов для конвертации")
        results = []
        if self.workers > 1:
            with Pool(processes=self.workers) as pool:
                for res in tqdm(pool.imap(self._process_document_worker, sorted(docx_files)), total=len(docx_files), desc="Обработка файлов", unit="файл"):
                    results.append(res)
        else:
            for docx_file in tqdm(sorted(docx_files), desc="Обработка файлов", unit="файл"):
                res = self._process_document_worker(docx_file)
                results.append(res)
        for docx_file_name, md_content, extracted_images, ai_analyses, errors in results:
            if md_content:
                if not self.dry_run:
                    md_file = self.md_dir / f"{Path(docx_file_name).stem}.md"
                    with open(md_file, 'w', encoding='utf-8') as f:
                        f.write(md_content)
                    logger.info(f"Создан файл: {md_file}")
                self.stats['processed_files'] += 1
                self.stats['extracted_images'] += extracted_images
                self.stats['ai_analyses'] += ai_analyses
            else:
                self.stats['errors'] += errors
        self._print_statistics()

    def _print_statistics(self):
        logger.info("=" * 50)
        logger.info("СТАТИСТИКА ОБРАБОТКИ")
        logger.info("=" * 50)
        logger.info(f"Обработано файлов: {self.stats['processed_files']}")
        logger.info(f"Извлечено изображений: {self.stats['extracted_images']}")
        logger.info(f"ИИ-анализов выполнено: {self.stats['ai_analyses']}")
        logger.info(f"Ошибок: {self.stats['errors']}")
        logger.info("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Конвертер .docx в .md с анализом изображений через Phi-3")
    parser.add_argument('filename', type=str, nargs='?', default=None,
                        help='Имя конкретного .docx файла для обработки. Если не указано, обрабатываются все файлы в папке docx/.')
    parser.add_argument('--input', type=str, default='docx', help='Папка с исходными .docx файлами (по умолчанию: docx)')
    parser.add_argument('--output', type=str, default='md_phi3', help='Папка для сохранения .md файлов (по умолчанию: md_phi3)')
    parser.add_argument('--filter', type=str, default='*.docx', help='Маска фильтрации файлов, например: "*report*.docx"')
    parser.add_argument('--dry-run', action='store_true', help='Не сохранять .md файлы, только выводить процесс')
    parser.add_argument('--workers', type=int, default=1, help='Число параллельных процессов (по умолчанию: 1, максимум: число ядер)')
    args = parser.parse_args()
    print("🚀 Конвертер .docx в .md с анализом изображений через Phi-3")
    print("=" * 60)
    try:
        converter = DocxConverterPhi3(input_dir=args.input, output_dir=args.output, dry_run=args.dry_run, workers=args.workers)
        converter.convert_all_documents(target_file=args.filename, file_filter=args.filter)
        print("✅ Конвертация завершена успешно!")
    except Exception as e:
        print(f"❌ Ошибка конвертации: {e}")
        logger.critical(f"Критическая ошибка: {e}", exc_info=True)

if __name__ == "__main__":
    main() 