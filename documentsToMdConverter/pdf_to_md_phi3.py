#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Конвертер PDF в .md с анализом изображений через Phi-3.
Аналог docx_to_md_phi3.py, но для PDF файлов.
"""

import os
import sys
import logging
import re
from pathlib import Path
from typing import List, Tuple, Dict
import io
from datetime import datetime
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import fitz  # PyMuPDF
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_conversion_phi3.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pdf_to_md_phi3")

class PdfConverterPhi3:
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
    
    def __init__(self, input_dir="pdf", output_dir="codeDocumentation", dry_run=False, workers=1):
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

    def _is_unwanted_text(self, text: str) -> bool:
        """Проверка на нежелательные секции"""
        txt = text.strip()
        return any(re.match(pat, txt, re.IGNORECASE)
                   for pat in self.UNWANTED_SECTION_HEADERS)

    def _extract_images_from_pdf(self, pdf_path: Path) -> List[Dict]:
        """Извлечение изображений из PDF"""
        images = []
        hash_counts = {}
        
        try:
            doc = fitz.open(str(pdf_path))
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                        else:  # CMYK: convert to RGB first
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("png")
                            pix1 = None
                        
                        h = hashlib.md5(img_data).hexdigest()
                        hash_counts[h] = hash_counts.get(h, 0) + 1
                        
                        images.append({
                            'data': img_data,
                            'size': len(img_data),
                            'hash': h,
                            'page': page_num + 1,
                            'index': img_index
                        })
                        
                        pix = None
                        
                    except Exception as e:
                        logger.warning(f"Ошибка извлечения изображения {img_index} со страницы {page_num + 1}: {e}")
                        continue
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Ошибка чтения PDF {pdf_path.name}: {e}")
        
        # Добавляем счетчик вхождений
        for img in images:
            img['count'] = hash_counts[img['hash']]
        
        return images

    def extract_text_and_images_from_pdf(self, pdf_path: Path) -> Tuple[str, List[Dict], Dict]:
        """
        Извлечение текста и изображений из PDF
        Возвращает:
         - очищенный текст
         - список изображений с метаданными
         - словарь hash->информация об изображении
        """
        try:
            doc = fitz.open(str(pdf_path))
            image_files = self._extract_images_from_pdf(pdf_path)
            
            text_parts = []
            image_hash_to_info = {}
            image_counter = 0
            skip_block = False
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Извлекаем текст
                text = page.get_text()
                
                # Разбиваем на параграфы
                paragraphs = text.split('\n\n')
                
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    
                    # Проверяем на нежелательные секции
                    if self._is_unwanted_text(para):
                        logger.info(f"Skip text section: {para[:100]}...")
                        skip_block = True
                        continue
                    elif skip_block and (
                        re.match(r'^\d+(\.\d+)*\s+', para) or
                        re.match(r'^[A-Z][A-Z\s]{8,}$', para)
                    ):
                        skip_block = False
                    
                    if not skip_block:
                        text_parts.append(para)
                
                # Обрабатываем изображения на странице
                page_images = [img for img in image_files if img['page'] == page_num + 1]
                
                for img in page_images:
                    h = img['hash']
                    if h not in image_hash_to_info:
                        image_counter += 1
                        image_hash_to_info[h] = {
                            'number': image_counter,
                            'data': img['data'],
                            'position': len(text_parts),
                            'page': img['page'],
                            'context': text_parts[-1] if text_parts else ""
                        }
                        text_parts.append(f"[IMAGE_{image_counter}]")
                    else:
                        num = image_hash_to_info[h]['number']
                        text_parts.append(f"[IMAGE_{num}]")
            
            doc.close()
            
            return '\n\n'.join(text_parts), image_files, image_hash_to_info
            
        except Exception as e:
            logger.error(f"Ошибка обработки PDF {pdf_path.name}: {e}")
            return "", [], {}

    def _is_meaningless_image(self, image_data: bytes) -> bool:
        """Проверка на бессмысленные изображения"""
        try:
            img = Image.open(io.BytesIO(image_data))
            
            # Слишком маленькое изображение
            if img.width < 50 or img.height < 50:
                return True
            
            # Проверка на почти белое/черное изображение
            if img.mode in ('RGB', 'RGBA'):
                pixels = list(img.getdata())
                if len(pixels) == 0:
                    return True
                
                # Подсчет белых и черных пикселей
                white_count = sum(1 for p in pixels if p[0] > 240 and p[1] > 240 and p[2] > 240)
                black_count = sum(1 for p in pixels if p[0] < 15 and p[1] < 15 and p[2] < 15)
                
                total_pixels = len(pixels)
                if total_pixels > 0:
                    white_ratio = white_count / total_pixels
                    black_ratio = black_count / total_pixels
                    
                    if white_ratio > 0.9 or black_ratio > 0.9:
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Ошибка анализа изображения: {e}")
            return True

    def analyze_image_with_phi3(self, image_data: bytes, context_text: str = "") -> str:
        """Анализ изображения с помощью Phi-3"""
        if not self.model or not self.tokenizer:
            return "Image analysis not available"
        
        try:
            # Проверяем на бессмысленные изображения
            if self._is_meaningless_image(image_data):
                return ""
            
            # Подготавливаем изображение
            img = Image.open(io.BytesIO(image_data))
            
            # Создаем промпт
            context_hint = f"Context: {context_text[:200]}..." if context_text else ""
            prompt = f"""<|user|>
Analyze this technical image (diagram, chart, table, or technical illustration) and provide a clear, concise description in English. Focus on the technical content, structure, and key information shown.

{context_hint}

Please describe what this image shows in a technical context.
<|assistant|>"""
            
            # Токенизация
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Генерация
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Декодирование
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем только ответ ассистента
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[-1].strip()
            
            # Очистка ответа
            response = self.postprocess_image_description(response)
            
            return response if response else ""
            
        except Exception as e:
            logger.error(f"Ошибка анализа изображения с Phi-3: {e}")
            return ""

    def postprocess_image_description(self, desc: str) -> str:
        """Постобработка описания изображения"""
        if not desc:
            return ""
        
        # Удаляем неанглийский текст
        lines = desc.split('\n')
        english_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Подсчитываем процент латинских символов
            latin_chars = sum(1 for c in line if c.isascii() and c.isprintable())
            total_chars = len(line)
            
            if total_chars > 0 and latin_chars / total_chars > 0.6:
                english_lines.append(line)
        
        result = ' '.join(english_lines)
        
        # Очистка от лишних пробелов
        result = re.sub(r'\s+', ' ', result).strip()
        
        return result

    def clean_and_structure_text(self, text: str) -> str:
        """Очистка и структурирование текста"""
        if not text:
            return ""
        
        # Разбиваем на параграфы
        paragraphs = text.split('\n\n')
        cleaned_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Удаляем лишние пробелы
            para = re.sub(r'\s+', ' ', para)
            
            # Удаляем технические маркеры
            para = re.sub(r'\[IMAGE_\d+\]', '', para)
            
            # Удаляем пустые параграфы
            if para and not para.isspace():
                cleaned_paragraphs.append(para)
        
        return '\n\n'.join(cleaned_paragraphs)

    def _process_document_worker(self, pdf_file):
        """Обработка одного PDF файла (для multiprocessing)"""
        try:
            pdf_path = Path(pdf_file)
            logger.info(f"Обработка: {pdf_path.name}")
            
            # Извлекаем текст и изображения
            text, images, image_hash_to_info = self.extract_text_and_images_from_pdf(pdf_path)
            
            if not text:
                logger.warning(f"Не удалось извлечь текст из {pdf_path.name}")
                return
            
            # Обрабатываем изображения
            image_descriptions = {}
            for h, info in image_hash_to_info.items():
                if info['data']:
                    desc = self.analyze_image_with_phi3(info['data'], info['context'])
                    if desc:
                        image_descriptions[info['number']] = desc
                        self.stats['ai_analyses'] += 1
            
            # Заменяем маркеры изображений на описания
            def replace_marker(match):
                img_num = int(match.group(1))
                return image_descriptions.get(img_num, "")
            
            text = re.sub(r'\[IMAGE_(\d+)\]', replace_marker, text)
            
            # Очищаем и структурируем текст
            text = self.clean_and_structure_text(text)
            
            # Создаем markdown контент
            md_content = self._create_markdown_content(pdf_path, text, len(image_descriptions))
            
            # Сохраняем файл
            if not self.dry_run:
                output_file = self.md_dir / f"{pdf_path.stem}.md"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                logger.info(f"Сохранен: {output_file}")
            
            self.stats['processed_files'] += 1
            self.stats['extracted_images'] += len(images)
            
        except Exception as e:
            logger.error(f"Ошибка обработки {pdf_file}: {e}")
            self.stats['errors'] += 1

    def _create_markdown_content(self, pdf_path: Path, text: str, image_count: int) -> str:
        """Создание markdown контента"""
        header = f"# {pdf_path.stem}\n\n"
        header += f"*Converted from PDF: {pdf_path.name}*\n"
        header += f"*Images analyzed: {image_count}*\n"
        header += f"*Conversion date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        
        return header + text

    def convert_all_documents(self, target_file: str = "", file_filter: str = "*.pdf"):
        """Конвертация всех PDF документов"""
        if target_file:
            pdf_files = [self.input_dir / target_file]
        else:
            pdf_files = list(self.input_dir.glob(file_filter))
        
        if not pdf_files:
            logger.warning(f"Не найдено PDF файлов в {self.input_dir}")
            return
        
        logger.info(f"Найдено {len(pdf_files)} PDF файлов для обработки")
        
        if self.workers == 1:
            # Последовательная обработка
            for pdf_file in tqdm(pdf_files, desc="Обработка PDF"):
                self._process_document_worker(pdf_file)
        else:
            # Параллельная обработка
            with Pool(processes=min(self.workers, len(pdf_files))) as pool:
                list(tqdm(
                    pool.imap(self._process_document_worker, pdf_files),
                    total=len(pdf_files),
                    desc="Обработка PDF"
                ))
        
        self._print_statistics()

    def _print_statistics(self):
        """Вывод статистики"""
        logger.info("=" * 50)
        logger.info("СТАТИСТИКА ОБРАБОТКИ PDF")
        logger.info("=" * 50)
        logger.info(f"Обработано файлов: {self.stats['processed_files']}")
        logger.info(f"Извлечено изображений: {self.stats['extracted_images']}")
        logger.info(f"AI-анализов: {self.stats['ai_analyses']}")
        logger.info(f"Ошибок: {self.stats['errors']}")
        logger.info("=" * 50)

def main():
    parser = argparse.ArgumentParser(description='Конвертер PDF в .md с анализом изображений через Phi-3')
    parser.add_argument('pdf_file', nargs='?', help='Конкретный PDF файл для обработки')
    parser.add_argument('--input', default='pdf', help='Папка с PDF файлами (по умолчанию: pdf)')
    parser.add_argument('--output', default='codeDocumentation', help='Папка для сохранения .md (по умолчанию: codeDocumentation)')
    parser.add_argument('--filter', default='*.pdf', help='Маска фильтрации файлов (по умолчанию: *.pdf)')
    parser.add_argument('--dry-run', action='store_true', help='Не сохранять .md файлы, только выводить процесс')
    parser.add_argument('--workers', type=int, default=1, help='Число параллельных процессов (по умолчанию: 1)')
    
    args = parser.parse_args()
    
    converter = PdfConverterPhi3(
        input_dir=args.input,
        output_dir=args.output,
        dry_run=args.dry_run,
        workers=args.workers
    )
    
    converter.convert_all_documents(args.pdf_file, args.filter)

if __name__ == "__main__":
    main() 