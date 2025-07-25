# Конвертер .docx → .md с анализом изображений (Phi-3)

Этот проект предназначен для пакетной конвертации документов .docx в markdown с автоматическим анализом технических изображений (схем, диаграмм, таблиц) через локальную модель Phi-3.

## Основные возможности
- Поддержка пакетной обработки .docx
- Извлечение текста, таблиц, технических изображений
- AI-анализ изображений (только для схем, диаграмм, графиков и т.п.)
- Фильтрация бессмысленных изображений (фото, логотипы, пустые картинки)
- Чистый markdown-вывод без технических маркеров
- Корректная обработка сложных таблиц
- Логирование процесса и статистика

## Быстрый старт
1. Установите зависимости (Python 3.8+, venv, requirements.txt)
2. Активируйте виртуальное окружение:
   ```
   . venv/Scripts/activate  # Windows
   source venv/bin/activate # Linux/Mac
   ```
3. Запустите конвертацию:
   ```
   python docx_to_md_phi3.py <имя_файла.docx> --input docx --output md_phi3 --workers 1
   ```
   Или для всех файлов:
   ```
   python docx_to_md_phi3.py --input docx --output md_phi3 --workers 1
   ```

## Структура вывода
- Все .md-файлы сохраняются в папке, указанной в --output (по умолчанию md_phi3)
- Вставки AI-описаний только для технических изображений
- Таблицы и структура документа максимально сохраняются

## Логирование и статистика
- Весь процесс логируется в conversion_phi3.log
- В конце работы выводится статистика: количество обработанных файлов, изображений, AI-анализов, ошибок

## Пример команды
```
python docx_to_md_phi3.py 02.docx --input docx --output md_phi3 --workers 1
```

## Контакты и поддержка
Для вопросов и предложений: [ваш email или github]

## Назначение
Этот инструмент предназначен для пакетной конвертации документов `.docx` из папки `docx/` в чистые, англоязычные markdown-файлы для базы знаний. Все изображения анализируются с помощью локальной модели Phi-3 (без облачных или платных API), а их описания вставляются непосредственно в текст в нужном месте, без технических маркеров и лишнего форматирования.

## Архитектура и принципы работы
- Все `.docx` файлы из папки `docx/` обрабатываются автоматически.
- Извлекается и очищается текст, сохраняется структура документа (заголовки, списки, таблицы).
- Все изображения заменяются на краткие, структурированные и полезные описания на английском языке, сгенерированные Phi-3.
- В итоговом markdown-файле нет картинок, только текст и описания изображений.
- Не используются технические маркеры (нет **IMAGE N:**, >, 'Document:' и т.д.), только чистое описание.
- Обработка быстрая, с поддержкой GPU (если доступно).
- Вся обработка бесплатна и выполняется локально.

## Возможности
- Пакетная обработка всех `.docx` из папки `docx/`
- Сохранение структуры markdown (заголовки, списки, таблицы)
- Автоматический анализ и описание всех изображений через Phi-3
- Нет технических маркеров и мусора в итоговом тексте
- Поддержка работы на GPU (если есть)
- Не требует подключения к интернету или облачным сервисам

## Установка
```bash
python -m venv venv
venv\Scripts\activate  # или source venv/bin/activate для Linux/Mac
pip install -r requirements.txt
```

## Использование
1. Поместите ваши `.docx` файлы в папку `docx/` (или другую, указав через --input).
2. Запустите конвертер с нужными параметрами:
```bash
python docx_to_md_phi3.py [имя_файла.docx] --input docx/ --output md_phi3/ --filter "*.docx" --dry-run --workers 4
```
- `имя_файла.docx` — (опционально) обработать только один файл
- `--input` — папка с исходными .docx (по умолчанию: docx)
- `--output` — папка для сохранения .md (по умолчанию: md_phi3)
- `--filter` — маска фильтрации файлов, например: "*report*.docx"
- `--dry-run` — не сохранять .md файлы, только выводить процесс
- `--workers` — число параллельных процессов (по умолчанию: 1, максимум: число ядер)

Во время работы отображается прогресс-бар (tqdm).

3. Готовые `.md` файлы появятся в папке, указанной через `--output` (по умолчанию `md_phi3/`).

## Пример результата
```
# Название документа

Раздел 1
Текст после очистки.

Иерархическая схема, показывающая структуру протокола обмена данными. Верхний уровень — 'Data Interchange', далее ветвление на 'Transfer Batch' и 'Notification'. В каждой ветке указаны поля, такие как Sender, Recipient, File Sequence Number, различные временные метки и индикаторы, с пометкой обязательности (M, O, C, R). Стрелки отражают иерархию и вложенность параметров.

Дальнейший текст...
```

## Примечания
- Весь текст и описания изображений — только на английском языке.
- Для анализа изображений используется только Phi-3, облачные и платные сервисы не применяются.
- Для максимальной производительности рекомендуется использовать компьютер с NVIDIA GPU и установленными драйверами CUDA.

## Требования
См. файл `requirements.txt` для полного списка зависимостей.

## Лицензия
MIT 