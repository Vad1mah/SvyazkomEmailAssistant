#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор ответов для ИИ-советника с использованием Knowledge Graph
Формат ответа: clarify, solution, routeTo
"""

import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
import time

from knowledge_graph_system import KnowledgeGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Генератор ответов с использованием Knowledge Graph"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.max_context_tokens = 4000
        self.max_response_tokens = 1000
        self.torch = None
        # --- Инициализация модели ---
        import warnings
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_id = "microsoft/Phi-3.5-mini-instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            ).to(self.device)
            if self.device == "cpu":
                logger.warning("Внимание: модель будет работать очень медленно на CPU. Рекомендуется использовать GPU.")
        except Exception as e:
            logger.error(f"Ошибка загрузки Phi-3.5-mini-instruct: {e}. Проверьте установку transformers и torch.")
            raise
        
    def generate_response(self, query: str) -> Dict:
        """Генерация ответа на основе Knowledge Graph"""
        start_time = time.time()
        
        # 1. Поиск релевантного контекста
        context = self._get_relevant_context(query)
        
        # 2. Формирование промпта для LLM
        prompt = self._create_prompt(query, context)
        
        # 3. Генерация ответа в нужном формате
        response = self._generate_llm_response(prompt)
        
        # 4. Парсинг ответа на компоненты
        parsed_response = self._parse_response(response)
        
        # 4.1. Автоматическая подстановка реального email, если routeTo некорректен
        route_to_llm = parsed_response.get("routeTo", "").strip().lower()
        # Список типовых заглушек и невалидных адресов
        bad_emails = {"email@example.com", "sales@example.com", "support@example.com", "info@example.com", "", None}
        is_bad = False
        if route_to_llm in bad_emails or not ("@" in route_to_llm and "." in route_to_llm):
            is_bad = True
        # Проверяем среди специалистов в контексте
        fallback_email = None
        specialist_responses = context.get("specialist_responses", [])
        for resp in specialist_responses:
            email = resp.get("route_to")
            if email and ("@" in email and "." in email):
                fallback_email = email
                break
        # Если не нашли — ищем в email_specialist
        if not fallback_email:
            general_context = context.get("general_context", {})
            email_specialist = general_context.get("email_specialist", [])
            for dialog in email_specialist:
                email = dialog.get("route_to")
                if email and ("@" in email and "." in email):
                    fallback_email = email
                    break
        # Если всё равно не нашли — оставляем routeTo как есть
        if is_bad and fallback_email:
            parsed_response["routeTo"] = fallback_email
        
        # 5. Оценка уверенности
        confidence = self._calculate_confidence(context, parsed_response)
        
        # 6. Определение маршрутизации
        route_to = self._determine_routing(context, parsed_response)
        
        processing_time = time.time() - start_time
        
        return {
            "clarify": parsed_response.get("clarify", ""),
            "solution": parsed_response.get("solution", ""),
            "routeTo": parsed_response.get("routeTo", ""),
            "confidence": confidence,
            "route_to": route_to,
            "context_tokens": len(prompt.split()),
            "sources": self._extract_sources(context),
            "processing_time": processing_time,
            "context_summary": self._create_context_summary(context)
        }
    
    def _get_relevant_context(self, query: str) -> Dict:
        """Получение релевантного контекста из Knowledge Graph"""
        # Получаем контекст с приоритизацией
        context = self.kg.get_context_for_query(query, max_nodes=10)
        
        # Дополнительно ищем ответы специалистов
        specialist_responses = self.kg.find_specialist_responses(query, max_results=5)
        
        return {
            "general_context": context,
            "specialist_responses": specialist_responses
        }
    
    def _create_prompt(self, query: str, context: Dict) -> str:
        """Создание промпта для LLM"""
        
        # Начинаем с инструкции
        prompt = f"""Ты - ИИ-советник по техническим запросам клиентов. Отвечай на основе предоставленного контекста в строго определенном формате.

ЗАПРОС КЛИЕНТА: {query}

КОНТЕКСТ ДЛЯ ОТВЕТА:
"""
        
        # Добавляем ответы специалистов (новая структура)
        specialist_responses = context.get("specialist_responses", [])
        if specialist_responses:
            prompt += "\n=== ПРИМЕРЫ ОТВЕТОВ СПЕЦИАЛИСТОВ ===\n"
            for i, response in enumerate(specialist_responses[:3], 1):
                prompt += f"""
{i}) Уточняющие вопросы: {response.get('instruction', '[нет уточняющих вопросов]')}
   Возможное решение: {response.get('response', '[нет решения]')}
   Перенаправление: {response.get('route_to', '[не указано]')}
"""
        
        # Добавляем общий контекст
        general_context = context.get("general_context", {})
        
        # Email от специалистов
        email_specialist = general_context.get("email_specialist", [])
        if email_specialist:
            prompt += "\n=== ПРИМЕРЫ EMAIL-ДИАЛОГОВ ===\n"
            for i, dialog in enumerate(email_specialist[:2], 1):
                prompt += f"""
{i}) Уточняющие вопросы: {dialog.get('instruction', '[нет уточняющих вопросов]')}
   Возможное решение: {dialog.get('response', '[нет решения]')}
   Перенаправление: {dialog.get('route_to', '[не указано]')}
"""
        
        # Документация
        documentation = general_context.get("documentation", [])
        if documentation:
            prompt += "\n=== ДОКУМЕНТАЦИЯ ===\n"
            for i, doc in enumerate(documentation[:2], 1):
                prompt += f"""
{i}. {doc['content'][:300]}...
"""
        
        # Код
        code = general_context.get("code", [])
        if code:
            prompt += "\n=== ТЕХНИЧЕСКАЯ ДОКУМЕНТАЦИЯ ===\n"
            for i, code_doc in enumerate(code[:2], 1):
                prompt += f"""
{i}. {code_doc['content'][:300]}...
"""
        
        # Инструкция для генерации ответа в нужном формате
        prompt += f"""

ИНСТРУКЦИЯ: На основе предоставленного контекста сформируй ответ в строго определенном формате JSON с тремя полями:

1. \"clarify\" - уточняющие вопросы для клиента (если нужны)
2. \"solution\" - возможное решение или ответ на запрос
3. \"routeTo\" - укажи реальный email специалиста или отдела из приведённых выше примеров или контекста (НЕ используй email@example.com или другие заглушки, если не уверен — выбери наиболее релевантный email из контекста).

Если есть ответы специалистов на похожие вопросы — используй их как основу.
Если есть релевантная документация — включи технические детали.
Если информации недостаточно — укажи это и предложи обратиться к специалисту.

ОТВЕТ (только JSON без дополнительного текста):"""
        
        return prompt
    
    def _generate_llm_response(self, prompt: str) -> str:
        """
        Генерация ответа с помощью Phi-3.5-mini-instruct
        """
        if self.torch is None:
            raise RuntimeError("torch не инициализирован. Проверьте установку.")
        # Формируем prompt в формате chat
        chat_prompt = f"<|user|>{prompt}<|end|><|assistant|>"
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=False
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Извлекаем только ответ ассистента
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        return response.strip()
    
    def _parse_response(self, response: str) -> Dict:
        """Парсинг ответа на компоненты clarify, solution, routeTo"""
        try:
            # Пытаемся найти JSON в ответе
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                return {
                    "clarify": parsed.get("clarify", ""),
                    "solution": parsed.get("solution", ""),
                    "routeTo": parsed.get("routeTo", "")
                }
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
        
        # Если не удалось распарсить JSON, разбиваем на части
        parts = response.split('\n')
        clarify = ""
        solution = ""
        routeTo = ""
        
        current_section = None
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            if "clarify" in part.lower() or "уточн" in part.lower():
                current_section = "clarify"
                clarify = part
            elif "solution" in part.lower() or "решен" in part.lower() or "ответ" in part.lower():
                current_section = "solution"
                solution = part
            elif "route" in part.lower() or "перенаправ" in part.lower():
                current_section = "routeTo"
                routeTo = part
            elif current_section:
                if current_section == "clarify":
                    clarify += " " + part
                elif current_section == "solution":
                    solution += " " + part
                elif current_section == "routeTo":
                    routeTo += " " + part
        
        return {
            "clarify": clarify.strip(),
            "solution": solution.strip(),
            "routeTo": routeTo.strip()
        }
    
    def _calculate_confidence(self, context: Dict, parsed_response: Dict) -> float:
        """Расчет уверенности в ответе"""
        confidence = 0.5  # Базовая уверенность
        
        # Повышаем уверенность если есть ответы специалистов
        specialist_responses = context.get("specialist_responses", [])
        if specialist_responses:
            confidence += 0.3
        
        # Повышаем если есть документация
        general_context = context.get("general_context", {})
        if general_context.get("documentation"):
            confidence += 0.1
        
        if general_context.get("code"):
            confidence += 0.1
        
        # Повышаем если есть конкретное решение
        if parsed_response.get("solution"):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _determine_routing(self, context: Dict, parsed_response: Dict) -> Optional[str]:
        """Определение маршрутизации"""
        # Если в ответе указан routeTo, используем его
        if parsed_response.get("routeTo"):
            return parsed_response["routeTo"]
        
        # Иначе определяем на основе контекста
        specialist_responses = context.get("specialist_responses", [])
        if specialist_responses:
            # Берем наиболее частый route_to из ответов специалистов
            route_counts = {}
            for response in specialist_responses:
                route = response.get("route_to")
                if route:
                    route_counts[route] = route_counts.get(route, 0) + 1
            
            if route_counts:
                return max(route_counts, key=route_counts.get)
        
        return None
    
    def _extract_sources(self, context: Dict) -> List[str]:
        """Извлечение источников"""
        sources = []
        
        general_context = context.get("general_context", {})
        
        if general_context.get("documentation"):
            sources.append("documentation")
        
        if general_context.get("code"):
            sources.append("code_documentation")
        
        if general_context.get("email_specialist"):
            sources.append("email_dialogs")
        
        specialist_responses = context.get("specialist_responses", [])
        if specialist_responses:
            sources.append("specialist_responses")
        
        return sources
    
    def _create_context_summary(self, context: Dict) -> Dict:
        """Создание сводки контекста"""
        summary = {
            "total_sources": 0,
            "documentation_sections": 0,
            "code_sections": 0,
            "email_dialogs": 0,
            "specialist_responses": 0
        }
        
        general_context = context.get("general_context", {})
        
        if general_context.get("documentation"):
            summary["documentation_sections"] = len(general_context["documentation"])
            summary["total_sources"] += 1
        
        if general_context.get("code"):
            summary["code_sections"] = len(general_context["code"])
            summary["total_sources"] += 1
        
        if general_context.get("email_specialist"):
            summary["email_dialogs"] = len(general_context["email_specialist"])
            summary["total_sources"] += 1
        
        specialist_responses = context.get("specialist_responses", [])
        if specialist_responses:
            summary["specialist_responses"] = len(specialist_responses)
            summary["total_sources"] += 1
        
        return summary

class ResponseAPI:
    """API для генерации ответов"""
    
    def __init__(self, knowledge_graph_path: str = "knowledge_graph.json"):
        self.kg_path = knowledge_graph_path
        self.kg = None
        self.generator = None
        self._initialize()
    
    def _initialize(self):
        """Инициализация системы"""
        try:
            # Загружаем Knowledge Graph
            self.kg = KnowledgeGraph()
            if Path(self.kg_path).exists():
                self.kg.load_graph(self.kg_path)
            else:
                self.kg.build_graph()
                self.kg.save_graph(self.kg_path)
            
            # Создаем генератор ответов
            self.generator = ResponseGenerator(self.kg)
            
            logger.info("Response API initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Response API: {e}")
            raise
    
    def get_response(self, query: str) -> Dict:
        """Получение ответа на запрос"""
        if self.generator is None:
            raise RuntimeError("Response generator not initialized")
        
        return self.generator.generate_response(query) 