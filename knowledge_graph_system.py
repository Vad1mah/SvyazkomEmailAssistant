#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Система Knowledge Graph для ИИ-советника по техническим запросам клиентов.
Интегрирует три источника знаний:
1. База знаний по документации (md_phi3)
2. База знаний по коду (codeDocumentation)
3. Email-пары (JSON)
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import defaultdict
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeNode:
    """Узел в графе знаний"""
    id: str
    content: str
    source: str  # 'documentation', 'code', 'email'
    metadata: Dict
    embeddings: Optional[List[float]] = None

@dataclass
class KnowledgeEdge:
    """Связь между узлами в графе знаний"""
    source_id: str
    target_id: str
    relationship_type: str
    weight: float = 1.0

class KnowledgeGraph:
    """Система Knowledge Graph для интеграции источников знаний"""
    
    def __init__(self, 
                 docs_path: str = "documentsToMdConverter/md_phi3",
                 code_path: str = "documentsToMdConverter/codeDocumentation",
                 email_path: str = "test_emails.jsonl"):
        self.docs_path = Path(docs_path)
        self.code_path = Path(code_path)
        self.email_path = Path(email_path)
        
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        self.node_embeddings: Dict[str, List[float]] = {}
        
        # Индексы для быстрого поиска
        self.content_index: Dict[str, List[str]] = defaultdict(list)
        self.source_index: Dict[str, List[str]] = defaultdict(list)
        
    def load_documentation_knowledge(self) -> List[KnowledgeNode]:
        """Загрузка базы знаний по документации"""
        nodes = []
        
        if not self.docs_path.exists():
            logger.warning(f"Documentation path not found: {self.docs_path}")
            return nodes
        
        for md_file in self.docs_path.glob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Разбиваем на секции
                sections = self._split_into_sections(content)
                
                for i, section in enumerate(sections):
                    if len(section.strip()) < 50:  # Пропускаем слишком короткие секции
                        continue
                    
                    node_id = f"doc_{md_file.stem}_{i}"
                    node = KnowledgeNode(
                        id=node_id,
                        content=section.strip(),
                        source="documentation",
                        metadata={
                            "file": md_file.name,
                            "section": i,
                            "type": "documentation"
                        }
                    )
                    nodes.append(node)
                    
            except Exception as e:
                logger.error(f"Error loading {md_file}: {e}")
        
        logger.info(f"Loaded {len(nodes)} documentation nodes")
        return nodes
    
    def load_code_knowledge(self) -> List[KnowledgeNode]:
        """Загрузка базы знаний по коду"""
        nodes = []
        
        if not self.code_path.exists():
            logger.warning(f"Code documentation path not found: {self.code_path}")
            return nodes
        
        for md_file in self.code_path.glob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Разбиваем на секции
                sections = self._split_into_sections(content)
                
                for i, section in enumerate(sections):
                    if len(section.strip()) < 50:
                        continue
                    
                    node_id = f"code_{md_file.stem}_{i}"
                    node = KnowledgeNode(
                        id=node_id,
                        content=section.strip(),
                        source="code",
                        metadata={
                            "file": md_file.name,
                            "section": i,
                            "type": "code_documentation"
                        }
                    )
                    nodes.append(node)
                    
            except Exception as e:
                logger.error(f"Error loading {md_file}: {e}")
        
        logger.info(f"Loaded {len(nodes)} code documentation nodes")
        return nodes
    
    def load_email_knowledge(self) -> List[KnowledgeNode]:
        """Загрузка базы знаний из email-диалогов (вопрос-ответ)"""
        nodes = []
        if not self.email_path.exists():
            logger.warning(f"Email data path not found: {self.email_path}")
            return nodes
        try:
            with open(self.email_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        data = json.loads(line.strip())
                        # Ожидаем поля: instruction, response, route_to, attachments
                        if not all(k in data for k in ("instruction", "response", "route_to")):
                            logger.warning(f"Invalid data structure at line {line_num}")
                            continue
                        metadata = {
                            "instruction": data["instruction"],
                            "response": data["response"],
                            "route_to": data["route_to"],
                            "line": line_num
                        }
                        attachments = data.get("attachments", [])
                        if attachments:
                            metadata["attachments"] = attachments
                        node = KnowledgeNode(
                            id=f"email_dialog_{line_num}",
                            content=f"Вопрос: {data['instruction']}\n\nОтвет: {data['response']}",
                            source="email",
                            metadata=metadata
                        )
                        nodes.append(node)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON at line {line_num}: {e}")
        except Exception as e:
            logger.error(f"Error loading email data: {e}")
        logger.info(f"Loaded {len(nodes)} email dialog nodes")
        return nodes
    
    def _split_into_sections(self, content: str) -> List[str]:
        """Разбиение контента на секции"""
        # Разбиваем по заголовкам markdown
        sections = re.split(r'\n#{1,6}\s+', content)
        
        # Если нет заголовков, разбиваем по параграфам
        if len(sections) == 1:
            sections = re.split(r'\n\n+', content)
        
        return [s.strip() for s in sections if s.strip()]
    
    def build_graph(self):
        """Построение графа знаний"""
        logger.info("Building knowledge graph...")
        
        # Загружаем все источники знаний
        doc_nodes = self.load_documentation_knowledge()
        code_nodes = self.load_code_knowledge()
        email_nodes = self.load_email_knowledge()
        
        # Добавляем узлы в граф
        all_nodes = doc_nodes + code_nodes + email_nodes
        
        for node in all_nodes:
            self.nodes[node.id] = node
            self.content_index[node.source].append(node.id)
            self.source_index[node.source].append(node.id)
        
        # Создаем связи между узлами
        self._create_edges()
        
        logger.info(f"Knowledge graph built: {len(self.nodes)} nodes, {len(self.edges)} edges")
    
    def _create_edges(self):
        """Создание связей между узлами"""
        # Связи по ключевым словам между документацией и кодом
        self._create_semantic_edges()
        
        # Связи по техническим терминам
        self._create_technical_edges()
        
        # Связи между email-диалогами по схожим темам
        self._create_email_similarity_edges()
    
    def _create_semantic_edges(self):
        """Создание семантических связей"""
        # Извлекаем ключевые слова из документации
        doc_keywords = self._extract_keywords([n for n in self.nodes.values() if n.source == "documentation"])
        code_keywords = self._extract_keywords([n for n in self.nodes.values() if n.source == "code"])
        
        # Создаем связи по общим ключевым словам
        for doc_node in self.nodes.values():
            if doc_node.source != "documentation":
                continue
                
            doc_words = set(self._extract_words(doc_node.content))
            
            for code_node in self.nodes.values():
                if code_node.source != "code":
                    continue
                
                code_words = set(self._extract_words(code_node.content))
                
                # Находим пересечение ключевых слов
                common_words = doc_words.intersection(code_words)
                
                if len(common_words) >= 2:  # Минимум 2 общих слова
                    weight = len(common_words) / max(len(doc_words), len(code_words))
                    
                    edge = KnowledgeEdge(
                        source_id=doc_node.id,
                        target_id=code_node.id,
                        relationship_type="semantic_similarity",
                        weight=weight
                    )
                    self.edges.append(edge)
    
    def _create_technical_edges(self):
        """Создание связей по техническим терминам"""
        technical_terms = [
            "oracle", "database", "clearing", "ftp", "sftp", "cdr", "tap",
            "billing", "partitioning", "backup", "restore", "monitoring",
            "server", "application", "gui", "web", "api", "schema",
            "table", "procedure", "function", "job", "scheduler"
        ]
        
        for term in technical_terms:
            term_nodes = []
            
            for node in self.nodes.values():
                if term.lower() in node.content.lower():
                    term_nodes.append(node)
            
            # Создаем связи между узлами с одинаковыми техническими терминами
            for i, node1 in enumerate(term_nodes):
                for node2 in term_nodes[i+1:]:
                    if node1.id != node2.id:
                        edge = KnowledgeEdge(
                            source_id=node1.id,
                            target_id=node2.id,
                            relationship_type="technical_term",
                            weight=0.8
                        )
                        self.edges.append(edge)
    
    def _create_email_similarity_edges(self):
        """Создание связей между email-диалогами по схожим темам"""
        email_nodes = [n for n in self.nodes.values() if n.source == "email"]
        
        # Создаем связи между диалогами с похожими темами
        for i, node1 in enumerate(email_nodes):
            for node2 in email_nodes[i+1:]:
                # Извлекаем слова из тем диалогов
                subject1 = node1.metadata.get('original_subject', '').lower()
                subject2 = node2.metadata.get('original_subject', '').lower()
                
                # Находим общие слова в темах
                words1 = set(self._extract_words(subject1))
                words2 = set(self._extract_words(subject2))
                common_words = words1.intersection(words2)
                
                # Если есть общие слова в теме, создаем связь
                if len(common_words) >= 2:
                    weight = len(common_words) / max(len(words1), len(words2), 1)
                    
                    edge = KnowledgeEdge(
                        source_id=node1.id,
                        target_id=node2.id,
                        relationship_type="similar_topic",
                        weight=weight
                    )
                    self.edges.append(edge)
    
    def _extract_keywords(self, nodes: List[KnowledgeNode]) -> List[str]:
        """Извлечение ключевых слов из узлов"""
        keywords = []
        
        for node in nodes:
            words = self._extract_words(node.content)
            keywords.extend(words)
        
        return keywords
    
    def _extract_words(self, text: str) -> List[str]:
        """Извлечение слов из текста"""
        # Удаляем специальные символы и разбиваем на слова
        words = re.findall(r'\b[a-zA-Zа-яА-Я]{3,}\b', text.lower())
        
        # Фильтруем стоп-слова
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'и', 'в', 'на', 'с', 'по', 'для', 'от', 'до', 'из', 'за', 'под', 'над',
            'при', 'про', 'без', 'через', 'между', 'около', 'вокруг', 'внутри',
            'вне', 'после', 'до', 'во', 'со', 'об', 'о', 'у', 'к', 'ко'
        }
        
        return [word for word in words if word not in stop_words]
    
    def search_relevant_nodes(self, query: str, top_k: int = 5) -> List[KnowledgeNode]:
        """Поиск релевантных узлов по запросу с приоритизацией email-диалогов специалистов"""
        query_words = set(self._extract_words(query))
        
        node_scores = []
        
        for node in self.nodes.values():
            node_words = set(self._extract_words(node.content))
            
            # Базовое совпадение по словам
            common_words = query_words.intersection(node_words)
            base_score = len(common_words) / max(len(query_words), 1)
            
            if base_score > 0:
                # Приоритизация email-диалогов с ответами специалистов
                priority_multiplier = 1.0
                
                if node.source == "email":
                    # Повышаем приоритет для диалогов с ответами специалистов
                    priority_multiplier = 3.0
                
                # Финальный скор с приоритизацией
                final_score = base_score * priority_multiplier
                node_scores.append((node, final_score))
        
        # Сортируем по релевантности
        node_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [node for node, score in node_scores[:top_k]]
    
    def get_context_for_query(self, query: str, max_nodes: int = 10) -> Dict:
        """Получение контекста для запроса (email-диалоги без лишних полей)"""
        relevant_nodes = self.search_relevant_nodes(query, top_k=max_nodes * 2)
        context = {
            "email": [],
            "documentation": [],
            "code": []
        }
        for node in relevant_nodes:
            if node.source == "email":
                entry = {
                    "instruction": node.metadata.get("instruction", ""),
                    "response": node.metadata.get("response", ""),
                    "route_to": node.metadata.get("route_to", ""),
                    "line": node.metadata.get("line", None)
                }
                if "attachments" in node.metadata:
                    entry["attachments"] = node.metadata["attachments"]
                context["email"].append(entry)
            else:
                context[node.source].append({
                    "content": node.content,
                    "metadata": node.metadata
                })
        for key in context:
            context[key] = context[key][:max_nodes]
        return context
    
    def find_specialist_responses(self, query: str, max_results: int = 5) -> List[Dict]:
        """Поиск ответов по запросу (без response_type)"""
        query_words = set(self._extract_words(query))
        responses = []
        for node in self.nodes.values():
            if node.source == "email":
                node_words = set(self._extract_words(node.content))
                common_words = query_words.intersection(node_words)
                score = len(common_words) / max(len(query_words), 1)
                if score > 0.1:
                    entry = {
                        "instruction": node.metadata.get("instruction", ""),
                        "response": node.metadata.get("response", ""),
                        "route_to": node.metadata.get("route_to", ""),
                        "relevance_score": score,
                        "line": node.metadata.get("line", None)
                    }
                    if "attachments" in node.metadata:
                        entry["attachments"] = node.metadata["attachments"]
                    responses.append(entry)
        responses.sort(key=lambda x: x["relevance_score"], reverse=True)
        return responses[:max_results]
    
    def save_graph(self, output_path: str):
        """Сохранение графа в файл"""
        graph_data = {
            "nodes": {
                node_id: {
                    "content": node.content,
                    "source": node.source,
                    "metadata": node.metadata
                }
                for node_id, node in self.nodes.items()
            },
            "edges": [
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relationship_type": edge.relationship_type,
                    "weight": edge.weight
                }
                for edge in self.edges
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Knowledge graph saved to {output_path}")
    
    def load_graph(self, input_path: str):
        """Загрузка графа из файла"""
        with open(input_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # Загружаем узлы
        for node_id, node_data in graph_data["nodes"].items():
            node = KnowledgeNode(
                id=node_id,
                content=node_data["content"],
                source=node_data["source"],
                metadata=node_data["metadata"]
            )
            self.nodes[node_id] = node
            self.content_index[node.source].append(node_id)
            self.source_index[node.source].append(node_id)
        
        # Загружаем связи
        for edge_data in graph_data["edges"]:
            edge = KnowledgeEdge(
                source_id=edge_data["source_id"],
                target_id=edge_data["target_id"],
                relationship_type=edge_data["relationship_type"],
                weight=edge_data["weight"]
            )
            self.edges.append(edge)
        
        logger.info(f"Knowledge graph loaded from {input_path}")

def main():
    """Основная функция для тестирования системы"""
    # Создаем экземпляр Knowledge Graph
    kg = KnowledgeGraph()
    
    # Строим граф
    kg.build_graph()
    
    # Сохраняем граф
    kg.save_graph("knowledge_graph.json")
    
    # Тестируем поиск
    test_query = "Как настроить доступ к базе данных Oracle?"
    context = kg.get_context_for_query(test_query)
    
    print(f"Query: {test_query}")
    print(f"Found {len(context['documentation'])} documentation nodes")
    print(f"Found {len(context['code'])} code nodes")
    print(f"Found {len(context['email'])} email nodes")
    
    # Показываем примеры
    if context['documentation']:
        print(f"\nDocumentation example:\n{context['documentation'][0]['content'][:200]}...")
    
    # Тестируем специальный поиск ответов специалистов
    print(f"\n=== Поиск ответов специалистов ===")
    specialist_responses = kg.find_specialist_responses(test_query, max_results=3)
    
    for i, response in enumerate(specialist_responses, 1):
        print(f"\n{i}. Специалист: {response['route_to']}")
        print(f"   Релевантность: {response['relevance_score']:.2f}")
        print(f"   Вопрос: {response['instruction'][:100]}...")
        print(f"   Ответ: {response['response'][:150]}...")

if __name__ == "__main__":
    main() 