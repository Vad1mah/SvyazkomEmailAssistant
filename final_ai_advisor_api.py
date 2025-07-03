#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Финальный API для ИИ-советника по техническим запросам клиентов
Объединяет Knowledge Graph и систему генерации ответов
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import logging
import json
from pathlib import Path
from datetime import datetime
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from knowledge_graph_system import KnowledgeGraph
from final_response_generator import ResponseGenerator, ResponseAPI

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем FastAPI приложение
app = FastAPI(
    title="ИИ-советник по техническим запросам",
    description="API для генерации ответов на основе Knowledge Graph",
    version="1.0.0"
)

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные экземпляры
knowledge_graph = None
response_api = None

# Pydantic модели
class AdvisorRequest(BaseModel):
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context_preference: Optional[str] = "balanced"  # "documentation", "code", "email", "balanced"

class AdvisorResponse(BaseModel):
    query: str
    clarify: str
    solution: str
    routeTo: str
    sources: List[str]
    confidence: float
    route_to: Optional[str]
    context_tokens: int
    processing_time: float
    timestamp: str

class SystemStatus(BaseModel):
    status: str
    knowledge_graph_nodes: int
    knowledge_graph_edges: int
    model_status: str
    last_update: str

class QueryHistory(BaseModel):
    query: str
    answer: str
    timestamp: str
    sources: List[str]

# История запросов (в памяти, в продакшене нужно использовать БД)
query_history = []

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    global knowledge_graph, response_api
    
    try:
        # Инициализируем Knowledge Graph
        knowledge_graph = KnowledgeGraph()
        
        # Проверяем существование файла графа
        if Path("knowledge_graph.json").exists():
            knowledge_graph.load_graph("knowledge_graph.json")
            logger.info("Knowledge Graph loaded from file")
        else:
            knowledge_graph.build_graph()
            knowledge_graph.save_graph("knowledge_graph.json")
            logger.info("Knowledge Graph built and saved")
        
        # Инициализируем Response API
        response_api = ResponseAPI("knowledge_graph.json")
        
        logger.info("AI Advisor initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize AI Advisor: {e}")
        raise

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "ИИ-советник по техническим запросам клиентов",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "ask": "/ask - Задать вопрос",
            "status": "/status - Статус системы",
            "history": "/history - История запросов",
            "rebuild": "/rebuild - Пересборка графа знаний"
        }
    }

@app.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Получение статуса системы"""
    if knowledge_graph is None or response_api is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Проверяем статус модели
    model_status = "unknown"
    try:
        # Простой тест модели
        test_response = response_api.get_response("test")
        model_status = "ready"
    except Exception as e:
        model_status = f"error: {str(e)}"
    
    return SystemStatus(
        status="running",
        knowledge_graph_nodes=len(knowledge_graph.nodes),
        knowledge_graph_edges=len(knowledge_graph.edges),
        model_status=model_status,
        last_update=datetime.now().isoformat()
    )

@app.post("/ask", response_model=AdvisorResponse)
async def ask_question(request: AdvisorRequest):
    """Основной эндпоинт для задавания вопросов"""
    if response_api is None:
        raise HTTPException(status_code=503, detail="Response API not initialized")
    
    import time
    start_time = time.time()
    
    try:
        # Получаем ответ
        response = response_api.get_response(request.query)
        
        processing_time = time.time() - start_time
        
        # Добавляем в историю
        history_entry = QueryHistory(
            query=request.query,
            answer=f"Clarify: {response['clarify']}\nSolution: {response['solution']}\nRouteTo: {response['routeTo']}",
            timestamp=datetime.now().isoformat(),
            sources=response['sources']
        )
        query_history.append(history_entry)
        
        # Ограничиваем историю
        if len(query_history) > 1000:
            query_history.pop(0)
        
        # Обеспечиваем, что clarify, solution и routeTo - строки
        clarify = response['clarify']
        if isinstance(clarify, list):
            clarify = ' '.join(clarify) if clarify else ""
        elif not isinstance(clarify, str):
            clarify = str(clarify) if clarify else ""
        
        solution = response['solution']
        if isinstance(solution, list):
            solution = ' '.join(solution) if solution else ""
        elif not isinstance(solution, str):
            solution = str(solution) if solution else ""
        
        routeTo = response['routeTo']
        if isinstance(routeTo, list):
            routeTo = ' '.join(routeTo) if routeTo else ""
        elif not isinstance(routeTo, str):
            routeTo = str(routeTo) if routeTo else ""
        
        return AdvisorResponse(
            query=request.query,
            clarify=clarify,
            solution=solution,
            routeTo=routeTo,
            sources=response['sources'],
            confidence=response['confidence'],
            route_to=response['route_to'],
            context_tokens=response['context_tokens'],
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Question processing error: {str(e)}")

@app.get("/history", response_model=List[QueryHistory])
async def get_query_history(limit: int = 50, user_id: Optional[str] = None):
    """Получение истории запросов"""
    # В реальной системе здесь была бы фильтрация по user_id
    return query_history[-limit:] if limit > 0 else query_history

@app.post("/rebuild")
async def rebuild_knowledge_graph(background_tasks: BackgroundTasks):
    """Пересборка графа знаний в фоновом режиме"""
    if knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Knowledge Graph not initialized")
    
    def rebuild_task():
        try:
            # Очищаем текущий граф
            knowledge_graph.nodes.clear()
            knowledge_graph.edges.clear()
            knowledge_graph.content_index.clear()
            knowledge_graph.source_index.clear()
            
            # Пересобираем граф
            knowledge_graph.build_graph()
            knowledge_graph.save_graph("knowledge_graph.json")
            
            # Переинициализируем Response API
            global response_api
            response_api = ResponseAPI("knowledge_graph.json")
            
            logger.info("Knowledge Graph rebuilt successfully")
            
        except Exception as e:
            logger.error(f"Error rebuilding Knowledge Graph: {e}")
    
    # Запускаем пересборку в фоне
    background_tasks.add_task(rebuild_task)
    
    return {
        "message": "Knowledge Graph rebuild started in background",
        "status": "processing"
    }

@app.get("/search")
async def search_knowledge(query: str, source: Optional[str] = None, limit: int = 10):
    """Поиск в графе знаний"""
    if knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Knowledge Graph not initialized")
    
    try:
        relevant_nodes = knowledge_graph.search_relevant_nodes(query, top_k=limit)
        
        results = []
        for node in relevant_nodes:
            if source and node.source != source:
                continue
                
            results.append({
                "id": node.id,
                "source": node.source,
                "content": node.content[:500] + "..." if len(node.content) > 500 else node.content,
                "metadata": node.metadata
            })
        
        return {
            "query": query,
            "results": results,
            "total_found": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching knowledge: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/stats")
async def get_detailed_stats():
    """Получение детальной статистики"""
    if knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Knowledge Graph not initialized")
    
    # Статистика по источникам
    source_stats = {}
    for node in knowledge_graph.nodes.values():
        source = node.source
        if source not in source_stats:
            source_stats[source] = 0
        source_stats[source] += 1
    
    # Статистика по типам связей
    edge_stats = {}
    for edge in knowledge_graph.edges:
        edge_type = edge.relationship_type
        if edge_type not in edge_stats:
            edge_stats[edge_type] = 0
        edge_stats[edge_type] += 1
    
    return {
        "total_nodes": len(knowledge_graph.nodes),
        "total_edges": len(knowledge_graph.edges),
        "nodes_by_source": source_stats,
        "edges_by_type": edge_stats,
        "query_history_count": len(query_history),
        "last_query_time": query_history[-1].timestamp if query_history else None
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья системы"""
    if knowledge_graph is None or response_api is None:
        return {"status": "unhealthy", "reason": "Components not initialized"}
    
    try:
        # Проверяем базовую функциональность
        test_response = response_api.get_response("test")
        return {
            "status": "healthy",
            "knowledge_graph": "ready",
            "response_generator": "ready",
            "nodes_count": len(knowledge_graph.nodes),
            "edges_count": len(knowledge_graph.edges)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "reason": str(e),
            "knowledge_graph": "ready" if knowledge_graph else "not ready",
            "response_generator": "error"
        }

# Дополнительные эндпоинты для мониторинга

@app.get("/metrics")
async def get_metrics():
    """Получение метрик системы"""
    return {
        "knowledge_graph_nodes": len(knowledge_graph.nodes) if knowledge_graph else 0,
        "knowledge_graph_edges": len(knowledge_graph.edges) if knowledge_graph else 0,
        "query_history_size": len(query_history),
        "uptime": "TODO: implement uptime tracking",
        "average_response_time": "TODO: implement response time tracking"
    }

@app.get("/config")
async def get_config():
    """Получение конфигурации системы"""
    return {
        "max_context_tokens": 4000,
        "model": "phi4",
        "temperature": 0.7,
        "max_response_tokens": 1000,
        "knowledge_sources": [
            "documentation",
            "code",
            "email"
        ]
    }

@app.post("/feedback")
async def submit_feedback(query: str, answer: str, rating: int, feedback: Optional[str] = None):
    """Отправка обратной связи по ответам"""
    # В реальной системе здесь была бы запись в БД
    logger.info(f"Feedback received: query='{query}', rating={rating}, feedback='{feedback}'")
    
    return {
        "message": "Feedback received",
        "status": "success"
    }

@app.get("/chat")
async def chat_page():
    """Отдаёт чат-интерфейс"""
    return FileResponse("chat_frontend.html", media_type="text/html")

if __name__ == "__main__":
    uvicorn.run(
        "final_ai_advisor_api:app",
        host="0.0.0.0",
        port=8001,  # Используем другой порт
        reload=True,
        log_level="info"
    ) 