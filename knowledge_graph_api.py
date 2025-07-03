#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI интерфейс для системы Knowledge Graph
Предоставляет API для работы с графом знаний и генерации ответов
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import logging
from knowledge_graph_system import KnowledgeGraph

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем FastAPI приложение
app = FastAPI(
    title="Knowledge Graph API",
    description="API для работы с графом знаний технической поддержки",
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

# Глобальный экземпляр Knowledge Graph
knowledge_graph = None

# Pydantic модели для API
class QueryRequest(BaseModel):
    query: str
    max_nodes: int = 10
    include_metadata: bool = True

class QueryResponse(BaseModel):
    query: str
    context: Dict
    total_nodes: int
    processing_time: float

class GraphInfo(BaseModel):
    total_nodes: int
    total_edges: int
    nodes_by_source: Dict[str, int]
    edges_by_type: Dict[str, int]

class NodeInfo(BaseModel):
    id: str
    source: str
    content_preview: str
    metadata: Dict

class EdgeInfo(BaseModel):
    source_id: str
    target_id: str
    relationship_type: str
    weight: float

class GraphStats(BaseModel):
    nodes: List[NodeInfo]
    edges: List[EdgeInfo]

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске"""
    global knowledge_graph
    try:
        knowledge_graph = KnowledgeGraph()
        knowledge_graph.build_graph()
        logger.info("Knowledge Graph initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Knowledge Graph: {e}")
        raise

@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Knowledge Graph API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья системы"""
    if knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Knowledge Graph not initialized")
    
    return {
        "status": "healthy",
        "nodes_count": len(knowledge_graph.nodes),
        "edges_count": len(knowledge_graph.edges)
    }

@app.get("/info", response_model=GraphInfo)
async def get_graph_info():
    """Получение информации о графе"""
    if knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Knowledge Graph not initialized")
    
    # Подсчитываем узлы по источникам
    nodes_by_source = {}
    for node in knowledge_graph.nodes.values():
        source = node.source
        nodes_by_source[source] = nodes_by_source.get(source, 0) + 1
    
    # Подсчитываем связи по типам
    edges_by_type = {}
    for edge in knowledge_graph.edges:
        edge_type = edge.relationship_type
        edges_by_type[edge_type] = edges_by_type.get(edge_type, 0) + 1
    
    return GraphInfo(
        total_nodes=len(knowledge_graph.nodes),
        total_edges=len(knowledge_graph.edges),
        nodes_by_source=nodes_by_source,
        edges_by_type=edges_by_type
    )

@app.post("/query", response_model=QueryResponse)
async def query_knowledge_graph(request: QueryRequest):
    """Поиск релевантной информации в графе знаний"""
    if knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Knowledge Graph not initialized")
    
    import time
    start_time = time.time()
    
    try:
        # Получаем контекст для запроса
        context = knowledge_graph.get_context_for_query(
            request.query, 
            max_nodes=request.max_nodes
        )
        
        processing_time = time.time() - start_time
        
        # Подсчитываем общее количество узлов
        total_nodes = sum(len(nodes) for nodes in context.values())
        
        return QueryResponse(
            query=request.query,
            context=context,
            total_nodes=total_nodes,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing error: {str(e)}")

@app.get("/search/{query}")
async def search_nodes(query: str, top_k: int = 5):
    """Поиск узлов по запросу"""
    if knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Knowledge Graph not initialized")
    
    try:
        relevant_nodes = knowledge_graph.search_relevant_nodes(query, top_k=top_k)
        
        results = []
        for node in relevant_nodes:
            results.append({
                "id": node.id,
                "source": node.source,
                "content": node.content[:500] + "..." if len(node.content) > 500 else node.content,
                "metadata": node.metadata,
                "score": 0.0  # TODO: добавить подсчет score
            })
        
        return {
            "query": query,
            "results": results,
            "total_found": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching nodes: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.get("/nodes", response_model=List[NodeInfo])
async def get_nodes(source: Optional[str] = None, limit: int = 100):
    """Получение списка узлов"""
    if knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Knowledge Graph not initialized")
    
    nodes = []
    count = 0
    
    for node in knowledge_graph.nodes.values():
        if source and node.source != source:
            continue
        
        if count >= limit:
            break
        
        nodes.append(NodeInfo(
            id=node.id,
            source=node.source,
            content_preview=node.content[:200] + "..." if len(node.content) > 200 else node.content,
            metadata=node.metadata
        ))
        count += 1
    
    return nodes

@app.get("/nodes/{node_id}")
async def get_node(node_id: str):
    """Получение конкретного узла по ID"""
    if knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Knowledge Graph not initialized")
    
    if node_id not in knowledge_graph.nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node = knowledge_graph.nodes[node_id]
    
    # Находим связанные узлы
    related_nodes = []
    for edge in knowledge_graph.edges:
        if edge.source_id == node_id:
            if edge.target_id in knowledge_graph.nodes:
                target_node = knowledge_graph.nodes[edge.target_id]
                related_nodes.append({
                    "id": target_node.id,
                    "source": target_node.source,
                    "relationship": edge.relationship_type,
                    "weight": edge.weight
                })
        elif edge.target_id == node_id:
            if edge.source_id in knowledge_graph.nodes:
                source_node = knowledge_graph.nodes[edge.source_id]
                related_nodes.append({
                    "id": source_node.id,
                    "source": source_node.source,
                    "relationship": edge.relationship_type,
                    "weight": edge.weight
                })
    
    return {
        "id": node.id,
        "source": node.source,
        "content": node.content,
        "metadata": node.metadata,
        "related_nodes": related_nodes
    }

@app.get("/edges", response_model=List[EdgeInfo])
async def get_edges(relationship_type: Optional[str] = None, limit: int = 100):
    """Получение списка связей"""
    if knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Knowledge Graph not initialized")
    
    edges = []
    count = 0
    
    for edge in knowledge_graph.edges:
        if relationship_type and edge.relationship_type != relationship_type:
            continue
        
        if count >= limit:
            break
        
        edges.append(EdgeInfo(
            source_id=edge.source_id,
            target_id=edge.target_id,
            relationship_type=edge.relationship_type,
            weight=edge.weight
        ))
        count += 1
    
    return edges

@app.post("/rebuild")
async def rebuild_graph():
    """Пересборка графа знаний"""
    if knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Knowledge Graph not initialized")
    
    try:
        # Очищаем текущий граф
        knowledge_graph.nodes.clear()
        knowledge_graph.edges.clear()
        knowledge_graph.content_index.clear()
        knowledge_graph.source_index.clear()
        
        # Пересобираем граф
        knowledge_graph.build_graph()
        
        return {
            "message": "Knowledge Graph rebuilt successfully",
            "nodes_count": len(knowledge_graph.nodes),
            "edges_count": len(knowledge_graph.edges)
        }
        
    except Exception as e:
        logger.error(f"Error rebuilding graph: {e}")
        raise HTTPException(status_code=500, detail=f"Rebuild error: {str(e)}")

@app.get("/stats", response_model=GraphStats)
async def get_detailed_stats():
    """Получение детальной статистики графа"""
    if knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Knowledge Graph not initialized")
    
    # Получаем первые 10 узлов каждого типа
    nodes_by_source = {}
    for node in knowledge_graph.nodes.values():
        source = node.source
        if source not in nodes_by_source:
            nodes_by_source[source] = []
        if len(nodes_by_source[source]) < 10:
            nodes_by_source[source].append(NodeInfo(
                id=node.id,
                source=node.source,
                content_preview=node.content[:200] + "..." if len(node.content) > 200 else node.content,
                metadata=node.metadata
            ))
    
    # Получаем первые 20 связей
    edges = []
    for edge in knowledge_graph.edges[:20]:
        edges.append(EdgeInfo(
            source_id=edge.source_id,
            target_id=edge.target_id,
            relationship_type=edge.relationship_type,
            weight=edge.weight
        ))
    
    return GraphStats(
        nodes=[node for nodes in nodes_by_source.values() for node in nodes],
        edges=edges
    )

# Дополнительные эндпоинты для работы с источниками знаний

@app.get("/sources/documentation")
async def get_documentation_nodes(limit: int = 50):
    """Получение узлов документации"""
    return await get_nodes(source="documentation", limit=limit)

@app.get("/sources/code")
async def get_code_nodes(limit: int = 50):
    """Получение узлов кода"""
    return await get_nodes(source="code", limit=limit)

@app.get("/sources/email")
async def get_email_nodes(limit: int = 50):
    """Получение узлов email"""
    return await get_nodes(source="email", limit=limit)

@app.get("/relationships/{relationship_type}")
async def get_relationships(relationship_type: str, limit: int = 50):
    """Получение связей определенного типа"""
    return await get_edges(relationship_type=relationship_type, limit=limit)

if __name__ == "__main__":
    uvicorn.run(
        "knowledge_graph_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 