import logging
from typing import Annotated, Optional

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
from zep_cloud.client import AsyncZep

from config import settings
from auth import require_auth, check_user_access, AuthUser

router = APIRouter()
logger = logging.getLogger("kwami-api.memory")


# =============================================================================
# Pydantic Models for Ontology
# =============================================================================

class EntityTypeDefinition(BaseModel):
    """Definition of an entity type for the knowledge graph."""
    name: str
    description: str


class EdgeTypeDefinition(BaseModel):
    """Definition of an edge (relationship) type for the knowledge graph."""
    name: str
    description: str


class OntologySchema(BaseModel):
    """Full ontology schema with entity and edge types."""
    entity_types: list[EntityTypeDefinition]
    edge_types: list[EdgeTypeDefinition]


# Default ontology for Kwami agents
DEFAULT_ENTITY_TYPES = [
    {"name": "Preference", "description": "User preferences, choices, opinions, or selections."},
    {"name": "Procedure", "description": "Multi-step instructions or workflows."},
    {"name": "Person", "description": "People mentioned in conversation."},
    {"name": "Organization", "description": "Companies, institutions, teams, or groups."},
    {"name": "Location", "description": "Physical places, cities, countries, venues."},
    {"name": "Event", "description": "Scheduled or past events, meetings, appointments."},
    {"name": "Project", "description": "Work projects, personal initiatives, creative endeavors."},
    {"name": "Topic", "description": "Subjects of interest or discussion themes."},
    {"name": "Product", "description": "Products, services, software, or tools."},
    {"name": "Skill", "description": "User skills, expertise, or competencies."},
    {"name": "Goal", "description": "User goals, objectives, or aspirations."},
]

DEFAULT_EDGE_TYPES = [
    {"name": "KNOWS", "description": "The user knows a person."},
    {"name": "WORKS_AT", "description": "Employment relationship with an organization."},
    {"name": "LIVES_IN", "description": "The user's residence or location."},
    {"name": "INTERESTED_IN", "description": "Topics or things the user is interested in."},
    {"name": "WORKING_ON", "description": "Projects the user is actively working on."},
    {"name": "HAS_SKILL", "description": "Skills the user possesses."},
    {"name": "WANTS_TO", "description": "Goals or desires the user has expressed."},
    {"name": "ATTENDED", "description": "Events the user has attended or will attend."},
    {"name": "USES", "description": "Products or tools the user uses."},
    {"name": "PREFERS", "description": "Preferences the user has expressed."},
]


async def get_zep_client():
    if not settings.zep_api_key:
        raise HTTPException(status_code=503, detail="Memory service not configured (ZEP_API_KEY missing)")
    return AsyncZep(api_key=settings.zep_api_key)


def verify_user_access(user: AuthUser, user_id: str):
    """Verify the authenticated user has access to the requested user_id."""
    if not check_user_access(user, user_id):
        raise HTTPException(
            status_code=403,
            detail="Access denied: You can only access your own memory data"
        )

@router.get("/debug/{user_id}")
async def debug_user_memory(
    user_id: str,
    user: Annotated[AuthUser, Depends(require_auth)],
    client: AsyncZep = Depends(get_zep_client),
):
    """Debug endpoint to check what memory exists for a user."""
    verify_user_access(user, user_id)
    result = {
        "user_id": user_id, 
        "facts": [], 
        "graph_edges": [],
        "graph_nodes": 0,
        "threads": [],
        "errors": []
    }
    
    # Try to get facts via graph.search (Zep v3 method)
    try:
        facts_response = await client.graph.search(
            user_id=user_id,
            query="user information",
            scope="edges",
            limit=20,
        )
        if facts_response and facts_response.edges:
            result["facts"] = [edge.fact for edge in facts_response.edges if hasattr(edge, 'fact') and edge.fact]
            result["graph_edges"] = [
                {"fact": edge.fact, "name": edge.name if hasattr(edge, 'name') else None}
                for edge in facts_response.edges if hasattr(edge, 'fact')
            ]
    except Exception as e:
        result["errors"].append(f"graph.search: {str(e)}")
    
    # Try graph nodes API
    try:
        nodes = await client.graph.node.get_by_user_id(user_id=user_id, limit=10)
        result["graph_nodes"] = len(nodes) if nodes else 0
        if nodes:
            result["node_names"] = [n.name for n in nodes if hasattr(n, 'name')]
    except Exception as e:
        result["errors"].append(f"graph.node: {str(e)}")
    
    # List ALL threads to see what exists
    try:
        threads_response = await client.thread.list_all(page_size=50)
        all_threads = []
        if threads_response:
            threads_list = threads_response.threads if hasattr(threads_response, 'threads') else threads_response
            for t in (threads_list or []):
                thread_id = t.thread_id if hasattr(t, 'thread_id') else (t.uuid if hasattr(t, 'uuid') else str(t))
                thread_user = t.user_id if hasattr(t, 'user_id') else None
                all_threads.append({
                    "thread_id": thread_id,
                    "user_id": thread_user,
                })
        result["all_threads"] = all_threads  # Show ALL threads for debugging
        
        # Now filter to this user and get context
        for t_info in all_threads:
            if t_info["user_id"] == user_id or (t_info["thread_id"] and user_id in str(t_info["thread_id"])):
                try:
                    ctx = await client.thread.get_context(thread_id=t_info["thread_id"])
                    if ctx and ctx.context:
                        t_info["context"] = ctx.context[:500] + "..." if len(ctx.context) > 500 else ctx.context
                except Exception as ctx_err:
                    t_info["context_error"] = str(ctx_err)[:100]
                result["threads"].append(t_info)
    except Exception as e:
        result["errors"].append(f"thread.list_all: {str(e)}")
    
    return result


@router.get("/{user_id}/facts")
async def get_user_facts(
    user_id: str,
    user: Annotated[AuthUser, Depends(require_auth)],
    client: AsyncZep = Depends(get_zep_client),
):
    """Get all facts stored for a user via graph search (Zep v3)."""
    verify_user_access(user, user_id)
    try:
        # Zep v3 stores facts on graph edges - search for them
        facts_response = await client.graph.search(
            user_id=user_id,
            query="user information facts preferences",
            scope="edges",
            limit=50,
        )
        if facts_response and facts_response.edges:
            return [edge.fact for edge in facts_response.edges if hasattr(edge, 'fact') and edge.fact]
        return []
    except Exception as e:
        # Check for 404 (user not found)
        if "404" in str(e):
            return []
        logger.error(f"Failed to fetch facts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{user_id}")
async def delete_user_memory(
    user_id: str,
    user: Annotated[AuthUser, Depends(require_auth)],
    client: AsyncZep = Depends(get_zep_client),
):
    """Delete all memory for a user (user, threads, and graph data).
    
    This is a destructive operation that removes:
    - All threads/sessions associated with the user
    - The user's knowledge graph (nodes and edges)
    - The user record itself
    
    Use with caution - this cannot be undone.
    Requires authentication when auth is enabled.
    """
    verify_user_access(user, user_id)
    logger.info(f"🗑️ Deleting all memory for user: {user_id}")
    deleted = {"threads": 0, "user": False, "errors": []}
    
    try:
        # 1. Delete all threads belonging to this user
        try:
            threads_response = await client.thread.list_all(page_size=100)
            if threads_response:
                threads_list = threads_response.threads if hasattr(threads_response, 'threads') else threads_response
                for t in (threads_list or []):
                    thread_id = t.thread_id if hasattr(t, 'thread_id') else (t.uuid if hasattr(t, 'uuid') else None)
                    thread_user = t.user_id if hasattr(t, 'user_id') else None
                    
                    # Delete threads that belong to this user
                    if thread_user == user_id or (thread_id and user_id in str(thread_id)):
                        try:
                            await client.thread.delete(thread_id=thread_id)
                            deleted["threads"] += 1
                            logger.info(f"🗑️ Deleted thread: {thread_id}")
                        except Exception as e:
                            deleted["errors"].append(f"Failed to delete thread {thread_id}: {str(e)}")
        except Exception as e:
            deleted["errors"].append(f"Failed to list threads: {str(e)}")
        
        # 2. Delete the user (this also deletes associated graph data in Zep)
        try:
            await client.user.delete(user_id=user_id)
            deleted["user"] = True
            logger.info(f"🗑️ Deleted user: {user_id}")
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                deleted["errors"].append(f"User {user_id} not found")
            else:
                deleted["errors"].append(f"Failed to delete user: {error_msg}")
        
        logger.info(f"🗑️ Deletion complete: {deleted['threads']} threads, user={deleted['user']}")
        return {
            "success": deleted["user"] or deleted["threads"] > 0,
            "user_id": user_id,
            "deleted_threads": deleted["threads"],
            "deleted_user": deleted["user"],
            "errors": deleted["errors"] if deleted["errors"] else None,
        }
        
    except Exception as e:
        logger.error(f"Failed to delete user memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}/messages")
async def get_user_messages(
    user_id: str,
    user: Annotated[AuthUser, Depends(require_auth)],
    client: AsyncZep = Depends(get_zep_client),
    limit: int = 100,
):
    """Get conversation messages for a user from their threads/sessions.
    
    This retrieves actual conversation history stored in Zep threads,
    which is where chat messages are stored via memory.add().
    """
    verify_user_access(user, user_id)
    logger.info(f"💬 Fetching messages for user: {user_id}")
    try:
        messages = []
        sessions = []
        
        # Get all threads and find ones belonging to this user
        try:
            threads_response = await client.thread.list_all(page_size=50)
            if threads_response:
                threads_list = threads_response.threads if hasattr(threads_response, 'threads') else threads_response
                for t in (threads_list or []):
                    thread_id = t.thread_id if hasattr(t, 'thread_id') else (t.uuid if hasattr(t, 'uuid') else None)
                    thread_user = t.user_id if hasattr(t, 'user_id') else None
                    
                    # Check if thread belongs to this user
                    if thread_user == user_id or (thread_id and user_id in str(thread_id)):
                        sessions.append({
                            "thread_id": thread_id,
                            "user_id": thread_user,
                            "created_at": str(t.created_at) if hasattr(t, 'created_at') and t.created_at else None,
                        })
                        
                        # Get messages from this thread
                        try:
                            msgs_response = await client.thread.get_messages(
                                thread_id=thread_id,
                                limit=limit,
                            )
                            if msgs_response and msgs_response.messages:
                                for msg in msgs_response.messages:
                                    messages.append({
                                        "uuid": msg.uuid if hasattr(msg, 'uuid') else None,
                                        "content": msg.content if hasattr(msg, 'content') else None,
                                        "role": msg.role if hasattr(msg, 'role') else (msg.role_type if hasattr(msg, 'role_type') else None),
                                        "role_type": msg.role_type if hasattr(msg, 'role_type') else None,
                                        "created_at": str(msg.created_at) if hasattr(msg, 'created_at') and msg.created_at else None,
                                        "thread_id": thread_id,
                                    })
                        except Exception as msg_err:
                            logger.warning(f"💬 Failed to get messages from thread {thread_id}: {msg_err}")
        except Exception as e:
            logger.warning(f"💬 thread.list_all failed: {e}")
        
        # Sort messages by created_at (newest first)
        messages.sort(key=lambda x: x.get('created_at') or '', reverse=True)
        
        logger.info(f"💬 Found {len(messages)} messages across {len(sessions)} sessions")
        return {
            "messages": messages[:limit],  # Limit total messages
            "message_count": len(messages),
            "sessions": sessions,
            "session_count": len(sessions),
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}/edges")
async def get_user_edges(
    user_id: str,
    user: Annotated[AuthUser, Depends(require_auth)],
    client: AsyncZep = Depends(get_zep_client),
    limit: int = 50,
):
    """Get all edges (facts with temporal data) for a user.
    
    Edges represent facts/relationships in the knowledge graph. Each edge includes:
    - fact: The relationship description
    - valid_at: When the fact became true
    - invalid_at: When the fact became false (if superseded)
    - created_at: When Zep learned the fact
    - expired_at: When Zep learned the fact was no longer true
    """
    verify_user_access(user, user_id)
    logger.info(f"🔗 Fetching edges for user: {user_id}")
    try:
        edges = []
        
        # Get edges directly from the graph API
        try:
            edges_response = await client.graph.edge.get_by_user_id(user_id=user_id, limit=limit)
            if edges_response:
                for edge in edges_response:
                    edges.append({
                        "uuid": edge.uuid_ if hasattr(edge, 'uuid_') else (edge.uuid if hasattr(edge, 'uuid') else None),
                        "fact": edge.fact if hasattr(edge, 'fact') else None,
                        "name": edge.name if hasattr(edge, 'name') else None,
                        "source_node_uuid": edge.source_node_uuid if hasattr(edge, 'source_node_uuid') else None,
                        "target_node_uuid": edge.target_node_uuid if hasattr(edge, 'target_node_uuid') else None,
                        # Temporal data
                        "created_at": str(edge.created_at) if hasattr(edge, 'created_at') and edge.created_at else None,
                        "valid_at": str(edge.valid_at) if hasattr(edge, 'valid_at') and edge.valid_at else None,
                        "invalid_at": str(edge.invalid_at) if hasattr(edge, 'invalid_at') and edge.invalid_at else None,
                        "expired_at": str(edge.expired_at) if hasattr(edge, 'expired_at') and edge.expired_at else None,
                    })
        except Exception as e:
            logger.warning(f"🔗 graph.edge.get_by_user_id failed: {e}")
            # Fallback to graph.search
            try:
                facts_response = await client.graph.search(
                    user_id=user_id,
                    query="*",
                    scope="edges",
                    limit=limit,
                )
                if facts_response and facts_response.edges:
                    for edge in facts_response.edges:
                        edges.append({
                            "uuid": edge.uuid_ if hasattr(edge, 'uuid_') else (edge.uuid if hasattr(edge, 'uuid') else None),
                            "fact": edge.fact if hasattr(edge, 'fact') else None,
                            "name": edge.name if hasattr(edge, 'name') else None,
                            "source_node_uuid": edge.source_node_uuid if hasattr(edge, 'source_node_uuid') else None,
                            "target_node_uuid": edge.target_node_uuid if hasattr(edge, 'target_node_uuid') else None,
                            "created_at": str(edge.created_at) if hasattr(edge, 'created_at') and edge.created_at else None,
                            "valid_at": str(edge.valid_at) if hasattr(edge, 'valid_at') and edge.valid_at else None,
                            "invalid_at": str(edge.invalid_at) if hasattr(edge, 'invalid_at') and edge.invalid_at else None,
                            "expired_at": str(edge.expired_at) if hasattr(edge, 'expired_at') and edge.expired_at else None,
                        })
            except Exception as search_err:
                logger.warning(f"🔗 Fallback graph.search also failed: {search_err}")
        
        logger.info(f"🔗 Found {len(edges)} edges")
        return {"edges": edges, "count": len(edges)}
        
    except Exception as e:
        logger.error(f"Failed to fetch edges: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}/nodes")
async def get_user_nodes(
    user_id: str,
    user: Annotated[AuthUser, Depends(require_auth)],
    client: AsyncZep = Depends(get_zep_client),
    limit: int = 50,
):
    """Get all nodes (entities with summaries) for a user.
    
    Nodes represent entities in the knowledge graph. Each node includes:
    - name: The entity name
    - summary: AI-generated overview of the entity
    - labels: Type categorization
    """
    verify_user_access(user, user_id)
    logger.info(f"🔵 Fetching nodes for user: {user_id}")
    try:
        nodes = []
        
        try:
            nodes_response = await client.graph.node.get_by_user_id(user_id=user_id, limit=limit)
            if nodes_response:
                for node in nodes_response:
                    nodes.append({
                        "uuid": node.uuid_ if hasattr(node, 'uuid_') else (node.uuid if hasattr(node, 'uuid') else None),
                        "name": node.name if hasattr(node, 'name') else None,
                        "summary": node.summary if hasattr(node, 'summary') else None,
                        "labels": list(node.labels) if hasattr(node, 'labels') and node.labels else [],
                        "created_at": str(node.created_at) if hasattr(node, 'created_at') and node.created_at else None,
                    })
        except Exception as e:
            logger.warning(f"🔵 graph.node.get_by_user_id failed: {e}")
        
        logger.info(f"🔵 Found {len(nodes)} nodes")
        return {"nodes": nodes, "count": len(nodes)}
        
    except Exception as e:
        logger.error(f"Failed to fetch nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}/ontology")
async def get_user_ontology(
    user_id: str,
    user: Annotated[AuthUser, Depends(require_auth)],
    client: AsyncZep = Depends(get_zep_client),
):
    """Get the ontology (entity/edge type schema) for a user's knowledge graph.
    
    Returns the configured entity types and edge types that Zep uses
    for extracting information from conversations.
    """
    verify_user_access(user, user_id)
    logger.info(f"📋 Fetching ontology for user: {user_id}")
    
    try:
        ontology = await client.graph.get_ontology(user_id=user_id)
        
        if ontology:
            return {
                "user_id": user_id,
                "entity_types": [
                    {"name": e.name, "description": e.description}
                    for e in (ontology.entity_types or [])
                ],
                "edge_types": [
                    {"name": e.name, "description": e.description}
                    for e in (ontology.edge_types or [])
                ],
            }
        else:
            # Return defaults if no ontology configured
            return {
                "user_id": user_id,
                "entity_types": DEFAULT_ENTITY_TYPES,
                "edge_types": DEFAULT_EDGE_TYPES,
                "is_default": True,
            }
            
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg:
            # No ontology configured, return defaults
            return {
                "user_id": user_id,
                "entity_types": DEFAULT_ENTITY_TYPES,
                "edge_types": DEFAULT_EDGE_TYPES,
                "is_default": True,
            }
        logger.error(f"Failed to fetch ontology: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{user_id}/ontology")
async def set_user_ontology(
    user_id: str,
    ontology: OntologySchema,
    user: Annotated[AuthUser, Depends(require_auth)],
    client: AsyncZep = Depends(get_zep_client),
):
    """Set the ontology (entity/edge type schema) for a user's knowledge graph.
    
    This configures what types of entities and relationships Zep will
    extract from conversations. Changes apply to future extractions.
    """
    verify_user_access(user, user_id)
    logger.info(f"📋 Setting ontology for user: {user_id}")
    
    try:
        await client.graph.set_ontology(
            user_id=user_id,
            entity_types=[e.model_dump() for e in ontology.entity_types],
            edge_types=[e.model_dump() for e in ontology.edge_types],
        )
        
        logger.info(
            f"📋 Ontology set: {len(ontology.entity_types)} entity types, "
            f"{len(ontology.edge_types)} edge types"
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "entity_types_count": len(ontology.entity_types),
            "edge_types_count": len(ontology.edge_types),
        }
        
    except Exception as e:
        logger.error(f"Failed to set ontology: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{user_id}/ontology/reset")
async def reset_user_ontology(
    user_id: str,
    user: Annotated[AuthUser, Depends(require_auth)],
    client: AsyncZep = Depends(get_zep_client),
):
    """Reset the ontology to default Kwami entity/edge types."""
    verify_user_access(user, user_id)
    logger.info(f"📋 Resetting ontology to defaults for user: {user_id}")
    
    try:
        await client.graph.set_ontology(
            user_id=user_id,
            entity_types=DEFAULT_ENTITY_TYPES,
            edge_types=DEFAULT_EDGE_TYPES,
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "message": "Ontology reset to defaults",
            "entity_types_count": len(DEFAULT_ENTITY_TYPES),
            "edge_types_count": len(DEFAULT_EDGE_TYPES),
        }
        
    except Exception as e:
        logger.error(f"Failed to reset ontology: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}/search")
async def search_graph(
    user_id: str,
    user: Annotated[AuthUser, Depends(require_auth)],
    client: AsyncZep = Depends(get_zep_client),
    q: str = Query(..., description="Search query"),
    scope: str = Query("nodes", description="Search scope: 'nodes', 'edges', or 'both'"),
    entity_types: Optional[str] = Query(None, description="Comma-separated entity types to filter by"),
    limit: int = Query(20, ge=1, le=100),
):
    """Search the knowledge graph with optional entity type filtering.
    
    This allows precise queries like "find all Preference entities about food"
    or "find Person entities named John".
    """
    verify_user_access(user, user_id)
    logger.info(f"🔍 Searching graph for user {user_id}: query='{q}', scope={scope}, types={entity_types}")
    
    try:
        results = {"nodes": [], "edges": [], "query": q, "scope": scope}
        
        # Parse entity types filter
        node_labels = None
        if entity_types:
            node_labels = [t.strip() for t in entity_types.split(",") if t.strip()]
        
        # Search nodes
        if scope in ("nodes", "both"):
            try:
                search_kwargs = {
                    "user_id": user_id,
                    "query": q,
                    "scope": "nodes",
                    "limit": limit,
                }
                if node_labels:
                    search_kwargs["node_labels"] = node_labels
                    
                nodes_response = await client.graph.search(**search_kwargs)
                
                if nodes_response and nodes_response.nodes:
                    for node in nodes_response.nodes:
                        results["nodes"].append({
                            "name": getattr(node, 'name', ''),
                            "type": node.labels[0].lower() if hasattr(node, 'labels') and node.labels else 'entity',
                            "labels": list(node.labels) if hasattr(node, 'labels') and node.labels else [],
                            "summary": getattr(node, 'summary', ''),
                            "uuid": getattr(node, 'uuid_', None) or getattr(node, 'uuid', None),
                            "score": getattr(node, 'score', 0),
                        })
            except Exception as e:
                logger.warning(f"🔍 Node search failed: {e}")
        
        # Search edges
        if scope in ("edges", "both"):
            try:
                edges_response = await client.graph.search(
                    user_id=user_id,
                    query=q,
                    scope="edges",
                    limit=limit,
                )
                
                if edges_response and edges_response.edges:
                    for edge in edges_response.edges:
                        results["edges"].append({
                            "fact": getattr(edge, 'fact', ''),
                            "relation": getattr(edge, 'name', 'related_to'),
                            "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', None),
                            "score": getattr(edge, 'score', 0),
                        })
            except Exception as e:
                logger.warning(f"🔍 Edge search failed: {e}")
        
        results["node_count"] = len(results["nodes"])
        results["edge_count"] = len(results["edges"])
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to search graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{user_id}/entities/{entity_type}")
async def get_entities_by_type(
    user_id: str,
    entity_type: str,
    user: Annotated[AuthUser, Depends(require_auth)],
    client: AsyncZep = Depends(get_zep_client),
    limit: int = Query(50, ge=1, le=100),
):
    """Get all entities of a specific type from the knowledge graph.
    
    Examples:
    - GET /memory/{user_id}/entities/Preference - Get all user preferences
    - GET /memory/{user_id}/entities/Person - Get all people mentioned
    - GET /memory/{user_id}/entities/Location - Get all locations
    """
    verify_user_access(user, user_id)
    logger.info(f"🏷️ Fetching {entity_type} entities for user: {user_id}")
    
    try:
        entities = []
        
        # Get all nodes and filter by type
        nodes_response = await client.graph.node.get_by_user_id(
            user_id=user_id,
            limit=limit * 2,  # Get more to filter
        )
        
        if nodes_response:
            for node in nodes_response:
                node_labels = list(node.labels) if hasattr(node, 'labels') and node.labels else []
                # Check if the entity type matches (case-insensitive)
                if any(label.lower() == entity_type.lower() for label in node_labels):
                    entities.append({
                        "name": getattr(node, 'name', ''),
                        "type": node_labels[0] if node_labels else 'entity',
                        "labels": node_labels,
                        "summary": getattr(node, 'summary', ''),
                        "uuid": getattr(node, 'uuid_', None) or getattr(node, 'uuid', None),
                        "created_at": str(node.created_at) if hasattr(node, 'created_at') and node.created_at else None,
                    })
                    if len(entities) >= limit:
                        break
        
        return {
            "entity_type": entity_type,
            "entities": entities,
            "count": len(entities),
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch entities by type: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _infer_node_type(name: str, summary: str | None, labels: list[str]) -> str:
    """Infer node type from Zep labels or fall back to keyword-based inference.
    
    Zep's free tier often returns generic labels, so we use smart inference.
    """
    # First, check if Zep provided a meaningful label
    if labels:
        label = labels[0].lower()
        # If it's a specific type (not generic), use it
        if label not in ("entity", "node", "unknown", ""):
            return label
    
    # Fall back to keyword-based inference
    name_lower = name.lower()
    text = f"{name_lower} {(summary or '').lower()}"
    
    # User/Assistant detection (highest priority)
    if "kwami_" in name_lower or name_lower == "user" or "identifies_as" in text:
        return "user"
    if name_lower in ("assistant", "ai", "bot") or "assistant" in name_lower:
        return "assistant"
    
    # Person detection
    person_indicators = ["person", "he ", "she ", "they ", "friend", "family", 
                        "brother", "sister", "mother", "father", "wife", "husband",
                        "colleague", "boss", "manager"]
    if any(p in text for p in person_indicators):
        return "person"
    
    # Pet/Animal detection
    pet_indicators = ["dog", "cat", "pet", "puppy", "kitten", "bird", "fish",
                     "labrador", "retriever", "shepherd", "poodle", "bulldog"]
    if any(p in text for p in pet_indicators):
        return "pet"
    
    # Location detection
    location_indicators = ["city", "country", "location", "lives in", "from", "born in",
                          "street", "address", "neighborhood", "district", "region",
                          "barcelona", "madrid", "london", "paris", "new york", "tokyo"]
    if any(loc in text for loc in location_indicators):
        return "location"
    
    # Place/Venue detection (more specific than location)
    place_indicators = ["park", "home", "house", "apartment", "office", "restaurant",
                       "café", "cafe", "bar", "gym", "school", "university", "hospital",
                       "store", "shop", "mall", "airport", "station"]
    if any(p in text for p in place_indicators):
        return "place"
    
    # Preference detection
    preference_indicators = ["likes", "loves", "enjoys", "prefers", "favorite", 
                            "favourite", "preference", "interested in", "passionate"]
    if any(p in text for p in preference_indicators):
        return "preference"
    
    # Skill/Profession detection
    skill_indicators = ["developer", "engineer", "designer", "artist", "musician",
                       "programmer", "software", "works as", "profession", "job",
                       "skill", "expertise", "experience in"]
    if any(s in text for s in skill_indicators):
        return "skill"
    
    # Topic/Interest detection
    topic_indicators = ["music", "sports", "art", "technology", "science", "cooking",
                       "gaming", "reading", "travel", "photography", "genre"]
    if any(t in text for t in topic_indicators):
        return "topic"
    
    # Event detection
    event_indicators = ["event", "meeting", "appointment", "birthday", "anniversary",
                       "conference", "party", "wedding", "trip", "vacation"]
    if any(e in text for e in event_indicators):
        return "event"
    
    # Project detection
    project_indicators = ["project", "working on", "building", "developing", "creating"]
    if any(p in text for p in project_indicators):
        return "project"
    
    # Product detection
    product_indicators = ["product", "app", "application", "tool", "service", "device"]
    if any(p in text for p in product_indicators):
        return "product"
    
    # Organization detection
    org_indicators = ["company", "organization", "team", "group", "corporation",
                     "startup", "business", "firm", "agency"]
    if any(o in text for o in org_indicators):
        return "organization"
    
    # Attribute/Property detection (colors, ages, etc.)
    if any(c in text for c in ["color", "colour", "brown", "black", "white", "red", "blue"]):
        return "attribute"
    if any(a in text for a in ["years old", "age", "height", "weight"]):
        return "attribute"
    
    # Default
    return "entity"


@router.get("/{user_id}/graph")
async def get_memory_graph(
    user_id: str,
    user: Annotated[AuthUser, Depends(require_auth)],
    client: AsyncZep = Depends(get_zep_client),
    limit: int = 50,
):
    """Get the knowledge graph representation of the user's memory from Zep.
    
    Uses Zep entity labels when available, with smart keyword inference as fallback.
    """
    verify_user_access(user, user_id)
    logger.info(f"📊 Fetching memory graph for user: {user_id}")
    try:
        nodes = []
        edges = []
        graph_edges_raw = []  # Store raw edges for relationship building
        
        # 1. Get edges via graph API (for relationships)
        try:
            edges_response = await client.graph.edge.get_by_user_id(user_id=user_id, limit=limit)
            if edges_response:
                logger.info(f"📊 Got {len(edges_response)} edges from graph.edge")
                for edge in edges_response:
                    edge_data = {
                        "fact": getattr(edge, 'fact', None),
                        "relation": getattr(edge, 'name', 'related_to'),
                        "source_node": getattr(edge, 'source_node_uuid', None),
                        "target_node": getattr(edge, 'target_node_uuid', None),
                    }
                    graph_edges_raw.append(edge_data)
        except Exception as e:
            logger.warning(f"📊 graph.edge failed, trying search: {e}")
            # Fallback to graph.search
            try:
                facts_response = await client.graph.search(
                    user_id=user_id,
                    query="*",
                    scope="edges",
                    limit=limit,
                )
                if facts_response and facts_response.edges:
                    for edge in facts_response.edges:
                        edge_data = {
                            "fact": getattr(edge, 'fact', None),
                            "relation": getattr(edge, 'name', 'related_to'),
                            "source_node": getattr(edge, 'source_node_uuid', None),
                            "target_node": getattr(edge, 'target_node_uuid', None),
                        }
                        graph_edges_raw.append(edge_data)
            except Exception as search_e:
                logger.warning(f"📊 graph.search edges also failed: {search_e}")
        
        # 2. Get nodes (entities) from graph.node API
        entity_nodes = []
        user_node_uuid = None  # Track the user/kwami node
        
        try:
            nodes_response = await client.graph.node.get_by_user_id(user_id=user_id, limit=limit)
            if nodes_response:
                logger.info(f"📊 Got {len(nodes_response)} nodes from graph.node")
                for node in nodes_response:
                    node_name = getattr(node, 'name', 'Unknown')
                    node_labels = list(node.labels) if hasattr(node, 'labels') and node.labels else []
                    node_uuid = getattr(node, 'uuid_', None) or getattr(node, 'uuid', None)
                    created_at = node.created_at if hasattr(node, 'created_at') else None
                    node_summary = getattr(node, 'summary', None)
                    
                    # Infer type using labels + keyword analysis
                    node_type = _infer_node_type(node_name, node_summary, node_labels)
                    
                    # Detect the user/kwami node
                    if node_type == "user" or "kwami_" in node_name.lower():
                        user_node_uuid = node_uuid
                        node_type = "user"
                    
                    entity_nodes.append({
                        "name": node_name,
                        "type": node_type,
                        "summary": node_summary,
                        "uuid": node_uuid,
                        "created_at": created_at,
                        "labels": node_labels,
                    })
        except Exception as e:
            logger.warning(f"📊 graph.node failed: {e}")
        
        # 3. Build the visualization graph
        # Map UUIDs to node IDs for edge building
        uuid_to_id = {}
        user_node_id = None
        
        # Add entity nodes with inferred types
        for i, entity in enumerate(entity_nodes):
            node_id = f"entity_{i}"
            if entity.get("uuid"):
                uuid_to_id[entity["uuid"]] = node_id
            
            # Track user node for potential edge connections
            if entity["type"] == "user":
                user_node_id = node_id
            
            # Shorten display label for user nodes
            label = entity["name"]
            if entity["type"] == "user" and len(label) > 20:
                label = "User"  # Cleaner display
            
            nodes.append({
                "id": node_id,
                "label": label,
                "type": entity["type"],
                "summary": entity["summary"],
                "uuid": entity["uuid"],
                "created_at": entity["created_at"],
                "labels": entity["labels"],
                "val": 25 if entity["type"] == "user" else 15
            })
        
        # 4. Build edges from raw graph edges (actual relationships)
        edges_added = set()
        connected_nodes = set()  # Track which nodes are connected
        
        for raw_edge in graph_edges_raw:
            source_uuid = raw_edge.get("source_node")
            target_uuid = raw_edge.get("target_node")
            relation = raw_edge.get("relation", "related_to")
            
            source_id = uuid_to_id.get(source_uuid)
            target_id = uuid_to_id.get(target_uuid)
            
            if source_id and target_id and source_id != target_id:
                edge_key = f"{source_id}-{target_id}-{relation}"
                if edge_key not in edges_added:
                    edges.append({
                        "source": source_id,
                        "target": target_id,
                        "relation": relation
                    })
                    edges_added.add(edge_key)
                    connected_nodes.add(source_id)
                    connected_nodes.add(target_id)
        
        # 5. Connect orphan nodes to user node if we have one
        if user_node_id:
            for node in nodes:
                if node["id"] != user_node_id and node["id"] not in connected_nodes:
                    edges.append({
                        "source": user_node_id,
                        "target": node["id"],
                        "relation": "related_to"
                    })
        
        logger.info(f"📊 Final graph: {len(nodes)} nodes, {len(edges)} edges")
        return {"nodes": nodes, "edges": edges}
        
    except Exception as e:
        logger.error(f"Failed to fetch memory graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


