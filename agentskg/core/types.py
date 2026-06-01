from typing import Any, Dict, List

from pydantic import BaseModel


class Entity(BaseModel):
    """Knowledge graph entity"""

    id: str
    name: str
    type: str
    attributes: Dict[str, Any] = {}


class Relation(BaseModel):
    """Knowledge graph relation"""

    id: str
    source: str
    target: str
    type: str
    attributes: Dict[str, Any] = {}


class KnowledgeGraphSchema(BaseModel):
    """Knowledge graph schema definition"""

    entity_types: List[str]
    relation_types: List[str]
    constraints: Dict[str, Any] = {}

class Triplet(BaseModel):
    """Knowledge graph triplet definition"""
    subject: str
    predicate: str
    object: str
    document_id: int

    def __hash__(self):
        return hash((self.subject, self.predicate, self.object))

    def __eq__(self, other):
        if not isinstance(other, Triplet):
            return False
        return (self.subject == other.subject and 
                self.predicate == other.predicate and 
                self.object == other.object)