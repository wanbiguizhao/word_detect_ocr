from pydantic import BaseModel
from typing import List, Optional


class Line(BaseModel):
    pos: int
    color: str


class AnnotationSubmit(BaseModel):
    lines: List[Line]


class BatchLabelItem(BaseModel):
    char: str
    charIndex: int


class BatchLabelSave(BaseModel):
    clusterId: int
    labels: List[BatchLabelItem]


class ClusterLabelSave(BaseModel):
    clusterId: int
    alias: Optional[str] = None
    char: Optional[str] = None
    charIndex: Optional[int] = None


class RecommendMode(BaseModel):
    mode: str = "global"