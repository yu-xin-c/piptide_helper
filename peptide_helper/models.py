from pydantic import BaseModel

class PhysChemResult(BaseModel):
    """理化性质返回结果模型"""
    mw: float
    pi: float

class ToxicityResult(BaseModel):
    """毒性预测返回结果模型"""
    is_toxic: bool
    score: float

class ActivityResult(BaseModel):
    """活性评估返回结果模型"""
    antimicrobial_score: float

class StructureResult(BaseModel):
    """3D结构预测返回结果模型"""
    pdb_id: str
    confidence: float
