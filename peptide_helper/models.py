from pydantic import BaseModel, Field

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


class PdbAtomRecord(BaseModel):
    """PDB ATOM 坐标记录"""
    serial: int
    atom_name: str
    residue_name: str
    chain_id: str
    residue_number: int
    x: float
    y: float
    z: float
    occupancy: float
    b_factor: float
    element: str = ""


class ResidueConfidence(BaseModel):
    """逐残基结构置信度"""
    position: int
    residue: str
    chain_id: str
    plddt: float


class ChainStructureSummary(BaseModel):
    """单条链的结构摘要"""
    chain_id: str
    residue_count: int
    atom_count: int
    mean_plddt: float
    min_plddt: float
    max_plddt: float


class CoordinateBounds(BaseModel):
    """三维坐标边界"""
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float


class LowConfidenceRegion(BaseModel):
    """连续低置信残基区间"""
    chain_id: str
    start: int
    end: int
    residue_count: int
    mean_plddt: float


class StructureResult(BaseModel):
    """3D结构预测返回结果模型"""
    pdb_id: str
    confidence: float
    source: str = "ESM Atlas"
    content_type: str = ""
    sequence_length: int = 0
    pdb_line_count: int = 0
    pdb_char_count: int = 0
    record_counts: dict[str, int] = Field(default_factory=dict)
    header: str = ""
    title: str = ""
    remarks: list[str] = Field(default_factory=list)
    atom_count: int = 0
    residue_count: int = 0
    chain_ids: list[str] = Field(default_factory=list)
    chain_summaries: list[ChainStructureSummary] = Field(default_factory=list)
    coordinate_bounds: CoordinateBounds | None = None
    min_plddt: float = 0.0
    max_plddt: float = 0.0
    high_confidence_residue_count: int = 0
    medium_confidence_residue_count: int = 0
    low_confidence_residue_count: int = 0
    very_low_confidence_residue_count: int = 0
    per_residue_plddt: list[ResidueConfidence] = Field(default_factory=list)
    low_confidence_residues: list[ResidueConfidence] = Field(default_factory=list)
    low_confidence_regions: list[LowConfidenceRegion] = Field(default_factory=list)
    atom_records: list[PdbAtomRecord] = Field(default_factory=list)
    structure_quality: str = ""
    pdb_preview: str = ""
    error: str = ""
