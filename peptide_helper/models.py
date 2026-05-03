from typing import Optional

from pydantic import BaseModel, Field


class ModelEvidence(BaseModel):
    """单个模型的标准化证据。"""

    name: str
    task: str
    label: str = "unknown"
    score: float | None = None
    weight: float = 1.0
    status: str = "available"
    source: str = ""
    error: str = ""


class PhysChemResult(BaseModel):
    """理化性质返回结果模型"""
    mw: float
    pi: float

class ToxicityResult(BaseModel):
    """毒性预测返回结果模型"""
    is_toxic: bool
    score: float
    consensus_level: str = "insufficient_models"
    evidence_summary: str = ""
    model_results: list[ModelEvidence] = Field(default_factory=list)

class ActivityResult(BaseModel):
    """活性评估返回结果模型"""
    antimicrobial_score: float
    consensus_level: str = "insufficient_models"
    evidence_summary: str = ""
    model_results: list[ModelEvidence] = Field(default_factory=list)


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
    coordinate_bounds: Optional[CoordinateBounds] = None
    min_plddt: float = 0.0
    max_plddt: float = 0.0
    high_confidence_residue_count: int = 0
    medium_confidence_residue_count: int = 0
    low_confidence_residue_count: int = 0
    very_low_confidence_residue_count: int = 0
    confident_fraction: float = 0.0
    per_residue_plddt: list[ResidueConfidence] = Field(default_factory=list)
    low_confidence_residues: list[ResidueConfidence] = Field(default_factory=list)
    low_confidence_regions: list[LowConfidenceRegion] = Field(default_factory=list)
    atom_records: list[PdbAtomRecord] = Field(default_factory=list)
    structure_quality: str = ""
    pdb_preview: str = ""
    error: str = ""

    def analysis_summary(self) -> dict:
        """面向后续 Agent 的结构摘要，避免把全量坐标塞进 LLM 上下文。"""

        return {
            "pdb_id": self.pdb_id,
            "confidence": self.confidence,
            "source": self.source,
            "content_type": self.content_type,
            "sequence_length": self.sequence_length,
            "pdb_line_count": self.pdb_line_count,
            "pdb_char_count": self.pdb_char_count,
            "record_counts": self.record_counts,
            "atom_count": self.atom_count,
            "residue_count": self.residue_count,
            "chain_ids": self.chain_ids,
            "chain_summaries": [item.model_dump() for item in self.chain_summaries],
            "coordinate_bounds": self.coordinate_bounds.model_dump() if self.coordinate_bounds else None,
            "min_plddt": self.min_plddt,
            "max_plddt": self.max_plddt,
            "high_confidence_residue_count": self.high_confidence_residue_count,
            "medium_confidence_residue_count": self.medium_confidence_residue_count,
            "low_confidence_residue_count": self.low_confidence_residue_count,
            "very_low_confidence_residue_count": self.very_low_confidence_residue_count,
            "confident_fraction": self.confident_fraction,
            "per_residue_plddt": [item.model_dump() for item in self.per_residue_plddt],
            "low_confidence_residues": [item.model_dump() for item in self.low_confidence_residues],
            "low_confidence_regions": [item.model_dump() for item in self.low_confidence_regions],
            "structure_quality": self.structure_quality,
            "error": self.error,
        }


class SequenceAnalysis(BaseModel):
    """单条多肽序列的分析结果集合"""
    sequence: str
    phys_chem_res: Optional[PhysChemResult] = None
    toxicity_res: Optional[ToxicityResult] = None
    activity_res: Optional[ActivityResult] = None
    structure_res: Optional[StructureResult] = None
