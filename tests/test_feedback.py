"""Unit tests for injury risk advisor and feedback generation."""

from runform.feedback.advisor import RiskAdvisor
from runform.schema import (
    BiomechanicalMetrics,
    FootStrikeType,
    InjuryRiskLevel,
)


def test_risk_advisor_optimal_form() -> None:
    advisor = RiskAdvisor()
    metrics = BiomechanicalMetrics(
        knee_flexion_left_deg=155.0,
        knee_flexion_right_deg=157.0,
        trunk_lean_deg=8.0,
    )
    feedback = advisor.evaluate(metrics, strike_type=FootStrikeType.MIDFOOT)
    assert feedback.risk_level == InjuryRiskLevel.LOW
    assert "Optimal Form & Alignment" in feedback.flags


def test_risk_advisor_overstriding_flag() -> None:
    advisor = RiskAdvisor(max_knee_extension_deg=165.0)
    metrics = BiomechanicalMetrics(
        knee_flexion_left_deg=175.0,  # Straight leg
        knee_flexion_right_deg=174.0,
        trunk_lean_deg=5.0,
    )
    feedback = advisor.evaluate(metrics, strike_type=FootStrikeType.HEEL)
    assert feedback.risk_level in (InjuryRiskLevel.MODERATE, InjuryRiskLevel.HIGH)
    assert any("Overstriding" in flag for flag in feedback.flags)


def test_risk_advisor_asymmetry_flag() -> None:
    advisor = RiskAdvisor(max_asymmetry_deg=10.0)
    metrics = BiomechanicalMetrics(
        knee_flexion_left_deg=145.0,
        knee_flexion_right_deg=162.0,  # 17 deg difference > 10
        trunk_lean_deg=5.0,
    )
    feedback = advisor.evaluate(metrics, strike_type=FootStrikeType.MIDFOOT)
    assert any("Asymmetry" in flag for flag in feedback.flags)
