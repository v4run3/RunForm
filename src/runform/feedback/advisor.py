"""Biomechanical injury-risk assessment and feedback engine for RunForm.

Evaluates calculated kinematics (knee flexion, trunk lean, bilateral symmetry)
and strike patterns against evidence-based clinical running biomechanics guidelines.
"""

from __future__ import annotations

from runform.schema import (
    BiomechanicalMetrics,
    FootStrikeType,
    FormFeedback,
    InjuryRiskLevel,
)


class RiskAdvisor:
    """Evaluates gait metrics and generates explainable injury risk assessments."""

    def __init__(
        self,
        max_knee_extension_deg: float = 165.0,
        max_trunk_lean_deg: float = 15.0,
        min_trunk_lean_deg: float = -2.0,
        max_asymmetry_deg: float = 12.0,
    ) -> None:
        """Initialize risk advisor with tunable clinical thresholds.

        Args:
            max_knee_extension_deg: Maximum allowable knee angle at contact before flagging overstriding.
            max_trunk_lean_deg: Maximum allowable forward trunk lean before flagging excessive flexion.
            min_trunk_lean_deg: Minimum trunk lean before flagging backward tilt.
            max_asymmetry_deg: Maximum bilateral joint angle difference before flagging asymmetry.
        """
        self.max_knee_ext = max_knee_extension_deg
        self.max_trunk_lean = max_trunk_lean_deg
        self.min_trunk_lean = min_trunk_lean_deg
        self.max_asymmetry = max_asymmetry_deg

    def evaluate(
        self,
        metrics: BiomechanicalMetrics,
        strike_type: FootStrikeType = FootStrikeType.UNKNOWN,
    ) -> FormFeedback:
        """Evaluate a runner's kinematic metrics and return actionable feedback.

        Args:
            metrics: Computed BiomechanicalMetrics object.
            strike_type: Primary classified FootStrikeType.

        Returns:
            FormFeedback object containing risk level, flags, and recommendations.
        """
        flags: list[str] = []
        recs: list[str] = []
        risk_score = 0  # 0: Low, 1-2: Moderate, 3+: High

        # 1. Overstriding & Knee Extension Check
        max_knee = max(
            [k for k in [metrics.knee_flexion_left_deg, metrics.knee_flexion_right_deg] if k is not None],
            default=None,
        )
        if max_knee is not None and max_knee > self.max_knee_ext:
            if strike_type == FootStrikeType.HEEL:
                flags.append("High Overstriding & Impact Risk (Extended Knee + Heel Strike)")
                recs.append(
                    "Try increasing your step rate (cadence) by 5-10% to land with a softer, "
                    "slightly bent knee closer under your hips."
                )
                risk_score += 2
            else:
                flags.append("Elevated Knee Extension at Contact")
                recs.append("Maintain a relaxed, soft knee bend at touchdown to absorb impact forces.")
                risk_score += 1

        # 2. Trunk Lean Check
        if metrics.trunk_lean_deg is not None:
            if metrics.trunk_lean_deg > self.max_trunk_lean:
                flags.append(f"Excessive Forward Trunk Lean ({metrics.trunk_lean_deg:.1f}°)")
                recs.append("Keep your chest open and tall; lean subtly from the ankles rather than bending from the hips.")
                risk_score += 1
            elif metrics.trunk_lean_deg < self.min_trunk_lean:
                flags.append(f"Backward Trunk Lean ({metrics.trunk_lean_deg:.1f}°)")
                recs.append("Avoid leaning backward as it increases braking forces on your joints.")
                risk_score += 1

        # 3. Bilateral Symmetry Check
        if metrics.knee_flexion_left_deg is not None and metrics.knee_flexion_right_deg is not None:
            diff = abs(metrics.knee_flexion_left_deg - metrics.knee_flexion_right_deg)
            if diff > self.max_asymmetry:
                flags.append(f"Bilateral Knee Asymmetry ({diff:.1f}° difference)")
                recs.append("Incorporate unilateral strength training (single-leg squats/lunges) to balance left/right loading.")
                risk_score += 1

        # Assign Risk Category
        if risk_score >= 2:
            risk_level = InjuryRiskLevel.HIGH if risk_score >= 3 else InjuryRiskLevel.MODERATE
        else:
            risk_level = InjuryRiskLevel.LOW

        if not flags:
            flags.append("Optimal Form & Alignment")
            recs.append("Great posture! Joint angles and landing patterns are within recommended efficiency ranges.")

        return FormFeedback(
            risk_level=risk_level,
            flags=flags,
            recommendations=recs,
        )
