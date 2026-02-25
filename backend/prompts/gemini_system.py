"""Gemini observation prompt and structured schema for video analysis."""

OBSERVATION_SCHEMA = {
    "stance": {
        "description": "Fighting stance observations",
        "fields": [
            "stance_width_relative_to_shoulders",
            "foot_stagger_depth",
            "body_angle_degrees",
            "weight_distribution_front_back",
            "knee_bend",
            "rear_heel_position",
        ],
    },
    "guard": {
        "description": "Guard position observations",
        "fields": [
            "lead_hand_height_relative_to_face",
            "rear_hand_height_relative_to_face",
            "elbow_tuck",
            "chin_position",
            "shoulder_position",
        ],
    },
    "strike_mechanics": {
        "description": "Mechanics of the executed strike(s)",
        "fields": [
            "extension_path",
            "fist_orientation_at_impact",
            "hip_rotation_degree",
            "shoulder_rotation",
            "rear_foot_pivot",
            "weight_transfer",
            "retraction_speed_vs_extension",
            "full_extension_achieved",
            "elbow_position_during_strike",
        ],
    },
    "non_striking_hand": {
        "description": "What the non-punching hand does during the strike",
        "fields": [
            "guard_maintained",
            "hand_drop_observed",
            "position_at_strike_peak",
        ],
    },
    "footwork": {
        "description": "Footwork and balance observations",
        "fields": [
            "feet_crossed",
            "overcommitment",
            "balance_throughout",
            "stance_recovery_after_strike",
        ],
    },
    "telegraphing": {
        "description": "Any telegraphing observed before strikes",
        "fields": [
            "hand_pullback_before_strike",
            "weight_shift_before_strike",
            "shoulder_dip",
            "other_tells",
        ],
    },
    "combination_flow": {
        "description": "Only for jab-cross combinations",
        "fields": [
            "transition_fluidity",
            "pause_between_strikes",
            "guard_during_transition",
            "rhythm",
        ],
    },
}


def _format_schema_for_prompt() -> str:
    lines = ["Provide your observations as JSON with these categories:\n"]
    for category, info in OBSERVATION_SCHEMA.items():
        lines.append(f"### {category}")
        lines.append(f"{info['description']}")
        for field in info["fields"]:
            lines.append(f"  - {field}: <describe what you observe>")
        lines.append("")
    return "\n".join(lines)


def get_gemini_prompt(strike_type: str) -> str:
    """Build the full Gemini observation prompt for a given strike type.

    Args:
        strike_type: One of "jab", "cross", or "jab_cross".

    Returns:
        Complete prompt string instructing Gemini to produce structured
        factual observations as JSON.
    """
    schema_text = _format_schema_for_prompt()

    return f"""You are an expert biomechanics analyst specializing in combat sports striking technique.

You are analyzing a video of a person performing a **{strike_type}** strike (Krav Maga context).

Reference images of correct technique are provided alongside the video. Compare the practitioner's form against these references.

## Your Task

Observe the practitioner's body mechanics in detail. Be FACTUAL and SPECIFIC — describe exactly what you see, with approximate angles, positions, and timing where possible. Do NOT provide coaching advice; only observations.

## Observation Categories

{schema_text}

## Important Notes

- If this is a single strike (jab or cross), skip the combination_flow category.
- For each field, describe what you actually observe. Use phrases like "approximately 45 degrees", "hand drops to chin level", "slight hip rotation ~20 degrees".
- Note the TIMING of movements — what happens first, what happens simultaneously, what is delayed.
- If you cannot clearly observe something (e.g., occluded by angle), say "not clearly visible" for that field.
- Compare against the reference images where applicable.

Respond with valid JSON only."""
