"""Claude coaching rubric and prompt assembly for technique feedback."""

import json

TECHNIQUE_RUBRIC = """## Krav Maga Striking Technique Rubric

### Stance & Base (6 checkpoints)
1. **Stance width**: Feet approximately shoulder-width apart, providing a stable base without overextending.
2. **Weight distribution**: Approximately 50/50 or slight front-foot bias; never heavily loaded on one foot.
3. **Foot stagger**: Lead foot forward, rear foot back ~1.5 shoulder widths; comfortable fighting stance depth.
4. **Body angle**: Torso bladed approximately 45 degrees to the target, reducing centerline exposure.
5. **Knee bend**: Both knees slightly bent, providing spring and mobility; not locked or overly crouched.
6. **Rear heel**: Rear heel raised off the ground, weight on ball of foot, enabling hip rotation and quick movement.

### Guard (5 checkpoints)
7. **Hand height**: Both hands at cheekbone height, protecting the face; not dropped to chin or chest level.
8. **Elbow tuck**: Elbows tucked close to the ribs, protecting the body and providing structural support for the guard.
9. **Chin position**: Chin tucked down toward the chest, reducing exposure to uppercuts and straights.
10. **Shoulder position**: Shoulders slightly raised to help protect the chin and jaw line.
11. **Guard symmetry**: Both hands in a balanced position; lead hand not excessively far forward or rear hand not dropped.

### Jab Mechanics (7 checkpoints)
12. **Extension path**: Lead hand travels in a straight line from guard to target — no looping, hooking, or arcing.
13. **Fist rotation**: Fist rotates to horizontal (palm down) at full extension for proper alignment.
14. **Shoulder cover**: Lead shoulder rises to protect the chin during extension.
15. **Non-punching hand**: Rear hand stays glued to the cheekbone throughout the jab; does not drop or pull back.
16. **Hip engagement**: Slight forward hip engagement to add power without overcommitting balance.
17. **Retraction**: Hand retracts at least as fast as it extends — "snap back" to guard position.
18. **Full extension**: Arm reaches full extension without locking the elbow; maximizes range.

### Cross Mechanics (8 checkpoints)
19. **Rear hand drive**: Rear hand drives straight from guard toward the target along the centerline.
20. **Hip rotation**: Full hip rotation from rear to front, generating power through the kinetic chain.
21. **Foot pivot**: Rear foot pivots on the ball, heel turning outward to enable full hip rotation.
22. **Weight transfer**: Weight shifts from rear foot to front foot during the strike, then recovers.
23. **Shoulder rotation**: Rear shoulder rotates through the centerline, extending reach and adding power.
24. **Full extension**: Arm reaches full extension without locking the elbow.
25. **Chin protection**: Chin stays tucked behind the lead shoulder during the cross.
26. **Return to guard**: Both hands return to guard position immediately after the cross lands.

### Jab-Cross Combination (5 checkpoints)
27. **Transition fluidity**: Smooth, continuous transition from jab to cross with no pause or reset.
28. **No telegraph**: No visible wind-up, hand pullback, or weight shift between strikes.
29. **Guard during transition**: Non-striking hand maintains guard position during each phase of the combination.
30. **Stance recovery**: Stance remains stable between strikes; feet do not drift or cross.
31. **Rhythm**: Natural 1-2 rhythm; the cross follows the jab without excessive delay or rushing.

### Footwork & Recovery (4 checkpoints)
32. **No overcommitment**: Striker does not lunge or fall forward past their base during strikes.
33. **Return to stance**: After striking, returns to proper fighting stance (width, stagger, balance).
34. **Balance**: Center of gravity stays over the base throughout; no stumbling or excessive leaning.
35. **Feet not crossed**: Feet never cross over each other during movement or striking.

### Common Errors to Watch For
- **Dropping guard**: Non-punching hand drops below chin level during strikes.
- **Telegraphing**: Pulling the hand back, dipping the shoulder, or shifting weight before striking.
- **Chicken-wing elbow**: Elbow flares out during the punch, creating a looping path instead of straight.
- **Leaning forward**: Upper body leans past the front knee, compromising balance and recovery.
- **Crossing feet**: Feet cross during movement, destroying base and balance.
- **Winding up**: Drawing the fist back or rotating the torso before initiating the strike.
- **No hip rotation**: Punching with arm only, without engaging hips for power generation.
- **Loose fist**: Fist not tightly clenched at impact, risking wrist injury.
- **Not returning to guard**: Leaving the striking hand extended or dropping it after the punch.
- **Locked elbow**: Hyperextending the elbow at full extension, risking joint injury.
- **Head stationary on centerline**: Head stays fixed on the centerline rather than shifting off-line with the punch.
"""

_OUTPUT_FORMAT = """{
  "summary": "One-line overall assessment of the practitioner's technique",
  "positives": [
    "Specific thing done well, referencing checkpoint number"
  ],
  "corrections": [
    {
      "issue": "What is wrong — be specific about body part and position",
      "why_it_matters": "Impact on power, safety, or defensive vulnerability",
      "how_to_fix": "One concrete, actionable instruction the practitioner can apply immediately",
      "priority": "high|medium|low"
    }
  ]
}"""


def get_claude_prompt(
    strike_type: str,
    observations: dict,
    rag_chunks: list,
    session_history: list,
) -> str:
    """Assemble the full Claude coaching prompt.

    Args:
        strike_type: One of "jab", "cross", or "jab_cross".
        observations: Structured observations dict from Gemini analysis.
        rag_chunks: List of relevant RAG text chunks (strings or dicts with "text" key).
        session_history: List of previous feedback dicts for progress tracking.

    Returns:
        Complete prompt string for Claude to generate coaching feedback.
    """
    sections = [
        "You are an expert Krav Maga striking coach. Your role is to analyze "
        "a practitioner's technique and provide clear, prioritized coaching feedback.\n",
        f"## Strike Being Analyzed: **{strike_type}**\n",
        "## Technique Rubric\n",
        TECHNIQUE_RUBRIC,
    ]

    # Gemini observations
    sections.append("\n## Video Observations (from biomechanics analysis)\n")
    if observations:
        sections.append(f"```json\n{json.dumps(observations, indent=2)}\n```\n")
    else:
        sections.append("No observations available.\n")

    # RAG chunks
    if rag_chunks:
        sections.append("## Relevant Coaching Reference Material\n")
        for i, chunk in enumerate(rag_chunks, 1):
            text = chunk["text"] if isinstance(chunk, dict) else chunk
            sections.append(f"**Source {i}:** {text}\n")

    # Session history
    if session_history:
        sections.append("## Session History (previous attempts)\n")
        sections.append(
            "Use this to track improvement and avoid repeating the same feedback.\n"
        )
        for i, entry in enumerate(session_history, 1):
            sections.append(f"**Attempt {i}:** {json.dumps(entry)}\n")

    # Output instructions
    sections.append("## Your Task\n")
    sections.append(
        "Analyze the observations against the rubric. Identify what the practitioner "
        "does well and what needs correction. Prioritize corrections by impact on "
        "technique effectiveness and safety.\n"
    )
    sections.append(
        "If session history is provided, acknowledge improvement and focus on "
        "remaining or new issues.\n"
    )
    sections.append(f"Respond with valid JSON only in this format:\n```json\n{_OUTPUT_FORMAT}\n```\n")
    sections.append(
        "Priority guide: **high** = safety risk or major power/technique loss, "
        "**medium** = noticeable technique gap, **low** = refinement detail.\n"
    )
    sections.append(
        "Limit corrections to the top 3-5 most impactful issues. "
        "Be specific and actionable — reference body parts, angles, and positions.\n"
    )

    return "\n".join(sections)
