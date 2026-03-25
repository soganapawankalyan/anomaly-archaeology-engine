import ollama

MODEL = "llama3.2:3b"

SYSTEM_PROMPT = (
    "You are a senior reliability engineer writing a structured incident report. "
    "Be direct and technical. Write in present tense. "
    "Format your response EXACTLY as follows with these exact headers:\n"
    "SEVERITY: [Critical/High/Medium/Low]\n"
    "PROBABLE CAUSE: [one sentence]\n"
    "CONTRIBUTING FACTORS: [bullet list of 2-3 factors]\n"
    "TIMELINE SUMMARY: [2-3 sentences describing the cascade]\n"
    "RECOMMENDED ACTIONS: [numbered list of 3 specific actions]\n"
    "Never use markdown bold. Never add extra sections."
)


def generate_incident_report(scenario: dict, investigation: dict) -> str:
    top_causes = investigation["root_causes"][:3]
    timeline   = investigation["timeline"][:6]

    cause_lines = "\n".join(
        f"  - {rc['signal']}: composite_score={rc['composite_score']:.3f}, "
        f"granger={rc['granger_score']:.3f}, first_detected_at_index={rc['first_anomaly']}"
        for rc in top_causes
    )
    timeline_lines = "\n".join(
        f"  [{ev['index']}] {ev['signal']}: {ev['event']} (value={ev['value']})"
        for ev in timeline
    )

    prompt = f"""Incident: {scenario['name']}
Severity: {scenario['severity']}
Incident type: {scenario['incident_type']}
Description: {scenario['description']}

Root cause analysis (ranked by composite score):
{cause_lines}

Evidence timeline:
{timeline_lines}

Top detected cause: {investigation['top_cause']}
True root cause signal: {scenario['root_cause_signal']}

Write the structured incident report."""

    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            options={"temperature": 0.2},
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"Report unavailable: {str(e)}"


def parse_report_sections(report_text: str) -> dict:
    sections = {
        "severity": "",
        "probable_cause": "",
        "contributing_factors": "",
        "timeline_summary": "",
        "recommended_actions": "",
        "raw": report_text,
    }
    current = None
    buffer  = []
    for line in report_text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("SEVERITY:"):
            sections["severity"] = stripped.replace("SEVERITY:", "").strip()
        elif stripped.startswith("PROBABLE CAUSE:"):
            current = "probable_cause"
            val = stripped.replace("PROBABLE CAUSE:", "").strip()
            if val:
                buffer = [val]
        elif stripped.startswith("CONTRIBUTING FACTORS:"):
            if current and buffer:
                sections[current] = "\n".join(buffer)
            current = "contributing_factors"
            buffer  = []
        elif stripped.startswith("TIMELINE SUMMARY:"):
            if current and buffer:
                sections[current] = "\n".join(buffer)
            current = "timeline_summary"
            buffer  = []
        elif stripped.startswith("RECOMMENDED ACTIONS:"):
            if current and buffer:
                sections[current] = "\n".join(buffer)
            current = "recommended_actions"
            buffer  = []
        elif stripped and current:
            buffer.append(stripped)
    if current and buffer:
        sections[current] = "\n".join(buffer)
    return sections
