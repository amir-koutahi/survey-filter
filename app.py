import io
import json
import os

import anthropic
import pandas as pd
import streamlit as st

DEPT_COLUMN = "Department/Program/Center"

CANONICAL_DEPARTMENTS = [
    "Department",
    "Faculty of Music",
    "Department of Statistical Sciences",
    "Faculty of Kinesiology & Physical Education",
    "Department of Curriculum, Teaching & Learning",
    "New College Programmes",
    "Department of Geography & Planning",
    "Department of Applied Psychology & Human Development",
    "Department of Sociology",
    "Department of Philosophy",
    "Department of Anthropology",
    "OISE Student Success Centre",
    "University College Programmes",
    "Department of Leadership, Higher & Adult Education",
    "Centre for Criminology & Sociolegal Studies",
    "Innis College Programmes",
    "Woodsworth College Programmes",
    "Department of Social Justice Education",
    "Joseph L. Rotman School of Management",
    "UTM: Mathematical & Computational Sciences",
    "Daniels Faculty of Architecture, Landscape & Design",
    "UTM: Biology",
    "UTM: Chemical & Physical Sciences",
    "UTM: Economics",
    "UTM: Management",
    "UTM: Institute of CCIT",
    "UTM: Psychology",
    "UTM: Anthropology",
    "UTM: Sociology",
    "UTM: Geography",
    "UTM: Dept. of Language Studies",
    "UTM: Historical Studies",
    "UTM: English & Drama",
    "UTM: Philosophy",
    "UTM: Political Science",
    "UTM: Dept of Visual Studies",
    "UTM: Institute for Management & Innovation",
    "Centre for Industrial Relations and Human Resources",
    "UTM: Forensic Science",
    "Institute for the History and Philosophy of Science and Technology",
    "UTM: Robert Gillespie Academic Skills Centre",
    "Centre for Teaching Support & Innovation",
    "UTM: Accessibility Services",
    "Transitional Year Program",
    "Graduate Centre for Academic Communication",
    "UTSC: Department of Physical & Environmental Science",
    "UTSC: Department of Computer & Mathematical Science",
    "UTSC: Department of Management",
    "Department of Psychology",
    "UTSC: Department of Biological Sciences",
    "Faculty of Pharmacy",
    "UTSC: Department of Psychology",
    "UTSC: Department of Arts, Culture & Media",
    "UTSC: Department of Health and Society",
    "UTSC: Department of Human Geography",
    "UTSC: Department of Political Science",
    "Faculty of Social Work",
    "UTSC: Department of Philosophy",
    "UTSC: Department of Historical & Cultural Studies",
    "UTSC: Department of Sociology",
    "UTSC: Department of English",
    "UTSC: Department of Language Studies",
    "UTSC: Department of Anthropology",
    "UTSC: Centre for Teaching & Learning",
    "UTSC: Department of Global Development Studies",
    "UTSC: AccessAbility Services",
    "UTSC: Office of the Registrar",
    "Department of Mathematics",
    "Department of Chemistry",
    "Department of Physics",
    "Department of Cell & Systems Biology",
    "Department of Ecology & Evolutionary Biology",
    "Department of Astronomy and Astrophysics",
    "Department of Pharmacology & Toxicology",
    "Faculty of Nursing",
    "Department of Molecular Genetics",
    "Department of Earth Sciences",
    "Dalla Lana School of Public Health",
    "Department of Physiology",
    "School of the Environment",
    "Department of Immunology",
    "Department of Biochemistry",
    "Department of Nutritional Sciences",
    "Institute of Forestry and Conservation",
    "Institute of Health Policy, Management & Evaluation",
    "Department of Occupational Science & Therapy",
    "Department of Laboratory Medicine & Pathobiology",
    "Institute of Medical Science",
    "Department of Surgery",
    "Canadian Institute for Theoretical Astrophysics",
    "Department of Physical Therapy",
    "Department of Speech-Language Pathology",
    "Department of Medical Biophysics",
    "Mathematical Finance Program",
    "Department of Economics",
    "Faculty of Information",
    "Department of Political Science",
    "Department of History",
    "Munk School of Global Affairs and Public Policy",
    "Department of English",
    "Cinema Studies Institute",
    "Department of Near & Middle East.Civilizations",
    "Department of Linguistics",
    "Department of East Asian Studies",
    "Department for the Study of Religion",
    "Department of Italian, Spanish, Portuguese & Latin American Studies",
    "Department of Art History",
    "Women and Gender Studies Institute",
    "Department of French",
    "Department of Classics",
    "Centre for Drama, Theatre and Performance Studies",
    "Centre for Medieval Studies",
    "Department of Slavic Languages & Literature",
    "Department of Germanic Language & Literatures",
    "Faculty of Law",
    "Centre for Jewish Studies",
    "Centre for Ethics",
    "Centre for Comparative Literature",
    "Centre for Indigenous Studies",
    "Centre for Diaspora & Transnational Studies",
    "Jackman Humanities Institute",
    "Department of Computer Science",
    "Department of Electrical & Computer Engineering",
    "Department of Mechanical & Industrial Engineering",
    "Department of Civil & Mineral Engineering",
    "Department of Chemical Engineering & Applied Chemistry",
    "Engineering Communication Programme",
    "Institute of Biomaterials & Biomedical Engineering",
    "Accommodated Testing Services",
    "Department of Materials Science & Engineering",
    "Division of Engineering Science",
    "Institute for Aerospace Studies",
    "APSC Cross-Disciplinary Programs",
    "APSC First Year Office",
    "Office of the Faculty Registrar",
]

CANONICAL_SET = set(CANONICAL_DEPARTMENTS)


def build_system_prompt() -> str:
    canonical_block = "\n".join(f"- {d}" for d in CANONICAL_DEPARTMENTS)
    return f"""You map user-entered department/program/centre names from a University of Toronto union survey to a fixed canonical list. Users have made typos, used abbreviations, dropped words, or written informal names — apply careful reasoning to recover the intended department.

CANONICAL LIST (the only valid match values):
{canonical_block}

Rules:
- Output the `matches` field as an array of canonical strings above, byte-for-byte. Copy exactly — punctuation, ampersands, capitalization, and abbreviations must match the list.
- If the input clearly references TWO OR MORE distinct departments (e.g., "Math and Stats", "Physics / Astronomy", "ECE & MIE"), return ALL of them in the `matches` array. Otherwise return a single-element array.
- Fix typos: "Physcs" -> ["Department of Physics"], "Depatment of Mathematics" -> ["Department of Mathematics"].
- Expand abbreviations using UofT context: "ECE" -> ["Department of Electrical & Computer Engineering"], "NMC" -> ["Department of Near & Middle East.Civilizations"], "Astro" -> ["Department of Astronomy and Astrophysics"].
- Match partial names: "Math" / "mathematics" -> ["Department of Mathematics"]; "Spanish" -> ["Department of Italian, Spanish, Portuguese & Latin American Studies"]; "Cinema Studies" -> ["Cinema Studies Institute"]; "History" -> ["Department of History"].
- Multi-department examples: "Math and Stats" -> ["Department of Mathematics", "Department of Statistical Sciences"]; "Physics/Astronomy" -> ["Department of Physics", "Department of Astronomy and Astrophysics"].
- For UTM-prefixed inputs, use the corresponding "UTM:" entry. For UTSC-prefixed inputs, use the corresponding "UTSC:" entry. "UNIVERSITY OF TORONTO MISSISSAUGA - COMPUTER SCIENCE" — UTM has no separate CS department; map to ["UTM: Mathematical & Computational Sciences"].
- "FAS" (Faculty of Arts and Science) is a faculty, not a department in the list — set `matches` to [original input] and confidence to "none".
- "null", empty-looking, or genuinely unmatchable values: set `matches` to [original input] and confidence to "none".
- Do not invent canonical entries. If nothing fits, return [original input verbatim] with confidence "none"."""


def map_departments(unique_values: list[str], api_key: str, model: str):
    client = anthropic.Anthropic(api_key=api_key)

    schema = {
        "type": "object",
        "properties": {
            "mappings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"},
                        "matches": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low", "none"],
                        },
                        "reasoning": {"type": "string"},
                    },
                    "required": ["input", "matches", "confidence", "reasoning"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["mappings"],
        "additionalProperties": False,
    }

    user_msg = "Map each of these survey responses to the canonical list:\n\n" + "\n".join(
        f"{i + 1}. {v!r}" for i, v in enumerate(unique_values)
    )

    kwargs = {
        "model": model,
        "max_tokens": 16000,
        "system": build_system_prompt(),
        "messages": [{"role": "user", "content": user_msg}],
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    }
    # Adaptive thinking + effort are supported on Opus/Sonnet but error on Haiku 4.5.
    if model.startswith(("claude-opus", "claude-sonnet")):
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"]["effort"] = "high"

    response = client.messages.create(**kwargs)

    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    return data["mappings"], response.usage


def main():
    st.set_page_config(page_title="UofT Department Standardizer", layout="wide")
    st.title("UofT Department Standardizer")
    st.caption(
        "Upload a Bargaining Survey CSV. The app maps every entry in the "
        f"`{DEPT_COLUMN}` column to the canonical UofT department list and returns an Excel file."
    )

    api_key = ""
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")

    POWER_OPTIONS = {
        "Lower Power": "claude-haiku-4-5",
        "Medium Power": "claude-sonnet-4-6",
        "Higher Power": "claude-opus-4-7",
    }

    with st.sidebar:
        st.header("Settings")
        power_label = st.selectbox(
            "Power Level",
            list(POWER_OPTIONS.keys()),
            index=0,
            help="Lower Power is cheapest and handles this task well; Higher Power is most accurate on tricky abbreviations.",
        )
        model = POWER_OPTIONS[power_label]
        st.info(
            "Disclaimer: the Higher Power costs some money, however for the final "
            "analysis where the data will get larger it will be needed. For now, "
            "lower powers work fine."
        )

    uploaded = st.file_uploader("Upload survey CSV", type=["csv"])
    if not uploaded:
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return

    if DEPT_COLUMN not in df.columns:
        st.error(
            f"CSV is missing the `{DEPT_COLUMN}` column. "
            f"Found columns: {list(df.columns)[:10]}..."
        )
        return

    st.success(f"Loaded {len(df):,} rows.")

    raw_series = df[DEPT_COLUMN].astype("string").str.strip()
    unique_vals = sorted(v for v in raw_series.dropna().unique() if v)
    st.info(f"Found {len(unique_vals)} unique non-empty values in `{DEPT_COLUMN}`.")

    with st.expander("Preview unique values"):
        st.write(unique_vals)

    if not api_key:
        st.error(
            "Server is missing the ANTHROPIC_API_KEY secret. "
            "The app owner needs to set it in Streamlit Cloud → Settings → Secrets."
        )
        return

    if not st.button("Standardize and download Excel", type="primary"):
        return

    with st.spinner(f"Mapping with {power_label}..."):
        try:
            mappings, usage = map_departments(unique_vals, api_key, model)
        except anthropic.APIError as e:
            st.error(f"Anthropic API error: {e}")
            return
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return

    invalid = sorted({
        x
        for m in mappings
        if m["confidence"] != "none"
        for x in m["matches"]
        if x not in CANONICAL_SET
    })
    if invalid:
        st.warning(
            "Model returned values that don't byte-match the canonical list "
            "(treated as 'none'): " + ", ".join(invalid)
        )
        for m in mappings:
            if m["confidence"] != "none" and any(x not in CANONICAL_SET for x in m["matches"]):
                m["confidence"] = "none"
                m["matches"] = [m["input"]]

    mapping_dict = {m["input"]: "; ".join(m["matches"]) for m in mappings}
    confidence_dict = {m["input"]: m["confidence"] for m in mappings}

    preview_df = pd.DataFrame(
        [
            {
                "input": m["input"],
                "match": "; ".join(m["matches"]),
                "num_matches": len(m["matches"]),
                "confidence": m["confidence"],
                "reasoning": m["reasoning"],
            }
            for m in mappings
        ]
    )
    st.subheader("Mapping preview")
    st.dataframe(preview_df, width="stretch")

    multi = preview_df[preview_df["num_matches"] > 1]
    if not multi.empty:
        st.info(f"{len(multi)} input(s) mapped to multiple departments (joined with `; `).")

    counts = preview_df["confidence"].value_counts().to_dict()
    cols = st.columns(4)
    cols[0].metric("High", counts.get("high", 0))
    cols[1].metric("Medium", counts.get("medium", 0))
    cols[2].metric("Low", counts.get("low", 0))
    cols[3].metric("Unmatched", counts.get("none", 0))

    st.caption(
        f"Tokens — input: {usage.input_tokens:,}, "
        f"output: {usage.output_tokens:,}"
    )

    def replace(v):
        if pd.isna(v):
            return v
        key = str(v).strip()
        return mapping_dict.get(key, v)

    df_out = df.copy()
    df_out[DEPT_COLUMN] = df_out[DEPT_COLUMN].apply(replace)

    confidence_col = f"{DEPT_COLUMN} (match confidence)"
    df_out[confidence_col] = df[DEPT_COLUMN].apply(
        lambda v: confidence_dict.get(str(v).strip(), "") if pd.notna(v) else ""
    )

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="Survey")
        preview_df.to_excel(writer, index=False, sheet_name="Mapping")
    buf.seek(0)

    base = uploaded.name.rsplit(".", 1)[0]
    st.download_button(
        "Download standardized Excel",
        data=buf,
        file_name=f"{base}-standardized.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )


if __name__ == "__main__":
    main()
