import io
import json
import os
import re
import zipfile

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


def match_stewards_to_respondents(stewards, respondents, api_key: str, model: str):
    client = anthropic.Anthropic(api_key=api_key)

    system = (
        "You match union stewards against survey respondents to determine which "
        "stewards have completed the survey.\n\n"
        "Match on name (allowing typos, nicknames, middle-name variations, "
        "transliteration differences) and email (case-insensitive). "
        "Email exact match is the strongest signal; a confident name match also counts.\n\n"
        "For each steward, return whether they have a matching respondent."
    )

    stewards_block = "\n".join(
        f"S{s['idx']}: {s['first']} {s['last']} <{s['email']}>" for s in stewards
    )
    respondents_block = "\n".join(
        f"R{r['idx']}: {r['first']} {r['last']} <{r['email']}>" for r in respondents
    )
    user_msg = (
        f"STEWARDS:\n{stewards_block}\n\nSURVEY RESPONDENTS:\n{respondents_block}\n\n"
        "For each steward, determine whether any respondent is plausibly the same person."
    )

    schema = {
        "type": "object",
        "properties": {
            "matches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "steward_idx": {"type": "integer"},
                        "matched": {"type": "boolean"},
                        "respondent_idx": {"type": ["integer", "null"]},
                        "confidence": {
                            "type": "string",
                            "enum": ["high", "medium", "low", "none"],
                        },
                    },
                    "required": ["steward_idx", "matched", "respondent_idx", "confidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["matches"],
        "additionalProperties": False,
    }

    kwargs = {
        "model": model,
        "max_tokens": 16000,
        "system": system,
        "messages": [{"role": "user", "content": user_msg}],
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    }
    if model.startswith(("claude-opus", "claude-sonnet")):
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"]["effort"] = "high"

    response = client.messages.create(**kwargs)
    text = next(b.text for b in response.content if b.type == "text")
    data = json.loads(text)
    return data["matches"], response.usage


def run_department_standardization(api_key: str, model: str, power_label: str):
    st.subheader("Department standardization")
    st.caption(
        "Upload a Bargaining Survey CSV. The app maps every entry in the "
        f"`{DEPT_COLUMN}` column to the canonical UofT department list and returns an Excel file."
    )

    uploaded = st.file_uploader("Upload survey CSV", type=["csv"], key="dept_survey")
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

    primary_dict = {m["input"]: m["matches"][0] for m in mappings}
    secondary_dict = {
        m["input"]: "; ".join(m["matches"][1:]) for m in mappings
    }
    confidence_dict = {m["input"]: m["confidence"] for m in mappings}

    preview_df = pd.DataFrame(
        [
            {
                "input": m["input"],
                "match": m["matches"][0],
                "second match": "; ".join(m["matches"][1:]),
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
        st.info(f"{len(multi)} input(s) mapped to multiple departments (second match goes in its own column).")

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

    def lookup(d, v, default=""):
        if pd.isna(v):
            return v if d is primary_dict else default
        return d.get(str(v).strip(), v if d is primary_dict else default)

    df_out = df.copy()
    df_out[DEPT_COLUMN] = df_out[DEPT_COLUMN].apply(lambda v: lookup(primary_dict, v))

    second_match_col = f"{DEPT_COLUMN} (second match)"
    df_out[second_match_col] = df[DEPT_COLUMN].apply(lambda v: lookup(secondary_dict, v))

    confidence_col = f"{DEPT_COLUMN} (match confidence)"
    df_out[confidence_col] = df[DEPT_COLUMN].apply(lambda v: lookup(confidence_dict, v))

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


def run_steward_audit(api_key: str, model: str, power_label: str):
    st.subheader("Stewards didnt complete survey")
    st.caption(
        "Upload the steward list and the survey CSV. The app returns an Excel "
        "of stewards who have NOT completed the survey."
    )

    col_a, col_b = st.columns(2)
    stewards_file = col_a.file_uploader("Steward list CSV", type=["csv"], key="audit_stewards")
    survey_file = col_b.file_uploader("Survey CSV", type=["csv"], key="audit_survey")
    if not stewards_file or not survey_file:
        return

    try:
        stewards_df = pd.read_csv(stewards_file)
    except Exception as e:
        st.error(f"Could not read steward CSV: {e}")
        return
    try:
        survey_df = pd.read_csv(survey_file)
    except Exception as e:
        st.error(f"Could not read survey CSV: {e}")
        return

    steward_first = "First Name"
    steward_last = "Last Name"
    steward_email = "Top Ranked Email"
    for col in (steward_first, steward_last, steward_email):
        if col not in stewards_df.columns:
            st.error(f"Steward CSV is missing the `{col}` column.")
            return

    survey_first = "Name"
    survey_last = "Last"
    survey_email = "Email"
    for col in (survey_first, survey_last, survey_email):
        if col not in survey_df.columns:
            st.error(f"Survey CSV is missing the `{col}` column.")
            return

    def norm_email(v):
        if pd.isna(v):
            return ""
        return str(v).strip().lower()

    def norm_str(v):
        if pd.isna(v):
            return ""
        return str(v).strip()

    stewards = []
    for i, row in stewards_df.iterrows():
        stewards.append({
            "idx": int(i),
            "first": norm_str(row[steward_first]),
            "last": norm_str(row[steward_last]),
            "email": norm_email(row[steward_email]),
        })

    respondents = []
    for i, row in survey_df.iterrows():
        respondents.append({
            "idx": int(i),
            "first": norm_str(row[survey_first]),
            "last": norm_str(row[survey_last]),
            "email": norm_email(row[survey_email]),
        })

    st.success(
        f"Loaded {len(stewards):,} stewards and {len(respondents):,} survey respondents."
    )

    placeholder_emails = {"", "null@null.com"}
    resp_emails = {r["email"] for r in respondents if r["email"] not in placeholder_emails}

    email_matched = {
        s["idx"] for s in stewards
        if s["email"] and s["email"] not in placeholder_emails and s["email"] in resp_emails
    }
    st.info(f"{len(email_matched)} stewards matched by exact email; checking the rest by name.")

    remaining = [s for s in stewards if s["idx"] not in email_matched]

    if not api_key:
        st.error(
            "Server is missing the ANTHROPIC_API_KEY secret. "
            "The app owner needs to set it in Streamlit Cloud → Settings → Secrets."
        )
        return

    if not st.button("Run audit and download Excel", type="primary"):
        return

    ai_matched = set()
    usage = None
    if remaining:
        with st.spinner(f"Matching remaining stewards with {power_label}..."):
            try:
                results, usage = match_stewards_to_respondents(
                    remaining, respondents, api_key, model
                )
            except anthropic.APIError as e:
                st.error(f"Anthropic API error: {e}")
                return
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                return
        ai_matched = {
            r["steward_idx"] for r in results
            if r["matched"] and r["confidence"] in ("high", "medium")
        }

    matched_idx = email_matched | ai_matched
    missing_mask = ~stewards_df.index.isin(matched_idx)
    missing_df = stewards_df.loc[missing_mask].copy()

    cols = st.columns(3)
    cols[0].metric("Total stewards", len(stewards))
    cols[1].metric("Completed survey", len(matched_idx))
    cols[2].metric("Missing", len(missing_df))

    if usage is not None:
        st.caption(
            f"Tokens — input: {usage.input_tokens:,}, output: {usage.output_tokens:,}"
        )

    output_cols = [c for c in (
        steward_first, steward_last, steward_email,
        "CUPE No.", "Position Entity", "Division Name", "Unit Name",
    ) if c in missing_df.columns]
    preview_df = missing_df[output_cols] if output_cols else missing_df

    st.subheader("Stewards who have not completed the survey")
    st.dataframe(preview_df, width="stretch")

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        preview_df.to_excel(writer, index=False, sheet_name="Missing Stewards")
    buf.seek(0)

    base = stewards_file.name.rsplit(".", 1)[0]
    st.download_button(
        "Download missing-stewards Excel",
        data=buf,
        file_name=f"{base}-missing-survey.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )


def run_dedupe_by_name():
    st.subheader("Deduplicate by Name")
    st.caption(
        "Upload a survey file (CSV or Excel). Rows with matching first + last name "
        "are deduplicated to the first occurrence — comparison ignores capitalization "
        "and whitespace. The removed duplicates are returned as a separate Excel file."
    )

    uploaded = st.file_uploader(
        "Upload survey file", type=["csv", "xlsx"], key="dedupe_survey"
    )
    if not uploaded:
        return

    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    first_col = "Name"
    last_col = "Last"
    for col in (first_col, last_col):
        if col not in df.columns:
            st.error(f"File is missing the `{col}` column.")
            return

    st.success(f"Loaded {len(df):,} rows.")

    def norm(v):
        if pd.isna(v):
            return ""
        return "".join(str(v).split()).lower()

    first_norm = df[first_col].apply(norm)
    last_norm = df[last_col].apply(norm)
    key = first_norm + "|" + last_norm
    has_name = (first_norm != "") | (last_norm != "")

    duplicate_mask = has_name & key.duplicated(keep="first")

    kept_df = df.loc[~duplicate_mask].copy()
    removed_df = df.loc[duplicate_mask].copy()

    cols = st.columns(3)
    cols[0].metric("Total rows", len(df))
    cols[1].metric("Kept (deduplicated)", len(kept_df))
    cols[2].metric("Removed duplicates", len(removed_df))

    if not removed_df.empty:
        st.subheader("Removed duplicate rows")
        front = [first_col, last_col]
        ordered = front + [c for c in removed_df.columns if c not in front]
        st.dataframe(removed_df[ordered], width="stretch")
    else:
        st.info("No duplicate names found.")

    base = uploaded.name.rsplit(".", 1)[0]

    kept_buf = io.BytesIO()
    with pd.ExcelWriter(kept_buf, engine="openpyxl") as writer:
        kept_df.to_excel(writer, index=False, sheet_name="Deduplicated")
    kept_buf.seek(0)

    removed_buf = io.BytesIO()
    with pd.ExcelWriter(removed_buf, engine="openpyxl") as writer:
        removed_df.to_excel(writer, index=False, sheet_name="Removed")
    removed_buf.seek(0)

    dl_cols = st.columns(2)
    dl_cols[0].download_button(
        "Download deduplicated Excel",
        data=kept_buf,
        file_name=f"{base}-deduplicated.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
    dl_cols[1].download_button(
        "Download removed-duplicates Excel",
        data=removed_buf,
        file_name=f"{base}-removed-duplicates.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=removed_df.empty,
    )


def _safe_filename(name: str, fallback: str = "Unknown") -> str:
    """Strip characters that aren't allowed in filenames on common OSes."""
    if name is None:
        return fallback
    cleaned = re.sub(r'[\\/*?:"<>|\r\n\t]+', " ", str(name)).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.rstrip(". ")
    return cleaned[:120] if cleaned else fallback


def run_group_by_department():
    st.subheader("Group by Department")
    st.caption(
        "Upload a member CSV (e.g. the `2025_09 to present active U1` export). "
        "The app groups rows by the `Department Name` column and returns a ZIP "
        "containing one Excel file per department with the full list of members."
    )

    group_col = "Department Name"

    uploaded = st.file_uploader(
        "Upload member file (CSV or XLSX)",
        type=["csv", "xlsx", "xls"],
        key="group_by_dept_csv",
    )
    if not uploaded:
        return

    try:
        df = _read_table(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    if group_col not in df.columns:
        st.error(
            f"CSV is missing the `{group_col}` column. "
            f"Found columns: {list(df.columns)[:15]}..."
        )
        return

    st.success(f"Loaded {len(df):,} rows.")

    dept_series = df[group_col].astype("string").fillna("").str.strip()
    df = df.assign(**{group_col: dept_series.replace("", pd.NA)})

    blanks = int(df[group_col].isna().sum())
    if blanks:
        st.info(
            f"{blanks} row(s) have an empty `{group_col}`. "
            f"They will be grouped under `(Unknown)`."
        )
    df[group_col] = df[group_col].fillna("(Unknown)")

    groups = list(df.groupby(group_col, sort=True))

    summary_df = pd.DataFrame(
        [{"Department Name": name, "Members": len(g)} for name, g in groups]
    )

    cols = st.columns(3)
    cols[0].metric("Departments", len(groups))
    cols[1].metric("Total members", len(df))
    cols[2].metric("Largest group", int(summary_df["Members"].max()) if len(summary_df) else 0)

    st.subheader("Department summary")
    st.dataframe(summary_df, width="stretch")

    base = uploaded.name.rsplit(".", 1)[0]

    zip_buf = io.BytesIO()
    used_names: set[str] = set()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for dept_name, group_df in groups:
            safe = _safe_filename(dept_name)
            # Avoid collisions if two departments sanitize to the same name.
            candidate = f"{safe}.xlsx"
            n = 2
            while candidate in used_names:
                candidate = f"{safe} ({n}).xlsx"
                n += 1
            used_names.add(candidate)

            xbuf = io.BytesIO()
            with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
                # Excel sheet names must be <= 31 chars and exclude :\\/?*[]
                sheet_name = re.sub(r"[:\\/?*\[\]]", " ", str(dept_name))[:31] or "Sheet1"
                group_df.to_excel(writer, index=False, sheet_name=sheet_name)
            xbuf.seek(0)
            zf.writestr(candidate, xbuf.getvalue())

        # Include the summary so the recipient knows what's inside.
        sbuf = io.BytesIO()
        with pd.ExcelWriter(sbuf, engine="openpyxl") as writer:
            summary_df.to_excel(writer, index=False, sheet_name="Summary")
        sbuf.seek(0)
        zf.writestr("_summary.xlsx", sbuf.getvalue())

    zip_buf.seek(0)

    st.download_button(
        "Download departments ZIP",
        data=zip_buf,
        file_name=f"{base}-by-department.zip",
        mime="application/zip",
        type="primary",
    )


def _read_table(uploaded):
    """Read an uploaded CSV or XLSX file into a DataFrame.

    Detects format by magic bytes so files mis-named (e.g. an .xlsx saved
    with a .csv extension) still load correctly.
    """
    name = (uploaded.name or "").lower()
    raw = uploaded.getvalue()
    # xlsx/xls files are zip archives starting with "PK"; old xls starts with D0 CF 11 E0.
    looks_xlsx = raw[:2] == b"PK" or raw[:4] == b"\xd0\xcf\x11\xe0"
    if looks_xlsx or name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("Could not read file")


def run_split_members_by_survey(api_key: str, model: str, power_label: str):
    st.subheader("Split members by survey completion")
    st.caption(
        "Upload the full member list and the survey file. The app returns a ZIP "
        "with two Excel files: members who completed the survey, and members who did not."
    )

    col_a, col_b = st.columns(2)
    members_file = col_a.file_uploader(
        "Members CSV/XLSX", type=["csv", "xlsx", "xls"], key="split_members"
    )
    survey_file = col_b.file_uploader(
        "Survey CSV/XLSX", type=["csv", "xlsx", "xls"], key="split_survey"
    )
    if not members_file or not survey_file:
        return

    try:
        members_df = _read_table(members_file)
    except Exception as e:
        st.error(f"Could not read members file: {e}")
        return
    try:
        survey_df = _read_table(survey_file)
    except Exception as e:
        st.error(f"Could not read survey file: {e}")
        return

    member_email = "Top Ranked Email"
    if member_email not in members_df.columns:
        st.error(f"Members file is missing the `{member_email}` column.")
        return

    survey_email = "Email"
    if survey_email not in survey_df.columns:
        st.error(f"Survey file is missing the `{survey_email}` column.")
        return

    def norm_email(v):
        if pd.isna(v):
            return ""
        return str(v).strip().lower()

    st.success(
        f"Loaded {len(members_df):,} members and {len(survey_df):,} survey respondents."
    )

    placeholder_emails = {"", "null@null.com"}
    resp_emails = {
        norm_email(v) for v in survey_df[survey_email]
        if norm_email(v) not in placeholder_emails
    }

    member_emails_norm = members_df[member_email].apply(norm_email)
    completed_mask = member_emails_norm.isin(resp_emails) & ~member_emails_norm.isin(placeholder_emails)

    if not st.button("Split and download ZIP", type="primary"):
        return

    completed_df = members_df.loc[completed_mask].copy()
    missing_df = members_df.loc[~completed_mask].copy()

    cols = st.columns(3)
    cols[0].metric("Total members", len(members_df))
    cols[1].metric("Completed survey", len(completed_df))
    cols[2].metric("Did not complete", len(missing_df))

    st.subheader("Members who completed the survey")
    st.dataframe(completed_df, width="stretch")
    st.subheader("Members who did NOT complete the survey")
    st.dataframe(missing_df, width="stretch")

    completed_buf = io.BytesIO()
    with pd.ExcelWriter(completed_buf, engine="openpyxl") as writer:
        completed_df.to_excel(writer, index=False, sheet_name="Completed Survey")
    completed_buf.seek(0)

    missing_buf = io.BytesIO()
    with pd.ExcelWriter(missing_buf, engine="openpyxl") as writer:
        missing_df.to_excel(writer, index=False, sheet_name="Did Not Complete")
    missing_buf.seek(0)

    base = members_file.name.rsplit(".", 1)[0]

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base}-completed-survey.xlsx", completed_buf.getvalue())
        zf.writestr(f"{base}-did-not-complete.xlsx", missing_buf.getvalue())
    zip_buf.seek(0)

    st.download_button(
        "Download ZIP",
        data=zip_buf,
        file_name=f"{base}-survey-split.zip",
        mime="application/zip",
        type="primary",
    )


def main():
    st.set_page_config(page_title="UofT Bargaining Tools", layout="wide")
    st.title("UofT Bargaining Tools")

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
        mode = st.radio(
            "Mode",
            [
                "Department Standardization",
                "Stewards didnt complete survey",
                "Deduplicate by Name",
                "Group by Department",
                "Split members by survey completion",
            ],
            index=0,
        )
        power_label = st.selectbox(
            "Power Level",
            list(POWER_OPTIONS.keys()),
            index=0,
            help="Lower Power is cheapest and handles this task well; Higher Power is most accurate on tricky cases.",
        )
        model = POWER_OPTIONS[power_label]
        st.info(
            "Disclaimer: the Higher Power costs some money, however for the final "
            "analysis where the data will get larger it will be needed. For now, "
            "lower powers work fine."
        )

    if mode == "Department Standardization":
        run_department_standardization(api_key, model, power_label)
    elif mode == "Stewards didnt complete survey":
        run_steward_audit(api_key, model, power_label)
    elif mode == "Deduplicate by Name":
        run_dedupe_by_name()
    elif mode == "Group by Department":
        run_group_by_department()
    else:
        run_split_members_by_survey(api_key, model, power_label)


if __name__ == "__main__":
    main()
