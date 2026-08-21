"""
UniHack - Product Intelligence Enrichment Pipeline
----------------------------------------------------
Takes raw distributor catalog rows (Mfg_Part_Num, Part_Desc, E1_Brand,
Unilog_Brand, DIB_Brand, Part_Manuf) and enriches them into a standardized,
LOV-aware, search-ready record schema -- scoped to the fields judges can
actually verify against the ground-truth 200-item Delivery Format file.

Swap the API call in call_llm() for a different provider if needed --
the rest of the pipeline (cleaning, matching, scoring) doesn't change.
"""

import streamlit as st
import pandas as pd
import json
import os
import re
import difflib
import requests

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="UniHack — Product Intelligence", page_icon="⚡", layout="wide")

API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"

PLACEHOLDER_VALUES = {
    "-- unbranded --", "-- no unilog brand --", "-- no dib brand --",
    "", "n/a", "na", "none", "null",
}

# The fields we actually try to generate + score.
# Kept deliberately narrower than the full 250-column schema so every
# field can be checked against ground truth -- depth over breadth.
TARGET_FIELDS = [
    "Classpath",
    "Invoice_Desc",     # <=40 char, CAPS
    "Mobile_Desc",      # 60-80 char
    "Short_Desc",
    "Long_Desc",
    "Attribute_1_Label", "Attribute_1_Value", "Attribute_1_UOM",
    "Attribute_2_Label", "Attribute_2_Value", "Attribute_2_UOM",
    "Attribute_3_Label", "Attribute_3_Value", "Attribute_3_UOM",
]

# Maps our internal field names to the ACTUAL Delivery Format column headers
# (confirmed from the real spec file -- these use different casing/spacing
# than a naive guess would, e.g. a space before the attribute number, not
# an underscore, and ALL CAPS for most description fields).
GT_COLUMN_MAP = {
    "Classpath": "Classpath",
    "Invoice_Desc": "INVOICE_DESC",
    "Mobile_Desc": "MOBILE_DESC",
    "Short_Desc": "SHORT_DESC",
    "Long_Desc": "LONG_DESC1",
    "Attribute_1_Label": "ATTRIBUTE_LABEL 1",
    "Attribute_1_Value": "ATTRIBUTE_VALUE 1",
    "Attribute_1_UOM": "ATTRIBUTE_UOM 1",
    "Attribute_2_Label": "ATTRIBUTE_LABEL 2",
    "Attribute_2_Value": "ATTRIBUTE_VALUE 2",
    "Attribute_2_UOM": "ATTRIBUTE_UOM 2",
    "Attribute_3_Label": "ATTRIBUTE_LABEL 3",
    "Attribute_3_Value": "ATTRIBUTE_VALUE 3",
    "Attribute_3_UOM": "ATTRIBUTE_UOM 3",
}

PROMPT_TEMPLATE = """You are a product data enrichment assistant for an industrial distribution catalog (Unilog).

You will be given ONE raw catalog row. Produce a JSON object with EXACTLY these keys:
- "Classpath": best-guess category path using ">" separators, e.g. "Power Tool Accessories>Abrasives>Cut-Off Discs". Null if you truly cannot infer one.
- "Invoice_Desc": ALL CAPS, <= 40 characters, abbreviated (this is what prints on an invoice line).
- "Mobile_Desc": 60-80 characters, plain sentence-case, brand + product type + key identifier.
- "Short_Desc": a product-title-style description (brand, series/model, key attributes).
- "Long_Desc": a fuller sentence-case description including any dimensions/specs found in the input.
- "Attribute_1_Label" / "Attribute_1_Value" / "Attribute_1_UOM": the single most important spec found (e.g. Label="Diameter", Value="12", UOM="in"). UOM must be a standard abbreviation (in, mm, V, A, W, dBA, lb, etc.), or null if the value has no unit.
- "Attribute_2_Label" / "Attribute_2_Value" / "Attribute_2_UOM": second most important spec, same rules.
- "Attribute_3_Label" / "Attribute_3_Value" / "Attribute_3_UOM": third most important spec, same rules.

STRICT RULES:
- Every value must be traceable to the input text. NEVER invent a spec, brand claim, or certification that isn't stated or strongly implied by the part number/description.
- If a field cannot be determined from the input, set it to null. Null is correct and expected often -- do not guess to fill a slot.
- Do not include units inside the Value field (Value="12", UOM="in" -- not Value="12 in").

Raw catalog row:
Mfg_Part_Num: {MPN}
Part_Desc: {DESC}
Manufacturer (raw): {MANUF}

Respond with ONLY the JSON object, no markdown fences, no commentary.
"""


# ============================================================
# CLEANING / NORMALIZATION
# ============================================================
def clean_placeholder(value) -> str:
    """Blank out known placeholder strings so they never reach the model or output."""
    if value is None:
        return ""
    v = str(value).strip()
    return "" if v.lower() in PLACEHOLDER_VALUES else v


def strip_manufacturer_code(raw_manuf: str) -> str:
    """'Freud Inc (2435)' -> 'Freud Inc' -- pulls the trailing code out so
    fuzzy matching against a brand master list isn't thrown off by it."""
    return re.sub(r"\s*\([A-Za-z0-9]+\)\s*$", "", raw_manuf or "").strip()


def match_manufacturer(raw_manuf: str, master_names: list[str]) -> tuple[str, float]:
    """Fuzzy-match a raw manufacturer string against an approved master list.
    Returns (best_match_or_original, similarity_score 0-1). If no master
    list is loaded, returns the cleaned raw string with score 0 (unverified)."""
    cleaned = strip_manufacturer_code(raw_manuf)
    if not master_names or not cleaned:
        return cleaned, 0.0
    match = difflib.get_close_matches(cleaned, master_names, n=1, cutoff=0.6)
    if match:
        score = difflib.SequenceMatcher(None, cleaned.lower(), match[0].lower()).ratio()
        return match[0], round(score, 2)
    return cleaned, 0.0


# ============================================================
# LLM CALL
# ============================================================
def call_llm(mpn: str, desc: str, manuf: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(MPN=mpn, DESC=desc, MANUF=manuf)
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        return json.loads(raw)
    except Exception as e:
        return {field: None for field in TARGET_FIELDS} | {"error": str(e)}


def needs_review(record: dict) -> tuple[bool, list[str]]:
    """Rule-based review flag -- NOT a self-reported LLM confidence score."""
    issues = []
    if not record.get("Classpath"):
        issues.append("No classpath inferred")
    if record.get("Invoice_Desc") and len(record["Invoice_Desc"]) > 40:
        issues.append("Invoice_Desc exceeds 40 char limit")
    if not any(record.get(f"Attribute_{i}_Label") for i in (1, 2, 3)):
        issues.append("No attributes extracted")
    if record.get("error"):
        issues.append(f"Extraction error: {record['error']}")
    return len(issues) > 0, issues


# ============================================================
# SCORING AGAINST GROUND TRUTH
# ============================================================
# Fields where exact string match is a fair test (short, structured, low
# wording freedom). Description fields are excluded from "accuracy" scoring
# because two correct sentences can be worded differently -- those need a
# human eyeball pass instead, so we surface them side-by-side rather than
# score them as right/wrong.
EXACT_MATCH_FIELDS = [
    "Classpath",
    "Attribute_1_Label", "Attribute_1_Value", "Attribute_1_UOM",
    "Attribute_2_Label", "Attribute_2_Value", "Attribute_2_UOM",
    "Attribute_3_Label", "Attribute_3_Value", "Attribute_3_UOM",
]
FREE_TEXT_FIELDS = ["Invoice_Desc", "Mobile_Desc", "Short_Desc", "Long_Desc"]


def score_against_ground_truth(results_df: pd.DataFrame, truth_df: pd.DataFrame, key_col: str):
    """Returns (score_df, side_by_side_df). score_df is exact-match % for
    structured fields only. side_by_side_df lets you manually compare the
    free-text description fields, which shouldn't be scored as exact match."""
    truth_renamed = truth_df.rename(columns={v: k for k, v in GT_COLUMN_MAP.items() if v in truth_df.columns})
    merged = results_df.merge(truth_renamed, on=key_col, suffixes=("_gen", "_truth"))

    rows = []
    for col in EXACT_MATCH_FIELDS:
        gen_col, truth_col = f"{col}_gen", f"{col}_truth"
        if gen_col not in merged.columns or truth_col not in merged.columns:
            continue
        gen = merged[gen_col].astype(str).str.strip().str.lower()
        truth = merged[truth_col].astype(str).str.strip().str.lower()
        match_rate = (gen == truth).mean() if len(merged) else 0.0
        rows.append({"Field": col, "Rows Compared": len(merged), "Exact Match %": round(match_rate * 100, 1)})
    score_df = pd.DataFrame(rows)

    side_cols = [key_col]
    for col in FREE_TEXT_FIELDS:
        for suffix in ("_gen", "_truth"):
            c = f"{col}{suffix}"
            if c in merged.columns:
                side_cols.append(c)
    side_by_side_df = merged[side_cols] if len(merged) else pd.DataFrame()

    return score_df, side_by_side_df


# ============================================================
# UI
# ============================================================
def apply_theme() -> None:
    """Style the dashboard using Streamlit's built-in Light/Dark theme tokens."""
    colors = {
        "bg": "var(--background-color)",
        "surface": "var(--secondary-background-color)",
        "surface_alt": "color-mix(in srgb, var(--secondary-background-color) 82%, var(--background-color))",
        "text": "var(--text-color)",
        "muted": "color-mix(in srgb, var(--text-color) 72%, transparent)",
        "border": "color-mix(in srgb, var(--text-color) 18%, transparent)",
        "accent": "var(--primary-color)",
        "accent_dark": "color-mix(in srgb, var(--primary-color) 18%, var(--secondary-background-color))",
        "shadow": "color-mix(in srgb, var(--text-color) 12%, transparent)",
    }

    st.markdown(f"""
    <style>
      html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], section[data-testid="stMain"] {{ background: {colors['bg']} !important; color: {colors['text']} !important; }}
      [data-testid="stHeader"] {{ background: {colors['bg']} !important; }}
      [data-testid="stSidebar"] {{ background: {colors['surface']}; border-right: 1px solid {colors['border']}; }}
      [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {{ color: {colors['muted']} !important; }}
      [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: {colors['text']} !important; }}
      [data-testid="stSidebarCollapsedControl"], [data-testid="stSidebarCollapseButton"] {{ opacity: 1 !important; }}
      [data-testid="stSidebarCollapsedControl"] button, [data-testid="stSidebarCollapseButton"] button {{ background: {colors['accent_dark']} !important; color: {colors['text']} !important; border: 1px solid {colors['border']} !important; border-radius: 12px !important; opacity: 1 !important; }}
      [data-testid="stSidebarCollapsedControl"] svg, [data-testid="stSidebarCollapseButton"] svg {{ fill: {colors['text']} !important; stroke: {colors['text']} !important; opacity: 1 !important; }}
      .block-container {{ max-width: 1280px; padding-top: 4.25rem; padding-bottom: 3rem; }}
      .brand-bar {{ display: flex; align-items: center; gap: 13px; margin-bottom: .3rem; }}
      .brand-mark {{ width: 38px; height: 38px; display: inline-flex; align-items: center; justify-content: center;
        border-radius: 11px; background: {colors['accent']}; color: #ffffff; font-weight: 800; font-size: 19px; }}
      .brand-name {{ color: {colors['text']}; font-size: 1.04rem; font-weight: 740; letter-spacing: -.02em; }}
      .brand-subtitle {{ color: {colors['muted']}; font-size: .8rem; margin-top: 1px; }}
      .hero {{ background: {colors['surface']}; border: 1px solid {colors['border']}; border-radius: 18px;
        padding: 28px 30px; margin: 1.2rem 0 20px; box-shadow: 0 12px 32px {colors['shadow']}; }}
      .eyebrow {{ color: {colors['accent']}; font-size: .73rem; font-weight: 760; letter-spacing: .10em; text-transform: uppercase; }}
      .hero h1 {{ color: {colors['text']}; font-size: 2rem; letter-spacing: -.045em; margin: 7px 0 7px; }}
      .hero p {{ color: {colors['muted']}; margin: 0; font-size: .98rem; }}
      .section-label {{ color: {colors['text']}; font-size: 1.08rem; font-weight: 700; margin: 20px 0 6px; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 28px; border-bottom: 1px solid {colors['border']}; }}
      .stTabs [data-baseweb="tab"] {{ color: {colors['muted']}; font-weight: 600; padding: 13px 2px; }}
      .stTabs [aria-selected="true"] {{ color: {colors['accent']} !important; }}
      .stTabs [data-baseweb="tab-highlight"] {{ background-color: {colors['accent']}; }}
      [data-testid="stMetric"] {{ background: {colors['surface']}; border: 1px solid {colors['border']}; border-radius: 13px; padding: 15px; }}
      [data-testid="stMetricLabel"] {{ color: {colors['muted']} !important; }}
      [data-testid="stMetricValue"] {{ color: {colors['text']} !important; }}
      .stButton > button, [data-testid="stDownloadButton"] > button {{ background: {colors['accent']} !important; color: #ffffff !important; border: 0; border-radius: 9px; font-weight: 700; }}
      .stButton > button *, [data-testid="stDownloadButton"] > button * {{ color: #ffffff !important; fill: #ffffff !important; }}
      .stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {{ background: {colors['accent']} !important; color: #ffffff !important; filter: brightness(1.08); }}
      [data-testid="stFileUploader"] {{ background: {colors['surface_alt']}; border-radius: 12px; padding: 10px; }}
      [data-testid="stFileUploader"] section {{ border-color: {colors['border']}; background: transparent; }}
      [data-testid="stFileUploader"] button {{ background: {colors['surface']} !important; color: {colors['text']} !important; border: 1px solid {colors['border']} !important; }}
      [data-testid="stFileUploader"] small {{ color: {colors['muted']} !important; }}
      input, textarea, [data-baseweb="input"] input {{ background: {colors['surface_alt']} !important; color: {colors['text']} !important; -webkit-text-fill-color: {colors['text']} !important; }}
      [data-baseweb="input"], [data-baseweb="base-input"], [data-testid="stNumberInput"] > div > div {{ background: {colors['surface_alt']} !important; border-color: {colors['border']} !important; }}
      [data-testid="stNumberInput"] button {{ background: {colors['surface_alt']} !important; color: {colors['text']} !important; }}
      [data-testid="stToggle"] label span {{ color: {colors['text']} !important; }}
      div[data-testid="stDataFrame"] {{ border: 1px solid {colors['border']}; border-radius: 12px; overflow: hidden; }}
      [data-testid="stDataFrame"] * {{ color: {colors['text']}; }}
      .stAlert {{ border-radius: 10px; color: {colors['text']}; }}
      p, label, .stCaption, [data-testid="stWidgetLabel"] p {{ color: {colors['muted']}; }}
      [data-testid="stFileUploader"] label, [data-testid="stNumberInput"] label {{ color: {colors['text']} !important; }}
    </style>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="brand-bar">
  <div class="brand-mark">U</div>
  <div><div class="brand-name">UniHack</div><div class="brand-subtitle">Product intelligence workspace</div></div>
</div>
""", unsafe_allow_html=True)

apply_theme()

st.markdown("""
<div class="hero">
  <div class="eyebrow">Catalog operations</div>
  <h1>Turn raw product data into ready-to-use intelligence.</h1>
  <p>Enrich catalog records, standardize manufacturer data, and validate delivery-quality output from one focused workspace.</p>
</div>
""", unsafe_allow_html=True)

if not API_KEY:
    st.error("GROQ_API_KEY is not configured. Add it to Streamlit Secrets or your local environment to run enrichment.")
    st.stop()

with st.sidebar:
    st.markdown("### Reference data")
    manuf_file = st.file_uploader("Manufacturer / brand master list", type=["xlsx", "csv"])
    st.caption("Optional. Used to normalize Part_Manuf against approved names.")

master_names = []
if manuf_file is not None:
    df_master = pd.read_excel(manuf_file) if manuf_file.name.endswith("xlsx") else pd.read_csv(manuf_file)
    name_col = next((c for c in df_master.columns if "manufacturer_name" in c.lower() or "brand_name" in c.lower()), df_master.columns[0])
    master_names = df_master[name_col].dropna().astype(str).unique().tolist()
    st.sidebar.success(f"Loaded {len(master_names)} approved manufacturer/brand names.")

tab_run, tab_score = st.tabs(["Run Extraction", "Accuracy Report"])

with tab_run:
    st.markdown('<div class="section-label">Create an enrichment run</div>', unsafe_allow_html=True)
    st.caption("Upload a catalog with Mfg_Part_Num, Part_Desc, and Part_Manuf. Each processed row makes one API call.")
    upload_col, settings_col = st.columns([2.2, 1])
    with upload_col:
        catalog_file = st.file_uploader("Catalog file", type=["xlsx", "csv"])
    with settings_col:
        row_limit = st.number_input("Rows to process", min_value=1, max_value=1000, value=10)

    if catalog_file is not None:
        df = pd.read_excel(catalog_file) if catalog_file.name.endswith("xlsx") else pd.read_csv(catalog_file)
        summary_a, summary_b, summary_c = st.columns(3)
        summary_a.metric("Catalog rows", f"{len(df):,}")
        summary_b.metric("Rows in this run", f"{min(len(df), int(row_limit)):,}")
        summary_c.metric("Master list", f"{len(master_names):,}" if master_names else "Not loaded")
        st.markdown('<div class="section-label">Source preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(5))

        if st.button("Extract Intelligence", type="primary"):
            subset = df.head(int(row_limit)).copy()
            results = []
            progress = st.progress(0.0, text="Starting...")

            for i, row in subset.iterrows():
                mpn = clean_placeholder(row.get("Mfg_Part_Num"))
                desc = clean_placeholder(row.get("Part_Desc"))
                manuf_raw = clean_placeholder(row.get("Part_Manuf"))
                manuf_matched, manuf_confidence = match_manufacturer(manuf_raw, master_names)

                extracted = call_llm(mpn, desc, manuf_raw)
                record = {"Mfg_Part_Num": mpn, "Part_Desc": desc,
                          "Manufacturer_Normalized": manuf_matched,
                          "Manufacturer_Match_Score": manuf_confidence}
                for field in TARGET_FIELDS:
                    record[field] = extracted.get(field)

                flagged, issues = needs_review(record)
                record["Needs_Review"] = flagged
                record["Review_Reasons"] = "; ".join(issues)
                results.append(record)

                progress.progress((i + 1) / len(subset), text=f"Processed {i + 1}/{len(subset)}")

            results_df = pd.DataFrame(results)
            st.session_state["results_df"] = results_df
            st.success(f"Done. {results_df['Needs_Review'].sum()} of {len(results_df)} rows flagged for review.")
            st.markdown('<div class="section-label">Enriched records</div>', unsafe_allow_html=True)
            st.dataframe(results_df)

            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download results CSV", csv, "enriched_output.csv", "text/csv")

with tab_score:
    st.markdown('<div class="section-label">Validate against delivery format</div>', unsafe_allow_html=True)
    st.caption("Upload the known-good Delivery Format sheet to review structured-field accuracy and compare generated descriptions.")
    truth_file = st.file_uploader("Ground truth file", type=["xlsx", "csv"], key="truth")

    if "results_df" not in st.session_state:
        st.info("Run an extraction in the first tab before scoring.")
    elif truth_file is not None:
        truth_df = pd.read_excel(truth_file) if truth_file.name.endswith("xlsx") else pd.read_csv(truth_file)
        score_df, side_by_side_df = score_against_ground_truth(
            st.session_state["results_df"], truth_df, key_col="Mfg_Part_Num"
        )
        if score_df.empty:
            st.warning(
                "No matching rows found. Double-check: (1) both files actually share "
                "Mfg_Part_Num values -- e.g. you extracted from the 1000-item file but "
                "uploaded the 200-item ground truth, which won't overlap much, and "
                "(2) the ground-truth file's column headers match GT_COLUMN_MAP in the code."
            )
        else:
            st.subheader("Structured fields (exact match)")
            st.dataframe(score_df)
            st.metric("Overall structured-field accuracy", f"{score_df['Exact Match %'].mean():.1f}%")

            st.subheader("Description fields (compare manually -- wording can differ and still be correct)")
            st.dataframe(side_by_side_df)
