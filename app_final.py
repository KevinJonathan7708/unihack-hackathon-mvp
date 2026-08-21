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
    """Apply layout-only styling; Streamlit owns all theme colors."""
    st.markdown(f"""
    <style>
      .block-container {{ max-width: 1280px; padding-top: 4.25rem; padding-bottom: 3rem; }}
      [data-testid="stSidebar"] {{ border-right: 1px solid rgba(128, 128, 128, .22); }}
      [data-testid="stSidebarCollapsedControl"], [data-testid="stSidebarCollapseButton"] {{ opacity: 1 !important; }}
      [data-testid="stSidebarCollapsedControl"] button, [data-testid="stSidebarCollapseButton"] button {{ background: rgba(127, 127, 127, .16) !important; border: 1px solid rgba(127, 127, 127, .32) !important; border-radius: 12px !important; opacity: 1 !important; }}
      [data-testid="stSidebarCollapsedControl"] svg, [data-testid="stSidebarCollapseButton"] svg {{ opacity: 1 !important; }}
      .brand-bar {{ display: flex; align-items: center; gap: 13px; margin-bottom: 1.2rem; }}
      .brand-mark {{ width: 40px; height: 40px; display: inline-flex; align-items: center; justify-content: center;
        border-radius: 11px; overflow: hidden; }}
      .brand-mark img {{ width: 100%; height: 100%; object-fit: contain; }}
      .brand-name {{ font-size: 1.04rem; font-weight: 740; letter-spacing: -.02em; }}
      .brand-subtitle {{ font-size: .8rem; margin-top: 1px; opacity: .72; }}
      .hero {{ background: rgba(127, 127, 127, .07); border: 1px solid rgba(127, 127, 127, .22); border-radius: 18px;
        padding: 28px 30px; margin: 0 0 20px; box-shadow: 0 12px 32px rgba(0, 0, 0, .08); }}
      .eyebrow {{ color: #217a50; font-size: .73rem; font-weight: 760; letter-spacing: .10em; text-transform: uppercase; }}
      .hero h1 {{ font-size: 2rem; letter-spacing: -.045em; margin: 7px 0 7px; }}
      .hero p {{ margin: 0; font-size: .98rem; opacity: .76; }}
      .section-label {{ font-size: 1.08rem; font-weight: 700; margin: 20px 0 6px; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 28px; border-bottom: 1px solid rgba(127, 127, 127, .22); }}
      .stTabs [data-baseweb="tab"] {{ font-weight: 600; padding: 13px 2px; }}
      [data-testid="stMetric"] {{ border: 1px solid rgba(127, 127, 127, .22); border-radius: 13px; padding: 15px; }}
      [data-testid="stFileUploader"] {{ border-radius: 12px; padding: 10px; }}
      [data-testid="stFileUploader"] section {{ border-color: rgba(127, 127, 127, .32); background: transparent; }}
      div[data-testid="stDataFrame"] {{ border: 1px solid rgba(127, 127, 127, .22); border-radius: 12px; overflow: hidden; }}
      .stAlert {{ border-radius: 10px; }}
    </style>
    """, unsafe_allow_html=True)


LOGO_B64 = "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"

st.markdown(f"""
<div class="brand-bar">
  <div class="brand-mark"><img src="data:image/png;base64,{LOGO_B64}" alt="Cookie Warrior logo" /></div>
  <div><div class="brand-name">Cookie Warrior</div><div class="brand-subtitle">Product intelligence workspace</div></div>
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
