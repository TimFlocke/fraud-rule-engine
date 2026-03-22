import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from rule_engine import (
    load_ach_data, extract_rules, evaluate_strategy, FEATURES, TARGET,
    suggest_rule_via_claude, evaluate_suggested_rule,
    FEATURE_DICTIONARY, FEATURE_QUALITY_WARNINGS,
    ALL_SELECTABLE_FEATURES,
)
from metrics import calc_rule_metrics

# ── Set fonts ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] h1 {
        font-size: 32px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="ACH Fraud Rule Engine", layout="wide")

# ── Title ────────────────────────────────────────────────────────────────────
# st.title("ACH Fraud Rule Engine")

# ── Data loading (cached) ────────────────────────────────────────────────────

@st.cache_data
def get_data():
    return load_ach_data()


@st.cache_data
def get_rules(_df_hash, max_depth, min_leaf, min_split, _feature_key):
    df = get_data()
    features = list(_feature_key)
    return extract_rules(
        df,
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        min_samples_split=min_split,
        feature_columns=features,
    )


def df_hash(df):
    return hash(df.shape)

# ── Session state defaults ───────────────────────────────────────────────────
if "min_samples_split" not in st.session_state:
    st.session_state["min_samples_split"] = 10
if "selected_features" not in st.session_state:
    st.session_state["selected_features"] = list(ALL_SELECTABLE_FEATURES)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("ACH Fraud Rule Engine")
page = st.sidebar.radio("Navigate", ["Data Health", "Rule Discovery", "Strategy Builder", "Suggest a Rule"])

try:
    df = get_data()
except FileNotFoundError:
    st.error("Place your CSV at `data/fraud_data.csv` and reload.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Data Health
# ══════════════════════════════════════════════════════════════════════════════
if page == "Data Health":
    st.header("Data Health")

    col1, col2, col3 = st.columns(3)
    col1.metric("ACH Transactions", f"{len(df):,}")
    fraud_rate = df[TARGET].mean()
    col2.metric("Fraud Rate", f"{fraud_rate:.2%}")
    col3.metric("Fraud Count", f"{int(df[TARGET].sum()):,}")

    # Missing values
    st.subheader("Missing Values")
    missing = df[FEATURES + [TARGET]].isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values in feature columns.")
    else:
        st.dataframe(missing.rename("missing_count").to_frame())

    # Feature explorer
    st.subheader("Feature Explorer")
    feat = st.selectbox("Select feature", ALL_SELECTABLE_FEATURES + [TARGET])

    # Data dictionary description
    if feat in FEATURE_DICTIONARY:
        st.info(f"**{feat}:** {FEATURE_DICTIONARY[feat]}")

    # Feature-specific data quality warning
    if feat in FEATURE_QUALITY_WARNINGS:
        warning_text = FEATURE_QUALITY_WARNINGS[feat](df)
        if warning_text:
            st.warning(warning_text)

    # Distribution chart
    if feat in df.columns:
        if df[feat].nunique() <= 10:
            counts = df[feat].value_counts().reset_index()
            counts.columns = [feat, "count"]
            fig = px.bar(counts, x=feat, y="count", title=f"{feat} Distribution")
        else:
            fig = px.histogram(
                df, x=feat, nbins=50, title=f"{feat} Distribution",
                color=df[TARGET].map({0: "legit", 1: "fraud"}),
                barmode="overlay", opacity=0.6,
            )
            fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Column `{feat}` not found in dataset.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Rule Discovery
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Rule Discovery":
    st.header("Auto-Generated Fraud Rules")

    # ── Feature selector ─────────────────────────────────────────────────────
    st.subheader("Feature Selection")
    selected_features = st.multiselect(
        "Select Features for Rule Generation",
        options=ALL_SELECTABLE_FEATURES,
        default=st.session_state["selected_features"],
        key="feature_multiselect",
    )
    # Persist selection
    st.session_state["selected_features"] = selected_features

    if not selected_features:
        st.warning("Select at least one feature to generate rules.")
        st.stop()

    regenerate = st.button("Regenerate Rules")

    # ── Tree parameters ──────────────────────────────────────────────────────
    with st.expander("Tree parameters", expanded=False):
        max_depth = st.slider("max_depth", 2, 6, 4)
        min_leaf = st.slider("min_samples_leaf", 10, 100, 30)
        st.session_state["min_samples_split"] = st.slider(
            "min_samples_split", 5, 50,
            value=st.session_state["min_samples_split"],
            step=1,
            key="min_split_slider",
            help="Minimum samples required to split an internal node. "
                 "Higher values produce simpler trees with fewer rules.",
        )

    min_split = st.session_state["min_samples_split"]
    feature_key = tuple(sorted(selected_features))

    rules = get_rules(df_hash(df), max_depth, min_leaf, min_split, feature_key)

    if regenerate:
        get_rules.clear()
        rules = get_rules(df_hash(df), max_depth, min_leaf, min_split, feature_key)
        st.success(f"Rules regenerated with {len(selected_features)} features")

    if not rules:
        st.warning("No rules extracted. Try adjusting tree parameters or adding more features.")
        st.stop()

    st.info(
        f"Extracted **{len(rules)}** fraud-leaning rules, sorted by recall. "
        f"Using **{len(selected_features)}** features | min_samples_split={min_split}"
    )

    # Full rule table — no truncation
    display_data = []
    for r in rules:
        display_data.append({
            "Rule Name": r["name"],
            "Rule": r["rule_str"],
            "Narrative": r["narrative"],
            "Precision": f"{r['precision']:.1%}",
            "Recall": f"{r['recall']:.1%}",
            "Escalation Rate": f"{r['escalation_rate']:.1%}",
            "Fraud Caught": r["fraud_caught"],
        })

    st.dataframe(
        pd.DataFrame(display_data),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rule": st.column_config.TextColumn("Rule", width="large"),
            "Narrative": st.column_config.TextColumn("Narrative", width="large"),
        },
    )

    # Detail expander per rule — full text, no truncation
    for r in rules:
        with st.expander(f"{r['name']}: {r['rule_str']}"):
            st.markdown(f"**Narrative:** {r['narrative']}")
            st.markdown(f"**Full rule condition:**")
            st.markdown(f"> {r['rule_str']}")
            st.markdown(f"- Precision: `{r['precision']:.2%}`")
            st.markdown(f"- Recall: `{r['recall']:.2%}`")
            st.markdown(f"- Escalation Rate: `{r['escalation_rate']:.2%}`")
            st.markdown(f"- Fraud Caught: `{r['fraud_caught']}` / `{r['total_fraud']}`")
            st.markdown(f"- Total Flagged: `{r['total_flagged']}` / `{r['total_transactions']}`")
            st.code(r["lambda_str"], language="python")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Strategy Builder
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Strategy Builder":
    st.header("Strategy Builder")
    st.markdown("Select rules, set thresholds, and see real-time Accept / Refer / Reject splits.")

    # ── Explainer section ────────────────────────────────────────────────────
    with st.expander("How Scoring Works", expanded=True):
        st.markdown(
            "**Risk Scoring:** Each selected rule fires (1) or doesn't (0) per transaction. "
            "The **Risk Score** = mean precision of all fired rules. "
            "Example: if Rule_1 (precision 0.87) and Rule_3 (precision 0.92) both fire, "
            "Risk Score = (0.87 + 0.92) / 2 = **0.895**."
        )
        st.markdown(
            "**Set Thresholds:**\n"
            "- **Accept Threshold** — transactions with risk score *below* this are considered safe → **Accept**\n"
            "- **Refer Threshold** — transactions with risk score *between* accept and refer thresholds "
            "are medium risk → **Refer** (add friction: step-up auth, manual review)\n"
            "- Transactions with risk score *at or above* the refer threshold are high risk → **Reject**"
        )

    min_split = st.session_state["min_samples_split"]
    feature_key = tuple(sorted(st.session_state["selected_features"]))
    rules = get_rules(df_hash(df), 4, 30, min_split, feature_key)
    if not rules:
        st.warning("No rules available. Check Rule Discovery page.")
        st.stop()

    # ── Rule selection — full text ───────────────────────────────────────────
    st.subheader("Select Rules")
    selected_indices = []
    cols = st.columns(2)
    for i, r in enumerate(rules):
        col = cols[i % 2]
        if col.checkbox(f"{r['name']}: {r['rule_str']}", key=f"strat_rule_{i}"):
            selected_indices.append(i)

    selected_rules = [rules[i] for i in selected_indices]

    # ── Threshold sliders with keys for real-time reactivity ─────────────────
    st.subheader("Thresholds")
    c1, c2, c3 = st.columns(3)
    accept_threshold = c1.slider(
        "Accept threshold", 0.0, 1.0, 0.3, 0.05,
        key="accept_thresh",
        help="Below this risk score → Accept",
    )
    refer_threshold = c2.slider(
        "Refer threshold", 0.0, 1.0, 0.7, 0.05,
        key="refer_thresh",
        help="Between accept and this → Refer; above → Reject",
    )
    esc_warn = c3.slider(
        "Escalation rate warning", 0.0, 0.5, 0.20, 0.01,
        key="esc_warn_thresh",
        help="Warn if combined escalation exceeds this",
    )

    if refer_threshold <= accept_threshold:
        st.error("Refer threshold must be greater than accept threshold.")
        st.stop()

    # ── Metrics ──────────────────────────────────────────────────────────────
    st.subheader("Strategy Performance")

    if not selected_rules:
        st.info("Select at least one rule above to see strategy metrics.")
    else:
        result = evaluate_strategy(df, selected_rules, accept_threshold, refer_threshold)

        m1, m2, m3 = st.columns(3)
        m1.metric("Accept %", f"{result['accept_pct']:.1%}")
        m2.metric("Refer %", f"{result['refer_pct']:.1%}")
        m3.metric("Reject %", f"{result['reject_pct']:.1%}")

        combined_esc = result["refer_pct"] + result["reject_pct"]
        if combined_esc > esc_warn:
            st.warning(
                f"Combined escalation (Refer + Reject) is **{combined_esc:.1%}**, "
                f"exceeding your {esc_warn:.0%} threshold."
            )

        # Fraud distribution across actions
        st.subheader("Fraud Distribution by Action")
        total_fraud = max(result["total_fraud"], 1)
        fraud_dist = pd.DataFrame([
            {"Action": "Accept", "Fraud": result["fraud_in_accept"],
             "Pct of Total Fraud": result["fraud_in_accept"] / total_fraud},
            {"Action": "Refer", "Fraud": result["fraud_in_refer"],
             "Pct of Total Fraud": result["fraud_in_refer"] / total_fraud},
            {"Action": "Reject", "Fraud": result["fraud_in_reject"],
             "Pct of Total Fraud": result["fraud_in_reject"] / total_fraud},
        ])
        fraud_dist["Pct of Total Fraud"] = fraud_dist["Pct of Total Fraud"].map("{:.1%}".format)
        st.dataframe(fraud_dist, use_container_width=True, hide_index=True)

        fig = px.pie(
            names=["Accept", "Refer", "Reject"],
            values=[result["accept_pct"], result["refer_pct"], result["reject_pct"]],
            title="Transaction Action Split",
            color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"],
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Baseline comparison ──────────────────────────────────────────────
        st.subheader("Baseline Comparison")
        fraud_caught_by_strategy = result["fraud_in_refer"] + result["fraud_in_reject"]
        fraud_caught_pct = fraud_caught_by_strategy / total_fraud

        baseline_data = pd.DataFrame([
            {
                "Scenario": "No Rules (Baseline)",
                "Accept %": "100.0%",
                "Refer %": "0.0%",
                "Reject %": "0.0%",
                "Fraud Caught": "0.0%",
                "Escalation Rate": "0.0%",
            },
            {
                "Scenario": "Your Strategy",
                "Accept %": f"{result['accept_pct']:.1%}",
                "Refer %": f"{result['refer_pct']:.1%}",
                "Reject %": f"{result['reject_pct']:.1%}",
                "Fraud Caught": f"{fraud_caught_pct:.1%}",
                "Escalation Rate": f"{combined_esc:.1%}",
            },
        ])
        st.dataframe(baseline_data, use_container_width=True, hide_index=True)

        st.success(
            f"Your strategy catches **{fraud_caught_pct:.1%}** of fraud "
            f"({fraud_caught_by_strategy} of {result['total_fraud']} cases) "
            f"while escalating **{combined_esc:.1%}** of transactions "
            f"(vs 0% baseline)."
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Suggest a Rule
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Suggest a Rule":
    st.header("Suggest a Rule")
    st.markdown(
        "Describe a fraud pattern in plain English. Claude will generate a rule, "
        "and we'll evaluate it on the ACH dataset."
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        api_key = st.text_input("Anthropic API key", type="password",
                                help="Set ANTHROPIC_API_KEY env var or enter here.")

    user_text = st.text_area(
        "Describe the fraud pattern",
        placeholder="e.g., Flag transactions where EMAIL_RISK_SCORE > 120 and rapid_velocity is true",
    )

    if st.button("Generate & Evaluate Rule", disabled=not (api_key and user_text)):
        with st.spinner("Asking Claude to generate a rule..."):
            try:
                lambda_str = suggest_rule_via_claude(user_text, api_key)
            except Exception as e:
                st.error(f"Claude API error: {e}")
                st.stop()

        st.code(lambda_str, language="python")

        with st.spinner("Evaluating rule on ACH data..."):
            result = evaluate_suggested_rule(df, lambda_str)

        if result is None:
            st.error("Could not evaluate the generated rule. It may have invalid syntax or blocked content.")
        else:
            st.success("Rule evaluated successfully!")
            st.markdown(f"**Narrative:** {result['narrative']}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precision", f"{result['precision']:.1%}")
            c2.metric("Recall", f"{result['recall']:.1%}")
            c3.metric("Escalation Rate", f"{result['escalation_rate']:.1%}")
            c4.metric("Fraud Caught", result["fraud_caught"])

            st.markdown(f"**Rule:** `{result['rule_str']}`")

            if "suggested_rules" not in st.session_state:
                st.session_state["suggested_rules"] = []

            if st.button("Add to Strategy Builder"):
                st.session_state["suggested_rules"].append(result)
                st.success(f"Added **{result['name']}** to your strategy. Switch to Strategy Builder to use it.")

    # Show previously suggested rules
    if st.session_state.get("suggested_rules"):
        st.subheader("Previously Suggested Rules")
        for i, r in enumerate(st.session_state["suggested_rules"]):
            st.markdown(f"**{i+1}.** `{r['rule_str']}` — Precision: {r['precision']:.1%}, Recall: {r['recall']:.1%}")
