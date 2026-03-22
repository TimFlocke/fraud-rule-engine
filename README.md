# ACH Fraud Rule Engine

A fraud rule engine for ACH (bank account) transactions that auto-generates interpretable rules from a decision tree, lets you build a three-action strategy (Accept / Refer / Reject), and supports natural-language rule suggestions via Claude API.

Built as a portfolio project for a Senior Risk Analyst interview.

## Features

- **Data Health Dashboard** — row counts, fraud rate, feature distributions, missing value checks, data quality warnings
- **Auto-Generated Rules** — decision tree extraction produces human-readable fraud rules with precision, recall, escalation rate, and plain-English narratives
- **Strategy Builder** — select rules, tune Accept/Refer/Reject thresholds, see real-time impact on fraud catch and customer friction
- **Claude-Powered Rule Suggestions** — describe a fraud pattern in plain English; Claude generates a rule and the engine evaluates it instantly

## Data Specification

Place your CSV at `data/fraud_data.csv`. The engine filters to `PAYMENT_METHOD_TYPE == 'bank_account'`.

| Column | Type | Description |
|--------|------|-------------|
| `EMAIL_RISK_SCORE` | float (0–164, spike at 500) | Third-party email domain risk score. The spike at 500 likely represents missing/unknown data from the vendor. |
| `TRANSFER_AMOUNT_USD` | float | Transaction amount in USD. |
| `ACCOUNT_AGE_AT_PURCHASE_DAYS` | float | Days since account creation. |
| `INTERNATIONAL_PH` | int (0/1) | 1 if international phone, 0 if domestic. |
| `rapid_velocity` | bool | True if < 5 min between this and prior transaction. |
| `prior_transfers` | int | Cumulative transaction count before this one. |
| `prior_unique_phone_cntry` | int | Unique countries the user's phone has been in. |
| `is_fraud` | int (0/1) | Target variable. |

## Run Locally

```bash
# Clone and enter the project
git clone <your-repo-url>
cd fraud-rule-engine

# Install dependencies
pip install -r requirements.txt

# Add your data
cp /path/to/fraud_data.csv data/fraud_data.csv

# Run the app
streamlit run app.py
```

For the "Suggest a Rule" page, set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Deploy to Streamlit Cloud

1. Push to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set `app.py` as the main file
4. Add `ANTHROPIC_API_KEY` in the Secrets section
5. Deploy

Note: You'll need to include your `data/fraud_data.csv` in the repo (or adjust `.gitignore`) for Streamlit Cloud to access it.

## Project Structure

```
fraud-rule-engine/
├── app.py              # Streamlit app (4 pages)
├── rule_engine.py      # FraudRuleEngine: tree extraction, Claude integration
├── metrics.py          # Precision, recall, escalation calculations
├── requirements.txt
├── .gitignore
├── README.md
└── data/
    └── fraud_data.csv  # You provide this
```

## Details

- **Why decision trees?** They produce interpretable rules that risk analysts and compliance teams can review and approve — critical in financial services where model explainability is a regulatory requirement.
- **Three-action system** mirrors real production fraud stacks: hard blocks are expensive (false positives lose customers), so a Refer/friction tier (step-up auth, manual review) captures ambiguous cases without burning good users.
- **Escalation rate** is the metric ops teams actually care about — it determines staffing and queue capacity. The threshold slider lets you simulate operational impact before deploying rules.
- **EMAIL_RISK_SCORE = 500 spike** — called this out as a data quality issue. In practice, you'd work with the vendor to understand the bucketing or impute differently. Demonstrates awareness that feature engineering and data quality matter more than model complexity.
- **Claude API integration** shows how LLMs can accelerate the rule ideation loop — analysts describe patterns in English instead of writing code, lowering the barrier to rapid experimentation.
- **`class_weight='balanced'`** on the tree handles the class imbalance inherent in fraud data without needing SMOTE or other resampling — simpler and more robust for rule extraction.

---

## Future Improvements (V2)

This fraud rule engine is designed to be extended. Planned enhancements include:

### 1. Dollar Impact Analysis
- Display total dollar amount of fraud caught by each rule (not just count)
- Show cost-benefit analysis: "This rule catches $45,000 in fraud while escalating $2,000 in legitimate transfers"
- Help stakeholders understand financial impact of rule selection

### 2. Rule Orthogonality Detection
- Measure overlap between rules to identify redundancy
- Calculate Jaccard similarity: Rules catching identical fraud cases are marked as overlapping
- Recommend complementary rules: "Rule_1 and Rule_3 are orthogonal (0% overlap) - good combination"
- Warn users: "Your selected rules have 65% overlap - consider choosing more complementary rules for better coverage"

### 3. Export Strategy as Python Code
- Generate copy-paste Python code that applies the selected strategy to new data
- Output includes rule definitions, thresholds, and action assignment logic
- Enable non-technical users to deploy strategy: "Copy this code into your production pipeline"

### 4. Dynamic Dataset Upload
- Allow users to upload their own CSV or connect to a data source
- Auto-detect fraud label column and feature types
- Regenerate rules on new datasets in real-time
- Enable the tool to work across different payment methods, institutions, or time periods
