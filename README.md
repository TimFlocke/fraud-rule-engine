# Fraud Rule Engine

A fraud rule engine that auto-generates interpretable rules from a decision tree, lets you build a three-action strategy (Accept / Refer / Reject), and supports natural-language rule suggestions via Claude API. Works with any labeled fraud dataset — a default ACH dataset is included, or upload your own CSV.

## Live Demo

🚀 **[Try the app live here](https://fraud-rule-engine.streamlit.app/)**

No setup required — just click and explore!

## Features

- **Ingest Data** — upload any CSV with an `is_fraud` target column, or use the default ACH dataset. Preview your data, select features for rule generation, and see record and column counts update in real time.
- **Data Health Dashboard** — row counts, fraud rate, feature distributions, missing value checks, and data quality warnings
- **Auto-Generated Rules** — decision tree extraction produces human-readable fraud rules with precision, recall, escalation rate, and plain-English narratives
- **Strategy Builder** — select rules, tune Accept/Refer/Reject thresholds, see real-time impact on fraud catch and customer friction
- **AI Analyst** — powered by [Concentrate.ai](https://concentrate.ai). Two modes: describe a fraud pattern in plain English and the engine generates and evaluates a rule instantly, or ask the analyst a question about your dataset, features, or rules and get a conversational response. Switch between Claude, GPT-4o, and Gemini via a model selector.

## Supported Datasets

The engine works with any CSV that has an `is_fraud` column (0/1). A default ACH bank transfer dataset is loaded automatically. To use your own data, upload a CSV on the Ingest Data tab — the engine will detect your features automatically.

For the default ACH dataset the engine filters to `PAYMENT_METHOD_TYPE == 'bank_account'` and uses the following features:

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
4. Add `ANTHROPIC_API_KEY` and your data in the Secrets section
5. Deploy

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
- **Dynamic dataset support** — the engine auto-detects feature columns and categorical variables from any uploaded dataset. Categorical columns with cardinality below 20 are automatically one-hot encoded for the decision tree.
- **Claude API integration** leverages LLM to accelerate the rule ideation loop — analysts describe patterns in English instead of writing code, lowering the barrier to rapid experimentation. The system prompt is built dynamically from the active dataset's feature columns.
- **`class_weight='balanced'`** on the tree handles class imbalance inherent in fraud data without needing SMOTE or other resampling — simpler and more robust for rule extraction.

---

## Future Improvements

This fraud rule engine is designed to be extended. Planned enhancements include:

### 1. Dollar Impact Analysis
- Display total dollar amount of fraud caught by each rule (not just count)
- Show cost-benefit analysis: "This rule catches $45,000 in fraud while escalating $2,000 in legitimate transactions"
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
