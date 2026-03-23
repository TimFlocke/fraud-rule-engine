import re
import textwrap

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from metrics import calc_rule_metrics, calc_strategy_metrics

FEATURES = [
    "EMAIL_RISK_SCORE",
    "TRANSFER_AMOUNT_USD",
    "ACCOUNT_AGE_AT_PURCHASE_DAYS",
    "INTERNATIONAL_PH",
    "rapid_velocity",
    "prior_transfers",
    "prior_unique_phone_cntry",
]

CATEGORICAL_FEATURES = [
    "EMAIL_DOMAIN_BIN",
    "TRANSFER_TYPE_BIN",
]

ALL_SELECTABLE_FEATURES = FEATURES + CATEGORICAL_FEATURES

TARGET = "is_fraud"

FEATURE_DICTIONARY = {
    "EMAIL_RISK_SCORE": "Third-party vendor fraud risk score for email address (0-164). Higher = riskier. Note: Large spike at 500 suggests missing/unknown data.",
    "TRANSFER_AMOUNT_USD": "Transaction amount in USD. Higher amounts indicate higher risk profile.",
    "ACCOUNT_AGE_AT_PURCHASE_DAYS": "Days since account creation. New accounts = higher fraud risk.",
    "INTERNATIONAL_PH": "1 if user's phone is international, 0 if domestic.",
    "rapid_velocity": "Boolean. True if transaction occurs < 5 minutes after user's prior transaction. Indicates suspicious rapid activity.",
    "prior_transfers": "Cumulative count of transactions user has made before this one. New users = lower count.",
    "prior_unique_phone_cntry": "Count of unique countries user's phone has been located in before this transaction. High values = frequent traveler or potential account compromise.",
    "is_fraud": "Target variable. 1 = fraudulent transaction, 0 = legitimate.",
    "EMAIL_DOMAIN_BIN": "Categorical. Binned email domain (e.g., gmail, yahoo, protonmail, other). Encoded via one-hot for rule generation.",
    "TRANSFER_TYPE_BIN": "Categorical. Transfer type category (e.g., buy, sell, send, receive). Encoded via one-hot for rule generation.",
}

FEATURE_QUALITY_WARNINGS = {
    "EMAIL_RISK_SCORE": lambda df: (
        f"**Data quality warning:** {(df['EMAIL_RISK_SCORE'] == 500).mean():.1%} of transactions have "
        f"EMAIL_RISK_SCORE = 500. This likely represents missing/unknown data from the vendor, "
        f"not genuinely extreme risk. Rules on this feature may have limited discriminative power for those records."
    ) if (df["EMAIL_RISK_SCORE"] == 500).mean() > 0.05 else None,
    "ACCOUNT_AGE_AT_PURCHASE_DAYS": lambda df: (
        f"**Data quality note:** {(df['ACCOUNT_AGE_AT_PURCHASE_DAYS'] == 0).mean():.1%} of transactions "
        f"have account age = 0 days (same-day accounts)."
    ) if (df["ACCOUNT_AGE_AT_PURCHASE_DAYS"] == 0).mean() > 0.01 else None,
    "rapid_velocity": lambda df: (
        f"**Distribution note:** {df['rapid_velocity'].mean():.1%} of transactions have rapid_velocity = True."
    ),
    "INTERNATIONAL_PH": lambda df: (
        f"**Distribution note:** {df['INTERNATIONAL_PH'].mean():.1%} of transactions have an international phone."
    ),
    "EMAIL_DOMAIN_BIN": lambda df: (
        f"**Distribution note:** {df['EMAIL_DOMAIN_BIN'].nunique()} unique email domain bins. "
        f"Top: {df['EMAIL_DOMAIN_BIN'].value_counts().head(3).to_dict()}"
    ) if "EMAIL_DOMAIN_BIN" in df.columns else None,
    "TRANSFER_TYPE_BIN": lambda df: (
        f"**Distribution note:** {df['TRANSFER_TYPE_BIN'].nunique()} unique transfer type bins. "
        f"Top: {df['TRANSFER_TYPE_BIN'].value_counts().head(3).to_dict()}"
    ) if "TRANSFER_TYPE_BIN" in df.columns else None,
}

# Binary features that get == True / == False in rule text
_BINARY_FEATURES = {"rapid_velocity", "INTERNATIONAL_PH"}


def load_ach_data(source: str | pd.DataFrame = "data/fraud_data.csv") -> pd.DataFrame:
    """Load ACH data from a file path or an already-loaded DataFrame."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)
    df = df[df["PAYMENT_METHOD_TYPE"] == "bank_account"].copy()
    df["rapid_velocity"] = df["rapid_velocity"].astype(int)
    df.reset_index(drop=True, inplace=True)
    return df


# ── One-hot encoding helpers ─────────────────────────────────────────────────

def _onehot_encode(df: pd.DataFrame, cat_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encode categorical columns, dropping first category per column.

    Returns the encoded DataFrame and the list of new dummy column names.
    """
    present = [c for c in cat_cols if c in df.columns]
    if not present:
        return pd.DataFrame(index=df.index), []
    dummies = pd.get_dummies(df[present], columns=present, drop_first=True, dtype=int)
    return dummies, list(dummies.columns)


def _is_onehot_col(col_name: str) -> bool:
    """Check if a column name looks like a one-hot encoded categorical."""
    for cat in CATEGORICAL_FEATURES:
        if col_name.startswith(cat + "_"):
            return True
    return False


def _onehot_readable(col_name: str) -> str:
    """Turn 'EMAIL_DOMAIN_BIN_gmail' into 'Gmail domain', etc."""
    for cat in CATEGORICAL_FEATURES:
        prefix = cat + "_"
        if col_name.startswith(prefix):
            value = col_name[len(prefix):]
            value_display = value.replace("_", " ").title()
            if cat == "EMAIL_DOMAIN_BIN":
                return f"{value_display} domain"
            elif cat == "TRANSFER_TYPE_BIN":
                return f"{value_display} transfer type"
            else:
                label = cat.replace("_BIN", "").replace("_", " ").title()
                return f"{value_display} {label}"
    return col_name


# ── Decision-tree rule extraction ────────────────────────────────────────────

def _path_to_rule_str(tree, feature_names, path_nodes):
    """Convert a root-to-leaf path into a human-readable rule string."""
    conditions = []
    for i in range(len(path_nodes) - 1):
        node = path_nodes[i]
        child = path_nodes[i + 1]
        feat = feature_names[tree.feature[node]]
        thresh = tree.threshold[node]

        if feat in _BINARY_FEATURES or _is_onehot_col(feat):
            if child == tree.children_left[node]:
                conditions.append(f"{feat} == False")
            else:
                conditions.append(f"{feat} == True")
        else:
            if child == tree.children_left[node]:
                conditions.append(f"{feat} <= {thresh:.2f}")
            else:
                conditions.append(f"{feat} > {thresh:.2f}")
    return " AND ".join(conditions)


def _path_to_lambda_str(tree, feature_names, path_nodes):
    """Build a pandas-compatible lambda string for eval."""
    parts = []
    for i in range(len(path_nodes) - 1):
        node = path_nodes[i]
        child = path_nodes[i + 1]
        feat = feature_names[tree.feature[node]]
        thresh = tree.threshold[node]

        if feat in _BINARY_FEATURES or _is_onehot_col(feat):
            if child == tree.children_left[node]:
                parts.append(f"(row['{feat}'] == 0)")
            else:
                parts.append(f"(row['{feat}'] == 1)")
        else:
            if child == tree.children_left[node]:
                parts.append(f"(row['{feat}'] <= {thresh:.2f})")
            else:
                parts.append(f"(row['{feat}'] > {thresh:.2f})")
    return "lambda row: " + " & ".join(parts)


_OP_PATTERN = re.compile(
    r'^\s*(?P<feat>[A-Za-z_][A-Za-z0-9_]*)\s*(?P<op>>=|<=|!=|==|>|<)\s*(?P<val>.+?)\s*$'
)


def _parse_condition(condition: str) -> tuple[str, str, str] | None:
    """Extract (feature, operator, value) from a condition string.

    Returns None if the condition can't be parsed.
    """
    m = _OP_PATTERN.match(condition)
    if m:
        return m.group("feat"), m.group("op"), m.group("val")
    return None


def _is_true_check(op: str, val: str) -> bool:
    """Return True if the operator/value pair represents a truthy check."""
    val_lower = val.strip().lower()
    return val_lower in ("true", "1") and op in ("==", ">=", ">")


def _is_false_check(op: str, val: str) -> bool:
    """Return True if the operator/value pair represents a falsy check."""
    val_lower = val.strip().lower()
    return (val_lower in ("false", "0") and op in ("==", "<=", "<")) or \
           (val_lower in ("true", "1") and op in ("!=",))


def _condition_narrative(condition: str) -> str:
    """Turn a single rule condition into a human-readable phrase."""
    try:
        condition = condition.strip()

        # One-hot encoded categorical features
        for cat in CATEGORICAL_FEATURES:
            prefix = cat + "_"
            if prefix in condition:
                parsed = _parse_condition(condition)
                feat_name = parsed[0] if parsed else condition.split()[0]
                readable = _onehot_readable(feat_name)
                if parsed and _is_true_check(parsed[1], parsed[2]):
                    return readable
                elif parsed and _is_false_check(parsed[1], parsed[2]):
                    return f"not {readable}"
                elif "True" in condition or "1" in condition:
                    return readable
                else:
                    return f"not {readable}"

        parsed = _parse_condition(condition)

        # INTERNATIONAL_PH — binary feature, check before numeric parsing
        if "INTERNATIONAL_PH" in condition:
            if parsed and (_is_true_check(parsed[1], parsed[2])):
                return "international phone number"
            elif parsed and (_is_false_check(parsed[1], parsed[2])):
                return "domestic phone number"
            elif "True" in condition:
                return "international phone number"
            return "domestic phone number"

        # rapid_velocity — binary feature
        if "rapid_velocity" in condition:
            if parsed and (_is_true_check(parsed[1], parsed[2])):
                return "rapid successive transactions (< 5 min apart)"
            elif parsed and (_is_false_check(parsed[1], parsed[2])):
                return "normal transaction spacing"
            elif "True" in condition:
                return "rapid successive transactions (< 5 min apart)"
            return "normal transaction spacing"

        # For numeric features, extract operator and value via regex
        if parsed:
            feat, op, val = parsed
            try:
                num_val = float(val)
            except ValueError:
                num_val = None

            is_high = op in (">", ">=")

            if "EMAIL_RISK_SCORE" in feat:
                if is_high:
                    return f"high email address risk score (vendor score {op} {val})"
                return f"low-to-moderate email address risk score (vendor score {op} {val})"

            if "TRANSFER_AMOUNT_USD" in feat:
                display = f"${num_val:,.0f}" if num_val is not None else val
                if is_high:
                    return f"high transfer amount ({op} {display})"
                return f"lower transfer amount ({op} {display})"

            if "ACCOUNT_AGE_AT_PURCHASE_DAYS" in feat:
                display = f"{num_val:.0f}" if num_val is not None else val
                if is_high:
                    return f"established account ({op} {display} days old)"
                return f"new account ({op} {display} days old)"

            if "prior_transfers" in feat:
                display = f"{num_val:.0f}" if num_val is not None else val
                if is_high:
                    return f"some transaction history ({op} {display} prior transfers)"
                return f"very few prior transactions ({op} {display})"

            if "prior_unique_phone_cntry" in feat:
                display = f"{num_val:.0f}" if num_val is not None else val
                if is_high:
                    return f"multi-country phone activity ({op} {display} countries)"
                return f"limited phone country diversity ({op} {display} countries)"

            # Generic numeric condition for unknown features
            display = f"{num_val:.2f}" if num_val is not None else val
            readable_feat = feat.replace("_", " ").lower()
            return f"{readable_feat} {op} {display}"

        # Fallback: return the raw condition
        return condition

    except Exception:
        return condition


def _generate_narrative(rule_str: str, metrics: dict) -> str:
    """Create a 2-3 sentence narrative explaining the fraud pattern."""
    try:
        # Split on AND / OR, handling both upper and mixed case
        parts = re.split(r'\s+AND\s+|\s+OR\s+', rule_str, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]

        if not parts:
            return "Custom fraud detection rule based on user suggestion."

        descriptions = []
        for p in parts:
            desc = _condition_narrative(p)
            if desc:
                descriptions.append(desc)

        if not descriptions:
            return "Custom fraud detection rule based on user suggestion."

        pattern = ", ".join(descriptions[:3])
        if len(descriptions) > 3:
            pattern += f", and {len(descriptions) - 3} more signal(s)"

        prec = metrics.get("precision", 0)
        esc = metrics.get("escalation_rate", 0)

        sentence1 = f"This rule targets transactions with {pattern}."

        if prec >= 0.5:
            sentence2 = f"With {prec:.0%} precision, it reliably identifies fraud when triggered."
        elif prec >= 0.2:
            sentence2 = f"At {prec:.0%} precision, it balances detection against false positives."
        else:
            sentence2 = (
                f"Precision is {prec:.0%}, so expect some false positives "
                f"— best used as a friction gate (Refer) rather than a hard block."
            )

        if esc > 0.15:
            sentence3 = (
                f"Escalation rate of {esc:.1%} is elevated; "
                f"consider pairing with other signals to reduce customer friction."
            )
        else:
            sentence3 = f"Low escalation rate ({esc:.1%}) means minimal impact on legitimate users."

        return f"{sentence1} {sentence2} {sentence3}"

    except Exception:
        return "Custom fraud detection rule based on user suggestion."


def _prepare_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Build the feature matrix, one-hot encoding any categorical columns.

    Returns (X DataFrame with all columns, list of actual column names used).
    """
    numeric_cols = [c for c in feature_columns if c not in CATEGORICAL_FEATURES]
    cat_cols = [c for c in feature_columns if c in CATEGORICAL_FEATURES]

    parts = []
    all_col_names = []

    if numeric_cols:
        parts.append(df[numeric_cols])
        all_col_names.extend(numeric_cols)

    if cat_cols:
        dummies, dummy_names = _onehot_encode(df, cat_cols)
        if dummy_names:
            parts.append(dummies)
            all_col_names.extend(dummy_names)

    if not parts:
        return pd.DataFrame(index=df.index), []

    X = pd.concat(parts, axis=1)
    return X, all_col_names


def extract_rules(
    df: pd.DataFrame,
    max_depth: int = 4,
    min_samples_leaf: int = 30,
    min_samples_split: int = 10,
    feature_columns: list[str] | None = None,
) -> list[dict]:
    """Train a decision tree and extract fraud-leaning leaf paths as rules."""
    feature_columns = feature_columns or FEATURES

    X, actual_feature_names = _prepare_features(df, feature_columns)
    if X.empty:
        return []
    y = df[TARGET].values

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X, y)

    tree = clf.tree_
    feature_names = actual_feature_names

    # Walk every root-to-leaf path
    def _walk(node, path):
        if tree.children_left[node] == tree.children_right[node]:
            yield path + [node]
        else:
            yield from _walk(tree.children_left[node], path + [node])
            yield from _walk(tree.children_right[node], path + [node])

    # We need the full X (with one-hot cols) attached to df for lambda eval
    df_eval = pd.concat([df, X[[c for c in actual_feature_names if c not in df.columns]]], axis=1)

    rules = []
    for idx, leaf_path in enumerate(_walk(0, [])):
        leaf = leaf_path[-1]
        counts = tree.value[leaf][0]
        total_in_leaf = counts.sum()
        fraud_in_leaf = counts[1]
        leaf_fraud_rate = fraud_in_leaf / total_in_leaf if total_in_leaf > 0 else 0

        base_rate = y.mean()
        if leaf_fraud_rate <= base_rate:
            continue

        rule_str = _path_to_rule_str(tree, feature_names, leaf_path)
        lambda_str = _path_to_lambda_str(tree, feature_names, leaf_path)

        if not rule_str:
            continue

        fn = eval(lambda_str)
        flagged = df_eval.apply(fn, axis=1).astype(bool)
        metrics = calc_rule_metrics(flagged, df[TARGET])
        narrative = _generate_narrative(rule_str, metrics)

        rules.append({
            "name": f"DT_Rule_{idx + 1}",
            "rule_str": rule_str,
            "lambda_str": lambda_str,
            "narrative": narrative,
            **metrics,
            "_flagged": flagged.values,
        })

    rules.sort(key=lambda r: r["recall"], reverse=True)

    for i, r in enumerate(rules):
        r["name"] = f"Rule_{i + 1}"

    return rules


# ── Suggested rule helpers ────────────────────────────────────────────────────

def add_suggested_rule(rule_dict: dict, suggested_rules: list[dict]) -> list[dict]:
    """Append a suggested rule to the suggested rules list.

    The rule_dict should have the same structure as auto-generated rules:
    name, rule_str, lambda_str, narrative, precision, recall,
    escalation_rate, fraud_caught, and _flagged.
    """
    suggested_rules.append(rule_dict)
    return suggested_rules


# ── Strategy evaluation ──────────────────────────────────────────────────────

def evaluate_strategy(
    df: pd.DataFrame,
    selected_rules: list[dict],
    accept_threshold: float = 0.3,
    refer_threshold: float = 0.7,
) -> dict:
    return calc_strategy_metrics(df, selected_rules, accept_threshold, refer_threshold)


# ── Claude API rule suggestion ───────────────────────────────────────────────

CLAUDE_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a fraud analyst. Given a user's description of a fraud pattern,
    generate a Python lambda rule that evaluates a single pandas DataFrame row.

    Available columns: EMAIL_RISK_SCORE, TRANSFER_AMOUNT_USD,
    ACCOUNT_AGE_AT_PURCHASE_DAYS, INTERNATIONAL_PH, rapid_velocity,
    prior_transfers, prior_unique_phone_cntry, is_fraud.

    Categorical one-hot columns (value is 0 or 1):
    EMAIL_DOMAIN_BIN_<value> (e.g., EMAIL_DOMAIN_BIN_gmail, EMAIL_DOMAIN_BIN_yahoo)
    TRANSFER_TYPE_BIN_<value> (e.g., TRANSFER_TYPE_BIN_buy, TRANSFER_TYPE_BIN_sell)

    Return ONLY the lambda function on one line, nothing else.
    Example: lambda row: (row['EMAIL_RISK_SCORE'] > 100) & (row['rapid_velocity'] == 1)
""")


def suggest_rule_via_claude(user_text: str, api_key: str) -> str:
    """Call Claude API to parse a natural-language rule suggestion into a lambda."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        system=CLAUDE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_text}],
    )
    return message.content[0].text.strip()


def evaluate_suggested_rule(df: pd.DataFrame, lambda_str: str) -> dict | None:
    """Evaluate a lambda string against the dataset and return metrics + narrative."""
    if not lambda_str.startswith("lambda row:"):
        return None

    blocked = ["import", "__", "exec", "eval", "open", "os.", "sys.", "subprocess"]
    lower = lambda_str.lower()
    for b in blocked:
        if b in lower:
            return None

    # Build eval df with one-hot columns available
    dummies, _ = _onehot_encode(df, CATEGORICAL_FEATURES)
    df_eval = pd.concat([df, dummies[[c for c in dummies.columns if c not in df.columns]]], axis=1)

    try:
        fn = eval(lambda_str)
        flagged = df_eval.apply(fn, axis=1).astype(bool)
    except Exception:
        return None

    metrics = calc_rule_metrics(flagged, df[TARGET])

    body = lambda_str.replace("lambda row:", "").strip()
    rule_str = body.replace("row['", "").replace("']", "").replace("(", "").replace(")", "")
    rule_str = rule_str.replace(" & ", " AND ").replace(" | ", " OR ")

    narrative = _generate_narrative(rule_str, metrics)

    return {
        "name": "Suggested_Rule",
        "rule_str": body,
        "lambda_str": lambda_str,
        "narrative": narrative,
        **metrics,
        "_flagged": flagged.values,
    }
