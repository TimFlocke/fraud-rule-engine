import numpy as np
import pandas as pd


def calc_rule_metrics(flagged: pd.Series, actual_fraud: pd.Series) -> dict:
    """Calculate precision, recall, escalation rate, and fraud caught for a rule.

    Args:
        flagged: Boolean series — True where the rule fires.
        actual_fraud: Binary series — 1 for fraud, 0 for legit.
    """
    tp = int((flagged & (actual_fraud == 1)).sum())
    fp = int((flagged & (actual_fraud == 0)).sum())
    fn = int((~flagged & (actual_fraud == 1)).sum())
    total = len(actual_fraud)
    total_fraud = int(actual_fraud.sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    escalation_rate = (tp + fp) / total if total > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "escalation_rate": round(escalation_rate, 4),
        "fraud_caught": tp,
        "total_flagged": tp + fp,
        "total_fraud": total_fraud,
        "total_transactions": total,
    }


def calc_strategy_metrics(
    df: pd.DataFrame,
    rule_results: list[dict],
    accept_threshold: float = 0.3,
    refer_threshold: float = 0.7,
) -> dict:
    """Calculate Accept / Refer / Reject splits for a set of selected rules.

    Risk score per transaction = mean precision of all matching rules (0 if none match).
    """
    n = len(df)
    if n == 0:
        return {"accept_pct": 0, "refer_pct": 0, "reject_pct": 0,
                "fraud_in_accept": 0, "fraud_in_refer": 0, "fraud_in_reject": 0}

    precisions = []
    flag_matrix = []
    for r in rule_results:
        precisions.append(r["precision"])
        flag_matrix.append(r["_flagged"])

    if not flag_matrix:
        return {"accept_pct": 1.0, "refer_pct": 0.0, "reject_pct": 0.0,
                "fraud_in_accept": int(df["is_fraud"].sum()),
                "fraud_in_refer": 0, "fraud_in_reject": 0}

    flag_arr = np.column_stack(flag_matrix)  # (n, num_rules)
    prec_arr = np.array(precisions)

    # For each transaction, mean precision of rules that fired
    matched = flag_arr.astype(float)
    weighted = matched * prec_arr[np.newaxis, :]
    sum_prec = weighted.sum(axis=1)
    count_matched = matched.sum(axis=1)
    risk_scores = np.where(count_matched > 0, sum_prec / count_matched, 0.0)

    actions = np.where(
        risk_scores < accept_threshold, "accept",
        np.where(risk_scores < refer_threshold, "refer", "reject"),
    )

    fraud = df["is_fraud"].values
    accept_mask = actions == "accept"
    refer_mask = actions == "refer"
    reject_mask = actions == "reject"

    return {
        "accept_pct": round(accept_mask.sum() / n, 4),
        "refer_pct": round(refer_mask.sum() / n, 4),
        "reject_pct": round(reject_mask.sum() / n, 4),
        "fraud_in_accept": int(fraud[accept_mask].sum()),
        "fraud_in_refer": int(fraud[refer_mask].sum()),
        "fraud_in_reject": int(fraud[reject_mask].sum()),
        "total_fraud": int(fraud.sum()),
        "risk_scores": risk_scores,
        "actions": actions,
    }
