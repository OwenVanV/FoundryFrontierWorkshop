"""
============================================================================
MODULE 05 — Evaluation Framework
============================================================================

WORKSHOP NARRATIVE:
    You've used Azure OpenAI to investigate a fraud ring and generated
    a comprehensive report. But how ACCURATE were the findings?

    In production AI systems, you can't just trust the model's output —
    you need to MEASURE it. This module introduces the evaluation
    framework: comparing LLM-generated findings against ground truth
    to calculate precision, recall, and F1 scores.

    This is the bridge between "cool AI demo" and "production-ready
    fraud detection system." Without evaluation, you're flying blind.

LEARNING OBJECTIVES:
    1. Understand precision, recall, and F1 for fraud detection
    2. Compare LLM findings against ground truth data
    3. Analyze false positives and false negatives
    4. Test prompt variants for detection improvement
    5. Use Azure AI Evaluation patterns for systematic quality measurement

AZURE SERVICES USED:
    - Azure OpenAI (evaluation analysis, prompt variant testing)
    - Azure AI Evaluation SDK patterns
    - OpenTelemetry (evaluation metrics tracking)

KEY CONCEPTS:
    - PRECISION: Of the accounts the LLM flagged as fraud, how many actually are?
      High precision = few false positives = fewer innocent accounts frozen.
    - RECALL: Of the actual fraud ring members, how many did the LLM find?
      High recall = few false negatives = fewer fraudsters escaping.
    - F1 SCORE: Harmonic mean of precision and recall. Balances both concerns.

ESTIMATED TIME: 20-25 minutes

============================================================================
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# PATH SETUP
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.azure_client import call_openai, call_openai_json, SYSTEM_PROMPTS, get_total_tokens
from utils.telemetry import init_telemetry, create_span, record_metric, estimate_cost

DATA_DIR = PROJECT_ROOT / "data"
MODULE_NAME = "05_evaluation_framework"


# ============================================================================
# STEP 1: Load Ground Truth and Module 04 Predictions
# ============================================================================
# The ground truth file (generated with the synthetic data) contains
# the ACTUAL fraud ring member IDs and shell merchant IDs.
# Module 04 findings contain the LLM's PREDICTIONS.
# We compare these two to measure accuracy.
# ============================================================================

def step_1_load_truth_and_predictions() -> dict:
    """Load ground truth and LLM predictions for comparison."""
    print("=" * 70)
    print("MODULE 05 — Evaluation Framework")
    print("=" * 70)
    print()

    init_telemetry(service_name=MODULE_NAME)

    # Load ground truth (answer key)
    gt_path = DATA_DIR / "ground_truth.json"
    if not gt_path.exists():
        print("  ✗ ground_truth.json not found!")
        print("    Run data/generate_synthetic_data.py first.")
        sys.exit(1)

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    fraud_ring_gt = ground_truth["fraud_ring"]
    actual_members = set(fraud_ring_gt["member_customer_ids"])
    actual_shells = set(fraud_ring_gt["shell_merchant_ids"])

    print(f"  Ground Truth:")
    print(f"    Fraud ring members: {len(actual_members)}")
    print(f"    Shell merchants: {len(actual_shells)}")
    print(f"    Signals: {len(fraud_ring_gt['signals'])}")

    # Load Module 04 predictions
    predictions_path = DATA_DIR / "module_04_findings.json"
    if predictions_path.exists():
        with open(predictions_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        predicted_members = set(predictions.get("identified_ring_members", []))
        predicted_shells = set(predictions.get("identified_shell_merchants", []))
        print(f"\n  Module 04 Predictions:")
        print(f"    Predicted ring members: {len(predicted_members)}")
        print(f"    Predicted shell merchants: {len(predicted_shells)}")
    else:
        print("\n  ⚠ Module 04 findings not found.")
        print("    Running evaluation with empty predictions.")
        print("    For full results, run modules 01-04 first.")
        predicted_members = set()
        predicted_shells = set()
        predictions = {}

    print()

    return {
        "ground_truth": ground_truth,
        "actual_members": actual_members,
        "actual_shells": actual_shells,
        "predicted_members": predicted_members,
        "predicted_shells": predicted_shells,
        "predictions": predictions,
    }


# ============================================================================
# STEP 2: Calculate Precision, Recall, F1
# ============================================================================
# This is the core evaluation logic. We compute metrics separately for:
# - Fraud ring member detection
# - Shell merchant detection
# - Combined (overall ring identification)
#
# WORKSHOP DISCUSSION POINT:
# "In fraud detection, what's worse — a false positive (flagging an
# innocent account) or a false negative (missing a fraudster)?
# The answer depends on the business context:
# - For account freezes: false positives cause customer harm
# - For regulatory reporting: false negatives cause compliance risk
# This tension is why F1 (balanced metric) is important."
# ============================================================================

def step_2_calculate_metrics(eval_data: dict) -> dict:
    """Calculate precision, recall, and F1 for fraud detection."""
    print("─" * 70)
    print("STEP 2: Calculating Evaluation Metrics")
    print("─" * 70)
    print()

    actual_members = eval_data["actual_members"]
    predicted_members = eval_data["predicted_members"]
    actual_shells = eval_data["actual_shells"]
    predicted_shells = eval_data["predicted_shells"]

    def compute_metrics(actual: set, predicted: set, label: str) -> dict:
        """Compute precision, recall, F1 for a binary classification."""
        true_positives = actual & predicted
        false_positives = predicted - actual
        false_negatives = actual - predicted

        tp = len(true_positives)
        fp = len(false_positives)
        fn = len(false_negatives)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics = {
            "label": label,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "true_positive_ids": sorted(true_positives),
            "false_positive_ids": sorted(false_positives),
            "false_negative_ids": sorted(false_negatives),
        }

        return metrics

    with create_span("compute_evaluation_metrics") as span:
        member_metrics = compute_metrics(actual_members, predicted_members, "fraud_ring_members")
        merchant_metrics = compute_metrics(actual_shells, predicted_shells, "shell_merchants")

        # Combined metrics (all entities)
        all_actual = actual_members | actual_shells
        all_predicted = predicted_members | predicted_shells
        combined_metrics = compute_metrics(all_actual, all_predicted, "combined")

        span.set_attribute("member_f1", member_metrics["f1_score"])
        span.set_attribute("merchant_f1", merchant_metrics["f1_score"])
        span.set_attribute("combined_f1", combined_metrics["f1_score"])

        record_metric("precision", member_metrics["precision"], "", {"entity": "members"})
        record_metric("recall", member_metrics["recall"], "", {"entity": "members"})
        record_metric("f1_score", member_metrics["f1_score"], "", {"entity": "members"})
        record_metric("precision", merchant_metrics["precision"], "", {"entity": "merchants"})
        record_metric("recall", merchant_metrics["recall"], "", {"entity": "merchants"})
        record_metric("f1_score", merchant_metrics["f1_score"], "", {"entity": "merchants"})

    # Display results
    print("  ┌─────────────────────────────────────────────────────────────────")
    print("  │ FRAUD RING MEMBER DETECTION")
    print(f"  │   Precision: {member_metrics['precision']:.2%}  ({member_metrics['true_positives']} TP / {member_metrics['true_positives'] + member_metrics['false_positives']} predicted)")
    print(f"  │   Recall:    {member_metrics['recall']:.2%}  ({member_metrics['true_positives']} TP / {member_metrics['true_positives'] + member_metrics['false_negatives']} actual)")
    print(f"  │   F1 Score:  {member_metrics['f1_score']:.2%}")
    if member_metrics['false_positives'] > 0:
        print(f"  │   False Positives: {member_metrics['false_positive_ids'][:5]}")
    if member_metrics['false_negatives'] > 0:
        print(f"  │   False Negatives: {member_metrics['false_negative_ids'][:5]}")
    print("  │")
    print("  │ SHELL MERCHANT DETECTION")
    print(f"  │   Precision: {merchant_metrics['precision']:.2%}  ({merchant_metrics['true_positives']} TP / {merchant_metrics['true_positives'] + merchant_metrics['false_positives']} predicted)")
    print(f"  │   Recall:    {merchant_metrics['recall']:.2%}  ({merchant_metrics['true_positives']} TP / {merchant_metrics['true_positives'] + merchant_metrics['false_negatives']} actual)")
    print(f"  │   F1 Score:  {merchant_metrics['f1_score']:.2%}")
    print("  │")
    print("  │ COMBINED (ALL ENTITIES)")
    print(f"  │   Precision: {combined_metrics['precision']:.2%}")
    print(f"  │   Recall:    {combined_metrics['recall']:.2%}")
    print(f"  │   F1 Score:  {combined_metrics['f1_score']:.2%}")
    print("  └─────────────────────────────────────────────────────────────────")
    print()

    return {
        "member_metrics": member_metrics,
        "merchant_metrics": merchant_metrics,
        "combined_metrics": combined_metrics,
    }


# ============================================================================
# STEP 3: Error Analysis with LLM
# ============================================================================
# We use the LLM to analyze WHY certain errors occurred:
# - Why were some legitimate accounts flagged? (false positives)
# - Why were some ring members missed? (false negatives)
#
# This analysis helps improve prompt engineering for the next iteration.
# ============================================================================

def step_3_error_analysis(eval_data: dict, metrics: dict) -> dict:
    """Use LLM to analyze false positives and false negatives."""
    print("─" * 70)
    print("STEP 3: Error Analysis")
    print("─" * 70)
    print()

    member_metrics = metrics["member_metrics"]
    merchant_metrics = metrics["merchant_metrics"]
    ground_truth = eval_data["ground_truth"]

    error_data = {
        "member_detection": {
            "true_positives": member_metrics["true_positive_ids"],
            "false_positives": member_metrics["false_positive_ids"],
            "false_negatives": member_metrics["false_negative_ids"],
        },
        "merchant_detection": {
            "true_positives": merchant_metrics["true_positive_ids"],
            "false_positives": merchant_metrics["false_positive_ids"],
            "false_negatives": merchant_metrics["false_negative_ids"],
        },
        "ground_truth_signals": ground_truth["fraud_ring"]["signals"],
    }

    print(f"  Analyzing {member_metrics['false_positives']} false positives and "
          f"{member_metrics['false_negatives']} false negatives...")
    print()

    with create_span("llm_error_analysis") as span:
        response = call_openai_json(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["evaluator"]},
                {"role": "user", "content": (
                    "Analyze the following fraud detection evaluation results. "
                    "The ground truth contains the actual fraud ring signals. "
                    "Our detection system produced the results shown.\n\n"
                    "For each error type, explain:\n"
                    "1. WHY these errors likely occurred\n"
                    "2. What prompt engineering changes could reduce them\n"
                    "3. What additional data or context would help\n\n"
                    "Output JSON:\n"
                    "{\n"
                    '  "false_positive_analysis": {\n'
                    '    "count": N,\n'
                    '    "likely_causes": ["..."],\n'
                    '    "mitigation_strategies": ["..."]\n'
                    "  },\n"
                    '  "false_negative_analysis": {\n'
                    '    "count": N,\n'
                    '    "missed_signals": ["..."],\n'
                    '    "detection_improvements": ["..."]\n'
                    "  },\n"
                    '  "prompt_engineering_recommendations": [\n'
                    "    {\n"
                    '      "current_approach": "...",\n'
                    '      "suggested_change": "...",\n'
                    '      "expected_impact": "..."\n'
                    "    }\n"
                    "  ],\n"
                    '  "overall_assessment": "..."\n'
                    "}\n\n"
                    f"Evaluation Data:\n{json.dumps(error_data, indent=2)}"
                )},
            ],
            module_name=MODULE_NAME,
        )

        span.set_attribute("tokens_used", response["usage"]["total_tokens"])
        record_metric("tokens_used", response["usage"]["total_tokens"], "tokens", {"step": "3"})
        record_metric("ttft_ms", response["ttft_ms"], "ms", {"step": "3"})

    result = response["parsed"]
    print()

    return result


# ============================================================================
# STEP 4: Prompt Variant Testing
# ============================================================================
# This step demonstrates how to systematically test different prompt
# strategies. We run the same detection task with different system
# prompts and compare accuracy.
#
# WORKSHOP KEY INSIGHT:
# "Small changes in prompts can dramatically affect detection accuracy.
# This is why evaluation frameworks are essential — you need to MEASURE
# the impact of prompt changes, not just guess."
# ============================================================================

def step_4_prompt_variant_test(eval_data: dict) -> dict:
    """Test different prompt variants for detection accuracy."""
    print("─" * 70)
    print("STEP 4: Prompt Variant Testing")
    print("─" * 70)
    print()

    # Define prompt variants to test
    prompt_variants = [
        {
            "name": "baseline",
            "system_prompt": SYSTEM_PROMPTS["pattern_detector"],
            "instruction_style": "standard",
        },
        {
            "name": "chain_of_thought",
            "system_prompt": (
                SYSTEM_PROMPTS["pattern_detector"] +
                "\n\nIMPORTANT: Think step by step. Before reaching any conclusion, "
                "explicitly list the evidence for and against. Consider alternative "
                "explanations before flagging as fraud."
            ),
            "instruction_style": "chain_of_thought",
        },
        {
            "name": "high_precision",
            "system_prompt": (
                SYSTEM_PROMPTS["pattern_detector"] +
                "\n\nIMPORTANT: Minimize false positives. Only flag accounts where "
                "you see strong evidence from MULTIPLE independent signals. A single "
                "anomaly is not sufficient — require at least 3 corroborating signals."
            ),
            "instruction_style": "high_precision",
        },
        {
            "name": "high_recall",
            "system_prompt": (
                SYSTEM_PROMPTS["pattern_detector"] +
                "\n\nIMPORTANT: Prioritize finding ALL fraud ring members, even at "
                "the cost of some false positives. Cast a wide net. Flag any account "
                "with even a single suspicious signal — better to over-flag than miss."
            ),
            "instruction_style": "high_recall",
        },
    ]

    # Prepare a focused test dataset
    # We use a subset of customer and transaction data to keep costs down
    import csv
    test_customers = []
    with open(DATA_DIR / "customers.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            test_customers.append(row)

    # Focus on registration patterns and email domains
    customer_summary = []
    for c in test_customers[:100]:  # First 100 (includes fraud ring)
        customer_summary.append({
            "customer_id": c["customer_id"],
            "email": c["email"],
            "city": c["city"],
            "state": c.get("state", ""),
            "registration_date": c["registration_date"],
            "risk_score": c["risk_score"],
        })

    actual_members = eval_data["actual_members"]
    variant_results = []

    for variant in prompt_variants:
        print(f"  Testing variant: '{variant['name']}'...")

        with create_span(f"prompt_variant_{variant['name']}") as span:
            response = call_openai_json(
                messages=[
                    {"role": "system", "content": variant["system_prompt"]},
                    {"role": "user", "content": (
                        "Review these customer profiles and identify which accounts "
                        "are likely part of a fraud ring. Look for synthetic identity "
                        "indicators: clustered registration dates, unusual email domains, "
                        "geographic inconsistencies.\n\n"
                        "Output JSON:\n"
                        '{"suspicious_accounts": ["CUS-..."], '
                        '"reasoning": "brief explanation"}\n\n'
                        f"Customer Profiles:\n{json.dumps(customer_summary, indent=2)}"
                    )},
                ],
                module_name=MODULE_NAME,
            )

            predicted = set(response["parsed"].get("suspicious_accounts", []))

            tp = len(actual_members & predicted)
            fp = len(predicted - actual_members)
            fn = len(actual_members - predicted)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            result = {
                "variant_name": variant["name"],
                "instruction_style": variant["instruction_style"],
                "predicted_count": len(predicted),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "tokens_used": response["usage"]["total_tokens"],
                "latency_ms": response["latency_ms"],
            }
            variant_results.append(result)

            span.set_attribute("variant", variant["name"])
            span.set_attribute("f1_score", f1)
            record_metric("variant_f1", f1, "", {"variant": variant["name"]})
            record_metric("ttft_ms", response["ttft_ms"], "ms", {"step": "4", "variant": variant["name"]})

        print(f"    P: {precision:.2%}  R: {recall:.2%}  F1: {f1:.2%}  "
              f"({tp}TP/{fp}FP/{fn}FN)  [{response['usage']['total_tokens']} tokens]")

    print()

    # Display comparison table
    print("  ┌─ Prompt Variant Comparison ─────────────────────────────────────")
    print("  │ {:<18} {:>9} {:>8} {:>8} {:>7} {:>7}".format(
        "Variant", "Precision", "Recall", "F1", "Tokens", "ms"))
    print("  │ " + "─" * 60)
    for r in variant_results:
        print("  │ {:<18} {:>8.1%} {:>8.1%} {:>8.1%} {:>7,} {:>7.0f}".format(
            r["variant_name"],
            r["precision"],
            r["recall"],
            r["f1_score"],
            r["tokens_used"],
            r["latency_ms"],
        ))
    print("  └────────────────────────────────────────────────────────────────")
    print()

    # Identify best variant
    best = max(variant_results, key=lambda r: r["f1_score"])
    print(f"  Best variant: '{best['variant_name']}' (F1: {best['f1_score']:.2%})")
    print()

    return {
        "variant_results": variant_results,
        "best_variant": best["variant_name"],
    }


# ============================================================================
# STEP 5: Summary
# ============================================================================

def step_5_summary(metrics: dict, error_analysis: dict, variant_results: dict):
    """Print evaluation summary."""
    print("=" * 70)
    print("MODULE 05 — Summary")
    print("=" * 70)

    totals = get_total_tokens()
    cost = estimate_cost(totals["prompt_tokens"], totals["completion_tokens"])

    combined = metrics["combined_metrics"]

    # Export full evaluation results
    eval_results = {
        "module": "05_evaluation_framework",
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metrics": metrics,
        "error_analysis": error_analysis,
        "prompt_variant_results": variant_results,
        "token_usage": totals,
        "estimated_cost_usd": cost,
    }

    output_path = DATA_DIR / "module_05_evaluation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, default=str)

    print(f"""
  Evaluation Results:
    ├─ Combined Precision:      {combined['precision']:.2%}
    ├─ Combined Recall:         {combined['recall']:.2%}
    ├─ Combined F1:             {combined['f1_score']:.2%}
    ├─ Best prompt variant:     {variant_results['best_variant']}
    └─ Results exported to:     {output_path}

  API Usage (this module):
    ├─ API Calls:               {totals['api_calls']}
    ├─ Total Tokens:            {totals['total_tokens']:,}
    └─ Estimated Cost:          ${cost:.4f}

  KEY TAKEAWAY:
  Evaluation transforms AI-powered fraud detection from a black box into
  a measurable, improvable system. By testing prompt variants and measuring
  precision/recall, you can systematically improve detection accuracy.

  The prompt variant test shows how instruction style (chain-of-thought,
  high-precision, high-recall) directly impacts detection performance.
  In production, you'd run these evaluations on every prompt change.

  NEXT: Run 06_observability_dashboard.py (operational telemetry)
""")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Execute Module 05: Evaluation Framework."""
    eval_data = step_1_load_truth_and_predictions()
    metrics = step_2_calculate_metrics(eval_data)
    error_analysis = step_3_error_analysis(eval_data, metrics)
    variant_results = step_4_prompt_variant_test(eval_data)
    step_5_summary(metrics, error_analysis, variant_results)


if __name__ == "__main__":
    main()
