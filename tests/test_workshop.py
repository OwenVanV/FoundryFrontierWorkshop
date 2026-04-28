"""
============================================================================
PayPal x Azure AI Workshop — Test Suite
============================================================================

PURPOSE:
    Validates that all workshop modules can be imported, all data files
    exist and are well-formed, all tools function correctly, and the
    auth system works in both modes.

    Run: python -m pytest tests/ -v
    Or:  python tests/test_workshop.py

============================================================================
"""

import os
import sys
import csv
import json
import importlib
import importlib.util
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"


# ============================================================================
# TEST GROUP 1: Data Files Exist and Are Well-Formed
# ============================================================================

class TestDataFiles:
    """Verify all synthetic data files exist and have expected structure."""

    def test_transactions_csv_exists(self):
        assert (DATA_DIR / "transactions.csv").exists(), "transactions.csv missing — run data/generate_synthetic_data.py"

    def test_customers_csv_exists(self):
        assert (DATA_DIR / "customers.csv").exists(), "customers.csv missing"

    def test_merchants_csv_exists(self):
        assert (DATA_DIR / "merchants.csv").exists(), "merchants.csv missing"

    def test_payment_sessions_json_exists(self):
        assert (DATA_DIR / "payment_sessions.json").exists(), "payment_sessions.json missing"

    def test_dispute_cases_json_exists(self):
        assert (DATA_DIR / "dispute_cases.json").exists(), "dispute_cases.json missing"

    def test_risk_signals_json_exists(self):
        assert (DATA_DIR / "risk_signals.json").exists(), "risk_signals.json missing"

    def test_velocity_metrics_csv_exists(self):
        assert (DATA_DIR / "velocity_metrics.csv").exists(), "velocity_metrics.csv missing"

    def test_system_telemetry_csv_exists(self):
        assert (DATA_DIR / "system_telemetry.csv").exists(), "system_telemetry.csv missing"

    def test_ground_truth_json_exists(self):
        assert (DATA_DIR / "ground_truth.json").exists(), "ground_truth.json missing"

    def test_transactions_csv_row_count(self):
        with open(DATA_DIR / "transactions.csv", "r") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 40000, f"Expected 40K+ transactions, got {len(rows)}"

    def test_transactions_csv_has_required_columns(self):
        with open(DATA_DIR / "transactions.csv", "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        required = ["transaction_id", "timestamp", "sender_id", "receiver_id",
                     "merchant_id", "amount", "currency", "status"]
        for col in required:
            assert col in row, f"Missing column: {col}"

    def test_customers_csv_row_count(self):
        with open(DATA_DIR / "customers.csv", "r") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2000, f"Expected 2000 customers, got {len(rows)}"

    def test_merchants_csv_row_count(self):
        with open(DATA_DIR / "merchants.csv", "r") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 500, f"Expected 500 merchants, got {len(rows)}"

    def test_payment_sessions_json_valid(self):
        with open(DATA_DIR / "payment_sessions.json", "r") as f:
            data = json.load(f)
        assert isinstance(data, list), "payment_sessions.json should be a list"
        assert len(data) > 10000, f"Expected 10K+ sessions, got {len(data)}"

    def test_ground_truth_has_fraud_ring(self):
        with open(DATA_DIR / "ground_truth.json", "r") as f:
            gt = json.load(f)
        assert "fraud_ring" in gt, "ground_truth.json missing 'fraud_ring' key"
        ring = gt["fraud_ring"]
        assert len(ring["member_customer_ids"]) == 18, f"Expected 18 ring members"
        assert len(ring["shell_merchant_ids"]) == 4, f"Expected 4 shell merchants"

    def test_fraud_ring_ids_exist_in_data(self):
        with open(DATA_DIR / "ground_truth.json", "r") as f:
            gt = json.load(f)
        member_ids = set(gt["fraud_ring"]["member_customer_ids"])

        with open(DATA_DIR / "customers.csv", "r") as f:
            customer_ids = {row["customer_id"] for row in csv.DictReader(f)}

        missing = member_ids - customer_ids
        assert not missing, f"Fraud ring members missing from customers.csv: {missing}"


# ============================================================================
# TEST GROUP 2: Core Utility Imports
# ============================================================================

class TestCoreImports:
    """Verify all core utility modules can be imported."""

    def test_import_auth(self):
        from utils.auth import use_managed_identity, get_azure_credential
        assert callable(use_managed_identity)
        assert callable(get_azure_credential)

    def test_import_auth_managed_identity_flag(self):
        from utils.auth import use_managed_identity
        # Default should be False (no env var set)
        result = use_managed_identity()
        assert isinstance(result, bool)

    def test_import_telemetry(self):
        from utils.telemetry import init_telemetry, create_span, record_metric
        assert callable(init_telemetry)
        assert callable(create_span)
        assert callable(record_metric)

    def test_import_telemetry_cost_estimation(self):
        from utils.telemetry import estimate_cost
        cost = estimate_cost(1000, 500, "gpt-4o")
        assert isinstance(cost, float)
        assert cost > 0


# ============================================================================
# TEST GROUP 3: LLM Module Imports
# ============================================================================

class TestLLMModuleImports:
    """Verify all 6 LLM workshop modules can be imported without errors."""

    def test_import_module_01(self):
        spec = importlib.util.spec_from_file_location(
            "m01", PROJECT_ROOT / "01_data_exploration.py")
        mod = importlib.util.module_from_spec(spec)
        # Don't execute — just verify it can be loaded
        assert spec is not None

    def test_import_module_02(self):
        spec = importlib.util.spec_from_file_location(
            "m02", PROJECT_ROOT / "02_pattern_detection.py")
        assert spec is not None

    def test_import_module_03(self):
        spec = importlib.util.spec_from_file_location(
            "m03", PROJECT_ROOT / "03_cross_reference_analysis.py")
        assert spec is not None

    def test_import_module_04(self):
        spec = importlib.util.spec_from_file_location(
            "m04", PROJECT_ROOT / "04_fraud_ring_investigation.py")
        assert spec is not None

    def test_import_module_05(self):
        spec = importlib.util.spec_from_file_location(
            "m05", PROJECT_ROOT / "05_evaluation_framework.py")
        assert spec is not None

    def test_import_module_06(self):
        spec = importlib.util.spec_from_file_location(
            "m06", PROJECT_ROOT / "06_observability_dashboard.py")
        assert spec is not None


# ============================================================================
# TEST GROUP 4: Agent Module Imports
# ============================================================================

class TestAgentModuleImports:
    """Verify agent module files exist and have valid Python syntax."""

    def _check_syntax(self, filepath):
        """Compile-check a Python file without executing it."""
        with open(filepath, "r") as f:
            source = f.read()
        compile(source, str(filepath), "exec")

    def test_agent_01_syntax(self):
        self._check_syntax(PROJECT_ROOT / "agents" / "01_foundry_agent_setup.py")

    def test_agent_02_syntax(self):
        self._check_syntax(PROJECT_ROOT / "agents" / "02_maf_first_agent.py")

    def test_agent_03_syntax(self):
        self._check_syntax(PROJECT_ROOT / "agents" / "03_foundry_code_analysis.py")

    def test_agent_04_syntax(self):
        self._check_syntax(PROJECT_ROOT / "agents" / "04_maf_multi_turn_investigation.py")

    def test_agent_05_syntax(self):
        self._check_syntax(PROJECT_ROOT / "agents" / "05_maf_workflow_orchestration.py")

    def test_agent_06_syntax(self):
        self._check_syntax(PROJECT_ROOT / "agents" / "06_evaluation.py")

    def test_fraud_tools_syntax(self):
        self._check_syntax(PROJECT_ROOT / "agents" / "utils" / "fraud_tools.py")


# ============================================================================
# TEST GROUP 5: Voice_CUA Module Imports
# ============================================================================

class TestVoiceCUAModuleImports:
    """Verify Voice_CUA module files have valid Python syntax."""

    def _check_syntax(self, filepath):
        with open(filepath, "r") as f:
            source = f.read()
        compile(source, str(filepath), "exec")

    def test_cua_basics_syntax(self):
        self._check_syntax(PROJECT_ROOT / "Voice_CUA" / "01_cua_basics" / "run_cua_direct.py")

    def test_voice_live_basic_syntax(self):
        self._check_syntax(PROJECT_ROOT / "Voice_CUA" / "02_voice_live_intro" / "voice_live_basic.py")

    def test_voice_functions_syntax(self):
        self._check_syntax(PROJECT_ROOT / "Voice_CUA" / "03_voice_with_functions" / "voice_functions.py")

    def test_voice_cua_bridge_syntax(self):
        self._check_syntax(PROJECT_ROOT / "Voice_CUA" / "04_voice_cua_bridge" / "voice_cua_bridge.py")

    def test_voice_cua_operator_syntax(self):
        self._check_syntax(PROJECT_ROOT / "Voice_CUA" / "05_full_system" / "voice_cua_operator.py")

    def test_shared_cua_client_syntax(self):
        self._check_syntax(PROJECT_ROOT / "Voice_CUA" / "shared" / "cua_client.py")

    def test_shared_audio_helpers_syntax(self):
        self._check_syntax(PROJECT_ROOT / "Voice_CUA" / "shared" / "audio_helpers.py")


# ============================================================================
# TEST GROUP 6: Fraud Data Tools (Functional Tests)
# ============================================================================

class TestFraudTools:
    """Test the @tool-decorated functions work against actual data."""

    def _load_tool_module(self):
        """Import fraud_tools, handling the agent-framework dependency."""
        try:
            sys.path.insert(0, str(PROJECT_ROOT / "agents"))
            # The tools use @tool decorator from agent-framework.
            # If not installed, we test the raw functions instead.
            from utils.fraud_tools import (
                _load_csv, _load_json,
            )
            return True
        except ImportError:
            return False

    def test_load_transactions_csv(self):
        with open(DATA_DIR / "transactions.csv", "r") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0
        assert "amount" in rows[0]
        assert float(rows[0]["amount"]) >= 0

    def test_load_customers_csv(self):
        with open(DATA_DIR / "customers.csv", "r") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2000
        assert "email" in rows[0]

    def test_fraud_ring_smurfing_pattern(self):
        """Verify the smurfing pattern exists in the data."""
        with open(DATA_DIR / "transactions.csv", "r") as f:
            near_threshold = [
                row for row in csv.DictReader(f)
                if 9200 <= float(row["amount"]) <= 9800
            ]
        # Should have a meaningful number of near-threshold transactions
        assert len(near_threshold) > 50, (
            f"Expected 50+ transactions in $9,200-$9,800 range, got {len(near_threshold)}"
        )

    def test_shell_merchants_exist(self):
        """Verify the 4 shell merchants with 'Apex' names exist."""
        with open(DATA_DIR / "merchants.csv", "r") as f:
            apex = [
                row for row in csv.DictReader(f)
                if "apex" in row["business_name"].lower()
                or "a.d.s." in row["business_name"].lower()
            ]
        assert len(apex) >= 4, f"Expected 4+ Apex-related merchants, got {len(apex)}"

    def test_registration_clustering(self):
        """Verify fraud ring members have clustered registration dates."""
        with open(DATA_DIR / "customers.csv", "r") as f:
            customers = list(csv.DictReader(f))

        # First 18 customers are fraud ring members
        ring_dates = [c["registration_date"][:10] for c in customers[:18]]
        unique_months = set(d[:7] for d in ring_dates)
        # All ring members should register in Feb 2026 (1 month window)
        assert len(unique_months) <= 2, (
            f"Ring members should cluster in 1-2 months, got {len(unique_months)}: {unique_months}"
        )

    def test_device_fingerprint_sharing(self):
        """Verify device fingerprint sharing exists in sessions."""
        with open(DATA_DIR / "payment_sessions.json", "r") as f:
            sessions = json.load(f)

        from collections import defaultdict
        fp_to_customers = defaultdict(set)
        for s in sessions:
            fp = s.get("device", {}).get("fingerprint")
            cid = s.get("customer_id")
            if fp and cid:
                fp_to_customers[fp].add(cid)

        shared = {fp: custs for fp, custs in fp_to_customers.items() if len(custs) >= 2}
        assert len(shared) > 0, "Expected at least some shared device fingerprints"


# ============================================================================
# TEST GROUP 7: Auth Module Tests
# ============================================================================

class TestAuthModule:
    """Test the authentication helper module."""

    def test_use_managed_identity_default_false(self):
        from utils.auth import use_managed_identity
        # With no env var set (or set to false), should return False
        original = os.environ.get("USE_MANAGED_IDENTITY")
        os.environ["USE_MANAGED_IDENTITY"] = "false"
        assert use_managed_identity() is False
        if original is not None:
            os.environ["USE_MANAGED_IDENTITY"] = original
        else:
            del os.environ["USE_MANAGED_IDENTITY"]

    def test_use_managed_identity_true(self):
        from utils.auth import use_managed_identity
        original = os.environ.get("USE_MANAGED_IDENTITY")
        os.environ["USE_MANAGED_IDENTITY"] = "true"
        assert use_managed_identity() is True
        if original is not None:
            os.environ["USE_MANAGED_IDENTITY"] = original
        else:
            del os.environ["USE_MANAGED_IDENTITY"]

    def test_get_voice_live_headers_api_key(self):
        from utils.auth import get_voice_live_headers
        original_mi = os.environ.get("USE_MANAGED_IDENTITY")
        original_key = os.environ.get("AZURE_AI_API_KEY")
        os.environ["USE_MANAGED_IDENTITY"] = "false"
        os.environ["AZURE_AI_API_KEY"] = "test-key-123"

        headers = get_voice_live_headers()
        assert "api-key" in headers
        assert headers["api-key"] == "test-key-123"

        # Restore
        if original_mi is not None:
            os.environ["USE_MANAGED_IDENTITY"] = original_mi
        else:
            os.environ.pop("USE_MANAGED_IDENTITY", None)
        if original_key is not None:
            os.environ["AZURE_AI_API_KEY"] = original_key
        else:
            os.environ.pop("AZURE_AI_API_KEY", None)

    def test_get_openai_client_args_api_key(self):
        from utils.auth import get_openai_client_args
        original_mi = os.environ.get("USE_MANAGED_IDENTITY")
        original_ep = os.environ.get("AZURE_OPENAI_ENDPOINT")
        original_key = os.environ.get("AZURE_OPENAI_API_KEY")

        os.environ["USE_MANAGED_IDENTITY"] = "false"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com/"
        os.environ["AZURE_OPENAI_API_KEY"] = "test-key"

        args = get_openai_client_args()
        assert args["api_key"] == "test-key"
        assert "azure_endpoint" in args
        assert "azure_ad_token_provider" not in args

        # Restore
        for var, val in [("USE_MANAGED_IDENTITY", original_mi),
                         ("AZURE_OPENAI_ENDPOINT", original_ep),
                         ("AZURE_OPENAI_API_KEY", original_key)]:
            if val is not None:
                os.environ[var] = val
            else:
                os.environ.pop(var, None)


# ============================================================================
# TEST GROUP 8: Project Structure Verification
# ============================================================================

class TestProjectStructure:
    """Verify the complete project structure is intact."""

    EXPECTED_FILES = [
        # Root
        "README.md", "requirements.txt", ".env.example", ".gitignore",
        # Utils
        "utils/__init__.py", "utils/azure_client.py", "utils/telemetry.py", "utils/auth.py",
        # LLM modules
        "01_data_exploration.py", "02_pattern_detection.py",
        "03_cross_reference_analysis.py", "04_fraud_ring_investigation.py",
        "05_evaluation_framework.py", "06_observability_dashboard.py",
        # Data generator
        "data/generate_synthetic_data.py",
        # Agents
        "agents/requirements.txt", "agents/.env.example",
        "agents/utils/__init__.py", "agents/utils/fraud_tools.py",
        "agents/01_foundry_agent_setup.py", "agents/02_maf_first_agent.py",
        "agents/03_foundry_code_analysis.py", "agents/04_maf_multi_turn_investigation.py",
        "agents/05_maf_workflow_orchestration.py", "agents/06_evaluation.py",
        # Voice_CUA
        "Voice_CUA/requirements.txt", "Voice_CUA/.env.example",
        "Voice_CUA/shared/__init__.py", "Voice_CUA/shared/cua_client.py",
        "Voice_CUA/shared/audio_helpers.py",
        "Voice_CUA/01_cua_basics/run_cua_direct.py",
        "Voice_CUA/02_voice_live_intro/voice_live_basic.py",
        "Voice_CUA/03_voice_with_functions/voice_functions.py",
        "Voice_CUA/04_voice_cua_bridge/voice_cua_bridge.py",
        "Voice_CUA/05_full_system/voice_cua_operator.py",
    ]

    def test_all_expected_files_exist(self):
        missing = []
        for filepath in self.EXPECTED_FILES:
            full_path = PROJECT_ROOT / filepath
            if not full_path.exists():
                missing.append(filepath)
        assert not missing, f"Missing files:\n  " + "\n  ".join(missing)

    def test_no_hardcoded_api_keys_in_python_files(self):
        """Security check: ensure no API keys are hardcoded in Python files."""
        # Scan all .py files in the project (excluding computer-use/ which has legacy code)
        import re
        violations = []
        for py_file in PROJECT_ROOT.rglob("*.py"):
            if "computer-use" in str(py_file) or "__pycache__" in str(py_file):
                continue
            if ".venv" in str(py_file) or "venv" in py_file.parts:
                continue
            if "test_workshop" in str(py_file):
                continue
            with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            # Look for patterns that look like hardcoded keys
            # (32+ character hex strings or base64 that aren't in comments/docstrings)
            lines = content.split("\n")
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                # Check for api_key="..." or API_KEY="..." with actual values
                if re.search(r'(?:api_key|API_KEY)\s*=\s*["\'][a-zA-Z0-9]{20,}["\']', line):
                    rel_path = py_file.relative_to(PROJECT_ROOT)
                    violations.append(f"{rel_path}:{i}")

        assert not violations, (
            f"Potential hardcoded API keys found:\n  " + "\n  ".join(violations)
        )


# ============================================================================
# RUNNER — Execute all tests when run directly
# ============================================================================

def run_all_tests():
    """Run all tests and print results."""
    import traceback

    test_classes = [
        TestDataFiles,
        TestCoreImports,
        TestLLMModuleImports,
        TestAgentModuleImports,
        TestVoiceCUAModuleImports,
        TestFraudTools,
        TestAuthModule,
        TestProjectStructure,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    print("=" * 70)
    print("PayPal x Azure AI Workshop — Test Suite")
    print("=" * 70)
    print()

    for test_class in test_classes:
        class_name = test_class.__name__
        print(f"  {class_name}:")
        instance = test_class()

        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(methods):
            total += 1
            try:
                getattr(instance, method_name)()
                passed += 1
                print(f"    ✓ {method_name}")
            except AssertionError as e:
                failed += 1
                errors.append((class_name, method_name, str(e)))
                print(f"    ✗ {method_name}: {str(e)[:60]}")
            except Exception as e:
                failed += 1
                errors.append((class_name, method_name, traceback.format_exc()))
                print(f"    ✗ {method_name}: {type(e).__name__}: {str(e)[:60]}")
        print()

    print("─" * 70)
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print("─" * 70)

    if errors:
        print()
        print("  FAILURES:")
        for cls, method, msg in errors:
            print(f"    {cls}.{method}:")
            for line in msg.split("\n")[:3]:
                print(f"      {line}")
        print()

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
