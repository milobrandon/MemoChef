"""
End-to-end test: run the full Memo Chef pipeline with:
  - EVER Lexington memo
  - Lexington-Limestone proforma
  - Synthetic market data workbook (different numbers)

Uses the same pipeline the Streamlit app uses (run_memo_pipeline).
"""
import os
import sys
import uuid

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load from memo_automator_app/.env which has the API key
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_project_root, "memo_automator_app", ".env"))
load_dotenv(os.path.join(_project_root, ".env"))
load_dotenv()

from memo_chef.models import RunRequest
from memo_chef.pipeline import run_memo_pipeline

# ---------- Paths ----------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMO = os.path.join(PROJECT_ROOT, "a. Sandbox", "EVER Lexington_20260205_TEST.pptx")
PROFORMA = os.path.join(PROJECT_ROOT, "a. Sandbox", "Proforma_Lexington-Limestone_20241021.xlsm")
MARKET_DATA = os.path.join(PROJECT_ROOT, "test_output_randomized", "synthetic_market_data.xlsx")
CONFIG = os.path.join(PROJECT_ROOT, "config.yaml")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_output_randomized", "market_test_run")

os.makedirs(OUTPUT_DIR, exist_ok=True)

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("ERROR: ANTHROPIC_API_KEY not set")
    sys.exit(1)

# Verify files exist
for label, path in [("Memo", MEMO), ("Proforma", PROFORMA), ("Market data", MARKET_DATA)]:
    if not os.path.isfile(path):
        print(f"ERROR: {label} not found: {path}")
        sys.exit(1)
    print(f"  {label}: {path}")

print(f"  Output: {OUTPUT_DIR}")
print()

# ---------- Build request ----------
request = RunRequest(
    memo_path=MEMO,
    proforma_path=PROFORMA,
    output_dir=OUTPUT_DIR,
    api_key=api_key,
    config_path=CONFIG,
    run_id=str(uuid.uuid4())[:8],
    property_name="EVER Lexington",
    property_rename_to="VERVE Lexington",
    market_data_path=MARKET_DATA,
    dry_run=False,
    skip_validation=False,
    resume_from_checkpoint=False,
)

# ---------- Progress callback ----------
def on_progress(update):
    """Callback receives a StageUpdate(key, label, percent, detail)."""
    print(f"  [{update.percent:3d}%] {update.key}: {update.label}")

# ---------- Run ----------
print("=" * 60)
print("RUNNING FULL PIPELINE (proforma + market data)")
print("=" * 60)

try:
    result = run_memo_pipeline(request, callback=on_progress)
    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    if hasattr(result, "changes_applied"):
        print(f"  Changes applied: {result.changes_applied}")
    if hasattr(result, "warnings"):
        for w in result.warnings:
            print(f"  WARNING: [{w.stage}] {w.message}")
    print(f"  Output dir: {OUTPUT_DIR}")
except Exception as e:
    print(f"\nPIPELINE FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
