from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path

import app_services
from memo_chef.models import RunManifest, RunRequest, StageUpdate
from memo_chef.theme import APP_TITLE, STAGES, app_css, hero_html


def test_run_request_name_properties():
    request = RunRequest(
        memo_path="C:/tmp/memo.pptx",
        proforma_path="C:/tmp/proforma.xlsx",
        output_dir="C:/tmp/out",
        api_key="test-key",
        config_path="config.yaml",
        run_id="abc123",
    )
    assert request.memo_name == "memo.pptx"
    assert request.proforma_name == "proforma.xlsx"


def test_stage_update_bounds_and_manifest_defaults():
    update = StageUpdate(key="extract", label="Extract", percent=25)
    manifest = RunManifest(
        run_id="run-1",
        memo_name="memo.pptx",
        proforma_name="proforma.xlsx",
    )
    assert update.percent == 25
    assert manifest.status == "running"
    assert manifest.outputs == {}
    assert manifest.stages == {}


def test_theme_assets_include_expected_brand_markers():
    css = app_css()
    hero = hero_html()
    assert APP_TITLE in hero
    assert "Memo Operations Platform" in hero
    assert "--accent" in css
    for stage in STAGES:
        assert stage in hero


def test_platform_health_reports_expected_components(monkeypatch):
    @contextmanager
    def fake_cursor():
        class FakeCursor:
            def execute(self, query):
                return None

            def fetchone(self):
                return (1,)

        yield FakeCursor()

    monkeypatch.setattr(app_services, "db_cursor", fake_cursor)
    monkeypatch.setattr(app_services.os.path, "exists", lambda path: path != "Subtext Brand Theme.thmx")
    health = app_services.get_platform_health()
    components = {row["Component"]: row for row in health}
    assert components["Database"]["Status"] == "healthy"
    assert components["Config"]["Status"] == "healthy"
    assert components["Theme"]["Status"] == "warning"


def test_storage_root_and_run_artifact_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(app_services, "get_storage_root", lambda: tmp_path)
    run_dir = app_services.get_run_storage_dir("run-123")
    assert run_dir == tmp_path / "run-123"
    assert run_dir.exists()
    (run_dir / "memo.pptx").write_bytes(b"memo")
    (run_dir / "change_log.md").write_text("log", encoding="utf-8")
    paths = app_services.get_run_artifact_paths("run-123")
    assert paths["memo"].endswith("memo.pptx")
    assert paths["change_log"].endswith("change_log.md")


def test_enqueue_job_serializes_binary_payload(monkeypatch):
    captured: dict[str, str] = {}

    @contextmanager
    def fake_cursor():
        class FakeCursor:
            def execute(self, query, params):
                captured["payload_json"] = params[2]

        yield FakeCursor()

    monkeypatch.setattr(app_services, "db_cursor", fake_cursor)
    payload = {
        "job_id": "job-1",
        "memo_name": "memo.pptx",
        "memo_bytes": b"\x00memo-bytes\xff",
        "market_data_bytes": None,
    }

    app_services.enqueue_job("alice", payload)

    stored = json.loads(captured["payload_json"])
    assert stored["memo_bytes"][app_services._BINARY_PAYLOAD_KEY] is True
    assert stored["market_data_bytes"] is None


def test_get_job_queue_restores_binary_payload(monkeypatch):
    payload = {
        "job_id": "job-1",
        "memo_name": "memo.pptx",
        "memo_bytes": b"\x00memo-bytes\xff",
        "nested": {"proforma_bytes": b"proforma-bytes"},
    }
    payload_json = json.dumps(app_services._json_safe_payload(payload))
    now = datetime.now()

    @contextmanager
    def fake_cursor():
        class FakeCursor:
            def execute(self, query, params):
                return None

            def fetchall(self):
                return [("job-1", "alice", "queued", payload_json, None, None, now, now)]

        yield FakeCursor()

    monkeypatch.setattr(app_services, "db_cursor", fake_cursor)

    queue = app_services.get_job_queue("alice")

    assert queue[0]["payload"] == payload
