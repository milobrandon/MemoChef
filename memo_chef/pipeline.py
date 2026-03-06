from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import anthropic
import yaml

from memo_automator import (
    _is_api_error,
    apply_branding,
    apply_updates,
    chunk_memo_by_pages,
    create_backup,
    extract_market_data,
    extract_memo_content,
    extract_proforma_data,
    extract_schedule_data,
    get_metric_mappings,
    load_config,
    normalize_layout,
    pre_validate_mappings,
    validate_mappings,
    write_change_log,
)

from .models import RunManifest, RunRequest, RunResult, RunWarning, StageRecord, StageUpdate

StageCallback = Callable[[StageUpdate], None] | None

LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"


class LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record) -> None:
        self.lines.append(self.format(record))


class CheckpointManager:
    def __init__(self, request: RunRequest) -> None:
        self.request = request
        self.path = Path(request.output_dir) / "run_manifest.json"
        self.manifest = self._load_or_create()

    def _load_or_create(self) -> RunManifest:
        if self.request.resume_from_checkpoint and self.path.exists():
            try:
                return RunManifest.model_validate_json(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return RunManifest(
            run_id=self.request.run_id,
            memo_name=Path(self.request.memo_path).name,
            proforma_name=Path(self.request.proforma_path).name,
            property_name=self.request.property_name,
            dry_run=self.request.dry_run,
            skip_validation=self.request.skip_validation,
        )

    def save(self) -> None:
        self.manifest.updated_at = datetime.now(UTC).isoformat()
        self.path.write_text(
            self.manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

    @contextmanager
    def stage(self, key: str, detail: str = ""):
        record = self.manifest.stages.get(key, StageRecord())
        record.status = "running"
        record.started_at = datetime.now(UTC).isoformat()
        record.detail = detail
        self.manifest.stages[key] = record
        self.save()
        started = time.time()
        try:
            yield record
            record.status = "completed"
        except Exception:
            record.status = "failed"
            raise
        finally:
            record.completed_at = datetime.now(UTC).isoformat()
            record.duration_seconds = round(time.time() - started, 2)
            self.manifest.stages[key] = record
            self.save()

    def add_warning(self, stage: str, message: str) -> None:
        self.manifest.warnings.append(RunWarning(stage=stage, message=message))
        self.save()

    def set_output(self, key: str, value: str) -> None:
        self.manifest.outputs[key] = value
        self.save()

    def set_count(self, key: str, value: int) -> None:
        self.manifest.counts[key] = value
        self.save()


# Approximate cost per million tokens (USD) by model prefix.
_TOKEN_RATES: dict[str, tuple[float, float]] = {
    "claude-opus": (15.0, 75.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (0.8, 4.0),
}


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    rate_in, rate_out = 3.0, 15.0  # default to Sonnet pricing
    for prefix, rates in _TOKEN_RATES.items():
        if prefix in model.lower():
            rate_in, rate_out = rates
            break
    return round((input_tokens * rate_in + output_tokens * rate_out) / 1_000_000, 6)


class _MessagesProxy:
    """Intercepts messages.create to accumulate token usage."""

    def __init__(self, client: "anthropic.Anthropic", tracker: "TokenTracker") -> None:
        self._client = client
        self._tracker = tracker

    def create(self, *args, **kwargs):
        response = self._client.messages.create(*args, **kwargs)
        if hasattr(response, "usage"):
            self._tracker.input_tokens += response.usage.input_tokens
            self._tracker.output_tokens += response.usage.output_tokens
            model = kwargs.get("model", "")
            self._tracker.estimated_cost_usd += _cost_usd(
                model, response.usage.input_tokens, response.usage.output_tokens
            )
        return response


class TokenTracker:
    """Wraps an Anthropic client and tracks cumulative token usage."""

    def __init__(self, client: "anthropic.Anthropic") -> None:
        self._client = client
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.estimated_cost_usd: float = 0.0
        self.messages = _MessagesProxy(client, self)

    def __getattr__(self, name: str):
        return getattr(self._client, name)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _emit(callback: StageCallback, key: str, label: str, percent: int, detail: str = "") -> None:
    if callback is not None:
        callback(StageUpdate(key=key, label=label, percent=percent, detail=detail))


def _write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _retry(
    func,
    *args,
    retries: int = 3,
    base_delay: float = 1.0,
    jitter: float = 0.25,
    checkpoint: CheckpointManager | None = None,
    stage: str = "",
    **kwargs,
):
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except Exception as err:
            attempt += 1
            if attempt > retries or not _is_api_error(err):
                raise
            wait_seconds = base_delay * (2 ** (attempt - 1)) + random.uniform(0, jitter)
            if checkpoint is not None:
                checkpoint.add_warning(stage, f"Retrying after API error: {err}")
            time.sleep(wait_seconds)


def _mapping_with_batching(
    client,
    proforma_data: str,
    memo_content: str,
    cfg: dict,
    property_name: str | None,
    callback: StageCallback,
    checkpoint: CheckpointManager,
) -> dict:
    batch_threshold = 80_000
    rate_limit_interval = 65
    prompt_size = len(proforma_data) + len(memo_content)
    if prompt_size <= batch_threshold:
        _emit(callback, "mapping", "Generate mappings", 52, "Sending full-deck mapping pass")
        mappings = _retry(
            get_metric_mappings,
            client,
            proforma_data,
            memo_content,
            cfg,
            property_name=property_name,
            checkpoint=checkpoint,
            stage="mapping",
        )
        mappings.pop("_truncated", None)
        return mappings

    memo_chunks = chunk_memo_by_pages(memo_content, pages_per_chunk=3)
    mappings = {"table_updates": [], "text_updates": [], "row_inserts": []}
    last_api_call = 0.0
    for index, chunk in enumerate(memo_chunks, start=1):
        percent = 50 + int((18 * index) / max(len(memo_chunks), 1))
        _emit(callback, "mapping", f"Generate mappings ({index}/{len(memo_chunks)})", percent)
        if index > 1 and last_api_call > 0:
            wait_seconds = rate_limit_interval - (time.time() - last_api_call)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
        last_api_call = time.time()
        batch = _retry(
            get_metric_mappings,
            client,
            proforma_data,
            chunk,
            cfg,
            property_name=property_name,
            checkpoint=checkpoint,
            stage="mapping",
        )
        if batch.pop("_truncated", False):
            covered_pages = {
                entry.get("page")
                for group in ("table_updates", "text_updates", "row_inserts")
                for entry in batch.get(group, [])
            }
            mappings["table_updates"].extend(batch.get("table_updates", []))
            mappings["text_updates"].extend(batch.get("text_updates", []))
            mappings["row_inserts"].extend(batch.get("row_inserts", []))
            sub_chunks = chunk_memo_by_pages(chunk, pages_per_chunk=1)
            for sub_chunk in sub_chunks:
                sub_pages = set(int(match) for match in re.findall(r"PAGE (\d+)", sub_chunk))
                if sub_pages and sub_pages.issubset(covered_pages):
                    continue
                wait_seconds = rate_limit_interval - (time.time() - last_api_call)
                if wait_seconds > 0:
                    time.sleep(wait_seconds)
                last_api_call = time.time()
                sub_batch = _retry(
                    get_metric_mappings,
                    client,
                    proforma_data,
                    sub_chunk,
                    cfg,
                    property_name=property_name,
                    checkpoint=checkpoint,
                    stage="mapping",
                )
                sub_batch.pop("_truncated", None)
                mappings["table_updates"].extend(sub_batch.get("table_updates", []))
                mappings["text_updates"].extend(sub_batch.get("text_updates", []))
                mappings["row_inserts"].extend(sub_batch.get("row_inserts", []))
            continue
        mappings["table_updates"].extend(batch.get("table_updates", []))
        mappings["text_updates"].extend(batch.get("text_updates", []))
        mappings["row_inserts"].extend(batch.get("row_inserts", []))
    return mappings


def run_memo_pipeline(request: RunRequest, callback: StageCallback = None) -> RunResult:
    os.makedirs(request.output_dir, exist_ok=True)
    checkpoint = CheckpointManager(request)
    logger = logging.getLogger("memo_automator")
    log_capture = LogCapture()
    log_capture.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(log_capture)

    try:
        checkpoint.manifest.status = "running"
        checkpoint.manifest.config_profile = request.config_override_path and Path(request.config_override_path).stem
        checkpoint.save()
        cfg = load_config(request.config_path)
        if request.config_override_path and Path(request.config_override_path).exists():
            with open(request.config_override_path, encoding="utf-8") as f:
                override = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, override)
        _raw_client = anthropic.Anthropic(
            api_key=request.api_key,
            max_retries=5,
            timeout=900.0,
        )
        client = TokenTracker(_raw_client)

        _emit(callback, "backup", "Create backup", 5)
        with checkpoint.stage("backup", "Creating backup copy"):
            backup_path = create_backup(request.memo_path, request.output_dir)
            checkpoint.set_output("backup_path", backup_path)

        _emit(callback, "extract_sources", "Extract source data", 12)
        with checkpoint.stage("extract_sources", "Extracting proforma, market, and schedule data"):
            proforma_data = extract_proforma_data(request.proforma_path, cfg)
            proforma_extract_path = os.path.join(request.output_dir, "proforma_extract.txt")
            Path(proforma_extract_path).write_text(proforma_data, encoding="utf-8")
            checkpoint.set_output("proforma_extract", proforma_extract_path)

            if request.schedule_path:
                schedule_data = extract_schedule_data(request.schedule_path, cfg)
                if schedule_data:
                    proforma_data += "\n\n" + schedule_data
                    schedule_extract_path = os.path.join(request.output_dir, "schedule_extract.txt")
                    Path(schedule_extract_path).write_text(schedule_data, encoding="utf-8")
                    checkpoint.set_output("schedule_extract", schedule_extract_path)

            if request.market_data_path:
                market_data = extract_market_data(request.market_data_path, cfg)
                if market_data:
                    proforma_data += "\n\n" + market_data
                    market_extract_path = os.path.join(request.output_dir, "market_data_extract.txt")
                    Path(market_extract_path).write_text(market_data, encoding="utf-8")
                    checkpoint.set_output("market_data_extract", market_extract_path)
                else:
                    checkpoint.add_warning(
                        "extract_sources",
                        "Market data file loaded but no dashboard tabs were extracted.",
                    )

        _emit(callback, "extract_memo", "Extract memo", 24)
        with checkpoint.stage("extract_memo", "Extracting memo deck contents"):
            memo_content = extract_memo_content(request.memo_path, cfg)
            memo_extract_path = os.path.join(request.output_dir, "memo_extract.txt")
            Path(memo_extract_path).write_text(memo_content, encoding="utf-8")
            checkpoint.set_output("memo_extract", memo_extract_path)

        _emit(callback, "mapping", "Generate mappings", 45)
        with checkpoint.stage("mapping", "Generating candidate updates"):
            mappings = _mapping_with_batching(
                client,
                proforma_data,
                memo_content,
                cfg,
                request.property_name,
                callback,
                checkpoint,
            )
            mappings["table_updates"] = [
                entry for entry in mappings["table_updates"]
                if entry.get("old_value") != entry.get("new_value")
            ]
            mappings["text_updates"] = [
                entry for entry in mappings["text_updates"]
                if entry.get("old_text") != entry.get("new_text")
            ]
            mappings = pre_validate_mappings(mappings, memo_content)
            raw_mapping_path = os.path.join(request.output_dir, "mappings_raw.json")
            _write_json(raw_mapping_path, mappings)
            checkpoint.set_output("mappings_raw", raw_mapping_path)

        _emit(callback, "validation", "Validate changes", 72)
        with checkpoint.stage("validation", "Validating mappings"):
            if request.skip_validation:
                validated = mappings
                validated.setdefault("rejected", [])
                validated.setdefault("missed", [])
            else:
                validated = _retry(
                    validate_mappings,
                    client,
                    mappings,
                    proforma_data,
                    memo_content,
                    cfg,
                    property_name=request.property_name,
                    checkpoint=checkpoint,
                    stage="validation",
                )
            validated_mapping_path = os.path.join(request.output_dir, "mappings_validated.json")
            _write_json(validated_mapping_path, validated)
            checkpoint.set_output("mappings_validated", validated_mapping_path)

        _emit(callback, "apply", "Apply updates", 84)
        with checkpoint.stage("apply", "Applying text, table, and chart updates"):
            changes = apply_updates(request.memo_path, validated, dry_run=request.dry_run)
            checkpoint.set_count("changes", len(changes))

        if not request.dry_run:
            _emit(callback, "branding", "Apply branding", 90)
            with checkpoint.stage("branding", "Applying visual refresh"):
                theme_path = cfg.get("branding", {}).get("theme_path", "")
                if not theme_path:
                    theme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Subtext Brand Theme.thmx")
                if os.path.exists(theme_path):
                    branded_count = apply_branding(request.memo_path, theme_path, cfg)
                    checkpoint.set_count("branded_runs", branded_count)
                else:
                    checkpoint.add_warning("branding", "Theme file not found; branding skipped.")

            _emit(callback, "layout", "Normalize layout", 94)
            with checkpoint.stage("layout", "Normalizing slide layout"):
                layout_summary = normalize_layout(request.memo_path, cfg)
                checkpoint.set_count("titles_snapped", int(layout_summary.get("titles_snapped", 0)))
                checkpoint.set_count(
                    "page_numbers_snapped",
                    int(layout_summary.get("page_numbers_snapped", 0)),
                )

        _emit(callback, "artifacts", "Write artifacts", 97)
        with checkpoint.stage("artifacts", "Writing change log and manifest"):
            log_path = write_change_log(
                request.output_dir,
                changes,
                validated,
                request.memo_path,
                request.proforma_path,
                checkpoint.manifest.outputs["backup_path"],
            )
            checkpoint.set_output("change_log", log_path)
            checkpoint.set_count("rejected", len(validated.get("rejected", [])))
            checkpoint.set_count("missed", len(validated.get("missed", [])))
            checkpoint.set_count("input_tokens", client.input_tokens)
            checkpoint.set_count("output_tokens", client.output_tokens)
            # Store cost as integer microdollars to avoid float precision issues
            checkpoint.set_count(
                "estimated_cost_microdollars",
                int(round(client.estimated_cost_usd * 1_000_000)),
            )

        checkpoint.manifest.status = "completed"
        checkpoint.save()
        _emit(callback, "complete", "Run complete", 100)

        memo_bytes = Path(request.memo_path).read_bytes()
        log_bytes = Path(checkpoint.manifest.outputs["change_log"]).read_bytes()
        manifest_bytes = checkpoint.path.read_bytes()
        return RunResult(
            manifest=checkpoint.manifest,
            memo_path=request.memo_path,
            log_path=checkpoint.manifest.outputs["change_log"],
            manifest_path=str(checkpoint.path),
            memo_bytes=memo_bytes,
            log_bytes=log_bytes,
            manifest_bytes=manifest_bytes,
            changes=changes,
            rejected=validated.get("rejected", []),
            missed=validated.get("missed", []),
            log_lines=log_capture.lines[:],
        )
    except Exception as err:
        checkpoint.manifest.status = "failed"
        checkpoint.add_warning("pipeline", str(err))
        checkpoint.save()
        raise
    finally:
        logger.removeHandler(log_capture)
