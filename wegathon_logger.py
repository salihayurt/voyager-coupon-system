from __future__ import annotations
import json
import time
import uuid
import queue
import threading
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests


class WegathonLogger:
    def __init__(
        self,
        team: str,
        endpoint: str = "http://wegathon-opensearch.uzlas.com:2021/teams-ingest-pipeline/ingest",
        default_user: Optional[str] = None,
        batch_size: int = 50,
        flush_interval_sec: float = 2.0,
        max_queue_size: int = 10000,
        request_timeout_sec: float = 5.0,
        request_retries: int = 2,
        extra_context: Optional[Dict[str, Any]] = None,
        enable_console_fallback: bool = True,
    ):
        self.team = team
        self.endpoint = endpoint
        self.default_user = default_user
        self.batch_size = batch_size
        self.flush_interval_sec = flush_interval_sec
        self.request_timeout_sec = request_timeout_sec
        self.request_retries = request_retries
        self.extra_context = extra_context or {}
        self.enable_console_fallback = enable_console_fallback

        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max_queue_size)
        self._session = requests.Session()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run_worker, name="WegathonLoggerWorker", daemon=True)
        self._worker.start()

    def set_context(self, **ctx: Any) -> None:
        self.extra_context.update(ctx or {})

    def log_api_request(
        self,
        user: Optional[str],
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float,
        message: str = "",
        **extra: Any,
    ) -> None:
        data = {
            "category": "api",
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time_ms": response_time_ms,
        }
        self._enqueue_record(
            action="api_request",
            user=user,
            message=message or f"{method} {endpoint} -> {status_code} in {response_time_ms:.1f}ms",
            extra=data | extra,
        )

    def log_user_action(self, user: Optional[str], action: str, message: str = "", **extra: Any) -> None:
        self._enqueue_record(
            action=action,
            user=user,
            message=message or f"user action: {action}",
            extra={"category": "user_action"} | extra,
        )

    def log_error(
        self,
        user: Optional[str],
        action: str,
        exception: Exception,
        message: str = "",
        include_stack: bool = True,
        **extra: Any,
    ) -> None:
        payload = {
            "category": "error",
            "error_type": type(exception).__name__,
            "error_message": str(exception),
        }
        if include_stack:
            payload["stack_trace"] = "".join(
                traceback.format_exception(type(exception), exception, exception.__traceback__)
            )
        self._enqueue_record(
            action=action,
            user=user,
            message=message or f"error in {action}: {exception}",
            extra=payload | extra,
        )

    def log_performance(self, user: Optional[str], metric_name: str, value: float, unit: str = "ms", **extra: Any) -> None:
        self._enqueue_record(
            action="performance",
            user=user,
            message=f"{metric_name}={value}{unit}",
            extra={"category": "performance", "metric": metric_name, "value": value, "unit": unit} | extra,
        )

    def log_business_event(self, user: Optional[str], event_name: str, message: str = "", **extra: Any) -> None:
        self._enqueue_record(
            action=event_name,
            user=user,
            message=message or f"business event: {event_name}",
            extra={"category": "business"} | extra,
        )

    def log_security_event(self, user: Optional[str], event_name: str, severity: str, message: str = "", **extra: Any) -> None:
        self._enqueue_record(
            action=event_name,
            user=user,
            message=message or f"security event: {event_name}",
            extra={"category": "security", "severity": severity} | extra,
        )

    def flush(self) -> None:
        self._flush_now()

    def close(self) -> None:
        self._stop_event.set()
        self._worker.join(timeout=self.request_timeout_sec + 2.0)
        self._flush_now()

    def _enqueue_record(self, action: str, user: Optional[str], message: str, extra: Dict[str, Any]) -> None:
        record = self._build_record(action=action, user=user, message=message, extra=extra)
        try:
            self._q.put_nowait(record)
        except queue.Full:
            if self.enable_console_fallback:
                print("[WegathonLogger][DROP] buffer full; record=", self._safe_dumps(record))

    def _build_record(self, action: str, user: Optional[str], message: str, extra: Dict[str, Any]) -> Dict[str, Any]:
        base = {
            "team": self.team,
            "user": user or self.default_user or "anonymous",
            "action": action,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": str(uuid.uuid4()),
        }
        return base | self.extra_context | extra

    def _run_worker(self) -> None:
        buf: List[Dict[str, Any]] = []
        last_flush = time.monotonic()
        while not self._stop_event.is_set():
            timeout = max(0.0, self.flush_interval_sec - (time.monotonic() - last_flush))
            try:
                item = self._q.get(timeout=timeout)
                buf.append(item)
                while len(buf) < self.batch_size:
                    try:
                        buf.append(self._q.get_nowait())
                    except queue.Empty:
                        break
                self._send_batch(buf)
                buf.clear()
                last_flush = time.monotonic()
            except queue.Empty:
                if buf:
                    self._send_batch(buf)
                    buf.clear()
                    last_flush = time.monotonic()
            except Exception as e:
                if self.enable_console_fallback:
                    print("[WegathonLogger][WORKER-ERROR]", repr(e))

        if buf:
            self._send_batch(buf)

    def _flush_now(self) -> None:
        buf: List[Dict[str, Any]] = []
        try:
            while True:
                buf.append(self._q.get_nowait())
                if len(buf) >= self.batch_size:
                    self._send_batch(buf)
                    buf.clear()
        except queue.Empty:
            pass
        if buf:
            self._send_batch(buf)

    def _send_batch(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            return
        payload = self._safe_dumps(records)
        headers = {"Content-Type": "application/json"}
        for attempt in range(self.request_retries + 1):
            try:
                resp = self._session.post(self.endpoint, data=payload, headers=headers, timeout=self.request_timeout_sec)
                if resp.status_code < 300:
                    return
                else:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                if attempt == self.request_retries:
                    if self.enable_console_fallback:
                        print("[WegathonLogger][SEND-FAIL]", repr(e), "payload=", payload[:1000])
                else:
                    time.sleep(min(1.0 * (attempt + 1), 3.0))

    @staticmethod
    def _safe_dumps(obj: Any) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return json.dumps({"serialization_error": True, "repr": repr(obj)})


