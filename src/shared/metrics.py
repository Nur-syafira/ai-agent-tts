"""
OpenTelemetry трейсинг и метрики для мониторинга латентности.
"""

import os
from typing import Optional
from contextlib import contextmanager
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource
import time


class TelemetryManager:
    """Менеджер для OpenTelemetry трейсинга и метрик."""

    def __init__(self, service_name: str):
        """
        Инициализация telemetry.
        
        Args:
            service_name: Имя сервиса
        """
        self.service_name = service_name
        self.enabled = os.getenv("OTEL_ENABLED", "false").lower() == "true"
        self.tracer: Optional[trace.Tracer] = None
        self.meter: Optional[metrics.Meter] = None

        if self.enabled:
            self._setup_tracing()
            self._setup_metrics()

    def _setup_tracing(self):
        """Настройка трейсинга."""
        resource = Resource.create({"service.name": self.service_name})
        
        tracer_provider = TracerProvider(resource=resource)
        
        otlp_endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )
        
        span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
        span_processor = BatchSpanProcessor(span_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        trace.set_tracer_provider(tracer_provider)
        self.tracer = trace.get_tracer(self.service_name)

    def _setup_metrics(self):
        """Настройка метрик."""
        resource = Resource.create({"service.name": self.service_name})
        
        otlp_endpoint = os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"
        )
        
        metric_exporter = OTLPMetricExporter(endpoint=otlp_endpoint, insecure=True)
        metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=60000)
        
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        
        self.meter = metrics.get_meter(self.service_name)

    @contextmanager
    def trace_span(self, span_name: str, attributes: Optional[dict] = None):
        """
        Context manager для создания span.
        
        Args:
            span_name: Имя span
            attributes: Атрибуты span
            
        Yields:
            Span объект (или None если трейсинг отключён)
        """
        if not self.enabled or not self.tracer:
            yield None
            return

        with self.tracer.start_as_current_span(span_name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span

    def record_latency(
        self, metric_name: str, latency_ms: float, attributes: Optional[dict] = None
    ):
        """
        Записывает метрику латентности.
        
        Args:
            metric_name: Имя метрики
            latency_ms: Латентность в миллисекундах
            attributes: Дополнительные атрибуты
        """
        if not self.enabled or not self.meter:
            return

        histogram = self.meter.create_histogram(
            name=f"{metric_name}_latency_ms",
            description=f"Latency for {metric_name}",
            unit="ms",
        )
        
        histogram.record(latency_ms, attributes=attributes or {})


class LatencyTracker:
    """Удобный трекер латентности для E2E измерений."""

    def __init__(self, telemetry: TelemetryManager):
        """
        Инициализация tracker.
        
        Args:
            telemetry: TelemetryManager объект
        """
        self.telemetry = telemetry
        self.checkpoints = {}

    def checkpoint(self, name: str):
        """
        Сохраняет checkpoint времени.
        
        Args:
            name: Имя checkpoint
        """
        self.checkpoints[name] = time.perf_counter()

    def measure(self, start_checkpoint: str, end_checkpoint: str, metric_name: str):
        """
        Измеряет время между двумя checkpoints и записывает метрику.
        
        Args:
            start_checkpoint: Начальный checkpoint
            end_checkpoint: Конечный checkpoint
            metric_name: Имя метрики
            
        Returns:
            Латентность в миллисекундах
        """
        if start_checkpoint not in self.checkpoints:
            raise ValueError(f"Start checkpoint '{start_checkpoint}' not found")
        if end_checkpoint not in self.checkpoints:
            raise ValueError(f"End checkpoint '{end_checkpoint}' not found")

        latency_ms = (
            self.checkpoints[end_checkpoint] - self.checkpoints[start_checkpoint]
        ) * 1000

        self.telemetry.record_latency(metric_name, latency_ms)

        return latency_ms

    def reset(self):
        """Очищает все checkpoints."""
        self.checkpoints.clear()

