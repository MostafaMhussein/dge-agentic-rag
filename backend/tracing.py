"""Phoenix tracing and observability.

Configures OpenTelemetry to send traces to Phoenix for debugging and monitoring.
"""
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

from .config import get_settings

logger = logging.getLogger(__name__)

_tracer: Optional[trace.Tracer] = None
_provider: Optional[TracerProvider] = None
_initialized = False


def init_phoenix_tracing() -> None:
    """Initialize OpenTelemetry tracing with Phoenix backend."""
    global _tracer, _provider, _initialized
    
    if _initialized:
        return
    
    settings = get_settings()
    
    try:
        resource = Resource.create({
            "service.name": settings.phoenix_project_name,
            "service.version": "1.0.0",
        })
        
        _provider = TracerProvider(resource=resource)
        
        exporter = OTLPSpanExporter(
            endpoint=settings.phoenix_endpoint,
            insecure=True,
        )
        _provider.add_span_processor(BatchSpanProcessor(exporter))
        
        trace.set_tracer_provider(_provider)
        _tracer = trace.get_tracer(__name__)
        
        # instrument LlamaIndex
        try:
            from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
            LlamaIndexInstrumentor().instrument(tracer_provider=_provider)
            logger.info("LlamaIndex instrumentation enabled")
        except Exception as e:
            logger.warning(f"LlamaIndex instrumentation failed: {e}")
        
        logger.info(f"Phoenix tracing initialized: {settings.phoenix_endpoint}")
        _initialized = True
        
    except Exception as e:
        logger.error(f"Tracing initialization failed: {e}")


def get_tracer() -> Optional[trace.Tracer]:
    """Get the tracer instance."""
    return _tracer


@contextmanager
def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Create a trace span for monitoring."""
    if _tracer is None:
        yield None
        return
    
    with _tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(key, value)
        yield span


def log_retrieval(query: str, num_results: int, latency_ms: float) -> None:
    """Log retrieval metrics."""
    with trace_span("retrieval_metrics", {
        "query": query[:100],
        "num_results": num_results,
        "latency_ms": latency_ms,
    }):
        pass
