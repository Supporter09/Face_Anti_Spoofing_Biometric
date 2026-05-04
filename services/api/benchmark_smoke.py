from time import perf_counter

from fas.schemas import LivenessInferRequest
from fas.service import LivenessService


def main() -> None:
    service = LivenessService()
    started_at = perf_counter()
    response = service.infer(LivenessInferRequest())
    total_ms = (perf_counter() - started_at) * 1000.0
    print(
        {
            'face_detected': response.face_detected,
            'liveness_label': response.liveness_label,
            'latency_ms': response.latency_ms,
            'benchmark_wrapper_ms': total_ms,
        }
    )


if __name__ == '__main__':
    main()
