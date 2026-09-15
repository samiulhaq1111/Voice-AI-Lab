"""Benchmark API endpoints (Phase 5B + 5C analytics)."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.benchmarks.runner import (
    aggregate_results,
    run_scenario,
    run_scenario_n_times,
)
from app.benchmarks.scenarios import get_scenario, list_scenarios
from app.core.database import get_db
from app.core.logging import logger
from app.schemas import (
    BenchmarkBatchResponse,
    BenchmarkOverallSummary,
    BenchmarkProviderSummary,
    BenchmarkRecentResult,
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    BenchmarkScenarioResponse,
    BenchmarkScenarioSummary,
)
from app.services.benchmark_analytics import BenchmarkAnalyticsService

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.get("/scenarios", response_model=list[BenchmarkScenarioResponse])
async def get_scenarios() -> list[BenchmarkScenarioResponse]:
    """List all available benchmark scenarios."""
    return [
        BenchmarkScenarioResponse(
            scenario_id=s.scenario_id,
            name=s.name,
            description=s.description,
            category=s.category,
            expected_tool_calls=s.expected_tool_calls,
            include_tts=s.include_tts,
        )
        for s in list_scenarios()
    ]


@router.post("/run", response_model=BenchmarkRunResponse)
async def execute_benchmark(
    request: BenchmarkRunRequest,
    db: Session = Depends(get_db),
) -> BenchmarkRunResponse:
    """Execute a benchmark scenario once.

    Returns the run result with metrics and validation.
    """
    scenario = get_scenario(request.scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"Scenario '{request.scenario_id}' not found")

    logger.info("[BENCH] API run request scenario=%s", request.scenario_id)

    try:
        result = await run_scenario(db, request.scenario_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("[BENCH] API run failed error=%s", e)
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {e}") from e

    return BenchmarkRunResponse(**result.to_api_dict())


@router.post("/run/batch", response_model=BenchmarkBatchResponse)
async def execute_benchmark_batch(
    request: BenchmarkRunRequest,
    db: Session = Depends(get_db),
) -> BenchmarkBatchResponse:
    """Execute a benchmark scenario N times sequentially.

    Returns individual run results plus basic aggregation statistics.
    """
    scenario = get_scenario(request.scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"Scenario '{request.scenario_id}' not found")

    logger.info(
        "[BENCH] API batch request scenario=%s repetitions=%d",
        request.scenario_id,
        request.repetitions,
    )

    try:
        results = await run_scenario_n_times(
            db, request.scenario_id, repetitions=request.repetitions,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("[BENCH] API batch failed error=%s", e)
        raise HTTPException(status_code=500, detail=f"Benchmark batch failed: {e}") from e

    aggregation = aggregate_results(results) if len(results) > 1 else None

    return BenchmarkBatchResponse(
        runs=[BenchmarkRunResponse(**r.to_api_dict()) for r in results],
        aggregation=aggregation.to_api_dict() if aggregation else None,
    )


# ---------------------------------------------------------------------------
# Analytics endpoints (Phase 5C)
# ---------------------------------------------------------------------------


@router.get("/summary", response_model=BenchmarkOverallSummary)
async def get_overall_summary(
    scenario_id: str | None = Query(None, description="Filter by scenario"),
    benchmark_mode: str | None = Query(None, description="Filter by benchmark mode"),
    date_from: datetime | None = Query(None, description="Start date (ISO 8601)"),
    date_to: datetime | None = Query(None, description="End date (ISO 8601)"),
    db: Session = Depends(get_db),
) -> BenchmarkOverallSummary:
    """Overall summary across all benchmark runs."""
    svc = BenchmarkAnalyticsService(db)
    summary = svc.get_overall_summary(
        scenario_id=scenario_id,
        benchmark_mode=benchmark_mode,
        date_from=date_from,
        date_to=date_to,
    )
    return BenchmarkOverallSummary(**summary.to_dict())


@router.get("/scenarios/summary", response_model=list[BenchmarkScenarioSummary])
async def get_scenario_summaries(
    benchmark_mode: str | None = Query(None, description="Filter by benchmark mode"),
    date_from: datetime | None = Query(None, description="Start date (ISO 8601)"),
    date_to: datetime | None = Query(None, description="End date (ISO 8601)"),
    db: Session = Depends(get_db),
) -> list[BenchmarkScenarioSummary]:
    """Summary for each benchmark scenario."""
    svc = BenchmarkAnalyticsService(db)
    summaries = svc.get_scenario_summaries(
        benchmark_mode=benchmark_mode,
        date_from=date_from,
        date_to=date_to,
    )
    return [BenchmarkScenarioSummary(**s.to_dict()) for s in summaries]


@router.get("/providers/summary", response_model=list[BenchmarkProviderSummary])
async def get_provider_summaries(
    benchmark_mode: str | None = Query(None, description="Filter by benchmark mode"),
    date_from: datetime | None = Query(None, description="Start date (ISO 8601)"),
    date_to: datetime | None = Query(None, description="End date (ISO 8601)"),
    db: Session = Depends(get_db),
) -> list[BenchmarkProviderSummary]:
    """Summaries grouped by provider + model for each stage."""
    svc = BenchmarkAnalyticsService(db)
    summaries = svc.get_provider_summaries(
        benchmark_mode=benchmark_mode,
        date_from=date_from,
        date_to=date_to,
    )
    return [BenchmarkProviderSummary(**s.to_dict()) for s in summaries]


@router.get("/results", response_model=list[BenchmarkRecentResult])
async def get_recent_results(
    limit: int = Query(20, ge=1, le=100, description="Max results to return"),
    scenario_id: str | None = Query(None, description="Filter by scenario"),
    benchmark_mode: str | None = Query(None, description="Filter by benchmark mode"),
    date_from: datetime | None = Query(None, description="Start date (ISO 8601)"),
    date_to: datetime | None = Query(None, description="End date (ISO 8601)"),
    db: Session = Depends(get_db),
) -> list[BenchmarkRecentResult]:
    """Recent benchmark results with key metrics."""
    svc = BenchmarkAnalyticsService(db)
    results = svc.get_recent_results(
        limit=limit,
        scenario_id=scenario_id,
        benchmark_mode=benchmark_mode,
        date_from=date_from,
        date_to=date_to,
    )
    return [BenchmarkRecentResult(**r.to_dict()) for r in results]
