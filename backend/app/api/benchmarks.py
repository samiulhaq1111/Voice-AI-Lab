"""Benchmark API endpoints (Phase 5B)."""

from fastapi import APIRouter, Depends, HTTPException
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
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    BenchmarkScenarioResponse,
)

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
