from fastapi import APIRouter
from sqlalchemy import text

from app.models.schemas import HealthResponse
from app.services.cache import cache_ping
from app.services.database import get_db_session
from app.config import get_settings

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    checks: dict[str, str] = {}

    # PostgreSQL
    try:
        async with get_db_session() as session:
            await session.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:
        checks["postgres"] = f"error: {exc}"

    # Redis
    checks["redis"] = "ok" if await cache_ping() else "error: unreachable"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"

    return HealthResponse(
        status=overall,
        version="1.0.0",
        environment=settings.environment,
        checks=checks,
    )
