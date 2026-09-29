import logging

from hindsight_client import Hindsight

from core.config import get_settings

logger = logging.getLogger(__name__)
_client: Hindsight | None = None


def get_hindsight_client() -> Hindsight | None:
    global _client
    settings = get_settings()
    if not settings.hindsight_base_url.strip():
        return None
    if _client is None:
        _client = Hindsight(
            base_url=settings.hindsight_base_url.strip().rstrip("/"),
            api_key=settings.hindsight_api_key.strip() or None if settings.hindsight_api_key else None,
            timeout=settings.hindsight_timeout_seconds,
        )
    return _client


async def ensure_hindsight_bank() -> None:
    client = get_hindsight_client()
    if client is None:
        logger.info("Hindsight is disabled; set HINDSIGHT_BASE_URL to enable it")
        return

    settings = get_settings()
    try:
        await client.acreate_bank(
            bank_id=settings.hindsight_bank_id,
            name="AegisAI Customer Success",
            mission=(
                "Remember customer-success recommendations, reviewer decisions, "
                "feedback, and measured outcomes to improve future analyses."
            ),
        )
        logger.info("Hindsight bank ready: %s", settings.hindsight_bank_id)
    except Exception as exc:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code == 409:
            logger.info("Hindsight bank already exists: %s", settings.hindsight_bank_id)
            return
        logger.warning("Could not initialize Hindsight bank: %s", exc)


async def close_hindsight_client() -> None:
    global _client
    if _client is not None:
        try:
            await _client.aclose()
        except Exception as exc:
            logger.warning("Could not close Hindsight client cleanly: %s", exc)
        finally:
            _client = None