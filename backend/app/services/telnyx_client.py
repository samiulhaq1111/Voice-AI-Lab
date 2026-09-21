"""Minimal Telnyx Call Control V2 HTTP client.

Uses httpx to call the Telnyx REST API for inbound call control.
Only the operations needed for the telephony integration are implemented.
"""

import httpx

from app.core.config import settings
from app.core.logging import logger

_TELNYX_API_BASE = "https://api.telnyx.com/v2"


class TelnyxCallControlClient:
    """Thin httpx wrapper for Telnyx Call Control V2 actions."""

    def __init__(self, *, api_key: str | None = None) -> None:
        resolved_key = api_key or settings.telnyx_api_key
        if not resolved_key:
            raise ValueError(
                "Telnyx API key not configured. Set TELNYX_API_KEY."
            )
        self._client = httpx.AsyncClient(
            base_url=_TELNYX_API_BASE,
            timeout=httpx.Timeout(10.0),
            headers={
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
            },
        )

    async def answer_call(self, call_control_id: str) -> dict:
        """Answer an inbound call via Telnyx Call Control.

        POST /v2/calls/{call_control_id}/actions/answer

        Returns the parsed JSON response body on success.
        Raises RuntimeError on failure.
        """
        url = f"/calls/{call_control_id}/actions/answer"
        logger.info(
            "[VOICE:TELNYX] answer_call start call_control_id=%s",
            call_control_id,
        )
        try:
            response = await self._client.post(url, json={})
            logger.info(
                "[VOICE:TELNYX] answer_call response status=%d",
                response.status_code,
            )
            response.raise_for_status()
            body = response.json()
            logger.info(
                "[VOICE:TELNYX] call.answered sent call_control_id=%s",
                call_control_id,
            )
            return body if isinstance(body, dict) else {}
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            error_detail = ""
            try:
                err_body = e.response.json()
                errors = err_body.get("errors", [])
                if errors and isinstance(errors, list):
                    error_detail = errors[0].get("detail", str(errors[0]))
                else:
                    error_detail = str(err_body)[:200]
            except Exception:
                error_detail = e.response.text[:200] if e.response.text else ""
            logger.error(
                "[VOICE:TELNYX] answer_call failed status=%d detail=%s",
                status,
                error_detail[:200],
            )
            raise RuntimeError(
                f"Telnyx answer failed (HTTP {status}): {error_detail}"
            ) from e
        except httpx.TimeoutException:
            logger.error("[VOICE:TELNYX] answer_call timed out")
            raise RuntimeError("Telnyx answer timed out")
        except httpx.HTTPError as e:
            logger.error(
                "[VOICE:TELNYX] answer_call network error type=%s",
                type(e).__name__,
            )
            raise RuntimeError(f"Telnyx HTTP error: {e}") from e

    async def speak(
        self, call_control_id: str, text: str, *, voice: str | None = None
    ) -> dict:
        """Issue a speak action via Telnyx Call Control.

        POST /v2/calls/{call_control_id}/actions/speak

        Args:
            call_control_id: The call to speak on.
            text: The text to synthesize.
            voice: Telnyx voice identifier. Defaults to settings.telnyx_speak_voice.

        Returns the parsed JSON response body on success.
        Raises RuntimeError on failure.
        """
        resolved_voice = voice or settings.telnyx_speak_voice
        url = f"/calls/{call_control_id}/actions/speak"
        logger.info(
            "[VOICE:TELNYX] speak start call_control_id=%s text_length=%d voice=%s",
            call_control_id,
            len(text),
            resolved_voice,
        )
        try:
            response = await self._client.post(
                url,
                json={
                    "payload": text,
                    "voice": resolved_voice,
                    "language": "en-US",
                    "service_level": "basic",
                },
            )
            logger.info(
                "[VOICE:TELNYX] speak response status=%d",
                response.status_code,
            )
            response.raise_for_status()
            body = response.json()
            logger.info(
                "[VOICE:TELNYX] speak sent call_control_id=%s",
                call_control_id,
            )
            return body if isinstance(body, dict) else {}
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            error_detail = ""
            try:
                err_body = e.response.json()
                errors = err_body.get("errors", [])
                if errors and isinstance(errors, list):
                    error_detail = errors[0].get("detail", str(errors[0]))
                else:
                    error_detail = str(err_body)[:200]
            except Exception:
                error_detail = e.response.text[:200] if e.response.text else ""
            logger.error(
                "[VOICE:TELNYX] speak failed status=%d detail=%s",
                status,
                error_detail[:200],
            )
            raise RuntimeError(
                f"Telnyx speak failed (HTTP {status}): {error_detail}"
            ) from e
        except httpx.TimeoutException:
            logger.error("[VOICE:TELNYX] speak timed out")
            raise RuntimeError("Telnyx speak timed out")
        except httpx.HTTPError as e:
            logger.error(
                "[VOICE:TELNYX] speak network error type=%s",
                type(e).__name__,
            )
            raise RuntimeError(f"Telnyx HTTP error: {e}") from e

    async def streaming_start(
        self,
        call_control_id: str,
        stream_url: str,
        *,
        stream_track: str = "both_tracks",
        bidirectional_mode: str = "rtp",
        bidirectional_codec: str = "PCMU",
        bidirectional_target_legs: str = "opposite",
    ) -> dict:
        """Start bidirectional media streaming on a call.

        POST /v2/calls/{call_control_id}/actions/streaming_start

        Returns the parsed JSON response body on success.
        Raises RuntimeError on failure.
        """
        url = f"/calls/{call_control_id}/actions/streaming_start"
        body = {
            "stream_url": stream_url,
            "stream_track": stream_track,
            "stream_bidirectional_mode": bidirectional_mode,
            "stream_bidirectional_codec": bidirectional_codec,
            "stream_bidirectional_target_legs": bidirectional_target_legs,
        }
        logger.info(
            "[VOICE:TELNYX] streaming_start call_control_id=%s stream_url=%s "
            "mode=%s codec=%s",
            call_control_id,
            stream_url,
            bidirectional_mode,
            bidirectional_codec,
        )
        try:
            response = await self._client.post(url, json=body)
            logger.info(
                "[VOICE:TELNYX] streaming_start response status=%d",
                response.status_code,
            )
            response.raise_for_status()
            body_resp = response.json()
            logger.info(
                "[VOICE:TELNYX] streaming started call_control_id=%s",
                call_control_id,
            )
            return body_resp if isinstance(body_resp, dict) else {}
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            error_detail = ""
            try:
                err_body = e.response.json()
                errors = err_body.get("errors", [])
                if errors and isinstance(errors, list):
                    error_detail = errors[0].get("detail", str(errors[0]))
                else:
                    error_detail = str(err_body)[:200]
            except Exception:
                error_detail = e.response.text[:200] if e.response.text else ""
            logger.error(
                "[VOICE:TELNYX] streaming_start failed status=%d detail=%s",
                status,
                error_detail[:200],
            )
            raise RuntimeError(
                f"Telnyx streaming_start failed (HTTP {status}): {error_detail}"
            ) from e
        except httpx.TimeoutException:
            logger.error("[VOICE:TELNYX] streaming_start timed out")
            raise RuntimeError("Telnyx streaming_start timed out")
        except httpx.HTTPError as e:
            logger.error(
                "[VOICE:TELNYX] streaming_start network error type=%s",
                type(e).__name__,
            )
            raise RuntimeError(f"Telnyx HTTP error: {e}") from e

    async def close(self) -> None:
        """Release the underlying httpx client."""
        await self._client.aclose()
