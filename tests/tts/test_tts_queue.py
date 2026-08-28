import asyncio

import pytest

from domain.models.tts import TTSChunkRequest
from services.tts.errors import QueueClosedError, TTSRequestCancelled
from services.tts.queue import BoundedTTSQueue


def _request(request_id: str, chunk_id: int) -> TTSChunkRequest:
    return TTSChunkRequest(
        request_id=request_id,
        request_epoch=0,
        chunk_id=chunk_id,
        raw_text=f"句子{chunk_id}",
        synthesis_text=f"句子{chunk_id}",
        language="Chinese",
        voice_profile_id="calm",
    )


def test_cancel_atomically_removes_pending_and_wakes_producer() -> None:
    async def run() -> None:
        queue = BoundedTTSQueue(maxsize=1)
        await queue.put(_request("cancel", 0))
        blocked = asyncio.create_task(queue.put(_request("other", 0)))
        await asyncio.sleep(0)
        assert not blocked.done()

        removed = await queue.cancel(("cancel", 0))
        await asyncio.wait_for(blocked, timeout=1)
        assert [item.chunk_id for item in removed] == [0]
        assert (await queue.get()).request_id == "other"

        with pytest.raises(TTSRequestCancelled):
            await queue.put(_request("cancel", 1))

        await queue.close()
        with pytest.raises(QueueClosedError):
            await queue.get()

    asyncio.run(run())
