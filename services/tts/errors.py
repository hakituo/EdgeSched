"""TTS 子系统显式错误。"""


class TTSServiceError(RuntimeError):
    """TTS 子系统基础错误。"""


class UnsupportedFeatureError(TTSServiceError):
    """当前 Base 模型不支持所请求的能力。"""


class VoiceProfileError(TTSServiceError):
    """音色画像缺失或校验失败。"""


class WorkerProtocolError(TTSServiceError):
    """主进程与 Worker 的协议不一致。"""


class WorkerCrashedError(TTSServiceError):
    """TTS Worker 异常退出。"""


class QueueClosedError(TTSServiceError):
    """有界队列已经关闭。"""


class TTSRequestCancelled(TTSServiceError):
    """请求已经取消，不能继续排队。"""
