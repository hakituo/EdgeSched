"""任务数据模型与执行上下文。

从 task_scheduler.py 拆出：TaskPriority / TaskStatus / TaskType / TaskInfo
以及 current_task_id 的 contextvar 管理。
task_scheduler.py 仍 re-export 这些符号，外部导入路径不变。
"""
import contextvars
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# Task context
_current_task_id_ctx = contextvars.ContextVar("current_task_id", default=None)


def get_current_task_id() -> Optional[str]:
    """Get current task ID"""
    return _current_task_id_ctx.get()


def set_current_task_id(task_id: str):
    """在当前上下文中设置 task_id，返回 token 用于 reset"""
    return _current_task_id_ctx.set(task_id)


def reset_current_task_id(token) -> None:
    """恢复 set_current_task_id 之前的上下文"""
    _current_task_id_ctx.reset(token)


class TaskPriority(Enum):
    """Task Priority Enum"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

class TaskStatus(Enum):
    """Task Status Enum"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskType(Enum):
    """Task Type Enum"""
    DEFAULT = "default"
    CPU_BOUND = "cpu"
    GPU_BOUND = "gpu"

@dataclass(order=True)
class TaskInfo:
    """Task Info Data Class"""
    task_id: str
    name: str
    priority: TaskPriority = field(compare=False)
    created_at: float = field(compare=False)
    status: TaskStatus = field(compare=False)
    task_type: TaskType = field(default=TaskType.DEFAULT, compare=False)
    result: Optional[Any] = field(default=None, compare=False)
    error: Optional[str] = field(default=None, compare=False)
    start_time: Optional[float] = field(default=None, compare=False)
    end_time: Optional[float] = field(default=None, compare=False)
    cancel_requested: bool = field(default=False, compare=False)
    trace_id: str = field(default="system", compare=False)
