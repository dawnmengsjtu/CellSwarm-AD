# -*- coding: utf-8 -*-
"""Task queue for scheduling simulation jobs."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
import time


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """A single schedulable task."""
    task_id: str
    name: str
    callable_fn: Any = None
    kwargs: Dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def run(self) -> Any:
        self.status = TaskStatus.RUNNING
        try:
            if self.callable_fn:
                self.result = self.callable_fn(**self.kwargs)
            self.status = TaskStatus.COMPLETED
            self.completed_at = time.time()
            return self.result
        except Exception as e:
            self.status = TaskStatus.FAILED
            self.error = str(e)
            self.completed_at = time.time()
            raise


class TaskQueue:
    """FIFO task queue with status tracking."""

    def __init__(self):
        self.tasks: List[Task] = []
        self._index: int = 0

    def add(self, task: Task) -> None:
        self.tasks.append(task)

    def next(self) -> Optional[Task]:
        while self._index < len(self.tasks):
            task = self.tasks[self._index]
            if task.status == TaskStatus.PENDING:
                return task
            self._index += 1
        return None

    def run_next(self) -> Optional[Task]:
        task = self.next()
        if task:
            task.run()
            self._index += 1
            return task
        return None

    def run_all(self) -> List[Task]:
        results = []
        while True:
            task = self.run_next()
            if task is None:
                break
            results.append(task)
        return results

    def get_status(self) -> Dict:
        counts = {}
        for t in self.tasks:
            s = t.status.value
            counts[s] = counts.get(s, 0) + 1
        return {"total": len(self.tasks), **counts}
