# -*- coding: utf-8 -*-
"""Signal bus for inter-cell communication."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from enum import Enum


@dataclass
class CellMessage:
    """A message passed between cells."""
    sender_id: str
    receiver_id: str
    signal_type: str
    payload: Dict = field(default_factory=dict)
    timestamp: float = 0.0


class SignalBus:
    """Central message bus for cell-to-cell communication."""

    def __init__(self):
        self.messages: List[CellMessage] = []
        self.subscribers: Dict[str, List[Callable]] = {}

    def send(self, message: CellMessage) -> None:
        self.messages.append(message)
        if message.receiver_id in self.subscribers:
            for callback in self.subscribers[message.receiver_id]:
                callback(message)

    def subscribe(self, cell_id: str, callback: Callable) -> None:
        if cell_id not in self.subscribers:
            self.subscribers[cell_id] = []
        self.subscribers[cell_id].append(callback)

    def get_messages_for(self, cell_id: str) -> List[CellMessage]:
        return [m for m in self.messages if m.receiver_id == cell_id]

    def broadcast(self, sender_id: str, signal_type: str, payload: Dict) -> None:
        for cell_id in self.subscribers:
            msg = CellMessage(sender_id=sender_id, receiver_id=cell_id,
                              signal_type=signal_type, payload=payload, timestamp=0.0)
            self.send(msg)

    def clear(self) -> None:
        self.messages.clear()

    def message_count(self) -> int:
        return len(self.messages)
