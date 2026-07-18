# -*- coding: utf-8 -*-
"""Experiment data schema."""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetInfo:
    """Metadata for a public AD dataset."""
    name: str
    url: str
    description: str
    data_types: List[str] = field(default_factory=list)
    species: str = "human"
    access: str = "public"       # public, restricted, application
    citation: str = ""

    def to_dict(self):
        return {
            "name": self.name,
            "url": self.url,
            "description": self.description,
            "data_types": self.data_types,
            "species": self.species,
            "access": self.access,
        }


@dataclass
class ExperimentData:
    """Container for experimental timeseries data."""
    name: str
    time_points: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    unit: str = ""
    source: str = ""

    def __post_init__(self):
        if len(self.time_points) != len(self.values):
            raise ValueError("time_points and values must have same length")
