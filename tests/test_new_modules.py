# -*- coding: utf-8 -*-
"""Tests for data platform and spatial modules."""
import pytest
import numpy as np
from data_platform.parameter_db import ADParameterDB, Parameter
from data_platform.experiment_db import ExperimentData, DatasetInfo, AD_DATASETS
from data_platform.validation import rmse, r_squared, mean_absolute_error, normalized_rmse, ModelComparator
from layer1_tissue.spatial import MultiSpeciesGrid, CellPlacer, CellMigration, SpatialStats


class TestParameterDB:
    def test_load_all(self):
        db = ADParameterDB()
        assert db.summary()["total_params"] >= 14

    def test_get_param(self):
        db = ADParameterDB()
        p = db.get("abeta_production_rate")
        assert p.value == 0.1
        assert p.unit == "nM/s"

    def test_get_missing_raises(self):
        db = ADParameterDB()
        with pytest.raises(KeyError):
            db.get("nonexistent")

    def test_query_by_category(self):
        db = ADParameterDB()
        abeta = db.query(category="abeta")
        assert len(abeta) >= 3
        assert all(p.category == "abeta" for p in abeta)

    def test_query_by_confidence(self):
        db = ADParameterDB()
        high = db.query(min_confidence=0.9)
        assert all(p.confidence >= 0.9 for p in high)

    def test_parameter_in_range(self):
        p = Parameter(name="test", value=0.5, unit="x", source="test", min_value=0.0, max_value=1.0)
        assert p.in_range(0.5)
        assert not p.in_range(1.5)

    def test_parameter_invalid_confidence(self):
        with pytest.raises(ValueError):
            Parameter(name="test", value=1.0, unit="x", source="test", confidence=1.5)


class TestExperimentDB:
    def test_datasets_exist(self):
        assert len(AD_DATASETS) >= 4

    def test_adni_info(self):
        adni = [d for d in AD_DATASETS if d.name == "ADNI"][0]
        assert "adni" in adni.url.lower()

    def test_experiment_data(self):
        ed = ExperimentData(name="test", time_points=[0, 1, 2], values=[0.1, 0.2, 0.3], unit="nM")
        assert len(ed.time_points) == 3

    def test_experiment_data_mismatch(self):
        with pytest.raises(ValueError):
            ExperimentData(name="test", time_points=[0, 1], values=[0.1])


class TestValidationMetrics:
    def test_rmse_perfect(self):
        assert rmse([1, 2, 3], [1, 2, 3]) == 0.0

    def test_rmse_known(self):
        r = rmse([1, 2, 3], [1.1, 2.1, 3.1])
        assert abs(r - 0.1) < 0.001

    def test_r_squared_perfect(self):
        assert r_squared([1, 2, 3], [1, 2, 3]) == 1.0

    def test_r_squared_bad(self):
        r2 = r_squared([3, 2, 1], [1, 2, 3])
        assert r2 < 0

    def test_mae(self):
        assert mean_absolute_error([1, 2, 3], [1.1, 2.1, 3.1]) == pytest.approx(0.1, abs=0.001)

    def test_nrmse(self):
        n = normalized_rmse([1, 2, 3], [1.1, 2.1, 3.1])
        assert n > 0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            rmse([], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            rmse([1, 2], [1])


class TestModelComparator:
    def test_compare_perfect(self):
        mc = ModelComparator(rmse_threshold=0.1, r2_threshold=0.9)
        r = mc.compare("test", [1, 2, 3], [1, 2, 3])
        assert r.passed

    def test_compare_bad(self):
        mc = ModelComparator(rmse_threshold=0.01)
        r = mc.compare("test", [1, 2, 3], [2, 3, 4])
        assert not r.passed

    def test_summary(self):
        mc = ModelComparator()
        mc.compare("a", [1, 2], [1, 2])
        mc.compare("b", [1, 2], [3, 4])
        s = mc.summary()
        assert s["total"] == 2


class TestMultiSpeciesGrid:
    def test_add_species(self):
        g = MultiSpeciesGrid(width=10, height=10)
        g.add_species("abeta", 0.1, 0.01)
        assert "abeta" in g.species

    def test_add_source(self):
        g = MultiSpeciesGrid(width=10, height=10)
        g.add_species("abeta")
        g.add_source("abeta", 5, 5, 1.0)
        assert g.get_concentration("abeta", 5, 5) == 1.0

    def test_step_diffuses(self):
        g = MultiSpeciesGrid(width=10, height=10)
        g.add_species("abeta", diffusion_rate=0.5, decay_rate=0.0)
        g.add_source("abeta", 5, 5, 10.0)
        g.step(dt=0.1)
        # Neighbors should have some concentration
        assert g.get_concentration("abeta", 5, 4) > 0

    def test_decay_reduces(self):
        g = MultiSpeciesGrid(width=10, height=10)
        g.add_species("abeta", diffusion_rate=0.0, decay_rate=0.5)
        g.add_source("abeta", 5, 5, 10.0)
        before = g.total("abeta")
        g.step(dt=0.1)
        assert g.total("abeta") < before

    def test_gradient(self):
        g = MultiSpeciesGrid(width=10, height=10)
        g.add_species("abeta")
        g.add_source("abeta", 7, 5, 10.0)
        gx, gy = g.get_gradient("abeta", 6, 5)
        assert gx > 0  # gradient points toward source

    def test_missing_species_raises(self):
        g = MultiSpeciesGrid(width=10, height=10)
        with pytest.raises(KeyError):
            g.get_concentration("nonexistent", 0, 0)


class TestCellPlacer:
    def test_uniform(self):
        cp = CellPlacer(width=50, height=50)
        pos = cp.uniform(20)
        assert len(pos) == 20
        assert all(0 <= x < 50 and 0 <= y < 50 for x, y in pos)

    def test_clustered(self):
        cp = CellPlacer(width=100, height=100)
        pos = cp.clustered(30, centers=[(25, 25), (75, 75)], sigma=5.0)
        assert len(pos) == 30

    def test_layered(self):
        cp = CellPlacer(width=100, height=100)
        pos = cp.layered(counts=[10, 20], y_ranges=[(0, 50), (50, 100)])
        assert len(pos) == 30


class TestCellMigration:
    def test_random_walk(self):
        cm = CellMigration(width=50, height=50)
        pos = [(25, 25), (10, 10)]
        new_pos = cm.random_walk(pos, step_size=2.0)
        assert len(new_pos) == 2
        assert all(0 <= x < 50 and 0 <= y < 50 for x, y in new_pos)

    def test_chemotaxis(self):
        g = MultiSpeciesGrid(width=20, height=20)
        g.add_species("abeta")
        g.add_source("abeta", 15, 10, 10.0)
        cm = CellMigration(width=20, height=20)
        pos = [(10, 10)]
        new_pos = cm.chemotaxis(pos, g, "abeta", strength=1.0)
        # Should move toward source (x=15)
        assert new_pos[0][0] >= 10


class TestSpatialStats:
    def test_nn_distances(self):
        pos = [(0, 0), (3, 4)]
        dists = SpatialStats.nearest_neighbor_distances(pos)
        assert abs(dists[0] - 5.0) < 0.01

    def test_clustering_index(self):
        pos = [(50, 50), (51, 51), (49, 49)]
        ci = SpatialStats.clustering_index(pos, 100, 100)
        assert ci < 1.0  # clustered

    def test_hotspot_count(self):
        grid = np.zeros((10, 10))
        grid[5, 5] = 1.0
        grid[3, 3] = 0.5
        assert SpatialStats.hotspot_count(grid, 0.3) == 2
