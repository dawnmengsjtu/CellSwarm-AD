# -*- coding: utf-8 -*-
"""Tests for drug models: library, combination, trial simulator."""
import pytest
from layer2_pathology.drug_models.pharmacokinetics import PKModel, PKParameters, TwoCompartmentPKModel, RouteOfAdministration
from layer2_pathology.drug_models.pharmacodynamics import PDModel, PDParameters, DrugType, InhibitionType
from layer2_pathology.drug_models.drug_library import DrugLibrary, get_drug, DRUG_LIBRARY, Drug
from layer2_pathology.drug_models.combination import (
    CombinationModel, bliss_independence, loewe_additivity, InteractionType
)
from layer2_pathology.drug_models.trial_simulator import (
    TrialSimulator, Patient, TrialArm, TrialResult
)


class TestDrugLibrary:
    def test_four_drugs_exist(self):
        lib = DrugLibrary()
        assert len(lib.list_drugs()) == 4

    def test_get_donepezil(self):
        d = get_drug("donepezil")
        assert d.brand_name == "Aricept"
        assert d.approved

    def test_get_lecanemab(self):
        d = get_drug("lecanemab")
        assert "protofibril" in d.mechanism.lower()

    def test_get_missing_raises(self):
        lib = DrugLibrary()
        with pytest.raises(KeyError):
            lib.get("aspirin")

    def test_by_mechanism(self):
        lib = DrugLibrary()
        antibodies = lib.by_mechanism("antibody")
        assert len(antibodies) == 2

    def test_by_target(self):
        lib = DrugLibrary()
        nmda = lib.by_target("NMDA")
        assert len(nmda) == 1
        assert nmda[0].name == "Memantine"

    def test_drug_summary(self):
        d = get_drug("aducanumab")
        s = d.summary()
        assert "name" in s
        assert s["approved"] is True

    def test_add_custom_drug(self):
        lib = DrugLibrary()
        custom = Drug(
            name="TestDrug",
            generic_name="test",
            brand_name="TestBrand",
            mechanism="test mechanism",
            target="test target",
            pk_params=PKParameters(dose=10, Vd=50, ke=0.1),
            pd_params=PDParameters(Emax=0.5, EC50=1.0),
            approved=False,
        )
        lib.add(custom)
        assert "testdrug" in lib.list_drugs()


class TestPKIntegration:
    def test_donepezil_pk(self):
        d = get_drug("donepezil")
        pk = PKModel(d.pk_params)
        result = pk.simulate(t_end=168)  # 1 week
        assert result.Cmax > 0
        assert result.half_life > 50  # ~70h

    def test_lecanemab_iv(self):
        d = get_drug("lecanemab")
        pk = TwoCompartmentPKModel(d.pk_params)
        result = pk.simulate(t_end=336)  # 2 weeks
        assert result.Cmax > 0

    def test_steady_state(self):
        d = get_drug("memantine")
        pk = PKModel(d.pk_params)
        css = pk.steady_state_concentration(12.0)  # BID
        assert css > 0


class TestPDIntegration:
    def test_donepezil_ache_inhibition(self):
        d = get_drug("donepezil")
        pd = PDModel(d.pd_params)
        effect = pd.effect(0.015)  # at EC50
        assert 0.3 < effect < 0.5  # ~50% of Emax at EC50

    def test_memantine_nmda_block(self):
        d = get_drug("memantine")
        pd = PDModel(d.pd_params)
        mods = pd.modify_nmda_params(0.5)
        assert mods["nmda_conductance"] < 1.0

    def test_lecanemab_amyloid_clearance(self):
        d = get_drug("lecanemab")
        pd = PDModel(d.pd_params)
        mods = pd.modify_amyloid_params(3.0, base_k_prod=0.1, base_k_clear=0.02)
        assert mods["k_clear"] > 0.02  # enhanced clearance


class TestBlissIndependence:
    def test_zero_effects(self):
        assert bliss_independence(0.0, 0.0) == 0.0

    def test_full_effects(self):
        assert bliss_independence(1.0, 1.0) == 1.0

    def test_one_zero(self):
        assert bliss_independence(0.5, 0.0) == 0.5

    def test_typical(self):
        result = bliss_independence(0.3, 0.4)
        assert abs(result - 0.58) < 0.01

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            bliss_independence(1.5, 0.5)


class TestLoeweAdditivity:
    def test_single_drug(self):
        ci = loewe_additivity(1.0, 0.0, ec50_a=1.0, ec50_b=1.0)
        assert abs(ci - 1.0) < 0.1

    def test_both_drugs(self):
        ci = loewe_additivity(0.5, 0.5, ec50_a=1.0, ec50_b=1.0)
        assert ci > 0  # should be calculable


class TestCombinationModel:
    def setup_method(self):
        self.pd_a = PDModel(PDParameters(Emax=0.8, EC50=1.0))
        self.pd_b = PDModel(PDParameters(Emax=0.6, EC50=2.0))
        self.combo = CombinationModel(self.pd_a, self.pd_b, "DrugA", "DrugB")

    def test_analyze_bliss(self):
        r = self.combo.analyze_bliss(1.0, 2.0)
        assert r.drug_a == "DrugA"
        assert 0 < r.expected_combined <= 1.0

    def test_analyze_loewe(self):
        r = self.combo.analyze_loewe(1.0, 2.0)
        assert r.combination_index > 0

    def test_checkerboard(self):
        matrix = self.combo.checkerboard([0.5, 1.0], [1.0, 2.0])
        assert len(matrix) == 2
        assert len(matrix[0]) == 2

    def test_result_to_dict(self):
        r = self.combo.analyze_bliss(1.0, 2.0)
        d = r.to_dict()
        assert "interaction" in d
        assert "CI" in d


class TestPatient:
    def test_default(self):
        p = Patient(patient_id=0, age=70)
        assert p.cognitive_score == 26.0

    def test_apoe4_increases_abeta(self):
        p0 = Patient(patient_id=0, age=70, apoe4_copies=0)
        p2 = Patient(patient_id=1, age=70, apoe4_copies=2)
        assert p2.baseline_abeta > p0.baseline_abeta

    def test_severe_stage(self):
        p_mild = Patient(patient_id=0, age=70, disease_stage="mild")
        p_severe = Patient(patient_id=1, age=70, disease_stage="severe")
        assert p_severe.baseline_abeta > p_mild.baseline_abeta


class TestTrialSimulator:
    def test_generate_population(self):
        sim = TrialSimulator(duration_weeks=4, seed=42)
        pop = sim.generate_population(20)
        assert len(pop) == 20
        assert all(55 <= p.age <= 85 for p in pop)

    def test_run_placebo_arm(self):
        sim = TrialSimulator(duration_weeks=26, seed=42)
        patients = sim.generate_population(20)
        arm = TrialArm(name="Placebo", is_placebo=True)
        result = sim.run_arm(patients, arm)
        assert result.n_patients == 20
        assert result.mean_cognitive_change < 0  # decline expected over 26 weeks

    def test_run_treatment_arm(self):
        sim = TrialSimulator(duration_weeks=4, seed=42)
        d = get_drug("lecanemab")
        patients = sim.generate_population(5)
        arm = TrialArm(
            name="Lecanemab",
            pk_model=TwoCompartmentPKModel(d.pk_params),
            pd_model=PDModel(d.pd_params),
            dosing_interval_h=336,  # q2w
        )
        result = sim.run_arm(patients, arm)
        assert result.n_patients == 5

    def test_run_trial(self):
        sim = TrialSimulator(duration_weeks=4, seed=42)
        d = get_drug("donepezil")
        arms = [
            TrialArm(name="Placebo", is_placebo=True),
            TrialArm(
                name="Donepezil",
                pk_model=PKModel(d.pk_params),
                pd_model=PDModel(d.pd_params),
                dosing_interval_h=24,
            ),
        ]
        results = sim.run_trial(arms, n_per_arm=5)
        assert "Placebo" in results
        assert "Donepezil" in results

    def test_compare_arms(self):
        r1 = TrialResult("Placebo", 50, -3.5, 1.2, 0.85)
        r2 = TrialResult("Drug", 50, -2.0, 1.1, 0.92)
        comp = TrialSimulator.compare_arms(r2, r1)
        assert comp["mean_diff"] > 0  # drug better
        assert "cohens_d" in comp

    def test_result_to_dict(self):
        r = TrialResult("Test", 10, -2.5, 1.0, 0.9)
        d = r.to_dict()
        assert d["arm"] == "Test"
        assert d["n"] == 10
