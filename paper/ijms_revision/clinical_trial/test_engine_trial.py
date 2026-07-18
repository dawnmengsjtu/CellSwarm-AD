import math

from run_engine_trial import ARMS, N_PER_ARM, VISITS, arm_definitions, dropout_schedule, pd_effect, simulate_trial
import numpy as np


def test_engine_trial_shape_and_dropout():
    data = simulate_trial(123)
    assert data.patient_id.nunique() == 4 * N_PER_ARM
    assert len(data) == 4 * N_PER_ARM * len(VISITS)
    roster = data.drop_duplicates("patient_id")
    assert roster.groupby("treatment").first_missing_week.apply(lambda x: x.notna().mean()).eq(.15).all()


def test_combination_is_repository_pd_bliss():
    definitions = arm_definitions()
    done = pd_effect(definitions["donepezil"])
    mem_arm = definitions["donepezil+memantine"]
    mem_pk, mem_pd = mem_arm.drugs[1]
    mem = mem_pd.effect(mem_pk.steady_state_concentration(24.0))
    assert math.isclose(pd_effect(mem_arm), 1 - (1-done)*(1-mem), abs_tol=1e-12)


def test_dropout_schedule_exact_quota():
    schedule = dropout_schedule(np.random.default_rng(1))
    assert (schedule != 999).sum() == 30
