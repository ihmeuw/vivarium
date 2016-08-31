# ~/ceam/ceam_tests/test_gbd_data/test_gbd_ms_functions.py

import pytest
import numpy as np
import ceam.gbd_data.gbd_ms_functions
from ceam.gbd_data.gbd_ms_functions import get_cause_level_prevalence
from ceam.gbd_data.gbd_ms_functions import get_relative_risks
from ceam.gbd_data.gbd_ms_functions import get_pafs
from ceam.gbd_data.gbd_ms_functions import get_exposures

def test_get_sbp_mean_sd_Kenya_2000():
    # set the parameters
    location_id = 180 # Kenya
    year_start = 2000
    year_end = 2015

    # load the sbp data
    df = ceam.gbd_data.gbd_ms_functions.get_sbp_mean_sd(location_id, year_start, year_end)

    df = df[['year_id', 'sex_id', 'age', 'log_mean_0']]

    # reshape it so it is easy to access
    df = df.groupby(['year_id', 'sex_id', 'age']).first()

    # check if the value for 25 year old males matches the csv
    assert np.allclose(df.loc[(2000, 1, 25), 'log_mean_0'], np.log(118.948299)), 'should match data loaded by @aflaxman on 8/4/2016'

@pytest.mark.xfail
def test_get_sbp_mean_sd_2001():
    # load the sbp data
    df = ceam.gbd_data.gbd_ms_functions.get_sbp_mean_sd(80, 2001, 2001, 0)
    # would be nice if this worked


# em's unit tests
# 1. get_modelable_entity_draws
    # no unit test needed for this function. will produce graphs instead


# 2. assign_cause_at_the_beginning_of_the_simulation
    # need to prove that total cause envelope (e.g. ihd) is correct
	# need to prove that each sequelae is correctly assigned
            # these should both be done graphically

# 2a. get_cause_level_prevalence
def test_get_cause_level_prevalence():
    # pass in a states dict with only two sequela and make sure for one age/sex/year combo that the value in cause_level_prevalence is equal to the sum of the two seq prevalences
    dict_of_disease_states = {'severe_heart_failure' : 1823, 'moderate_heart_failure' : 1822}    
    
    # pick a random draw to test
    draw_number = np.random.randint(low=0, high=1000, size=1)[0]
    cause_level, seq_level_dict = get_cause_level_prevalence(dict_of_disease_states, 180, 1990, draw_number)

    # pick a random age and sex to test
    sex = np.random.randint(low=1, high=3, size=1)[0]
    age = np.random.randint(low=1, high=81, size=1)[0]

    # get a prevaelnce estimate for the random age and sex that we want to test
    moderate_heart_failure = seq_level_dict['moderate_heart_failure'].query("age == {a} and sex_id =={s}".format(a=age, s=sex))
    seq_prevalence_1 = moderate_heart_failure['draw_{}'.format(draw_number)].values[0]
    severe_heart_failure = seq_level_dict['severe_heart_failure'].query("age == {a} and sex_id =={s}".format(a=age, s=sex))
    seq_prevalence_2 = severe_heart_failure['draw_{}'.format(draw_number)].values[0]
    
    # add up the prevalences of the 2 sequela to see if we get cause-level prevalence
    cause_level = cause_level.query("age == {a} and sex_id =={s}".format(a=age, s=sex))
    cause_prev = cause_level['draw_{}'.format(draw_number)].values[0]    
    
    assert cause_prev == seq_prevalence_1 + seq_prevalence_2, 'get_cause_level_prevalence error. seq prevs need to add up to cause prev'


# 3. get_relative_risks
def test_get_relative_risks():
    df = get_relative_risks(180, 1990, 1990, 107, 493)

    # pick a random draw to test
    draw_number = np.random.randint(low=0, high=1000, size=1)[0]
    
    # pick a random age and sex to test
    sex = np.random.randint(low=1, high=3, size=1)[0]
    age = np.random.randint(low=1, high=25, size=1)[0]

    # assert that relative risks are 1 for people under age 
    df_filter = df.query("age == {a} and sex_id == {s}".format(a=age, s=sex))
    rr = df['rr_{}'.format(draw_number)].values[0]

    assert rr == 1.0, 'get_relative_risks should return rr=1 for younger ages for the risks which dont estimate relative risk for all ages'


# 4. get_pafs
def test_get_pafs():
    df = get_pafs(180, 1990, 1990, 107, 493) 

    # pick a random draw to test
    draw_number = np.random.randint(low=0, high=1000, size=1)[0]

    # pick a random age and sex to test
    sex = np.random.randint(low=1, high=3, size=1)[0]
    age = np.random.randint(low=1, high=25, size=1)[0]

    # assert that relative risks are 1 for people under age
    df_filter = df.query("age == {a} and sex_id == {s}".format(a=age, s=sex))

    pafs = df['draw_{}'.format(draw_number)].values[0]

    assert pafs == 0, 'get_pafs should return paf=0 for the ages for which we do not have GBD estimates'

# get_exposures
def test_get_exposures():
def test_get_pafs():
    df = get_exposures(180, 1990, 1990, 107)

    # pick a random draw to test
    draw_number = np.random.randint(low=0, high=1000, size=1)[0]

    # pick a random age and sex to test
    sex = np.random.randint(low=1, high=3, size=1)[0]
    age = np.random.randint(low=1, high=25, size=1)[0]

    # assert that relative risks are 1 for people under age
    df_filter = df.query("age == {a} and sex_id == {s}".format(a=age, s=sex))

    exposure = df['draw_{}'.format(draw_number)].values[0]

    assert exposure ==0, 'get_exposure should return exposure=0 for the ages for which we do not have GBD estimates'


