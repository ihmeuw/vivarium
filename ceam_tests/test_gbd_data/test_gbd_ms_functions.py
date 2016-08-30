# ~/ceam/ceam_tests/test_gbd_data/test_gbd_ms_functions.py

import pytest
import numpy as np
import ceam.gbd_data.gbd_ms_functions

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


# 3. sum_up_csmrs_for_all_causes_in_microsim
    # prove that all of the csmrs are being summed up correctly

# 4. get_cause_deleted_mortality_rate
    # prove that subtraction is working correctly
	# e.g. do for one loc/sex/age combo and prove that each step equals what it should

# 5. get_heart_failure_incidence_draws
    # prove that function and multiplication are working correctly by doing for loc/sex/age combination 

# 6. get_relative_risks
    # prove that function is working by doing one loc/sex/age combination
	# make sure function is correctly filling in the ages we don't have from GBD correctly

# 7. get_pafs
    # same as relative risks test

# 8. get_exposures
    # same as relative risks

# 9. get_sbp_mean_sd
    # prove that function is working by running through calculations for one loc/sex/age combination
	# prove that np.log and log standard deviation functions are working


# End.

