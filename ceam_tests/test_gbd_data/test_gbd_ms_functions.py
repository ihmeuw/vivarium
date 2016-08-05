# ~/ceam/ceam_tests/test_gbd_data/test_gbd_ms_functions.py

import numpy as np
import ceam.gbd_data.gbd_ms_functions

def test_get_sbp_mean_sd_Kenya_2000():
    # set the parameters
    location_id = 180 # Kenya
    year_start = 2000
    year_end = 2015

    # load the sbp data
    df = ceam.gbd_data.gbd_ms_functions.get_sbp_mean_sd(location_id, year_start, year_end)

    # reshape it so it is easy to access
    df = df.groupby(['year_id', 'sex_id', 'age']).first()

    # check if the value for 25 year old males matches the csv
    assert np.allclose(df.loc[(2000, 1, 25), 'log_mean'], np.log(118.948299)), 'should match data loaded by @aflaxman on 8/4/2016'

# End.
