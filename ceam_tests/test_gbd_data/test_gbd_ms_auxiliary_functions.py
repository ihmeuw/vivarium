# em's first pass at unit testing
# 1. set_age_year_index
from ceam.gbd_data.gbd_ms_auxiliary_functions import set_age_year_index
from ceam.gbd_data.gbd_ms_auxiliary_functions import interpolate_linearly_over_years_then_ages
from ceam.gbd_data.gbd_ms_auxiliary_functions import get_age_from_age_group_id
from ceam.gbd_data.gbd_ms_auxiliary_functions import create_age_column
from ceam.gbd_data.gbd_ms_auxiliary_functions import get_populations
from ceam.gbd_data.gbd_ms_auxiliary_functions import normalize_for_simulation
from ceam.gbd_data.gbd_ms_auxiliary_functions import expand_grid
from ceam.gbd_data.gbd_ms_auxiliary_functions import extrapolate_ages
from ceam.gbd_data.gbd_ms_auxiliary_functions import assign_sex_id
from scipy import stats
import pandas as pd
import numpy as np


def test_set_age_year_index():
    input_df = pd.DataFrame({'age' : np.arange(0, 6), 'year_id': np.arange(1800, 1806), "qty_of_interest" : np.random.randint(100, size =6)})
    
    indexer_test = set_age_year_index(input_df, 0, 10, 1800, 1805)

    assert indexer_test.columns.ravel() == 'qty_of_interest'
    
    # assert indexer_test.index.get_level_values('age') == np.repeat(np.arange(0, 11), 6)

    assert indexer_test.index.get_level_values('year_id') == np.repeat(np.arange(1800, 1806), 11)
    
    # assert that qty_of_interest is not null for age 0-5
    initial_ages = indexer_test[indexer_test.notnull(indexer_test['qty_of_interest'])]
    assert initial_ages.age.values.ravel() == np.arange(0, 5)

    # assert that qty_of_interest is null for ages outside of age 0-5
    post_index_ages = indexer_test[indexer_test.isnull(indexer_test['qty_of_interest'])]
    assert post_index_ages.age.values.ravel() == np.arage(6, 10)


# 2. interpolate_linearly_over_years_then_ages
def test_interpolate_linearly_over_years_then_ages():    
    
    # read in some real data, pick some years to interpolate between, and make sure the results make sense
    # use np.allclose to make sure that the interpolated value between 2 points is correctly interpolated
    interp_test = pd.read_csv("/share/costeffectiveness/CEAM/cache/draws_for_location163_for_meid2412.csv")
    interp_test = get_age_from_age_group_id(interp_test)
    interp_test = interp_test.query("sex_id ==1 and measure_id ==5")
    interp_test = interp_test[interp_test.age.isin([10, 20])]
    indexed_data = set_age_year_index(interp_test, 10, 20, 1990, 2015)
    interpolated_data = interpolate_linearly_over_years_then_ages(indexed_data, 'draw')[['draw_0']]
    interpolated_data.reset_index(level=0, inplace=True)
    interpolated_data.reset_index(level=1, inplace=True)
    value_at_fifteen = interpolated_data.query("age == 15 and year_id == 2015").draw_0.values
    value_at_ten = interpolated_data.query("age == 10 and year_id == 2015").draw_0.values
    value_at_twenty = interpolated_data.query("age == 20 and year_id == 2015").draw_0.values

    assert np.allclose(value_at_fifteen, (value_at_ten + value_at_twenty) *.5 ), "if interpolated correctly, these should be the same"    

# 3. create_age_column
def test_create_age_column():

    # see if you can use a fake population here instead

    pop = get_populations(180, 1990, 3)
    total_pop_value = pop.sum()['pop_scaled']
    pop['proportion_of_total_pop'] = pop['pop_scaled'] / total_pop_value
    simulants = pd.DataFrame({'simulant_id': range(0, 500000)})
    simulants = create_age_column(simulants, pop, 500000)
    simulants['count'] = 1
    simulants = simulants[['age', 'count']].groupby('age').sum()
    simulants['proportion'] = simulants['count'] / 500000
    
    # now check that the output proportions are close to the input proportions (em arbitrarily picked .1)
    # might need to revamp py test to allow for periodic transient failures since this function is based on randomness
    assert np.allclose(simulants.proportion.values, pop.proportion_of_total_pop.values, .1), 'need to make sure input/output proportions are close'


# 4. normalize_for_simulation
def test_normalize_for_simulation():
    df = pd.DataFrame({'sex_id': [1, 2], 'year_id': [1990, 1995]})
    df = normalize_for_simulation(df)

    assert df.columns.tolist() == ['year', 'sex'], 'normalize_for_simulation column names should be year and sex'
    assert df.sex.tolist() == ['Male', 'Female'], 'sex values should take categorical values Male, Female, or Both'

# 5. get_age_from_age_group_id
def test_get_age_from_age_group_id():
    df = pd.DataFrame({'age_group_id': np.arange(2, 22)})
    df = get_age_from_age_group_id(df)
    df.age = df.age.astype(int)
    assert df.columns.tolist() == ['age_group_id', 'age'], "get_age_from_age_group_id did not work. did not create an age column"
    assert df.age.tolist() == [0, 0, 0, 1] + [ x for x in np.arange(5, 81, 5)], "get_age_from_age_group_id didn't return the expected values in the age column"
 

# 6. expand_grid
def test_expand_grid():
    ages = pd.Series([0, 1, 2, 3, 4, 5])
    years = pd.Series([1990, 1991, 1992])

    df = expand_grid(ages, years)
    
    assert df.year_id.tolist() == np.repeat([1990, 1991, 1992], 6), "expand_grid should expand a df to get row for each age/year combo"
    assert df.age.tolist() == [0, 1, 2, 3, 4, 5] + [0, 1, 2, 3, 4, 5] + [0, 1, 2, 3, 4, 5], "expand_grid should expand a df to get row for each age/year combo"


# 7. extrapolate_ages
def test_extrapolate_ages():
    df = pd.DataFrame({'age': [80], 'qty_of_interest': [100], 'year_id': [2000]})
    indexed_df = set_age_year_index(df, 80, 80, 2000, 2000)
    extrapolated_df = extrapolate_ages(indexed_df, 100, 2000, 2000)

    assert extrapolated_df.age.tolist() == [x for x in range(80, 101)], "extrapolate ages did not extrapolate properly"

# 8. assign_sex_id
# create fake populations of men/women and assign sex id while making sure it's correlated with age
def test_assign_sex_id():
    male_pop = pd.DataFrame({'age': [0, 5, 10, 15, 20], 'pop_scaled': [0, 25000, 50000, 75000, 10000]})
    female_pop = pd.DataFrame({'age': [0, 5, 10, 15, 20], 'pop_scaled': [100000, 75000, 50000, 25000, 0]})

    simulants = pd.DataFrame({'simulant_id': range(0, 500000), 'age': np.repeat([0, 5, 10, 15, 20], 100000)})

    df = assign_sex_id(simulants, male_pop, female_pop)

    # age 0 should be all women, 5 should be 75%, 10 should be 50%, and so on
    df['count'] = 1
    grouped = df[['sex_id', 'age', 'count']].groupby(['sex_id', 'age']).sum()
    grouped['proportion'] = grouped['count'] / 100000
    
    assert np.allclose(grouped.proportion.tolist(), [x for x in np.arange(.25, 1.25, .25)] + [x for x in np.arange(0, 1.25, .25)][4:0:-1], .1), 'assign_sex_id needs to assign sexes so that they are correlated with age'


# End.

