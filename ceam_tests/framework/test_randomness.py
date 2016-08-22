
def test_filter_for_probability():
    pop = pd.DataFrame(dict(age=[0]*10000, simulant_id=range(10000)))

    sub_pop = filter_for_probability(pop, 0.5)
    assert round(len(sub_pop)/len(pop), 1) == 0.5

    sub_sub_pop = filter_for_probability(sub_pop, 0.5)
    assert round(len(sub_sub_pop)/len(sub_pop), 1) == 0.5

