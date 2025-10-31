import cr_nematic_structure as crn


def test_run_sim():
    config = crn.Configuration()
    config.t_max = 0.1
    crn.run_simulation(config)
