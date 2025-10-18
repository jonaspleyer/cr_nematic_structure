import cr_nematic_structure as crn

if __name__ == "__main__":
    config = crn.Configuration()
    config.n_vertices = 3
    crn.run_simulation(config)
