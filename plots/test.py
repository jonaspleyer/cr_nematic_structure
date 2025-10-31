import cr_nematic_structure as crn

µm = crn.MICRO_METRE

if __name__ == "__main__":
    config = crn.Configuration()
    config.t_max /= 6
    config.save_interval = config.t_max / 20
    config.n_agents = [(3, 3), (3, 7)]  # [(3, 2), (6, 7)]
    config.domain_size = (400 * µm, 400 * µm, 20 * µm)
    config.initial_domain_size = (50 * µm, 50 * µm, 0.0)
    config.initial_domain_middle = (200 * µm, 200 * µm, 10 * µm)
    config.damping = 0.2

    output_path = crn.run_simulation(config).parent

    crn.plot_all_spheres(
        output_path,
        config,
        transparent_background=True,
        overwrite=True,
    )
