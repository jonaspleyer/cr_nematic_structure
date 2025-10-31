use cellular_raza::prelude::*;
use pyo3::prelude::*;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

pub const MICRO_METRE: f64 = 1e-6;

pub const MINUTE: f64 = 1.0;
pub const HOUR: f64 = 60.0 * MINUTE;

#[derive(CellAgent, Clone, Deserialize, Serialize)]
pub struct Agent {
    #[Mechanics]
    mechanics: RodMechanics<f64, 3>,

    // Interaction
    interaction: RodInteraction<MorsePotential>,

    // Cycle
    growth_rate: f64,
    neighbors: usize,
    neighbor_cap: usize,
    spring_length_threshold: f64,
}

impl
    cellular_raza::concepts::Interaction<
        nalgebra::MatrixXx3<f64>,
        nalgebra::MatrixXx3<f64>,
        nalgebra::MatrixXx3<f64>,
        f64,
    > for Agent
{
    fn calculate_force_between(
        &self,
        own_pos: &nalgebra::MatrixXx3<f64>,
        own_vel: &nalgebra::MatrixXx3<f64>,
        ext_pos: &nalgebra::MatrixXx3<f64>,
        ext_vel: &nalgebra::MatrixXx3<f64>,
        ext_info: &f64,
    ) -> Result<(nalgebra::MatrixXx3<f64>, nalgebra::MatrixXx3<f64>), CalcError> {
        self.interaction
            .calculate_force_between(own_pos, own_vel, ext_pos, ext_vel, ext_info)
    }

    fn is_neighbor(
        &self,
        own_pos: &nalgebra::MatrixXx3<f64>,
        ext_pos: &nalgebra::MatrixXx3<f64>,
        _ext_inf: &f64,
    ) -> Result<bool, CalcError> {
        for own_point in own_pos.row_iter() {
            for ext_point in ext_pos.row_iter() {
                if (own_point - ext_point).norm() < 2.0 * self.interaction.0.radius {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    fn react_to_neighbors(&mut self, neighbors: usize) -> Result<(), CalcError> {
        self.neighbors = neighbors;
        Ok(())
    }
}

impl InteractionInformation<f64> for Agent {
    fn get_interaction_information(&self) -> f64 {
        self.interaction.0.radius
    }
}

impl Cycle<Agent> for Agent {
    fn update_cycle(
        _rng: &mut rand_chacha::ChaCha8Rng,
        dt: &f64,
        cell: &mut Agent,
    ) -> Option<cellular_raza::prelude::CycleEvent> {
        // Set the growth rate depending on the number of neighbors
        let growth_rate = if cell.neighbors > 0 {
            (cell.growth_rate * (cell.neighbor_cap - cell.neighbors) as f64
                / cell.neighbor_cap as f64)
                .max(0.0)
        } else {
            cell.growth_rate
        };

        if cell.mechanics.pos.nrows() == 1 {
            cell.interaction.0.radius += growth_rate * dt;
            if cell.interaction.0.radius > cell.spring_length_threshold {
                return Some(CycleEvent::Division);
            }
        } else {
            cell.mechanics.spring_length += growth_rate * dt;
            if cell.mechanics.spring_length > cell.spring_length_threshold {
                return Some(CycleEvent::Division);
            }
        }
        None
    }

    fn divide(
        rng: &mut rand_chacha::ChaCha8Rng,
        cell: &mut Agent,
    ) -> Result<Agent, cellular_raza::prelude::DivisionError> {
        let c2_mechanics = cell.mechanics.divide(rng, cell.interaction.0.radius)?;
        if c2_mechanics.pos.nrows() == 1 {
            cell.interaction.0.radius /= core::f64::consts::SQRT_2;
        }
        let mut c2 = cell.clone();
        c2.mechanics = c2_mechanics;
        Ok(c2)
    }
}

#[test]
fn test_divide_2_vertices() {
    let mut p1 = nalgebra::MatrixXx3::zeros(2);
    p1.set_row(0, &[0.0; 3].into());
    p1.set_row(1, &[2.0, 0.0, 0.0].into());
    let mut agent = Agent {
        mechanics: RodMechanics {
            pos: p1.clone(),
            vel: 0.0 * p1,
            diffusion_constant: 0.0,
            spring_tension: 1.0,
            rigidity: 1.0,
            spring_length: 2.0,
            damping: 0.1,
        },
        interaction: RodInteraction(MorsePotential {
            radius: 0.5,
            potential_stiffness: 0.5,
            cutoff: 1.0,
            strength: 0.1,
        }),
        growth_rate: 0.1,
        neighbors: 0,
        neighbor_cap: 1,
        spring_length_threshold: 2.0,
    };
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0);
    let new_agent =
        <Agent as cellular_raza::concepts::Cycle>::divide(&mut rng, &mut agent).unwrap();

    let dist = new_agent.mechanics.pos.row(0) - agent.mechanics.pos.row(1);
    assert_eq!(
        agent.mechanics.spring_length,
        new_agent.mechanics.spring_length
    );
    assert_eq!(agent.interaction.0.radius, new_agent.interaction.0.radius);
    assert!((dist.norm() - 2.0 * agent.interaction.0.radius).abs() < 1e-2);
}

short_default::default! {
#[pyo3_stub_gen::derive::gen_stub_pyclass]
#[pyclass(get_all, set_all)]
pub struct Configuration {
    spring_tension: f64 = 10.0 / MINUTE.powf(2.0),
    rigidity: f64 = 1.0 * MICRO_METRE / MINUTE.powf(2.0),
    damping: f64 = 1.5 / MINUTE,
    spring_length: f64 = 4.0 * MICRO_METRE,
    radius: f64 = 3.0 * MICRO_METRE,
    potential_stiffness: f64 = 0.5 / MICRO_METRE,
    strength: f64 = 0.1 * MICRO_METRE.powf(2.0) / MINUTE.powf(2.0),
    cutoff: f64 = 5.0 * MICRO_METRE,
    spring_length_threshold: f64 = 8.0 * MICRO_METRE,
    growth_rate: f64 = 0.1 * MICRO_METRE / MINUTE,
    neighbor_cap: usize = 8,
    t0: f64 = 0.0 * MINUTE,
    dt: f64 = 0.1 * MINUTE,
    save_interval: f64 = 25.0 * MINUTE,
    t_max: f64 = 25.0 * HOUR,
    domain_size: (f64, f64, f64) = (200.0 * MICRO_METRE, 100.0 * MICRO_METRE, 15.0 * MICRO_METRE),
    initial_domain_size: (f64, f64, f64) =
        (20.0 * MICRO_METRE, 20.0 * MICRO_METRE, 10.0 * MICRO_METRE),
    initial_domain_middle: (f64, f64, f64) =
        (100.0 * MICRO_METRE, 50.0 * MICRO_METRE, 7.5 * MICRO_METRE),
    domain_segments: [usize; 3] = [8, 1, 1],
    n_threads: usize = 1,
    n_agents: Vec<(usize, usize)> = vec![(5, 1), (5, 8)],
}}

#[pymethods]
impl Configuration {
    #[new]
    fn new() -> Self {
        Self::default()
    }
}

#[pyo3_stub_gen::derive::gen_stub_pyfunction]
#[pyfunction]
pub fn run_simulation(config: &Configuration) -> Result<std::path::PathBuf, SimulationError> {
    // Define initial random seed
    use rand::Rng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);

    let Configuration {
        spring_tension,
        rigidity,
        damping,
        spring_length,
        radius,
        potential_stiffness,
        strength,
        cutoff,
        spring_length_threshold,
        growth_rate,
        neighbor_cap,
        t0,
        dt,
        save_interval,
        t_max,
        domain_size,
        initial_domain_size,
        initial_domain_middle,
        domain_segments,
        n_threads,
        n_agents: _,
    } = *config;

    // Give agent default values
    let agent = Agent {
        mechanics: RodMechanics {
            pos: nalgebra::MatrixXx3::zeros(1), // This number of vertices is a dummy values
            vel: nalgebra::MatrixXx3::zeros(1), // It will be replaced by the code below
            diffusion_constant: 0.0 * MICRO_METRE.powf(2.0) / MINUTE,
            spring_tension,
            rigidity,
            damping,
            spring_length,
        },
        interaction: RodInteraction(MorsePotential {
            radius,
            potential_stiffness,
            strength,
            cutoff,
        }),
        growth_rate,
        neighbors: 0,
        neighbor_cap,
        spring_length_threshold,
    };

    // Place agents in simulation domain
    let dxl = [
        initial_domain_middle.0 - initial_domain_size.0 / 2.0,
        initial_domain_middle.1 - initial_domain_size.1 / 2.0,
        initial_domain_middle.2 - initial_domain_size.2 / 2.0,
    ];
    let dxu = [
        initial_domain_middle.0 + initial_domain_size.0 / 2.0,
        initial_domain_middle.1 + initial_domain_size.1 / 2.0,
        initial_domain_middle.2 + initial_domain_size.2 / 2.0,
    ];

    // Use domain decomposition techniques to distribute initial agents
    let n_agents_total: usize = config.n_agents.iter().map(|&(n_agents, _)| n_agents).sum();
    let rectangle = spatial_decomposition::Rectangle {
        min: [dxl[0], dxl[1]],
        max: [dxu[0], dxu[1]],
    };
    let cuboids =
        spatial_decomposition::kmr_decompose(&rectangle, n_agents_total.try_into().unwrap());

    let mut n_agents_remaining = config
        .n_agents
        .iter()
        .copied()
        .filter(|&(n, _)| n > 0)
        .collect::<Vec<_>>();

    let agents = cuboids
        .into_iter()
        .enumerate()
        .map(|(n, spatial_decomposition::Cuboid { min, max })| {
            let ind = n % n_agents_remaining.len();
            let (mut n_remaining, n_vertices) = n_agents_remaining.get_mut(ind).unwrap();
            n_remaining -= 1;
            let n_vertices = *n_vertices;
            if n_remaining == 0 {
                n_agents_remaining.remove(ind);
            }

            let mut new_agent = agent.clone();
            // new_agent.mechanics.spring_length = rng.random_range(1.5..2.5) * MICRO_METRE;
            let mut pos = nalgebra::MatrixXx3::zeros(n_vertices);
            pos[(0, 0)] = rng.random_range(min[0]..max[0]);
            pos[(0, 1)] = rng.random_range(min[1]..max[1]);
            if dxl[2] == dxu[2] {
                pos[(0, 2)] = dxl[2]
            } else {
                pos[(0, 2)] = rng.random_range(dxl[2]..dxu[2]);
            }
            let theta = rng.random_range(0.0..2.0 * std::f64::consts::PI);
            for i in 1..pos.nrows() {
                let phi = theta
                    + rng.random_range(-std::f64::consts::FRAC_PI_8..std::f64::consts::FRAC_PI_8);
                let mut direction = nalgebra::Vector3::zeros();
                direction[0] = phi.cos();
                direction[1] = phi.sin();
                let new_pos =
                    pos.row(i - 1) + agent.mechanics.spring_length * (direction).transpose();
                use core::ops::AddAssign;
                pos.row_mut(i).add_assign(new_pos);
            }
            // Shift such that the average is in the middle
            let pos_avg = pos.row_mean();
            pos.row_iter_mut().for_each(|mut row| {
                row[0] += (min[0] + max[0]) / 2.0 - pos_avg[0];
                row[1] += (min[1] + max[1]) / 2.0 - pos_avg[1];
            });

            new_agent.mechanics.set_pos(&pos);
            new_agent.mechanics.set_velocity(&(0.0 * pos));
            new_agent
        })
        .collect::<Vec<_>>();

    // Domain Setup
    // let domain_sizes = [4.0 * domain_size, delta_x, 3.0 * delta_x];
    let domain = CartesianCuboidRods {
        domain: CartesianCuboid::from_boundaries_and_n_voxels(
            [0.0; 3],
            domain_size,
            domain_segments,
        )?,
        gel_pressure: 0.,
        surface_friction: 0.,
        surface_friction_distance: 1.,
    };

    // Storage Setup
    let storage_builder = cellular_raza::prelude::StorageBuilder::new().location("./out");

    // Time Setup
    let time_stepper = cellular_raza::prelude::time::FixedStepsize::from_partial_save_interval(
        t0,
        dt,
        t_max,
        save_interval,
    )?;

    let settings = Settings {
        n_threads: n_threads.try_into().unwrap(),
        time: time_stepper,
        storage: storage_builder,
        progressbar: Some("".into()),
    };

    let storage = run_simulation!(
        domain: domain,
        agents: agents,
        settings: settings,
        aspects: [Mechanics, Interaction, Cycle],
        zero_force_default: |c: &Agent| {
            nalgebra::MatrixXx3::zeros(c.mechanics.pos.nrows())
        },
        parallelizer: Rayon,
    )?;
    Ok(storage.cells.extract_builder().get_full_path())
}

use cudarc::{
    driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

#[test]
fn main() -> Result<(), DriverError> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // You can load a function from a pre-compiled PTX like so:
    let module = ctx.load_module(Ptx::from_file("./src/sin.ptx"))?;

    // and then load a function from it:
    let f = module.load_function("sin_kernel").unwrap();

    let a_host = [1.0, 2.0, 3.0];

    let a_dev = stream.memcpy_stod(&a_host)?;
    let mut b_dev = a_dev.clone();

    // we use a buidler pattern to launch kernels.
    let n = 3i32;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let mut launch_args = stream.launch_builder(&f);
    launch_args.arg(&mut b_dev);
    launch_args.arg(&a_dev);
    launch_args.arg(&n);
    unsafe { launch_args.launch(cfg) }?;

    let a_host_2 = stream.memcpy_dtov(&a_dev)?;
    let b_host = stream.memcpy_dtov(&b_dev)?;

    println!("Found {b_host:?}");
    println!("Expected {:?}", a_host.map(f32::sin));
    assert_eq!(&a_host, a_host_2.as_slice());

    Ok(())
}
