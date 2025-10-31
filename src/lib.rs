use pyo3::prelude::*;

mod simulation;

use simulation::*;

#[pymodule]
#[pyo3(name = "cr_nematic_structure")]
fn cr_nematic_structure(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_simulation, m)?)?;
    m.add_class::<Configuration>()?;

    // Constants
    m.add("MICRO_METRE", MICRO_METRE)?;
    m.add("MINUTE", MINUTE)?;
    m.add("HOUR", HOUR)?;

    Ok(())
}

pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
