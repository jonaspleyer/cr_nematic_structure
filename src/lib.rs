use pyo3::prelude::*;

mod simulation;

use simulation::*;

#[pymodule]
#[pyo3(name = "cr_nematic_structure")]
fn cr_nematic_structure(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_simulation, m)?)?;
    m.add_class::<Configuration>()?;
    Ok(())
}
