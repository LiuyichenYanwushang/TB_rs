use ndarray::*;
use num_complex::Complex;
use numpy::{IntoPyArray, PyArray, PyReadonlyArray, ToPyArray};
use pyo3::prelude::{
    pyclass, pyfunction, pymethods, pymodule, wrap_pyfunction, Bound, Py, PyModule, PyResult,
    Python,
};
use pyo3::types::PyComplex;
use pyo3::types::PyComplexMethods;
use pyo3::types::PyModuleMethods;
use Rustb::{self, Model as RustbModel}; // Import specific modules from Rustb

/// Python bindings for Rustb - Tight-Binding Model Library
///
/// This module provides Python interfaces for tight-binding model calculations
/// including band structure, transport properties, and topological invariants.

#[pyclass]
#[derive(Clone, Debug)]
pub struct Model {
    inner: Rustb::Model,
}

#[pyclass]
#[derive(Debug)]
pub struct Atom {
    #[pyo3(get, set)]
    pub position: Py<PyArray<f64, Ix1>>,
    #[pyo3(get, set)]
    pub name: AtomType,
    #[pyo3(get, set)]
    pub atom_list: usize,
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrbProj {
    S,
    Px,
    Py,
    Pz,
    Dxy,
    Dyz,
    Dxz,
    Dz2,
    Dx2y2,
    Fz3,
    Fxz2,
    Fyz2,
    Fzx2y2,
    Fxyz,
    Fxx23y2,
    Fy3x2y2,
    Sp1,
    Sp2,
    Sp21,
    Sp22,
    Sp23,
    Sp31,
    Sp32,
    Sp33,
    Sp34,
    Sp3d1,
    Sp3d2,
    Sp3d3,
    Sp3d4,
    Sp3d5,
    Sp3d21,
    Sp3d22,
    Sp3d23,
    Sp3d24,
    Sp3d25,
    Sp3d26,
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AtomType {
    H,
    He,
    Li,
    Be,
    B,
    C,
    N,
    O,
    F,
    Ne,
    Na,
    Mg,
    Al,
    Si,
    P,
    S,
    Cl,
    Ar,
    K,
    Ca,
    Sc,
    Ti,
    V,
    Cr,
    Mn,
    Fe,
    Co,
    Ni,
    Cu,
    Zn,
    Ga,
    Ge,
    As,
    Se,
    Br,
    Kr,
    Rb,
    Sr,
    Y,
    Zr,
    Nb,
    Mo,
    Tc,
    Ru,
    Rh,
    Pd,
    Ag,
    Cd,
    In,
    Sn,
    Sb,
    Te,
    I,
    Xe,
    Cs,
    Ba,
    La,
    Ce,
    Pr,
    Nd,
    Pm,
    Sm,
    Eu,
    Gd,
    Tb,
    Dy,
    Ho,
    Er,
    Tm,
    Yb,
    Lu,
    Hf,
    Ta,
    W,
    Re,
    Os,
    Ir,
    Pt,
    Au,
    Hg,
    Tl,
    Pb,
    Bi,
    Po,
    At,
    Rn,
    Fr,
    Ra,
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpinDirection {
    None,
    X,
    Y,
    Z,
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Gauge {
    Lattice,
    Atom,
}

#[pymethods]
impl Atom {
    pub fn position(&self, py: Python<'_>) -> Py<PyArray<f64, Ix1>> {
        self.position.clone_ref(py)
    }
    pub fn norb(&self) -> usize {
        self.atom_list
    }
    pub fn atom_type(&self) -> AtomType {
        self.name
    }
    pub fn push_orb(&mut self) {
        self.atom_list += 1;
    }
    pub fn remove_orb(&mut self) {
        self.atom_list -= 1;
    }
    pub fn change_type(&mut self, new_type: AtomType) {
        self.name = new_type;
    }
    #[new]
    pub fn new(position: PyReadonlyArray<f64, Ix1>, atom_list: usize, name: AtomType) -> Atom {
        Python::attach(|py| {
            let position_array = position.as_array().to_owned().to_pyarray(py).into();
            Atom {
                position: position_array,
                atom_list,
                name,
            }
        })
    }
}

#[pymethods]
impl Model {
    #[new]
    pub fn new(
        dim_r: usize,
        lat: PyReadonlyArray<f64, Ix2>,
        orb: PyReadonlyArray<f64, Ix2>,
        spin: bool,
    ) -> PyResult<Self> {
        Python::attach(|_py| {
            let lat_array = lat.as_array().to_owned();
            let orb_array = orb.as_array().to_owned();

            let inner = Rustb::Model::tb_model(dim_r, lat_array, orb_array, spin, None)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            Ok(Model { inner })
        })
    }

    /// Set hopping parameter between orbitals
    pub fn set_hop(
        &mut self,
        value: Py<PyComplex>,
        i: usize,
        j: usize,
        r: PyReadonlyArray<isize, Ix1>,
        spin_dir: SpinDirection,
    ) -> PyResult<()> {
        Python::attach(|py| {
            let complex_val = value.bind(py);
            let rust_complex = Complex::new(complex_val.real(), complex_val.imag());
            let r_array = r.as_array().to_owned();
            let spin_dir_rust = match spin_dir {
                SpinDirection::None => Rustb::SpinDirection::None,
                SpinDirection::X => Rustb::SpinDirection::x,
                SpinDirection::Y => Rustb::SpinDirection::y,
                SpinDirection::Z => Rustb::SpinDirection::z,
            };

            self.inner
                .set_hop(rust_complex, i, j, &r_array, spin_dir_rust);
            Ok(())
        })
    }

    /// Add hopping parameter (accumulate to existing value)
    pub fn add_hop(
        &mut self,
        value: Py<PyComplex>,
        i: usize,
        j: usize,
        r: PyReadonlyArray<isize, Ix1>,
        spin_dir: SpinDirection,
    ) -> PyResult<()> {
        Python::attach(|py| {
            let complex_val = value.bind(py);
            let rust_complex = Complex::new(complex_val.real(), complex_val.imag());
            let r_array = r.as_array().to_owned();
            let spin_dir_rust = match spin_dir {
                SpinDirection::None => Rustb::SpinDirection::None,
                SpinDirection::X => Rustb::SpinDirection::x,
                SpinDirection::Y => Rustb::SpinDirection::y,
                SpinDirection::Z => Rustb::SpinDirection::z,
            };

            self.inner
                .add_hop(rust_complex, i, j, &r_array, spin_dir_rust);
            Ok(())
        })
    }

    /// Set on-site energy for orbitals
    pub fn set_onsite(
        &mut self,
        onsite: PyReadonlyArray<f64, Ix1>,
        spin_dir: SpinDirection,
    ) -> PyResult<()> {
        Python::attach(|_py| {
            let onsite_array = onsite.as_array().to_owned();
            let spin_dir_rust = match spin_dir {
                SpinDirection::None => Rustb::SpinDirection::None,
                SpinDirection::X => Rustb::SpinDirection::x,
                SpinDirection::Y => Rustb::SpinDirection::y,
                SpinDirection::Z => Rustb::SpinDirection::z,
            };

            self.inner.set_onsite(&onsite_array, spin_dir_rust);
            Ok(())
        })
    }

    /// Generate k-path for band structure calculations
    pub fn k_path(
        &self,
        path: PyReadonlyArray<f64, Ix2>,
        nk: usize,
    ) -> PyResult<(
        Py<PyArray<f64, Ix2>>,
        Py<PyArray<f64, Ix1>>,
        Py<PyArray<f64, Ix1>>,
    )> {
        Python::attach(|py| {
            let path_array = path.as_array().to_owned();
            let (kvec, kdist, knode) = self
                .inner
                .k_path(&path_array, nk)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            let kvec_py = Py::from(kvec.into_pyarray(py).to_owned());
            let kdist_py = Py::from(kdist.into_pyarray(py).to_owned());
            let knode_py = Py::from(knode.into_pyarray(py).to_owned());

            Ok((kvec_py, kdist_py, knode_py))
        })
    }

    /// Solve eigenvalue problem at single k-point
    pub fn solve_onek(
        &self,
        kvec: PyReadonlyArray<f64, Ix1>,
    ) -> PyResult<(Py<PyArray<f64, Ix1>>, Py<PyArray<Complex<f64>, Ix2>>)> {
        Python::attach(|py| {
            let kvec_array = kvec.as_array().to_owned();
            let (eval, evec) = self.inner.solve_onek(&kvec_array);

            let eval_py = Py::from(eval.into_pyarray(py).to_owned());
            let evec_py = Py::from(evec.into_pyarray(py).to_owned());

            Ok((eval_py, evec_py))
        })
    }

    /// Solve eigenvalue problems for multiple k-points
    pub fn solve_all(
        &self,
        kvec: PyReadonlyArray<f64, Ix2>,
    ) -> PyResult<(Py<PyArray<f64, Ix2>>, Py<PyArray<Complex<f64>, Ix3>>)> {
        Python::attach(|py| {
            let kvec_array = kvec.as_array().to_owned();
            let (eval, evec) = self.inner.solve_all(&kvec_array);

            let eval_py = Py::from(eval.into_pyarray(py).to_owned());
            let evec_py = Py::from(evec.into_pyarray(py).to_owned());

            Ok((eval_py, evec_py))
        })
    }

    /// Solve eigenvalue problems in parallel
    pub fn solve_all_parallel(
        &self,
        kvec: PyReadonlyArray<f64, Ix2>,
    ) -> PyResult<(Py<PyArray<f64, Ix2>>, Py<PyArray<Complex<f64>, Ix3>>)> {
        Python::attach(|py| {
            let kvec_array = kvec.as_array().to_owned();
            let (eval, evec) = self.inner.solve_all_parallel(&kvec_array);

            let eval_py = Py::from(eval.into_pyarray(py).to_owned());
            let evec_py = Py::from(evec.into_pyarray(py).to_owned());

            Ok((eval_py, evec_py))
        })
    }

    /// Generate Hamiltonian at specific k-point
    pub fn gen_ham(
        &self,
        kvec: PyReadonlyArray<f64, Ix1>,
        gauge: Gauge,
    ) -> PyResult<Py<PyArray<Complex<f64>, Ix2>>> {
        Python::attach(|py| {
            let kvec_array = kvec.as_array().to_owned();
            let gauge_rust = match gauge {
                Gauge::Lattice => Rustb::Gauge::Lattice,
                Gauge::Atom => Rustb::Gauge::Atom,
            };

            let ham = self.inner.gen_ham(&kvec_array, gauge_rust);

            Ok(Py::from(ham.into_pyarray(py).to_owned()))
        })
    }

    /// Calculate density of states
    pub fn dos(
        &self,
        kmesh: PyReadonlyArray<usize, Ix1>,
        e_min: f64,
        e_max: f64,
        e_n: usize,
        eta: f64,
    ) -> PyResult<(Py<PyArray<f64, Ix1>>, Py<PyArray<f64, Ix1>>)> {
        Python::attach(|py| {
            let kmesh_array = kmesh.as_array().to_owned();
            let (e0, dos) = self
                .inner
                .dos(&kmesh_array, e_min, e_max, e_n, eta)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

            let e0_py = Py::from(e0.into_pyarray(py).to_owned());
            let dos_py = Py::from(dos.into_pyarray(py).to_owned());

            Ok((e0_py, dos_py))
        })
    }

    /// Calculate Hall conductivity
    pub fn hall_conductivity(
        &self,
        kmesh: PyReadonlyArray<usize, Ix1>,
        dir_1: PyReadonlyArray<f64, Ix1>,
        dir_2: PyReadonlyArray<f64, Ix1>,
        mu: f64,
        t: f64,
        spin: usize,
        eta: f64,
    ) -> PyResult<f64> {
        Python::attach(|_py| {
            let kmesh_array = kmesh.as_array().to_owned();
            let dir_1_array = dir_1.as_array().to_owned();
            let dir_2_array = dir_2.as_array().to_owned();

            self.inner
                .Hall_conductivity(&kmesh_array, &dir_1_array, &dir_2_array, mu, t, spin, eta)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Get model properties
    pub fn dim_r(&self) -> usize {
        self.inner.dim_r()
    }

    pub fn norb(&self) -> usize {
        self.inner.norb()
    }

    pub fn nsta(&self) -> usize {
        self.inner.nsta()
    }

    pub fn natom(&self) -> usize {
        self.inner.natom()
    }
}

/// Create model from Wannier90 Hamiltonian file
#[pyfunction]
pub fn from_hr(path: &str, file_name: &str, zero_energy: f64) -> PyResult<Model> {
    let inner = RustbModel::from_hr(path, file_name, zero_energy)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(Model { inner })
}

/// Python module initialization
#[pymodule]
fn Rustb4py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_class::<Atom>()?;
    m.add_class::<OrbProj>()?;
    m.add_class::<AtomType>()?;
    m.add_class::<SpinDirection>()?;
    m.add_class::<Gauge>()?;
    m.add_function(wrap_pyfunction!(from_hr, m)?)?;

    Ok(())
}
