# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rustb4py is a Python extension module built with PyO3 that provides Python bindings for the Rustb tight-binding library. It allows Python users to access Rustb's tight-binding model calculations and transport property computations.

## Codebase Structure

This is a workspace containing multiple Rust projects:
- **Rustb4py** (current directory): Python bindings using PyO3
- **Rustb**: Core tight-binding library (local dependency at `../Rustb/`)
- **TB_rs**: Main application for transport calculations

### Key Directories
- `src/`: Rust source code for Python bindings
- `../Rustb/src/`: Core tight-binding library implementation
- `../Rustb/examples/`: Example usage of Rustb library
- `../Rustb/tests/`: Test files for Rustb
- `.github/workflows/`: CI/CD configuration

## Key Dependencies

### Core Dependencies
- **pyo3** (0.26.0): Python-Rust bindings with extension module support
- **numpy** (0.26.0): NumPy array support for Python integration
- **ndarray** (0.16.1): Multi-dimensional arrays with rayon and serde features
- **ndarray-linalg** (0.17.0): Linear algebra operations
- **num-complex** (0.4.4): Complex number support
- **Rustb** (local): Core tight-binding library dependency

### BLAS/LAPACK Backends
Configure using Cargo features:
- **Intel MKL**: `intel-mkl-static` or `intel-mkl-system`
- **OpenBLAS**: `openblas-static` or `openblas-system`
- **Netlib**: `netlib-static` or `netlib-system`

## Build Commands

### Standard Builds
```bash
# Debug build
cargo build

# Release build with optimizations
cargo build --release

# Build with specific BLAS backend
cargo build --release --features intel-mkl-static
```

### Python Packaging with Maturin
```bash
# Build Python wheels
maturin build

# Build and install for development
maturin develop

# Build for specific Python version
maturin build --interpreter python3.11
```

### Testing
```bash
# Run Rust tests
cargo test

# Run specific test
cargo test --testname

# Run tests with verbose output
cargo test -- --nocapture
```

## Python Integration Architecture

### Core Python Classes
- `Model`: Main tight-binding model container (wraps Rustb::Model)
- `Atom`: Atomic position and orbital information
- `OrbProj`: Orbital projection types (s, px, py, pz, dxy, etc.)
- `AtomType`: Chemical element types (H through Ra)
- `SpinDirection`: Spin polarization directions (None, x, y, z)
- `Gauge`: Gauge choices for Hamiltonian generation (Lattice, Atom)

### Key Python Methods
- `Model.new()`: Create new tight-binding model
- `Model.set_hop()`: Set hopping parameters
- `Model.solve_onek()`: Solve eigenvalue problem at single k-point
- `Model.solve_all()`: Solve eigenvalue problems for multiple k-points
- `Model.gen_ham()`: Generate Hamiltonian at specific k-point
- `Model.dos()`: Calculate density of states
- `Model.hall_conductivity()`: Calculate Hall conductivity

### Data Conversion Patterns
- Python NumPy arrays ↔ Rust ndarray arrays
- Python complex numbers ↔ Rust num_complex::Complex
- Python strings ↔ Rust &str/String

## Development Workflow

### Adding New Python Bindings
1. Add new `#[pyfunction]` or `#[pymethods]` in `src/lib.rs`
2. Implement proper error handling with `PyResult`
3. Add Python docstrings for documentation
4. Test Python integration with example scripts

### Testing Strategy
1. **Rust Unit Tests**: `cargo test` for core functionality
2. **Python Integration**: Import and test the built module
3. **Example Validation**: Verify examples in `../Rustb/examples/` work

### Common Development Tasks
- Adding new Python bindings for Rustb functions
- Improving Python API ergonomics and documentation
- Testing Python-Rust data conversion
- Performance optimization for Python interface
- Adding error handling and validation

## Key Files

- `src/lib.rs`: Main Python module implementation with PyO3 bindings
- `Cargo.toml`: Project configuration, dependencies, and features
- `pyproject.toml`: Python packaging configuration for maturin
- `../Rustb/Cargo.toml`: Core library configuration
- `../Rustb/src/model_struct.rs`: Core Model implementation
- `../Rustb/src/conductivity.rs`: Transport property calculations

## Performance Considerations

- Use `--release` builds for production performance
- Enable appropriate BLAS/LAPACK backend for linear algebra
- Consider parallel computation with Rayon where applicable
- Optimize data conversion between Python and Rust

## Debugging Tips

- Use `cargo build --verbose` for detailed build output
- Enable `RUST_BACKTRACE=1` for detailed error traces
- Test Python imports with simple scripts
- Check BLAS/LAPACK configuration for performance issues

## CI/CD Pipeline

The GitHub Actions workflow in `.github/workflows/CI.yml`:
- Builds wheels for Linux, Windows, and macOS
- Uses maturin-action for Python packaging
- Uploads artifacts for distribution
- Supports PyPI publishing for tagged releases