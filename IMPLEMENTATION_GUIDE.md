# FlexAIDdS Implementation Guide
## Three Core Improvements

This guide documents the implementation of three key architectural improvements to FlexAIDdS:

1. **BindingMode → StatMechEngine Integration**
2. **Python Bindings with pybind11**
3. **ENCoM Vibrational Entropy Integration**

---

## 1. BindingMode → StatMechEngine Refactor

### Overview

Replace manual Boltzmann weight summation in `BindingMode` with rigorous statistical mechanics via `StatMechEngine`, exposing full thermodynamic quantities (F, S, H, C_v) instead of just energy.

### Architecture

```cpp
class BindingMode {
    // Core ensemble
    std::vector<Pose> Poses;
    
    // NEW: Thermodynamic engine
    mutable statmech::StatMechEngine engine_;
    mutable bool thermo_cache_valid_;
    
    // Lazy evaluation
    void rebuild_engine() const;
    
    // NEW API
    statmech::Thermodynamics get_thermodynamics() const;
    double get_free_energy() const;         // F = -kT ln Z
    double get_heat_capacity() const;       // C_v
    std::vector<double> get_boltzmann_weights() const;
    
    // LEGACY API (backward compatible)
    double compute_energy() const;          // → get_free_energy()
    double compute_enthalpy() const;        // → mean_energy
    double compute_entropy() const;         // → S = (H-F)/T
};
```

### Implementation Pattern

**Lazy Evaluation:**
```cpp
void BindingMode::rebuild_engine() const {
    if (thermo_cache_valid_) return;  // Already current
    
    engine_.clear();
    for (const auto& pose : Poses) {
        engine_.add_sample(pose.CF, 1);  // Energy, multiplicity
    }
    thermo_cache_valid_ = true;
}
```

**New API Usage:**
```cpp
auto thermo = binding_mode.get_thermodynamics();
std::cout << "Free Energy F = " << thermo.free_energy << " kcal/mol\n";
std::cout << "Entropy S = " << thermo.entropy << " kcal/(mol·K)\n";
std::cout << "Heat Capacity Cv = " << thermo.heat_capacity << "\n";
std::cout << "Energy σ = " << thermo.std_energy << " kcal/mol\n";
```

**Backward Compatibility:**
```cpp
// Old code continues to work
double F = binding_mode.compute_energy();     // Uses StatMechEngine internally
double H = binding_mode.compute_enthalpy();   // mean_energy from engine
double S = binding_mode.compute_entropy();    // entropy from engine
```

### Changes to BindingMode.cpp

1. **Constructor:** Initialize `engine_` and `thermo_cache_valid_`
   ```cpp
   BindingMode::BindingMode(BindingPopulation* pop) 
       : Population(pop), 
         energy(0.0),
         engine_(static_cast<double>(pop->Temperature)),
         thermo_cache_valid_(false)
   {}
   ```

2. **add_Pose:** Invalidate cache
   ```cpp
   void BindingMode::add_Pose(Pose& pose) {
       Poses.push_back(pose);
       thermo_cache_valid_ = false;  // Force rebuild
   }
   ```

3. **compute_* methods:** Delegate to StatMechEngine
   ```cpp
   double BindingMode::compute_enthalpy() const {
       rebuild_engine();
       return engine_.compute().mean_energy;
   }
   
   double BindingMode::compute_entropy() const {
       rebuild_engine();
       return engine_.compute().entropy;
   }
   
   double BindingMode::compute_energy() const {
       return get_free_energy();
   }
   ```

4. **PDB Output:** Add thermodynamic REMARK annotations
   ```cpp
   auto thermo = get_thermodynamics();
   sprintf(tmpremark, "REMARK Free Energy F = %10.5f kcal/mol\n", thermo.free_energy);
   sprintf(tmpremark, "REMARK Enthalpy ⟨E⟩ = %10.5f kcal/mol\n", thermo.mean_energy);
   sprintf(tmpremark, "REMARK Entropy S = %10.5f kcal/(mol·K)\n", thermo.entropy);
   sprintf(tmpremark, "REMARK Heat Capacity Cv = %10.5f\n", thermo.heat_capacity);
   ```

### BindingPopulation Extensions

```cpp
class BindingPopulation {
    // NEW: Compare binding modes
    double compute_delta_G(const BindingMode& mode1, const BindingMode& mode2) const;
    
    // NEW: Global ensemble statistics
    statmech::StatMechEngine get_global_ensemble() const;
};
```

**Usage:**
```cpp
// Relative binding free energy
double ΔG = population.compute_delta_G(binding_mode_1, binding_mode_2);

// Global ensemble thermodynamics
auto global_engine = population.get_global_ensemble();
auto global_thermo = global_engine.compute();
```

---

## 2. Python Bindings with pybind11

### Architecture

**Directory Structure:**
```
FlexAIDdS/
├── LIB/                    # C++ core
│   ├── BindingMode.{h,cpp}
│   ├── statmech.{h,cpp}
│   └── ...
├── python/
│   ├── flexaidds/
│   │   ├── __init__.py
│   │   ├── core.py         # High-level Pythonic API
│   │   └── _bindings.cpp   # pybind11 bindings
│   ├── setup.py
│   └── pyproject.toml
└── examples/
    └── python_workflow.py
```

### pybind11 Bindings Layer

**File: `python/flexaidds/_bindings.cpp`**

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "BindingMode.h"
#include "statmech.h"

namespace py = pybind11;

// Thermodynamics struct
void bind_statmech(py::module& m) {
    py::class_<statmech::Thermodynamics>(m, "Thermodynamics")
        .def_readonly("temperature", &statmech::Thermodynamics::temperature)
        .def_readonly("log_Z", &statmech::Thermodynamics::log_Z)
        .def_readonly("free_energy", &statmech::Thermodynamics::free_energy)
        .def_readonly("mean_energy", &statmech::Thermodynamics::mean_energy)
        .def_readonly("heat_capacity", &statmech::Thermodynamics::heat_capacity)
        .def_readonly("entropy", &statmech::Thermodynamics::entropy)
        .def_readonly("std_energy", &statmech::Thermodynamics::std_energy)
        .def("__repr__", [](const statmech::Thermodynamics& t) {
            return "<Thermodynamics F=" + std::to_string(t.free_energy) + 
                   " H=" + std::to_string(t.mean_energy) +
                   " S=" + std::to_string(t.entropy) + ">";
        });
    
    py::class_<statmech::StatMechEngine>(m, "StatMechEngine")
        .def(py::init<double>(), py::arg("temperature") = 300.0)
        .def("add_sample", &statmech::StatMechEngine::add_sample,
             py::arg("energy"), py::arg("multiplicity") = 1)
        .def("compute", &statmech::StatMechEngine::compute)
        .def("boltzmann_weights", &statmech::StatMechEngine::boltzmann_weights)
        .def("delta_G", &statmech::StatMechEngine::delta_G)
        .def_property_readonly("temperature", &statmech::StatMechEngine::temperature)
        .def_property_readonly("size", &statmech::StatMechEngine::size)
        .def("clear", &statmech::StatMechEngine::clear);
}

// BindingMode
void bind_bindingmode(py::module& m) {
    py::class_<BindingMode>(m, "BindingMode")
        .def("get_thermodynamics", &BindingMode::get_thermodynamics)
        .def("get_free_energy", &BindingMode::get_free_energy)
        .def("get_heat_capacity", &BindingMode::get_heat_capacity)
        .def("get_boltzmann_weights", &BindingMode::get_boltzmann_weights)
        .def("compute_energy", &BindingMode::compute_energy)
        .def("compute_enthalpy", &BindingMode::compute_enthalpy)
        .def("compute_entropy", &BindingMode::compute_entropy)
        .def("get_size", &BindingMode::get_BindingMode_size);
}

// Module definition
PYBIND11_MODULE(_flexaidds, m) {
    m.doc() = "FlexAIDdS Python bindings - thermodynamic docking engine";
    
    bind_statmech(m);
    bind_bindingmode(m);
}
```

### High-Level Python API

**File: `python/flexaidds/core.py`**

```python
from ._flexaidds import StatMechEngine, BindingMode, Thermodynamics
import numpy as np
from typing import List, Tuple

class DockingEnsemble:
    """High-level Pythonic wrapper for binding mode analysis."""
    
    def __init__(self, binding_mode: BindingMode):
        self.mode = binding_mode
        self._thermo = None
    
    @property
    def thermodynamics(self) -> Thermodynamics:
        """Lazy-loaded thermodynamic properties."""
        if self._thermo is None:
            self._thermo = self.mode.get_thermodynamics()
        return self._thermo
    
    @property
    def free_energy(self) -> float:
        """Helmholtz free energy (kcal/mol)."""
        return self.thermodynamics.free_energy
    
    @property
    def entropy(self) -> float:
        """Configurational entropy (kcal/mol·K)."""
        return self.thermodynamics.entropy
    
    @property
    def heat_capacity(self) -> float:
        """Heat capacity C_v (kcal/mol·K²)."""
        return self.thermodynamics.heat_capacity
    
    def boltzmann_distribution(self) -> np.ndarray:
        """Get Boltzmann weights as numpy array."""
        return np.array(self.mode.get_boltzmann_weights())
    
    def to_dict(self) -> dict:
        """Export thermodynamics as dictionary."""
        t = self.thermodynamics
        return {
            'temperature_K': t.temperature,
            'free_energy_kcal_mol': t.free_energy,
            'enthalpy_kcal_mol': t.mean_energy,
            'entropy_kcal_mol_K': t.entropy,
            'heat_capacity': t.heat_capacity,
            'energy_std_kcal_mol': t.std_energy,
            'log_partition_function': t.log_Z,
            'ensemble_size': self.mode.get_size()
        }
```

### Build Configuration

**File: `python/setup.py`**

```python
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os

flex_sources = [
    "flexaidds/_bindings.cpp",
    "../LIB/BindingMode.cpp",
    "../LIB/statmech.cpp"
]

ext_modules = [
    Pybind11Extension(
        "flexaidds._flexaidds",
        sources=flex_sources,
        include_dirs=["../LIB"],
        cxx_std=20,
        extra_compile_args=["-O3", "-march=native", "-ffast-math"]
    )
]

setup(
    name="flexaidds",
    version="1.5.0",
    author="LP More",
    description="Thermodynamic molecular docking with statistical mechanics",
    packages=["flexaidds"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.8",
    install_requires=["numpy>=1.20.0"],
)
```

### Usage Example

```python
import flexaidds as flex
import numpy as np

# Create statistical mechanics engine
engine = flex.StatMechEngine(temperature=300.0)

# Add energy samples (from GA chromosomes)
energies = [-5.2, -4.8, -5.0, -4.9, -5.1]
for E in energies:
    engine.add_sample(E, multiplicity=1)

# Compute thermodynamics
thermo = engine.compute()
print(f"Free Energy: {thermo.free_energy:.3f} kcal/mol")
print(f"Entropy: {thermo.entropy:.6f} kcal/(mol·K)")
print(f"Heat Capacity: {thermo.heat_capacity:.6f}")

# Get Boltzmann weights
weights = engine.boltzmann_weights()
print(f"Boltzmann weights: {weights}")
```

---

## 3. ENCoM Vibrational Entropy Integration

### Overview

Integrate ENCoM (Elastic Network Contact Model) normal mode amplitudes and eigenvectors to compute vibrational entropy contributions, complementing the configurational entropy from pose ensembles.

### Architecture

```cpp
namespace encom {

struct NormalMode {
    int mode_id;
    double frequency;        // cm⁻¹
    double eigenvalue;       // (kcal/mol·Å²)
    std::vector<double> eigenvector;  // 3N Cartesian displacements
};

class VibrationalEntropy {
public:
    explicit VibrationalEntropy(double temperature_K = 300.0);
    
    // Add normal mode
    void add_mode(const NormalMode& mode, double amplitude);
    
    // Compute vibrational entropy (kcal/mol·K)
    double compute_entropy() const;
    
    // Compute zero-point energy (kcal/mol)
    double compute_zpe() const;
    
    // Quasi-harmonic entropy via covariance matrix
    static double quasi_harmonic_entropy(
        const Eigen::MatrixXd& covariance,
        double temperature);
    
private:
    double T_;
    std::vector<NormalMode> modes_;
    std::vector<double> amplitudes_;
};

}  // namespace encom
```

### Integration with BindingMode

```cpp
class BindingMode {
    // ...
    
    // NEW: Vibrational entropy component
    std::optional<encom::VibrationalEntropy> vib_entropy_;
    
    // Set ENCoM modes
    void set_vibrational_modes(
        const std::vector<encom::NormalMode>& modes,
        const std::vector<double>& amplitudes);
    
    // Total entropy = configurational + vibrational
    double compute_total_entropy() const;
};
```

### Input Pipeline

**Config keywords:**
```
NMAMOD    <int>      # Number of ENCoM modes to include
NMAEIG    <file>     # ENCoM eigenvector file (.eigvec)
NMAMP     <file>     # Mode amplitude file (.amplitudes)
```

**File formats:**
```
# .eigvec format (one mode per block)
MODE 1 FREQ 125.34 EIGENVALUE 0.0523
1  0.0234 -0.0156  0.0089
2 -0.0112  0.0201 -0.0045
...

# .amplitudes format
1  0.52
2  0.48
3  0.31
...
```

### Entropy Decomposition

```cpp
double BindingMode::compute_total_entropy() const {
    // Configurational entropy from pose ensemble
    double S_conf = compute_entropy();  // StatMechEngine
    
    // Vibrational entropy from ENCoM modes
    double S_vib = vib_entropy_ ? vib_entropy_->compute_entropy() : 0.0;
    
    return S_conf + S_vib;
}
```

### PDB Output

```cpp
sprintf(tmpremark, "REMARK Total Entropy S_total = %10.5f kcal/(mol·K)\n", 
        compute_total_entropy());
sprintf(tmpremark, "REMARK   Configurational S_conf = %10.5f\n", 
        compute_entropy());
if (vib_entropy_) {
    sprintf(tmpremark, "REMARK   Vibrational S_vib = %10.5f\n", 
            vib_entropy_->compute_entropy());
}
```

---

## Build Integration

Update `CMakeLists.txt`:

```cmake
# Python bindings (optional)
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python bindings" OFF)

if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
    
    pybind11_add_module(_flexaidds
        python/flexaidds/_bindings.cpp
        LIB/BindingMode.cpp
        LIB/statmech.cpp
    )
    
    target_include_directories(_flexaidds PRIVATE LIB)
    target_compile_options(_flexaidds PRIVATE -O3 -march=native)
    
    install(TARGETS _flexaidds DESTINATION python/flexaidds)
endif()

# ENCoM vibrational entropy module
option(ENABLE_ENCOM "Enable ENCoM vibrational entropy" OFF)

if(ENABLE_ENCOM)
    target_sources(FlexAID PRIVATE
        LIB/encom/VibrationalEntropy.cpp
        LIB/encom/ENCoMReader.cpp
    )
    target_compile_definitions(FlexAID PRIVATE FLEXAIDS_HAS_ENCOM)
    
    # Requires Eigen3 for covariance analysis
    if(NOT FLEXAIDS_USE_EIGEN)
        message(FATAL_ERROR "ENCoM support requires Eigen3")
    endif()
endif()
```

---

## Testing

### C++ Unit Tests

```cpp
#include <gtest/gtest.h>
#include "BindingMode.h"
#include "statmech.h"

TEST(BindingModeTest, StatMechIntegration) {
    // Create mock population
    // ...
    
    BindingMode mode(&population);
    
    // Add poses
    Pose p1(chrom1, 0, 0, 1.0, 300, {});
    mode.add_Pose(p1);
    
    // Get thermodynamics
    auto thermo = mode.get_thermodynamics();
    
    EXPECT_GT(thermo.free_energy, 0.0);
    EXPECT_GT(thermo.entropy, 0.0);
}

TEST(StatMechEngineTest, PartitionFunction) {
    statmech::StatMechEngine engine(300.0);
    engine.add_sample(-5.0, 1);
    engine.add_sample(-4.5, 1);
    
    auto thermo = engine.compute();
    
    EXPECT_LT(thermo.free_energy, -4.5);  // Should be lower than lowest
}
```

### Python Tests

```python
import pytest
import flexaidds as flex
import numpy as np

def test_statmech_engine():
    engine = flex.StatMechEngine(300.0)
    engine.add_sample(-5.0)
    engine.add_sample(-4.5)
    
    thermo = engine.compute()
    assert thermo.free_energy < -4.5
    assert thermo.entropy > 0

def test_boltzmann_weights():
    engine = flex.StatMechEngine(300.0)
    for E in [-5.0, -4.0, -3.0]:
        engine.add_sample(E)
    
    weights = engine.boltzmann_weights()
    assert len(weights) == 3
    assert np.isclose(sum(weights), 1.0)
    assert weights[0] > weights[1] > weights[2]  # Lowest E → highest weight
```

---

## Migration Path

### Phase 1: Core Refactor (This PR)
- ✅ Update `BindingMode.h` with StatMechEngine integration
- ✅ Implement new thermodynamic methods in `BindingMode.cpp`
- ✅ Add `compute_delta_G()` and `get_global_ensemble()` to `BindingPopulation`
- ✅ Update PDB output with full thermodynamic annotations
- ⚠️  Maintain backward compatibility (all existing code continues to work)

### Phase 2: Python Bindings (Next PR)
- Create `python/flexaidds/` directory structure
- Implement `_bindings.cpp` with pybind11
- Write `core.py` high-level API
- Add `setup.py` and `pyproject.toml`
- Write Python test suite

### Phase 3: ENCoM Integration (Future PR)
- Implement `VibrationalEntropy` class
- Add ENCoM file readers
- Extend `BindingMode` with vibrational modes
- Update config parser for `NMAMOD`, `NMAEIG`, `NMAMP` keywords

---

## Performance Considerations

1. **Lazy Evaluation:** `rebuild_engine()` only runs when cache invalid
2. **Minimal Overhead:** StatMechEngine uses log-sum-exp for numerical stability with O(N) complexity
3. **Memory:** `mutable` engine allows const method caching without breaking API
4. **Backward Compatibility:** Zero performance regression for existing code paths

---

## Summary

These three improvements transform FlexAIDdS from a CF-scoring GA into a rigorous thermodynamic docking engine:

1. **Statistical Mechanics:** Full ensemble thermodynamics (F, S, H, C_v) replace ad-hoc energy calculations
2. **Python Ecosystem:** pybind11 bindings enable Jupyter notebooks, ML pipelines, data analysis
3. **Vibrational Entropy:** ENCoM integration captures conformational flexibility beyond discrete pose sampling

All changes maintain backward compatibility while exposing modern APIs for advanced workflows.
