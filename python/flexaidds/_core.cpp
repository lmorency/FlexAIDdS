#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "statmech.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Python bindings for the FlexAID∆S thermodynamic core";

    py::class_<statmech::Thermodynamics>(m, "Thermodynamics")
        .def_readonly("free_energy", &statmech::Thermodynamics::free_energy)
        .def_readonly("mean_energy", &statmech::Thermodynamics::mean_energy)
        .def_readonly("entropy", &statmech::Thermodynamics::entropy)
        .def_readonly("heat_capacity", &statmech::Thermodynamics::heat_capacity)
        .def_readonly("std_energy", &statmech::Thermodynamics::std_energy);

    py::class_<statmech::StatMechEngine>(m, "StatMechEngine")
        .def(py::init<double>(), py::arg("temperature"))
        .def("clear", &statmech::StatMechEngine::clear,
             "Remove all states from the ensemble.")
        .def("add_sample",
             &statmech::StatMechEngine::add_sample,
             py::arg("energy"),
             py::arg("multiplicity") = 1.0,
             "Add a state with a given energy and multiplicity.")
        .def("set_temperature",
             &statmech::StatMechEngine::set_temperature,
             py::arg("temperature"),
             "Set ensemble temperature in Kelvin.")
        .def("compute",
             &statmech::StatMechEngine::compute,
             "Compute and return thermodynamic observables.")
        .def("boltzmann_weights",
             &statmech::StatMechEngine::boltzmann_weights,
             "Return normalized Boltzmann weights for the current ensemble.");
}
