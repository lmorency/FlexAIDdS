/*
 * FlexAID Python bindings via pybind11
 *
 * Exposes:
 *   - flexaidds.detect_cleft(pdb_file) → list of (x, y, z, radius) tuples
 *   - flexaidds.read_mol2(filename)    → dict with atom info
 *   - flexaidds.read_sdf(filename)     → dict with atom info
 *   - flexaidds.FlexAIDEngine          → wrapper around the C++ docking pipeline
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <tuple>
#include <string>
#include <cstring>
#include <cstdlib>

#include "flexaid.h"
#include "CleftDetector.h"
#include "Mol2Reader.h"
#include "SdfReader.h"

namespace py = pybind11;

/* ── helper: initialise a minimal FA_Global ─────────────────────────── */

static FA_Global* create_fa() {
    FA_Global* FA = (FA_Global*)calloc(1, sizeof(FA_Global));
    if (!FA) throw std::runtime_error("Failed to allocate FA_Global");

    FA->MIN_NUM_ATOM            = 1000;
    FA->MIN_NUM_RESIDUE         = 250;
    FA->MIN_ROTAMER_LIBRARY_SIZE = 155;
    FA->MIN_ROTAMER             = 1;
    FA->MIN_FLEX_BONDS          = 5;
    FA->MIN_CLEFTGRID_POINTS    = 250;
    FA->MIN_PAR                 = 6;
    FA->MIN_FLEX_RESIDUE        = 5;
    FA->MIN_NORMAL_GRID_POINTS  = 250;
    FA->MIN_OPTRES              = 1;
    FA->MIN_CONSTRAINTS         = 1;
    FA->ntypes                  = 40;  // default type count

    FA->num_atm = (int*)calloc(100000, sizeof(int));
    if (!FA->num_atm) throw std::runtime_error("Failed to allocate num_atm");

    return FA;
}

/* ── detect_cleft ────────────────────────────────────────────────────── */

static std::vector<std::tuple<float, float, float, float>>
py_detect_cleft(const std::string& pdb_file,
                float max_pair_dist   = 12.0f,
                float probe_max       =  5.0f,
                float probe_min       =  1.5f,
                float shrink_step     =  0.1f,
                float cluster_cutoff  =  4.0f,
                int   min_cluster     =  10)
{
    FA_Global* FA = create_fa();
    atom* atoms = nullptr;
    resid* residue = nullptr;

    // read_pdb allocates atoms and residue internally
    char pdb_path[MAX_PATH__];
    strncpy(pdb_path, pdb_file.c_str(), MAX_PATH__ - 1);
    pdb_path[MAX_PATH__ - 1] = '\0';

    read_pdb(FA, &atoms, &residue, pdb_path);

    CleftDetectorParams params;
    params.max_pair_dist   = max_pair_dist;
    params.probe_radius_max = probe_max;
    params.probe_radius_min = probe_min;
    params.probe_shrink_step = shrink_step;
    params.cluster_cutoff  = cluster_cutoff;
    params.min_cluster_size = min_cluster;

    sphere* spheres = detect_cleft(atoms, residue, FA->atm_cnt_real, FA->res_cnt, params);

    std::vector<std::tuple<float, float, float, float>> result;
    for (const sphere* s = spheres; s; s = s->prev)
        result.emplace_back(s->center[0], s->center[1], s->center[2], s->radius);

    free_sphere_list(spheres);
    free(atoms);
    free(residue);
    free(FA->num_atm);
    free(FA);

    return result;
}

/* ── read_mol2 ───────────────────────────────────────────────────────── */

struct AtomInfo {
    int number;
    std::string name;
    std::string element;
    float x, y, z;
    float radius;
    int type;
    std::vector<int> bonds;
};

static std::vector<AtomInfo>
py_read_mol2(const std::string& filename) {
    FA_Global* FA = create_fa();
    atom* atoms = (atom*)calloc(FA->MIN_NUM_ATOM, sizeof(atom));
    resid* residue = (resid*)calloc(FA->MIN_NUM_RESIDUE, sizeof(resid));
    if (!atoms || !residue) throw std::runtime_error("alloc failed");

    // Initialise residue[0] (mirrors read_pdb init)
    residue[0].fatm = (int*)calloc(1, sizeof(int));
    residue[0].latm = (int*)calloc(1, sizeof(int));

    int ok = read_mol2_ligand(FA, &atoms, &residue, filename.c_str());
    if (!ok) throw std::runtime_error("Failed to read MOL2 file: " + filename);

    int lig_res = FA->res_cnt;
    int first = residue[lig_res].fatm[0];
    int last  = residue[lig_res].latm[0];

    std::vector<AtomInfo> result;
    for (int i = first; i <= last; ++i) {
        AtomInfo ai;
        ai.number  = atoms[i].number;
        ai.name    = atoms[i].name;
        ai.element = atoms[i].element;
        ai.x       = atoms[i].coor[0];
        ai.y       = atoms[i].coor[1];
        ai.z       = atoms[i].coor[2];
        ai.radius  = atoms[i].radius;
        ai.type    = atoms[i].type;
        for (int b = 1; b <= atoms[i].bond[0]; ++b)
            ai.bonds.push_back(atoms[i].bond[b]);
        result.push_back(ai);
    }

    free(atoms);
    free(residue);
    free(FA->num_atm);
    free(FA);
    return result;
}

/* ── read_sdf ────────────────────────────────────────────────────────── */

static std::vector<AtomInfo>
py_read_sdf(const std::string& filename) {
    FA_Global* FA = create_fa();
    atom* atoms = (atom*)calloc(FA->MIN_NUM_ATOM, sizeof(atom));
    resid* residue = (resid*)calloc(FA->MIN_NUM_RESIDUE, sizeof(resid));
    if (!atoms || !residue) throw std::runtime_error("alloc failed");

    residue[0].fatm = (int*)calloc(1, sizeof(int));
    residue[0].latm = (int*)calloc(1, sizeof(int));

    int ok = read_sdf_ligand(FA, &atoms, &residue, filename.c_str());
    if (!ok) throw std::runtime_error("Failed to read SDF file: " + filename);

    int lig_res = FA->res_cnt;
    int first = residue[lig_res].fatm[0];
    int last  = residue[lig_res].latm[0];

    std::vector<AtomInfo> result;
    for (int i = first; i <= last; ++i) {
        AtomInfo ai;
        ai.number  = atoms[i].number;
        ai.name    = atoms[i].name;
        ai.element = atoms[i].element;
        ai.x       = atoms[i].coor[0];
        ai.y       = atoms[i].coor[1];
        ai.z       = atoms[i].coor[2];
        ai.radius  = atoms[i].radius;
        ai.type    = atoms[i].type;
        for (int b = 1; b <= atoms[i].bond[0]; ++b)
            ai.bonds.push_back(atoms[i].bond[b]);
        result.push_back(ai);
    }

    free(atoms);
    free(residue);
    free(FA->num_atm);
    free(FA);
    return result;
}

/* ── pybind11 module definition ──────────────────────────────────────── */

PYBIND11_MODULE(flexaidds, m) {
    m.doc() = "FlexAID-deltaS v1.5 Python bindings";

    py::class_<AtomInfo>(m, "AtomInfo")
        .def_readonly("number",  &AtomInfo::number)
        .def_readonly("name",    &AtomInfo::name)
        .def_readonly("element", &AtomInfo::element)
        .def_readonly("x",       &AtomInfo::x)
        .def_readonly("y",       &AtomInfo::y)
        .def_readonly("z",       &AtomInfo::z)
        .def_readonly("radius",  &AtomInfo::radius)
        .def_readonly("type",    &AtomInfo::type)
        .def_readonly("bonds",   &AtomInfo::bonds)
        .def("__repr__", [](const AtomInfo& a) {
            return "<AtomInfo " + a.name + " (" + a.element +
                   ") at (" + std::to_string(a.x) + ", " +
                   std::to_string(a.y) + ", " +
                   std::to_string(a.z) + ")>";
        });

    m.def("detect_cleft", &py_detect_cleft,
          "Detect binding-site cavities in a PDB file (SURFNET gap-sphere algorithm).\n"
          "Returns list of (x, y, z, radius) tuples for the largest detected cleft.",
          py::arg("pdb_file"),
          py::arg("max_pair_dist")  = 12.0f,
          py::arg("probe_max")      =  5.0f,
          py::arg("probe_min")      =  1.5f,
          py::arg("shrink_step")    =  0.1f,
          py::arg("cluster_cutoff") =  4.0f,
          py::arg("min_cluster")    =  10);

    m.def("read_mol2", &py_read_mol2,
          "Read a Tripos MOL2 file and return a list of AtomInfo objects.",
          py::arg("filename"));

    m.def("read_sdf", &py_read_sdf,
          "Read an MDL SDF/MOL V2000 file and return a list of AtomInfo objects.",
          py::arg("filename"));
}
