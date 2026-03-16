"""Tests for energy_matrix.py — pure Python, no C++ bindings needed."""

import math
import os
import struct
import tempfile

import numpy as np
import pytest

import warnings

from flexaidds.energy_matrix import (
    MATRIX_256_SIZE,
    SHNN_MAGIC,
    SHNN_VERSION,
    SYBYL_RADII,
    SYBYL_TYPE_NAMES,
    DensityPoint,
    EnergyMatrix,
    MatrixEntry,
    base_to_sybyl,
    decode_256_type,
    encode_256_type,
    parse_dat_file,
    sybyl_to_base,
    write_dat_file,
)


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_10type_dat(tmp_dir):
    """Create a minimal 10-type scalar energy matrix .dat file."""
    path = os.path.join(tmp_dir, "test_10.dat")
    lines = []
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    val = 1.0
    for i in range(10):
        for j in range(i, 10):
            label = f"{roman[i]}-{roman[j]}".rjust(10)
            lines.append(f"{label} = {val:10.4f}\n")
            val += 0.5
    with open(path, "w") as fh:
        fh.writelines(lines)
    return path


@pytest.fixture
def sample_density_dat(tmp_dir):
    """Create a 3-type matrix with density function entries."""
    path = os.path.join(tmp_dir, "test_density.dat")
    lines = [
        "       1-1 =     1.5000\n",                           # scalar
        "       1-2 = 0.1 -2.0 0.3 0.0 0.5 3.0\n",            # density
        "       1-3 =     0.7500\n",                           # scalar
        "       2-2 = 0.0 1.0 0.2 2.0 0.4 -1.0 0.8 0.5\n",   # density
        "       2-3 =    -1.2500\n",                           # scalar
        "       3-3 =     2.0000\n",                           # scalar
    ]
    with open(path, "w") as fh:
        fh.writelines(lines)
    return path


@pytest.fixture
def sample_256_matrix():
    """Create a simple 256×256 matrix with known values."""
    rng = np.random.RandomState(42)
    matrix = rng.randn(256, 256).astype(np.float64)
    # Symmetrise
    matrix = (matrix + matrix.T) / 2.0
    return EnergyMatrix(256, matrix)


# ── 256-type encoding tests ─────────────────────────────────────────────────

class TestEncode256Type:
    def test_basic_encoding(self):
        code = encode_256_type(base_type=3, charge_bin=2, hbond_flag=True)
        base, charge, hbond = decode_256_type(code)
        assert base == 3
        assert charge == 2
        assert hbond is True

    def test_zero_encoding(self):
        code = encode_256_type(0, 0, False)
        assert code == 0
        base, charge, hbond = decode_256_type(code)
        assert base == 0
        assert charge == 0
        assert hbond is False

    def test_max_encoding(self):
        code = encode_256_type(31, 3, True)
        assert code == 255
        base, charge, hbond = decode_256_type(code)
        assert base == 31
        assert charge == 3
        assert hbond is True

    def test_all_256_codes_roundtrip(self):
        for code in range(256):
            base, charge, hbond = decode_256_type(code)
            reconstructed = encode_256_type(base, charge, hbond)
            assert reconstructed == code

    def test_clamping(self):
        code = encode_256_type(50, 10, True)  # out of range
        base, charge, hbond = decode_256_type(code)
        assert 0 <= base <= 31
        assert 0 <= charge <= 3


# ── SYBYL projection tests ──────────────────────────────────────────────────

class TestSYBYLProjection:
    def test_sybyl_type_names_length(self):
        assert len(SYBYL_TYPE_NAMES) == 40

    def test_sybyl_radii_keys(self):
        assert set(SYBYL_RADII.keys()) == set(range(1, 41))

    def test_base_to_sybyl_range(self):
        for base in range(32):
            sybyl = base_to_sybyl(base)
            assert 1 <= sybyl <= 40, f"base={base} → sybyl={sybyl} out of range"

    def test_sybyl_to_base_range(self):
        for sybyl in range(1, 41):
            base = sybyl_to_base(sybyl)
            assert 0 <= base <= 31

    def test_roundtrip_identity_for_canonical_types(self):
        # First 22 base types should round-trip through sybyl
        for base in range(22):
            sybyl = base_to_sybyl(base)
            recovered = sybyl_to_base(sybyl)
            assert recovered == base, (
                f"base={base} → sybyl={sybyl} → base={recovered}")

    def test_carbon_ar_hetadj(self):
        # Base type 26 (C_ar_hetadj) maps to SYBYL C.AR (4)
        assert base_to_sybyl(26) == 4

    def test_carbon_pi_bridging(self):
        # Base type 27 (C_pi_bridging) maps to SYBYL C.2 (2)
        assert base_to_sybyl(27) == 2


# ── MatrixEntry evaluate tests ──────────────────────────────────────────────

class TestMatrixEntry:
    def test_scalar_evaluate(self):
        entry = MatrixEntry(type1=0, type2=1, is_scalar=True,
                            scalar_value=2.5)
        assert entry.evaluate(0.0) == 2.5
        assert entry.evaluate(0.5) == 2.5
        assert entry.evaluate(1.0) == 2.5

    def test_density_evaluate_interpolation(self):
        entry = MatrixEntry(type1=0, type2=0, is_scalar=False,
                            density_points=[
                                DensityPoint(0.1, -2.0),
                                DensityPoint(0.3, 0.0),
                                DensityPoint(0.5, 3.0),
                            ])
        # Below first point
        assert entry.evaluate(0.05) == 0.0
        # At first point: C++ get_yval uses <= for left-bound check,
        # so exact boundary returns 0.0 (same as "below")
        assert entry.evaluate(0.1) == 0.0
        # Just past first point: enters first interval
        assert entry.evaluate(0.2) == pytest.approx(-1.0, abs=1e-6)
        # At second point
        assert entry.evaluate(0.3) == pytest.approx(0.0, abs=1e-6)
        # Beyond last point
        assert entry.evaluate(0.8) == pytest.approx(3.0, abs=1e-6)

    def test_density_empty(self):
        entry = MatrixEntry(type1=0, type2=0, is_scalar=False,
                            density_points=[])
        assert entry.evaluate(0.5) == 0.0


# ── Legacy .dat I/O tests ───────────────────────────────────────────────────

class TestLegacyDatIO:
    def test_parse_scalar_dat(self, sample_10type_dat):
        mat = EnergyMatrix.from_dat_file(sample_10type_dat)
        assert mat.ntypes == 10
        assert mat.matrix.shape == (10, 10)
        assert mat.is_symmetric

    def test_roundtrip_scalar(self, sample_10type_dat, tmp_dir):
        mat = EnergyMatrix.from_dat_file(sample_10type_dat)
        out_path = os.path.join(tmp_dir, "roundtrip.dat")
        mat.to_dat_file(out_path)
        mat2 = EnergyMatrix.from_dat_file(out_path)
        np.testing.assert_allclose(mat.matrix, mat2.matrix, atol=1e-3)

    def test_parse_density_dat(self, sample_density_dat):
        mat = EnergyMatrix.from_dat_file(sample_density_dat)
        assert mat.ntypes == 3
        # Check scalar entry
        assert mat.entries[(0, 0)].is_scalar
        assert mat.entries[(0, 0)].scalar_value == pytest.approx(1.5)
        # Check density entry
        assert not mat.entries[(0, 1)].is_scalar
        assert len(mat.entries[(0, 1)].density_points) == 3
        # Symmetry
        assert mat.entries[(1, 0)].density_points == mat.entries[(0, 1)].density_points

    def test_evaluate_density_entry(self, sample_density_dat):
        mat = EnergyMatrix.from_dat_file(sample_density_dat)
        # Evaluate density function at known point
        val = mat.evaluate(0, 1, 0.2)  # midpoint of first density
        assert val == pytest.approx(-1.0, abs=1e-6)

    def test_convenience_functions(self, sample_10type_dat, tmp_dir):
        ntypes, matrix = parse_dat_file(sample_10type_dat)
        assert ntypes == 10
        assert matrix.shape == (10, 10)
        out = os.path.join(tmp_dir, "conv.dat")
        write_dat_file(out, ntypes, matrix)
        nt2, m2 = parse_dat_file(out)
        np.testing.assert_allclose(matrix, m2, atol=1e-3)

    def test_invalid_line_count(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad.dat")
        with open(path, "w") as fh:
            fh.write("1.0\n2.0\n3.0\n4.0\n")  # 4 lines, not N*(N+1)/2
        with pytest.raises(ValueError, match="Invalid line count"):
            EnergyMatrix.from_dat_file(path)


# ── 256×256 binary I/O tests ────────────────────────────────────────────────

class TestBinaryIO:
    def test_roundtrip(self, sample_256_matrix, tmp_dir):
        path = os.path.join(tmp_dir, "test.shnn")
        sample_256_matrix.to_binary(path)
        loaded = EnergyMatrix.from_binary(path)
        assert loaded.ntypes == 256
        # float32 precision loss expected
        np.testing.assert_allclose(
            loaded.matrix, sample_256_matrix.matrix, atol=1e-5)

    def test_invalid_magic(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad.shnn")
        with open(path, "wb") as fh:
            fh.write(b"BAAD")
            fh.write(struct.pack("<II", 1, 256))
            fh.write(np.zeros(256 * 256, dtype=np.float32).tobytes())
        with pytest.raises(ValueError, match="Invalid magic"):
            EnergyMatrix.from_binary(path)

    def test_invalid_dimension(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad_dim.shnn")
        with open(path, "wb") as fh:
            fh.write(SHNN_MAGIC)
            fh.write(struct.pack("<II", 1, 128))  # wrong dim
            fh.write(np.zeros(128 * 128, dtype=np.float32).tobytes())
        with pytest.raises(ValueError, match="Expected 256"):
            EnergyMatrix.from_binary(path)

    def test_wrong_ntypes_for_binary(self):
        mat = EnergyMatrix(10, np.zeros((10, 10)))
        with pytest.raises(ValueError, match="Binary format requires"):
            mat.to_binary("/tmp/should_not_exist.shnn")

    def test_binary_header_format(self, sample_256_matrix, tmp_dir):
        path = os.path.join(tmp_dir, "header_check.shnn")
        sample_256_matrix.to_binary(path)
        with open(path, "rb") as fh:
            magic = fh.read(4)
            version, dim = struct.unpack("<II", fh.read(8))
        assert magic == SHNN_MAGIC
        assert version == SHNN_VERSION
        assert dim == 256

    def test_file_size(self, sample_256_matrix, tmp_dir):
        path = os.path.join(tmp_dir, "size_check.shnn")
        sample_256_matrix.to_binary(path)
        expected = 4 + 4 + 4 + 256 * 256 * 4  # magic + version + dim + data
        assert os.path.getsize(path) == expected


# ── projection tests ─────────────────────────────────────────────────────────

class TestProjection:
    def test_project_to_40_shape(self, sample_256_matrix):
        proj = sample_256_matrix.project_to_40()
        assert proj.ntypes == 40
        assert proj.matrix.shape == (40, 40)

    def test_project_to_40_symmetry(self, sample_256_matrix):
        proj = sample_256_matrix.project_to_40()
        assert proj.is_symmetric

    def test_project_requires_256(self):
        mat = EnergyMatrix(10, np.zeros((10, 10)))
        with pytest.raises(ValueError, match="project_to_40 requires"):
            mat.project_to_40()

    def test_project_preserves_sign_pattern(self):
        # All positive 256×256 → all positive 40×40
        mat = EnergyMatrix(256, np.ones((256, 256)))
        proj = mat.project_to_40()
        assert np.all(proj.matrix >= 0)

    def test_projection_averages_correctly(self):
        # Set all entries for base_type=0 (C_sp, SYBYL=1) to 5.0
        mat = np.zeros((256, 256))
        for code_i in range(256):
            base_i = code_i & 0x1F
            if base_i == 0:
                for code_j in range(256):
                    base_j = code_j & 0x1F
                    if base_j == 0:
                        mat[code_i, code_j] = 5.0
        em = EnergyMatrix(256, mat)
        proj = em.project_to_40()
        # SYBYL type 1 (C.1) = base_to_sybyl(0) = 1, 0-indexed = 0
        assert proj.matrix[0, 0] == pytest.approx(5.0, abs=1e-6)


# ── lookup / evaluate tests ─────────────────────────────────────────────────

class TestLookup:
    def test_lookup_matches_numpy(self, sample_256_matrix):
        mat = sample_256_matrix
        for i in range(0, 256, 17):
            for j in range(0, 256, 19):
                assert mat.lookup(i, j) == mat.matrix[i, j]

    def test_evaluate_scalar(self, sample_10type_dat):
        mat = EnergyMatrix.from_dat_file(sample_10type_dat)
        val = mat.evaluate(0, 0, 0.5)
        assert val == pytest.approx(mat.matrix[0, 0], abs=1e-3)


# ── symmetry and utilities ───────────────────────────────────────────────────

class TestUtilities:
    def test_symmetrise(self):
        mat = EnergyMatrix(3, np.array([[1, 2, 3],
                                         [4, 5, 6],
                                         [7, 8, 9]], dtype=np.float64))
        assert not mat.is_symmetric
        mat.symmetrise()
        assert mat.is_symmetric
        assert mat.matrix[0, 1] == pytest.approx(3.0)  # (2+4)/2
        assert mat.matrix[1, 0] == pytest.approx(3.0)

    def test_repr(self, sample_256_matrix):
        r = repr(sample_256_matrix)
        assert "256" in r
        assert "symmetric" in r


# ── ContactTable tests ──────────────────────────────────────────────────────

from flexaidds.energy_matrix import (
    ContactTable,
    KnowledgeBasedTrainer,
    DockingBenchmarkCase,
    EnergyMatrixOptimizer,
    OptimizationResult,
)


class TestContactTable:
    def test_save_load_roundtrip(self, tmp_dir):
        counts = np.array([[10, 5], [5, 8]], dtype=np.int64)
        totals = np.array([100, 80], dtype=np.int64)
        table = ContactTable(
            ntypes=2, counts=counts, type_totals=totals,
            n_structures=3, distance_cutoff=6.0,
        )
        path = os.path.join(tmp_dir, "contacts.json")
        table.save(path)
        loaded = ContactTable.load(path)
        assert loaded.ntypes == 2
        assert loaded.n_structures == 3
        assert loaded.distance_cutoff == 6.0
        np.testing.assert_array_equal(loaded.counts, counts)
        np.testing.assert_array_equal(loaded.type_totals, totals)

    def test_merge_tables(self):
        t1 = ContactTable(
            ntypes=3,
            counts=np.ones((3, 3), dtype=np.int64),
            type_totals=np.array([10, 20, 30], dtype=np.int64),
            n_structures=1,
        )
        t2 = ContactTable(
            ntypes=3,
            counts=np.ones((3, 3), dtype=np.int64) * 2,
            type_totals=np.array([5, 10, 15], dtype=np.int64),
            n_structures=2,
        )
        trainer = KnowledgeBasedTrainer(ntypes=3)
        trainer.add_contact_table(t1)
        trainer.add_contact_table(t2)
        result = trainer.get_contact_table()
        np.testing.assert_array_equal(result.counts, t1.counts + t2.counts)
        np.testing.assert_array_equal(result.type_totals, t1.type_totals + t2.type_totals)
        assert result.n_structures == 3


# ── KnowledgeBasedTrainer tests ─────────────────────────────────────────────

class TestKnowledgeBasedTrainer:
    def test_single_structure_contacts(self):
        """Adding one structure accumulates correct contact counts."""
        trainer = KnowledgeBasedTrainer(ntypes=3, distance_cutoff=5.0)
        # 4 atoms: types 0, 1, 2, 0
        # Distances: 0-1=1.0, 0-2=3.0, 0-3=2.0, 1-2=2.0, 1-3=2.24, 2-3=2.24
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ])
        types = np.array([0, 1, 2, 0])
        trainer.add_structure(coords, types)
        table = trainer.get_contact_table()
        assert table.n_structures == 1
        # All distances < 5.0, so all 6 pairs are contacts
        assert np.sum(table.counts) > 0
        # type 0 appears twice
        assert table.type_totals[0] == 2
        assert table.type_totals[1] == 1
        assert table.type_totals[2] == 1

    def test_pseudocount_prevents_inf(self):
        """Zero-count pairs produce finite energies thanks to pseudocount."""
        trainer = KnowledgeBasedTrainer(ntypes=4, pseudocount=1)
        # Only contacts between types 0 and 1
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        types = np.array([0, 1])
        trainer.add_structure(coords, types)
        matrix = trainer.derive_potential()
        # All entries should be finite
        assert np.all(np.isfinite(matrix.matrix))

    def test_high_frequency_negative_energy(self):
        """Frequently-observed contacts should have negative (favorable) energy."""
        trainer = KnowledgeBasedTrainer(ntypes=3, distance_cutoff=3.0,
                                         pseudocount=1)
        # Deterministic contacts: types 0 and 1 close, type 2 far away
        for _ in range(50):
            coords = np.array([
                [0.0, 0.0, 0.0],   # type 0
                [1.0, 0.0, 0.0],   # type 1 — within 3Å of type 0
                [100.0, 0.0, 0.0], # type 2 — far from both
            ])
            types = np.array([0, 1, 2])
            trainer.add_structure(coords, types)

        matrix = trainer.derive_potential()
        # 0-1 (many contacts) should be more negative than 0-2 (no contacts)
        assert matrix.matrix[0, 1] < matrix.matrix[0, 2]

    def test_inter_chain_filtering(self):
        """With chain_mask, only inter-chain contacts are counted."""
        trainer = KnowledgeBasedTrainer(ntypes=2, distance_cutoff=5.0)
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # same chain as atom 0
            [2.0, 0.0, 0.0],  # different chain
        ])
        types = np.array([0, 0, 1])
        chain_mask = np.array([0, 0, 1])  # atoms 0,1 in chain 0; atom 2 in chain 1
        trainer.add_structure(coords, types, chain_mask=chain_mask)
        table = trainer.get_contact_table()
        # Intra-chain (0-0) pair should NOT be counted
        # Only inter-chain contacts: 0-2 and 1-2
        assert table.counts[0, 0] == 0  # no intra-chain 0-0 contacts
        assert table.counts[0, 1] > 0   # inter-chain 0-1 contacts exist

    def test_derived_matrix_is_symmetric(self):
        """Output matrix should be symmetric."""
        trainer = KnowledgeBasedTrainer(ntypes=4)
        rng = np.random.RandomState(123)
        coords = rng.randn(10, 3)
        types = rng.randint(0, 4, size=10)
        trainer.add_structure(coords, types)
        matrix = trainer.derive_potential()
        assert matrix.is_symmetric

    def test_parse_pdb_contacts(self, tmp_dir):
        """Test PDB parsing convenience method."""
        pdb_content = (
            "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C\n"
            "HETATM    3  C1  LIG B   1       5.000   6.000   7.000  1.00  0.00           C\n"
            "END\n"
        )
        pdb_path = os.path.join(tmp_dir, "test.pdb")
        with open(pdb_path, "w") as fh:
            fh.write(pdb_content)

        type_mapping = {"ALA:N": 0, "ALA:CA": 1, "LIG:C1": 2}
        coords, atom_types, chain_mask = KnowledgeBasedTrainer.parse_pdb_contacts(
            pdb_path, type_mapping)
        assert coords.shape == (3, 3)
        assert list(atom_types) == [0, 1, 2]
        # Chains A and B → two distinct chain IDs
        assert len(set(chain_mask.tolist())) == 2


# ── EnergyMatrixOptimizer tests ─────────────────────────────────────────────

class TestEnergyMatrixOptimizer:
    def test_vector_matrix_roundtrip(self):
        """Flattening and reconstructing preserves values."""
        mat = EnergyMatrix(4, np.random.RandomState(42).randn(4, 4))
        mat.symmetrise()

        # Create a minimal optimizer just for the roundtrip test
        opt = EnergyMatrixOptimizer.__new__(EnergyMatrixOptimizer)
        opt._reference = mat

        vec = opt._matrix_to_vector(mat)
        assert len(vec) == 4 * 5 // 2  # upper triangle = 10

        recovered = opt._vector_to_matrix(vec)
        np.testing.assert_allclose(recovered.matrix, mat.matrix, atol=1e-12)

    def test_auc_perfect_separation(self):
        """AUC is 1.0 when all actives score lower than all decoys."""
        auc = EnergyMatrixOptimizer._compute_auc(
            [-5.0, -4.0, -3.0],  # actives (lower)
            [1.0, 2.0, 3.0],     # decoys (higher)
        )
        assert auc == pytest.approx(1.0)

    def test_auc_inverted(self):
        """AUC is 0.0 when all actives score higher than all decoys."""
        auc = EnergyMatrixOptimizer._compute_auc(
            [5.0, 6.0, 7.0],     # actives (higher = bad)
            [-1.0, -2.0, -3.0],  # decoys (lower)
        )
        assert auc == pytest.approx(0.0)

    def test_auc_random(self):
        """AUC is ~0.5 for interleaved scores."""
        auc = EnergyMatrixOptimizer._compute_auc(
            [-3.0, -1.0, 1.0, 3.0],   # actives
            [-2.0, 0.0, 2.0, 4.0],    # decoys
        )
        assert 0.3 < auc < 0.7

    def test_auc_empty(self):
        """AUC is 0.5 when either list is empty."""
        assert EnergyMatrixOptimizer._compute_auc([], [1.0]) == 0.5
        assert EnergyMatrixOptimizer._compute_auc([1.0], []) == 0.5

    def test_load_benchmark_directory(self, tmp_dir):
        """load_benchmark finds receptor, actives, and decoys."""
        bdir = os.path.join(tmp_dir, "bench")
        os.makedirs(os.path.join(bdir, "actives"))
        os.makedirs(os.path.join(bdir, "decoys"))

        # Create receptor
        with open(os.path.join(bdir, "receptor.pdb"), "w") as fh:
            fh.write("ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00  0.00\nEND\n")

        # Create actives
        for i in range(3):
            with open(os.path.join(bdir, "actives", f"active_{i}.mol2"), "w") as fh:
                fh.write("@<TRIPOS>MOLECULE\ntest\n")

        # Create decoys
        for i in range(5):
            with open(os.path.join(bdir, "decoys", f"decoy_{i}.mol2"), "w") as fh:
                fh.write("@<TRIPOS>MOLECULE\ntest\n")

        cases = EnergyMatrixOptimizer.load_benchmark(bdir)
        assert len(cases) == 8
        actives = [c for c in cases if c.is_active]
        decoys = [c for c in cases if not c.is_active]
        assert len(actives) == 3
        assert len(decoys) == 5

    def test_density_function_matrix_raises(self):
        """Optimizer refuses density-function matrices."""
        mat = EnergyMatrix(2, np.zeros((2, 2)))
        mat.entries[(0, 1)] = MatrixEntry(
            type1=0, type2=1, is_scalar=False,
            density_points=[DensityPoint(0.1, 1.0), DensityPoint(0.5, 2.0)],
        )
        with pytest.raises(ValueError, match="all-scalar"):
            EnergyMatrixOptimizer(
                reference_matrix=mat,
                benchmark=[],
            )


# ── CLI tests ───────────────────────────────────────────────────────────────

class TestCLI:
    def test_build_parser(self):
        from flexaidds.energy_matrix_cli import build_parser
        parser = build_parser()
        # Should accept 'train' subcommand
        args = parser.parse_args(["train", "--contacts", "c.json", "-o", "out.dat"])
        assert args.command == "train"
        assert args.contacts == "c.json"

    def test_convert_subcommand(self, sample_10type_dat, tmp_dir):
        from flexaidds.energy_matrix_cli import _cmd_convert
        import argparse
        out = os.path.join(tmp_dir, "converted.json")
        args = argparse.Namespace(
            input=sample_10type_dat,
            output=out,
            format="json",
        )
        ret = _cmd_convert(args)
        assert ret == 0
        import json
        data = json.loads(open(out).read())
        assert data["ntypes"] == 10
        assert len(data["matrix"]) == 10

    def test_train_subcommand(self, tmp_dir):
        from flexaidds.energy_matrix_cli import _cmd_train
        import argparse

        # Create a contact table
        table = ContactTable(
            ntypes=3,
            counts=np.array([[5, 10, 2], [10, 3, 7], [2, 7, 4]], dtype=np.int64),
            type_totals=np.array([50, 40, 30], dtype=np.int64),
            n_structures=5,
        )
        contacts_path = os.path.join(tmp_dir, "contacts.json")
        table.save(contacts_path)

        out_path = os.path.join(tmp_dir, "trained.dat")
        args = argparse.Namespace(
            contacts=contacts_path,
            output=out_path,
            temperature=300.0,
            pseudocount=1,
            no_labels=False,
        )
        ret = _cmd_train(args)
        assert ret == 0
        # Verify output is valid
        mat = EnergyMatrix.from_dat_file(out_path)
        assert mat.ntypes == 3
        assert mat.is_symmetric


# ── Robustness: FileNotFoundError tests ──────────────────────────────────────

class TestFileNotFound:
    def test_from_dat_file_missing(self):
        with pytest.raises(FileNotFoundError):
            EnergyMatrix.from_dat_file("/nonexistent/path/matrix.dat")

    def test_from_binary_missing(self):
        with pytest.raises(FileNotFoundError):
            EnergyMatrix.from_binary("/nonexistent/path/matrix.shnn")

    def test_contact_table_load_missing(self):
        with pytest.raises(FileNotFoundError):
            ContactTable.load("/nonexistent/path/contacts.json")

    def test_parse_pdb_contacts_missing(self):
        with pytest.raises(FileNotFoundError):
            KnowledgeBasedTrainer.parse_pdb_contacts(
                "/nonexistent/path/test.pdb", {"ALA:CA": 0})


# ── Robustness: Edge case tests ──────────────────────────────────────────────

class TestEdgeCases:
    def test_ntypes_zero_rejected(self):
        with pytest.raises(ValueError, match="ntypes must be positive"):
            EnergyMatrix(0, np.zeros((0, 0)))

    def test_ntypes_negative_rejected(self):
        with pytest.raises(ValueError, match="ntypes must be positive"):
            EnergyMatrix(-1, np.zeros((0, 0)))

    def test_ntypes_one_accepted(self):
        mat = EnergyMatrix(1, np.array([[2.5]]))
        assert mat.ntypes == 1
        assert mat.matrix[0, 0] == 2.5

    def test_matrix_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="does not match"):
            EnergyMatrix(3, np.zeros((2, 2)))

    def test_nan_matrix_rejected(self):
        mat = np.array([[1.0, float("nan")], [float("nan"), 1.0]])
        with pytest.raises(ValueError, match="non-finite"):
            EnergyMatrix(2, mat)

    def test_inf_matrix_rejected(self):
        mat = np.array([[1.0, float("inf")], [float("inf"), 1.0]])
        with pytest.raises(ValueError, match="non-finite"):
            EnergyMatrix(2, mat)

    def test_trainer_ntypes_zero_rejected(self):
        with pytest.raises(ValueError, match="ntypes must be positive"):
            KnowledgeBasedTrainer(ntypes=0)

    def test_trainer_temperature_zero_rejected(self):
        with pytest.raises(ValueError, match="temperature must be positive"):
            KnowledgeBasedTrainer(ntypes=3, temperature=0.0)

    def test_trainer_negative_pseudocount_rejected(self):
        with pytest.raises(ValueError, match="pseudocount must be non-negative"):
            KnowledgeBasedTrainer(ntypes=3, pseudocount=-1)

    def test_trainer_negative_cutoff_rejected(self):
        with pytest.raises(ValueError, match="distance_cutoff must be positive"):
            KnowledgeBasedTrainer(ntypes=3, distance_cutoff=-1.0)


# ── Robustness: Error path tests ─────────────────────────────────────────────

class TestErrorPaths:
    def test_dat_invalid_token_count(self, tmp_dir):
        """Odd number of tokens (not 1, not even) raises ValueError."""
        path = os.path.join(tmp_dir, "bad_tokens.dat")
        with open(path, "w") as fh:
            fh.write("       1-1 = 0.1 0.2 0.3\n")  # 3 tokens: not 1, not even
        with pytest.raises(ValueError, match="Invalid token count"):
            EnergyMatrix.from_dat_file(path)

    def test_binary_truncated(self, tmp_dir):
        path = os.path.join(tmp_dir, "truncated.shnn")
        with open(path, "wb") as fh:
            fh.write(SHNN_MAGIC)
            fh.write(struct.pack("<II", 1, 256))
            fh.write(np.zeros(100, dtype=np.float32).tobytes())  # too short
        with pytest.raises(ValueError, match="Truncated"):
            EnergyMatrix.from_binary(path)

    def test_contact_table_corrupted_json(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad.json")
        with open(path, "w") as fh:
            fh.write('{"ntypes": 2}')  # missing required keys
        with pytest.raises(ValueError, match="missing required keys"):
            ContactTable.load(path)

    def test_malformed_pdb_coordinates(self, tmp_dir):
        pdb_path = os.path.join(tmp_dir, "bad.pdb")
        with open(pdb_path, "w") as fh:
            fh.write("ATOM      1  CA  ALA A   1       XXXX   2.000   3.000  1.00  0.00           C\n")
        with pytest.raises(ValueError, match="Malformed PDB coordinates"):
            KnowledgeBasedTrainer.parse_pdb_contacts(pdb_path, {"ALA:CA": 0})


# ── Robustness: Integration tests ────────────────────────────────────────────

class TestIntegration:
    def test_full_workflow_contacts_to_roundtrip(self, tmp_dir):
        """contacts -> train -> derive -> write .dat -> read back -> verify."""
        trainer = KnowledgeBasedTrainer(ntypes=3, distance_cutoff=5.0, pseudocount=1)
        rng = np.random.RandomState(42)
        for _ in range(20):
            coords = rng.randn(6, 3) * 2.0
            types = rng.randint(0, 3, size=6)
            trainer.add_structure(coords, types)

        matrix = trainer.derive_potential()
        assert matrix.ntypes == 3
        assert matrix.is_symmetric
        assert np.all(np.isfinite(matrix.matrix))

        # Write and read back
        dat_path = os.path.join(tmp_dir, "derived.dat")
        matrix.to_dat_file(dat_path)
        loaded = EnergyMatrix.from_dat_file(dat_path)
        np.testing.assert_allclose(loaded.matrix, matrix.matrix, atol=1e-3)

    def test_256_binary_roundtrip_with_projection(self, tmp_dir):
        """256 binary -> read -> project_to_40 -> verify shape and symmetry."""
        rng = np.random.RandomState(99)
        mat256 = rng.randn(256, 256)
        mat256 = (mat256 + mat256.T) / 2.0
        em = EnergyMatrix(256, mat256)

        path = os.path.join(tmp_dir, "test256.shnn")
        em.to_binary(path)
        loaded = EnergyMatrix.from_binary(path)
        proj = loaded.project_to_40()
        assert proj.ntypes == 40
        assert proj.is_symmetric
        assert np.all(np.isfinite(proj.matrix))


# ── Robustness: OptimizationResult tests ─────────────────────────────────────

class TestOptimizationResult:
    def test_to_dict(self):
        mat = EnergyMatrix(2, np.zeros((2, 2)))
        result = OptimizationResult(
            best_matrix=mat,
            best_score=0.85,
            history=[(0, 0.5), (1, 0.7), (2, 0.85)],
            n_evaluations=100,
            convergence_reason="converged",
        )
        d = result.to_dict()
        assert d["best_score"] == 0.85
        assert d["n_evaluations"] == 100
        assert d["convergence_reason"] == "converged"
        assert len(d["history"]) == 3


# ── Robustness: Negative case tests ──────────────────────────────────────────

class TestNegativeCases:
    def test_lookup_out_of_bounds(self, sample_256_matrix):
        with pytest.raises((IndexError, ValueError)):
            sample_256_matrix.lookup(300, 0)

    def test_mismatched_chain_mask_length(self):
        trainer = KnowledgeBasedTrainer(ntypes=2, distance_cutoff=5.0)
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        types = np.array([0, 1])
        chain_mask = np.array([0])  # wrong length
        with pytest.raises((ValueError, IndexError)):
            trainer.add_structure(coords, types, chain_mask=chain_mask)

    def test_incompatible_ntypes_merge(self):
        trainer = KnowledgeBasedTrainer(ntypes=3)
        table = ContactTable(
            ntypes=5,
            counts=np.ones((5, 5), dtype=np.int64),
            type_totals=np.ones(5, dtype=np.int64),
            n_structures=1,
        )
        with pytest.raises(ValueError, match="does not match"):
            trainer.add_contact_table(table)

    def test_empty_benchmark_rejected(self):
        mat = EnergyMatrix(2, np.zeros((2, 2)))
        with pytest.raises(ValueError, match="non-empty"):
            EnergyMatrixOptimizer(reference_matrix=mat, benchmark=[])


# ── Robustness: Reproducibility test ─────────────────────────────────────────

class TestReproducibility:
    def test_derive_potential_deterministic(self):
        """Same inputs -> identical output matrices."""
        def build():
            trainer = KnowledgeBasedTrainer(ntypes=3, pseudocount=1)
            rng = np.random.RandomState(7)
            for _ in range(10):
                coords = rng.randn(5, 3)
                types = rng.randint(0, 3, size=5)
                trainer.add_structure(coords, types)
            return trainer.derive_potential()

        m1 = build()
        m2 = build()
        np.testing.assert_array_equal(m1.matrix, m2.matrix)


# ── Robustness: Binary version warning test ──────────────────────────────────

class TestBinaryVersionWarning:
    def test_version_mismatch_warns(self, tmp_dir):
        """Loading a binary with wrong version emits a warning."""
        path = os.path.join(tmp_dir, "future_version.shnn")
        with open(path, "wb") as fh:
            fh.write(SHNN_MAGIC)
            fh.write(struct.pack("<II", 99, 256))  # version 99
            fh.write(np.zeros(256 * 256, dtype=np.float32).tobytes())
        with pytest.warns(UserWarning, match="version 99"):
            EnergyMatrix.from_binary(path)
