"""Tests for the Boltz-2 NIM client module."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from flexaidds.boltz2 import (
    Boltz2AffinityResult,
    Boltz2Client,
    Boltz2Error,
    Boltz2Ligand,
    Boltz2Polymer,
    Boltz2PredictionResult,
    PocketConstraint,
    PocketContact,
    _build_payload,
    _parse_response,
    _validate_inputs,
)


# ---------------------------------------------------------------------------
# Dataclass construction
# ---------------------------------------------------------------------------


class TestBoltz2Polymer:
    def test_basic_construction(self):
        p = Boltz2Polymer(id="A", molecule_type="protein", sequence="ACGT")
        assert p.id == "A"
        assert p.molecule_type == "protein"
        assert p.sequence == "ACGT"
        assert p.cyclic is False

    def test_to_dict_minimal(self):
        p = Boltz2Polymer(id="A", molecule_type="protein", sequence="MKT")
        d = p.to_dict()
        assert d == {"id": "A", "molecule_type": "protein", "sequence": "MKT"}
        assert "cyclic" not in d

    def test_to_dict_cyclic(self):
        p = Boltz2Polymer(id="B", molecule_type="rna", sequence="AUGC", cyclic=True)
        d = p.to_dict()
        assert d["cyclic"] is True

    def test_frozen(self):
        p = Boltz2Polymer(id="A", molecule_type="protein", sequence="M")
        with pytest.raises(AttributeError):
            p.id = "B"


class TestBoltz2Ligand:
    def test_smiles(self):
        lig = Boltz2Ligand(id="L1", smiles="CCO")
        d = lig.to_dict()
        assert d == {"id": "L1", "smiles": "CCO"}
        assert "ccd" not in d
        assert "predict_affinity" not in d

    def test_ccd(self):
        lig = Boltz2Ligand(id="L2", ccd="ATP")
        d = lig.to_dict()
        assert d == {"id": "L2", "ccd": "ATP"}

    def test_affinity(self):
        lig = Boltz2Ligand(id="L1", smiles="CCO", predict_affinity=True)
        d = lig.to_dict()
        assert d["predict_affinity"] is True


class TestPocketConstraint:
    def test_to_dict(self):
        constraint = PocketConstraint(
            binder="L1",
            contacts=(
                PocketContact(id="A", residue_index=10),
                PocketContact(id="A", residue_index=25),
            ),
        )
        d = constraint.to_dict()
        assert d["constraint_type"] == "pocket"
        assert d["binder"] == "L1"
        assert len(d["contacts"]) == 2
        assert d["contacts"][0] == {"id": "A", "residue_index": 10}


class TestBoltz2AffinityResult:
    def test_construction(self):
        aff = Boltz2AffinityResult(
            ligand_id="L1",
            pic50=(6.5, 6.8),
            pred_value=(-6.5, -6.8),
            probability_binary=(0.95, 0.97),
        )
        assert aff.ligand_id == "L1"
        assert len(aff.pic50) == 2


class TestBoltz2PredictionResult:
    def test_construction(self):
        result = Boltz2PredictionResult(
            structures=("data_structure_1",),
            affinities={},
        )
        assert len(result.structures) == 1
        assert result.affinities == {}
        assert result.scores is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_empty_polymers(self):
        with pytest.raises(ValueError, match="At least one polymer"):
            _validate_inputs([], None)

    def test_too_many_polymers(self):
        polymers = [
            Boltz2Polymer(id=chr(65 + i), molecule_type="protein", sequence="M")
            for i in range(13)
        ]
        with pytest.raises(ValueError, match="Maximum 12"):
            _validate_inputs(polymers, None)

    def test_invalid_molecule_type(self):
        p = Boltz2Polymer(id="A", molecule_type="lipid", sequence="M")
        with pytest.raises(ValueError, match="Invalid molecule_type"):
            _validate_inputs([p], None)

    def test_empty_sequence(self):
        p = Boltz2Polymer(id="A", molecule_type="protein", sequence="")
        with pytest.raises(ValueError, match="1–4096"):
            _validate_inputs([p], None)

    def test_too_long_sequence(self):
        p = Boltz2Polymer(id="A", molecule_type="protein", sequence="M" * 4097)
        with pytest.raises(ValueError, match="1–4096"):
            _validate_inputs([p], None)

    def test_too_many_ligands(self):
        p = Boltz2Polymer(id="A", molecule_type="protein", sequence="M")
        ligands = [
            Boltz2Ligand(id=f"L{i}", smiles="C") for i in range(21)
        ]
        with pytest.raises(ValueError, match="Maximum 20"):
            _validate_inputs([p], ligands)

    def test_multiple_affinity_ligands(self):
        p = Boltz2Polymer(id="A", molecule_type="protein", sequence="M")
        ligands = [
            Boltz2Ligand(id="L1", smiles="C", predict_affinity=True),
            Boltz2Ligand(id="L2", smiles="CC", predict_affinity=True),
        ]
        with pytest.raises(ValueError, match="At most one ligand"):
            _validate_inputs([p], ligands)

    def test_ligand_neither_smiles_nor_ccd(self):
        p = Boltz2Polymer(id="A", molecule_type="protein", sequence="M")
        ligands = [Boltz2Ligand(id="L1")]
        with pytest.raises(ValueError, match="exactly one of"):
            _validate_inputs([p], ligands)

    def test_ligand_both_smiles_and_ccd(self):
        p = Boltz2Polymer(id="A", molecule_type="protein", sequence="M")
        ligands = [Boltz2Ligand(id="L1", smiles="C", ccd="ATP")]
        with pytest.raises(ValueError, match="exactly one of"):
            _validate_inputs([p], ligands)

    def test_valid_input_passes(self):
        p = Boltz2Polymer(id="A", molecule_type="protein", sequence="MKTAYIAKQ")
        lig = Boltz2Ligand(id="L1", smiles="CCO")
        _validate_inputs([p], [lig])  # should not raise


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------


class TestBuildPayload:
    def test_protein_only(self):
        p = Boltz2Polymer(id="A", molecule_type="protein", sequence="MKT")
        payload = _build_payload(
            [p], None, None,
            recycling_steps=3, sampling_steps=50, diffusion_samples=1,
            step_scale=1.638, output_format="mmcif",
            sampling_steps_affinity=200, diffusion_samples_affinity=5,
            affinity_mw_correction=False, write_full_pae=False,
        )
        assert len(payload["polymers"]) == 1
        assert "ligands" not in payload
        assert "constraints" not in payload
        assert payload["recycling_steps"] == 3

    def test_with_ligand_and_affinity(self):
        p = Boltz2Polymer(id="A", molecule_type="protein", sequence="MKT")
        lig = Boltz2Ligand(id="L1", smiles="CCO", predict_affinity=True)
        payload = _build_payload(
            [p], [lig], None,
            recycling_steps=3, sampling_steps=50, diffusion_samples=1,
            step_scale=1.638, output_format="mmcif",
            sampling_steps_affinity=200, diffusion_samples_affinity=5,
            affinity_mw_correction=True, write_full_pae=False,
        )
        assert payload["ligands"][0]["predict_affinity"] is True
        assert payload["sampling_steps_affinity"] == 200
        assert payload["diffusion_samples_affinity"] == 5
        assert payload["affinity_mw_correction"] is True

    def test_with_pocket_constraint(self):
        p = Boltz2Polymer(id="A", molecule_type="protein", sequence="MKT")
        lig = Boltz2Ligand(id="L1", smiles="CCO")
        constraint = PocketConstraint(
            binder="L1",
            contacts=(PocketContact(id="A", residue_index=1),),
        )
        payload = _build_payload(
            [p], [lig], [constraint],
            recycling_steps=3, sampling_steps=50, diffusion_samples=1,
            step_scale=1.638, output_format="mmcif",
            sampling_steps_affinity=200, diffusion_samples_affinity=5,
            affinity_mw_correction=False, write_full_pae=False,
        )
        assert payload["constraints"][0]["constraint_type"] == "pocket"
        assert payload["constraints"][0]["binder"] == "L1"

    def test_no_affinity_params_without_affinity(self):
        p = Boltz2Polymer(id="A", molecule_type="protein", sequence="MKT")
        lig = Boltz2Ligand(id="L1", smiles="CCO")  # predict_affinity=False
        payload = _build_payload(
            [p], [lig], None,
            recycling_steps=3, sampling_steps=50, diffusion_samples=1,
            step_scale=1.638, output_format="mmcif",
            sampling_steps_affinity=200, diffusion_samples_affinity=5,
            affinity_mw_correction=False, write_full_pae=False,
        )
        assert "sampling_steps_affinity" not in payload


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_structure_list(self):
        data = {
            "output": [
                {"structure": "data_mol\n_cell.length_a 1.0"},
            ],
        }
        result = _parse_response(data)
        assert len(result.structures) == 1
        assert "data_mol" in result.structures[0]

    def test_with_affinities(self):
        data = {
            "output": [{"structure": "cif_data"}],
            "affinities": {
                "L1": {
                    "affinity_pic50": [6.5, 6.8],
                    "affinity_pred_value": [-6.5, -6.8],
                    "affinity_probability_binary": [0.95, 0.97],
                }
            },
        }
        result = _parse_response(data)
        assert "L1" in result.affinities
        aff = result.affinities["L1"]
        assert aff.pic50 == (6.5, 6.8)
        assert aff.probability_binary == (0.95, 0.97)

    def test_empty_response(self):
        result = _parse_response({})
        assert result.structures == ()
        assert result.affinities == {}

    def test_string_structures(self):
        data = {"output": ["raw_cif_string"]}
        result = _parse_response(data)
        assert result.structures == ("raw_cif_string",)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class TestBoltz2Client:
    def test_default_init(self):
        client = Boltz2Client()
        assert client._base_url == "http://localhost:8000"
        assert client._timeout == 300

    def test_api_key_from_env(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-test123"}):
            client = Boltz2Client()
            assert client._api_key == "nvapi-test123"

    def test_explicit_api_key_overrides_env(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-env"}):
            client = Boltz2Client(api_key="nvapi-explicit")
            assert client._api_key == "nvapi-explicit"

    def test_trailing_slash_stripped(self):
        client = Boltz2Client(base_url="http://localhost:8000/")
        assert client._base_url == "http://localhost:8000"

    @patch("flexaidds.boltz2.urllib.request.urlopen")
    def test_health_check_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"status": "ready"}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        client = Boltz2Client()
        assert client.health_check() is True

    @patch("flexaidds.boltz2.urllib.request.urlopen")
    def test_health_check_failure(self, mock_urlopen):
        mock_urlopen.side_effect = ConnectionError("refused")
        client = Boltz2Client()
        assert client.health_check() is False

    @patch("flexaidds.boltz2.urllib.request.urlopen")
    def test_predict_protein_ligand(self, mock_urlopen):
        response_data = {
            "output": [{"structure": "data_complex\n_cell.length_a 1.0"}],
            "affinities": {
                "L1": {
                    "affinity_pic50": [7.2],
                    "affinity_pred_value": [-7.2],
                    "affinity_probability_binary": [0.99],
                }
            },
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(response_data).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        client = Boltz2Client()
        result = client.predict_protein_ligand(
            protein_sequence="MKTAYIAKQ",
            ligand_smiles="CCO",
            predict_affinity=True,
            pocket_residues=[1, 5],
        )

        assert len(result.structures) == 1
        assert "L1" in result.affinities
        assert result.affinities["L1"].pic50 == (7.2,)

        # Verify the request was made correctly
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        payload = json.loads(req.data.decode())
        assert payload["polymers"][0]["sequence"] == "MKTAYIAKQ"
        assert payload["ligands"][0]["smiles"] == "CCO"
        assert payload["ligands"][0]["predict_affinity"] is True
        assert payload["constraints"][0]["constraint_type"] == "pocket"
        assert len(payload["constraints"][0]["contacts"]) == 2

    @patch("flexaidds.boltz2.urllib.request.urlopen")
    def test_predict_api_error(self, mock_urlopen):
        error = urllib_http_error(400, "Bad Request", b'{"detail": "invalid"}')
        mock_urlopen.side_effect = error

        client = Boltz2Client()
        with pytest.raises(Boltz2Error) as exc_info:
            client.predict_protein_ligand(
                protein_sequence="MKT",
                ligand_smiles="C",
            )
        assert exc_info.value.status_code == 400

    @patch("flexaidds.boltz2.urllib.request.urlopen")
    def test_predict_connection_error(self, mock_urlopen):
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        client = Boltz2Client()
        with pytest.raises(ConnectionError, match="Failed to connect"):
            client.predict_protein_ligand(
                protein_sequence="MKT",
                ligand_smiles="C",
            )

    @patch("flexaidds.boltz2.urllib.request.urlopen")
    def test_auth_header_sent(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"output": []}'
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        client = Boltz2Client(api_key="nvapi-test")
        client.predict(
            polymers=[Boltz2Polymer(id="A", molecule_type="protein", sequence="M")],
        )

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer nvapi-test"

    def test_validation_error_before_request(self):
        client = Boltz2Client()
        with pytest.raises(ValueError, match="At least one polymer"):
            client.predict(polymers=[])


# ---------------------------------------------------------------------------
# Test for import from flexaidds
# ---------------------------------------------------------------------------


class TestImport:
    def test_import_from_package(self):
        from flexaidds import Boltz2Client, Boltz2Polymer, Boltz2Ligand
        assert Boltz2Client is not None
        assert Boltz2Polymer is not None
        assert Boltz2Ligand is not None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def urllib_http_error(code, reason, body):
    """Create a mock urllib HTTPError."""
    import urllib.error
    import io as _io
    err = urllib.error.HTTPError(
        url="http://localhost:8000/biology/mit/boltz2/predict",
        code=code,
        msg=reason,
        hdrs={},
        fp=_io.BytesIO(body),
    )
    return err
