"""Tests for flexaidds.updater — self-update checking.

Pure Python, no C++ bindings required.
"""

from __future__ import annotations

import io
import json
from unittest import mock

import pytest

from flexaidds.updater import (
    AssetInfo,
    UpdateInfo,
    _parse_version,
    _version_newer,
    check_for_updates,
    get_current_version,
    select_asset_for_platform,
)


class TestGetCurrentVersion:
    def test_returns_tuple(self):
        v = get_current_version()
        assert isinstance(v, tuple)
        assert all(isinstance(x, int) for x in v)

    def test_has_at_least_three_parts(self):
        v = get_current_version()
        assert len(v) >= 3


class TestParseVersion:
    def test_simple_tag(self):
        nums, rank = _parse_version("v1.2.3")
        assert nums == (1, 2, 3)
        assert rank == 3  # release

    def test_no_prefix(self):
        nums, rank = _parse_version("1.0.0")
        assert nums == (1, 0, 0)
        assert rank == 3

    def test_alpha(self):
        nums, rank = _parse_version("v1.0.0-alpha")
        assert nums == (1, 0, 0)
        assert rank == 0

    def test_beta(self):
        nums, rank = _parse_version("2.1.0-beta.1")
        assert nums == (2, 1, 0)
        assert rank == 1

    def test_rc(self):
        nums, rank = _parse_version("v3.0.0-rc.2")
        assert nums == (3, 0, 0)
        assert rank == 2


class TestVersionNewer:
    def test_newer_major(self):
        assert _version_newer("v2.0.0", (1, 0, 0, "alpha", 0)) is True

    def test_same_version_release_vs_alpha(self):
        # v1.0.0 (release) is newer than 1.0.0-alpha
        assert _version_newer("v1.0.0", (1, 0, 0, "alpha", 0)) is True

    def test_same_version_same_prerelease(self):
        assert _version_newer("v1.0.0-alpha", (1, 0, 0, "alpha", 0)) is False

    def test_older_version(self):
        assert _version_newer("v0.9.0", (1, 0, 0, "alpha", 0)) is False

    def test_newer_minor(self):
        assert _version_newer("v1.1.0", (1, 0, 0)) is True

    def test_newer_patch(self):
        assert _version_newer("v1.0.1", (1, 0, 0)) is True


class TestCheckForUpdates:
    FAKE_RELEASE = {
        "tag_name": "v99.0.0",
        "html_url": "https://github.com/LeBonhommePharma/FlexAIDdS/releases/tag/v99.0.0",
        "body": "Release notes here",
        "published_at": "2026-01-15T00:00:00Z",
        "assets": [
            {
                "name": "flexaidds-linux-x86_64.tar.gz",
                "browser_download_url": "https://example.com/flexaidds-linux-x86_64.tar.gz",
                "size": 5000000,
                "content_type": "application/gzip",
            }
        ],
    }

    def test_success(self):
        response_data = json.dumps(self.FAKE_RELEASE).encode()
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = response_data
        mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            info = check_for_updates()

        assert info is not None
        assert isinstance(info, UpdateInfo)
        assert info.latest_version == "v99.0.0"
        assert info.update_available is True
        assert len(info.assets) == 1
        assert info.assets[0].name == "flexaidds-linux-x86_64.tar.gz"

    def test_network_error(self):
        import urllib.error

        with mock.patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Network unreachable"),
        ):
            info = check_for_updates()

        assert info is None

    def test_no_tag(self):
        response_data = json.dumps({"tag_name": ""}).encode()
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = response_data
        mock_resp.__enter__ = mock.MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = mock.MagicMock(return_value=False)

        with mock.patch("urllib.request.urlopen", return_value=mock_resp):
            info = check_for_updates()

        assert info is None


class TestSelectAssetForPlatform:
    def test_linux_x86_64(self):
        assets = [
            AssetInfo("flexaidds-linux-x86_64.tar.gz", "url1", 100, ""),
            AssetInfo("flexaidds-darwin-arm64.tar.gz", "url2", 100, ""),
        ]
        with (
            mock.patch("platform.system", return_value="Linux"),
            mock.patch("platform.machine", return_value="x86_64"),
        ):
            result = select_asset_for_platform(assets)
        assert result is not None
        assert result.name == "flexaidds-linux-x86_64.tar.gz"

    def test_macos_arm64(self):
        assets = [
            AssetInfo("flexaidds-linux-x86_64.tar.gz", "url1", 100, ""),
            AssetInfo("flexaidds-darwin-arm64.tar.gz", "url2", 100, ""),
        ]
        with (
            mock.patch("platform.system", return_value="Darwin"),
            mock.patch("platform.machine", return_value="arm64"),
        ):
            result = select_asset_for_platform(assets)
        assert result is not None
        assert result.name == "flexaidds-darwin-arm64.tar.gz"

    def test_no_match(self):
        assets = [
            AssetInfo("flexaidds-linux-x86_64.tar.gz", "url1", 100, ""),
        ]
        with (
            mock.patch("platform.system", return_value="Windows"),
            mock.patch("platform.machine", return_value="AMD64"),
        ):
            result = select_asset_for_platform(assets)
        assert result is None
