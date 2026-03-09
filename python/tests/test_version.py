"""Tests for flexaidds.__version__ – package metadata constants."""

from __future__ import annotations

from flexaidds.__version__ import (
    __author__,
    __email__,
    __license__,
    __url__,
    __version__,
    __version_info__,
)


class TestVersion:
    def test_version_is_string(self):
        assert isinstance(__version__, str)

    def test_version_non_empty(self):
        assert len(__version__) > 0

    def test_version_info_is_tuple(self):
        assert isinstance(__version_info__, tuple)

    def test_version_info_first_element_is_int(self):
        assert isinstance(__version_info__[0], int)

    def test_author_is_string(self):
        assert isinstance(__author__, str) and __author__

    def test_email_contains_at(self):
        assert "@" in __email__

    def test_license_is_string(self):
        assert isinstance(__license__, str) and __license__

    def test_url_is_string(self):
        assert isinstance(__url__, str) and __url__
