#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest
from dataclasses import dataclass

from sources.utils import redact_url, remove_unsafe_chars


class UtilsTestCase(unittest.TestCase):
    def test_redact_url(self) -> None:
        @dataclass
        class UrlTestCase:
            msg: str
            url: str
            redacted_url: str

        test_cases = [
            UrlTestCase(
                url="https://a:b@c/d",
                redacted_url="https://a:*****@c/d",
                msg="HTTPS URL with password",
            ),
            UrlTestCase(
                url="https://c/d",
                redacted_url="https://c/d",
                msg="HTTPS URL without password",
            ),
            UrlTestCase(
                url="https://a:b@c:1234/d",
                redacted_url="https://a:*****@c:1234/d",
                msg="HTTPS URL with password and with port number",
            ),
            UrlTestCase(
                url="https://c:1234/d",
                redacted_url="https://c:1234/d",
                msg="HTTPS URL without password and with port number",
            ),
            UrlTestCase(
                url="git://a:b@c/d",
                redacted_url="git://a:*****@c/d",
                msg="git URL with password",
            ),
            UrlTestCase(
                url="git://c/d",
                redacted_url="git://c/d",
                msg="git URL without password",
            ),
        ]
        for case in test_cases:
            with self.subTest(msg=case.msg):
                self.assertEqual(redact_url(case.url), case.redacted_url)

    def test_unsafe_char_removal(self) -> None:
        """Test the remove_unsafe_chars function with chosen input strings."""

        @dataclass
        class UrlTestCase:
            s: str
            safe_s: str

        test_cases = [
            UrlTestCase(
                s="kernel-patches-bpf",
                safe_s="kernel-patches-bpf",
            ),
            UrlTestCase(
                s="kernel-patches.git",
                safe_s="kernel-patchesgit",
            ),
            UrlTestCase(
                s="https://a:b@c:1234/d",
                safe_s="httpsabc1234d",
            ),
        ]
        for case in test_cases:
            with self.subTest(msg=case.s):
                self.assertEqual(remove_unsafe_chars(case.s), case.safe_s)
