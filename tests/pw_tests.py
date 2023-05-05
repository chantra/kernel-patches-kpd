#!/usr/bin/env python3

import copy
import datetime
import json
import unittest
from dataclasses import dataclass
from typing import Any, Dict, Final, List, Optional, Set, Union
from unittest.mock import patch

import requests

from freezegun import freeze_time
from pyre_extensions import none_throws
from requests.models import PreparedRequest, Response
from requests.structures import CaseInsensitiveDict
from sources.patchwork import (
    IRRELEVANT_STATES,
    parse_tags,
    Patchwork,
    RELEVANT_STATES,
    Subject,
    TTL,
)
from sources.stats import STATS_KEY_BUG

DEFAULT_CHECK_CTX: Final[str] = "some_context"
DEFAULT_CHECK_CTX_QUERY: Final[
    str
] = f"?context={DEFAULT_CHECK_CTX}&order=-date&per_page=1"
PROJECT: Final[int] = 1234
DELEGATE: Final[int] = 12345


class PatchworkMock(Patchwork):
    def __init__(self, *args, **kwargs) -> None:
        # Force patchwork to use loopback/port 0 for whatever network access
        # we fail to mock
        # kwargs["server"] = "https://127.0.0.1:0"
        super().__init__(*args, **kwargs)
        # Initialize counters by calling `drop_counters()`. Failing to do so
        # would result in stats key be missing and would only bump the
        # STATS_KEY_BUG_OCCURENCE key which will prevent us from accessing
        # stats counters to validate code behaviour in unittests.
        self.drop_counters()


class ResponseMock(Response):
    def __init__(
        self,
        json_content: bytes,
        status_code: int,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()
        if headers is None:
            headers = {}
        self.headers = CaseInsensitiveDict(data=headers)
        self._content = json_content
        self.status_code = status_code


def get_default_pw_client() -> PatchworkMock:
    return PatchworkMock(
        server="127.0.0.1",
        api_version="1.1",
        search_patterns=[{"archived": False, "project": PROJECT, "delegate": DELEGATE}],
        auth_token="mocktoken",
    )


def pw_response_generator(data: Dict[str, Any]):
    """
    Generates a function suitable to pass to `_pw_get_patcher.side_effect`.
    It takes a dictionary as input that uses the called URL as key and contains the
    value that we would expect from converting the response's json to native python type.

    If the query look like a search (e.g has query parameters), and we don't find a hit. We
    will return an empty list.
    If the query is not a search, and there is no key hit, we will return a 404.

    Under the hood, the value is converted back to json, but by using native types when
    generating the content for unittests, it makes it either and natively handle by the
    tooling than if we were writing raw json blobs. For instance, we can benefit from the linter,
    we can use variables/constants, we can use comments to explain why a blob of data is used....
    """

    def response_fetcher(*args, **kwargs):
        # Generate the URL from the GET requests using request.get's `url`, and `params`, arguments.
        # e.g take the query string args and add the to the URL.
        # The generated URL is used to fetch the response associated.
        pr = PreparedRequest()
        pr.prepare_url(**kwargs)
        url = pr.url

        if url not in data:
            # If we don't have the key and there is no search parameters assume
            # we are querying a specific URL and it is not found.
            if not kwargs.get("params"):
                return ResponseMock(
                    json_content=b'{"detail": "Not found."}', status_code=404
                )
            # if there is query parameter, assume search and return an empty search result.
            return ResponseMock(json_content=b"[]", status_code=200)

        return ResponseMock(
            json_content=json.dumps(data[url]).encode(), status_code=200
        )

    return response_fetcher


class PatchworkTestCase(unittest.TestCase):
    """
    Base TestCase class that ensure any patchwork related test cases are properly initialized.
    """

    def setUp(self) -> None:
        self._pw = get_default_pw_client()
        self._pw_post_patcher = patch.object(requests.Session, "post").start()
        self._pw_get_patcher = patch.object(requests.Session, "get").start()


class TestPatchwork(PatchworkTestCase):
    def test_ensure_bug_key(self) -> None:
        """
        Stats accounting relies on the presence of the STATS_KEY_BUG key. Failing to initialize stat counters cause
        a stack overflow.
        This test ensures that we do not crash in case the stats counters were not initialized. Logging can be used to tracked bogus keys.
        """
        # simulate unitialized keys
        self._pw.drop_counters()
        self._pw.increment_counter("somerandomekey")
        self.assertEqual(self._pw.stats[STATS_KEY_BUG], 1)

    def test_get_wrapper(self) -> None:
        """
        Simple test to ensure that GET requests are properly mocked.
        """
        self._pw_get_patcher.return_value = ResponseMock(
            b"""{"key1": "value1", "key2": 2}""", 200
        )
        resp = self._pw._Patchwork__get("object").json()
        self._pw_get_patcher.assert_called_once()
        self.assertEqual(resp["key1"], "value1")

    def test_post_wrapper(self) -> None:
        """
        Simple test to ensure that POST requests are properly mocked.
        """
        self._pw_post_patcher.return_value = ResponseMock(
            b"""{"key1": "value1", "key2": 2}""", 200
        )
        # Make sure user and token are set so the requests is actually posted.
        self._pw.pw_token = "1234567890"
        self._pw.pw_user = "somerandomuser"
        resp = self._pw._Patchwork__post("some/random/url", "somerandomdata").json()
        self._pw_post_patcher.assert_called_once()
        self.assertEqual(resp["key1"], "value1")

    def test_get_objects_recursive(self) -> None:
        @dataclass
        class TestCase:
            name: str
            pages: List[ResponseMock]
            expected: List[Any]
            get_calls: int
            filters: Optional[Dict[str, Union[str, List[str]]]] = None

        test_cases = [
            TestCase(
                name="single page",
                pages=[ResponseMock(json_content=b'["a","b","c"]', status_code=200)],
                expected=["a", "b", "c"],
                get_calls=1,
            ),
            TestCase(
                name="Multiple pages with proper formatting",
                pages=[
                    ResponseMock(
                        json_content=b'["a"]',
                        headers={
                            "Link": '<https://127.0.0.1:0/api/1.1/projects/?page=2>; rel="next"'
                        },
                        status_code=200,
                    ),
                    ResponseMock(
                        json_content=b'["b"]',
                        headers={
                            "Link": '<https://127.0.0.1:0/api/1.1/projects/?page=3>; rel="next", <https://127.0.0.1:0/api/1.1/projects/>; rel="prev"'
                        },
                        status_code=200,
                    ),
                    ResponseMock(
                        json_content=b'["c"]',
                        headers={
                            "Link": '<https://127.0.0.1:0/api/1.1/projects/?page=2>; rel="prev"'
                        },
                        status_code=200,
                    ),
                ],
                expected=["a", "b", "c"],
                get_calls=3,
            ),
            TestCase(
                name="single page with single filters",
                pages=[ResponseMock(json_content=b'["a","b","c"]', status_code=200)],
                expected=["a", "b", "c"],
                get_calls=1,
                filters={"k1": "v1", "k2": "v2"},
            ),
            TestCase(
                name="single page with list filters",
                pages=[ResponseMock(json_content=b'["a","b","c"]', status_code=200)],
                expected=["a", "b", "c"],
                get_calls=1,
                filters={"k1": "v1", "k2": ["v2", "v2.2"]},
            ),
        ]

        for case in test_cases:
            with self.subTest(msg=case.name):
                self._pw_get_patcher.reset_mock()
                self._pw_get_patcher.side_effect = case.pages
                resp = self._pw._Patchwork__get_objects_recursive(
                    "foo", params=case.filters
                )
                self.assertEqual(resp, case.expected)
                self.assertEqual(self._pw_get_patcher.call_count, case.get_calls)
                # Check that our filters are passed to request.get
                params = self._pw_get_patcher.call_args_list[0].kwargs["params"]
                self.assertEqual(params, case.filters)

    def test_try_post_nocred_nomutation(self) -> None:
        """
        When pw_token is not set or is an empty string, we will not call post.
        """

        self._pw.auth_token = None
        self._pw._Patchwork__try_post(
            "https://127.0.0.1:0/some/random/url", "somerandomdata"
        )
        self._pw_post_patcher.assert_not_called()

        self._pw.auth_token = ""
        self._pw._Patchwork__try_post(
            "https://127.0.0.1:0/some/random/url", "somerandomdata"
        )
        self._pw_post_patcher.assert_not_called()

    def test_metered(self) -> None:
        """
        Test the `metered` decorator and ensured the right stats are set.
        """

        # Test the decorator when used with only 1 parameter.
        self._pw.drop_counters()
        self._pw._Patchwork__get_object_by_id("patches", 1234)
        self.assertEqual(self._pw.stats["patches_by_id_count"], 1)

        # Test the decorator when used with 2 parameters (as does `get_blob`).
        self._pw.drop_counters()
        self._pw.get_blob("some fake url")
        self.assertEqual(self._pw.stats["non_api_count"], 1)

    def test_format_since(self) -> None:
        @dataclass
        class TestCase:
            name: str
            now: str
            expected: str
            lookback: int = 3

        PDT_UTC_OFFSET = -7
        test_cases = [
            TestCase(
                name="1st of Jan 00:00 UTC",
                now="2010-01-01T00:00:00",
                expected="2009-12-29T00:00:00",
            ),
            TestCase(
                name="1st of Jan 00:00 PDT",
                now=f"2010-01-01T00:00:00{PDT_UTC_OFFSET:+03}:00",
                expected="2009-12-29T00:00:00",
            ),
            TestCase(
                name="1st of Jan 23:00 UTC",
                now="2010-01-01T23:00:00",
                expected="2009-12-29T00:00:00",
            ),
        ]
        for case in test_cases:
            with self.subTest(msg=case.name):
                # Force local time to be PDT
                with freeze_time(case.now, tz_offset=PDT_UTC_OFFSET):
                    self.assertEqual(
                        self._pw.format_since(case.lookback),
                        case.expected,
                    )

    def test_parse_tags(self) -> None:
        @dataclass
        class TestCase:
            name: str
            title: str
            expected: Set[str]

        test_cases = [
            TestCase(
                name="title without tags",
                title="title",
                expected=set(),
            ),
            TestCase(
                name="title with all valid tags",
                title="[tag1, tag2]title",
                expected={"tag1", "tag2"},
            ),
            TestCase(
                name="title without ignorable tags",
                title="[tag, 1/18, v4]title",
                expected={"tag"},
            ),
            TestCase(
                name="tags with extra spaces",
                title="[tag3, tag1  , tag2]title",
                expected={"tag1", "tag2", "tag3"},
            ),
        ]
        for case in test_cases:
            with self.subTest(msg=case.name):
                self.assertEqual(
                    parse_tags(case.title),
                    case.expected,
                )


def get_dict_key(d: Dict[Any, Any], idx: int = 0) -> Any:
    """
    Given a dictionary, get a list of keys and return the one a `idx`.
    """
    return list(d)[idx]


DEFAULT_FREEZE_DATE = "2010-07-23T00:00:00"
FOO_SERIES_FIRST = 2
FOO_SERIES_LAST = 10
DEFAULT_TEST_RESPONSES = {
    "https://127.0.0.1/api/1.1/series/?q=foo": [
        # Does not match the subject name
        {
            "id": 1,
            "name": "foo bar",
            "date": "2010-07-20T01:00:00",
            "patches": [{"id": 10}],
            "version": 0,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "mbox": "https://example.com",
        },
        # Matches and has relevant diff and is neither the oldest, nor the newest serie. Appears before FOO_SERIES_FIRST to ensure sorting is performed.
        {
            "id": 6,
            "name": "foo",
            "date": "2010-07-20T01:00:00",
            "patches": [{"id": 11}],
            "version": 0,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "mbox": "https://example.com",
        },
        # Matches and has relevant diff
        {
            "id": FOO_SERIES_FIRST,
            "name": "foo",
            "date": "2010-07-20T00:00:00",
            "patches": [{"id": 11}],
            "version": 0,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "mbox": "https://example.com",
        },
        # Matches and has only non-relevant diffs
        {
            "id": 3,
            "name": "foo",
            "date": "2010-07-21T00:00:00",
            "patches": [{"id": 12}, {"id": 13}, {"id": 14}, {"id": 15}, {"id": 16}],
            "version": 0,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "mbox": "https://example.com",
        },
        # Matches and has one relevant diffs
        {
            "id": FOO_SERIES_LAST,
            "name": "foo",
            "date": "2010-07-21T01:00:00",
            "patches": [{"id": 11}, {"id": 13}, {"id": 14}],
            "version": 0,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "mbox": "https://example.com",
        },
        # Matches and has one relevant diffs, not the most recent series, appears after FOO_SERIES_LAST to ensure sorting is performed.
        {
            "id": 4,
            "name": "foo",
            "date": "2010-07-21T00:00:00",
            "patches": [{"id": 11}, {"id": 13}, {"id": 14}],
            "version": 0,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "mbox": "https://example.com",
        },
        # Matches and has only non-relevant diffs
        {
            "id": 5,
            "name": "foo",
            "date": "2010-07-21T02:00:00",
            "patches": [{"id": 12}, {"id": 13}, {"id": 14}, {"id": 15}, {"id": 16}],
            "version": 0,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "mbox": "https://example.com",
        },
    ],
    # Multiple relevant series to test our guess_pr logic.
    "https://127.0.0.1/api/1.1/series/?q=barv2": [
        # Matches and has relevant diff.
        {
            "id": 6,
            "name": "barv2",
            "date": "2010-07-20T01:00:00",
            "patches": [{"id": 11}],
            "version": 1,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "mbox": "https://example.com",
        },
        # Matches, has one relevant diffs, and is the most recent series.
        {
            "id": 9,
            "name": "[v2] barv2",
            "date": "2010-07-21T00:00:00",
            "patches": [{"id": 11}, {"id": 13}, {"id": 14}],
            "version": 2,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "mbox": "https://example.com",
        },
    ],
    # Single relevant series to test our guess_pr logic.
    "https://127.0.0.1/api/1.1/series/?q=code": [
        # Matches, has one relevant diffs, and is the most recent series.
        {
            "id": 9,
            "name": "[v2] barv2",
            "date": "2010-07-21T00:00:00",
            "patches": [{"id": 11}, {"id": 13}, {"id": 14}],
            "version": 2,
            "url": "https://example.com",
            "web_url": "https://example.com",
            "mbox": "https://example.com",
        },
    ],
    # Correct project and delegate
    "https://127.0.0.1/api/1.1/patches/11/": {
        "id": 11,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": False,
    },
    # wrong project
    "https://127.0.0.1/api/1.1/patches/12/": {
        "id": 12,
        "project": {"id": PROJECT + 1},
        "delegate": {"id": DELEGATE},
        "archived": False,
    },
    # Wrong delegate
    "https://127.0.0.1/api/1.1/patches/13/": {
        "id": 13,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE + 1},
        "archived": False,
    },
    # Correct project/delegate but archived
    "https://127.0.0.1/api/1.1/patches/14/": {
        "id": 14,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
    },
    # None project
    "https://127.0.0.1/api/1.1/patches/15/": {
        "id": 15,
        "project": None,
        "delegate": {"id": DELEGATE},
        "archived": False,
    },
    # None delegate
    "https://127.0.0.1/api/1.1/patches/16/": {
        "id": 16,
        "project": {"id": PROJECT},
        "delegate": None,
        "archived": False,
    },
    #####################
    # Series test cases #
    #####################
    # An open series, is a series that has no patch in irrelevant state.
    "https://127.0.0.1/api/1.1/series/665/": {
        "id": 665,
        "name": "[a/b] this series is *NOT* closed!",
        "date": "2010-07-20T01:00:00",
        "patches": [{"id": 6651}, {"id": 6652}, {"id": 6653}],
        "cover_letter": {"name": "[cover letter tag, duplicate tag] cover letter name"},
        "version": 4,
        "url": "https://example.com",
        "web_url": "https://example.com",
        "mbox": "https://example.com",
    },
    # Patch in an relevant state.
    "https://127.0.0.1/api/1.1/patches/6651/": {
        "id": 6651,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": get_dict_key(RELEVANT_STATES),
        # No tag in name.
        "name": "foo",
    },
    # Patch in a relevant state.
    "https://127.0.0.1/api/1.1/patches/6652/": {
        "id": 6652,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": get_dict_key(RELEVANT_STATES, 1),
        # Multiple tags. 0/5 and 1/2 should be ignored.
        # Same for v42 and v24, and patch.
        # only "first patch tag", "second patch tag", "some stuff with spaces" should be valid.
        "name": "[0/5, 1/2 , v42, V24, first patch tag, second patch tag, patch , some stuff with spaces , patch] bar",
    },
    # Patch in an relevant state.
    "https://127.0.0.1/api/1.1/patches/6653/": {
        "id": 6653,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": get_dict_key(RELEVANT_STATES),
        # Single tag, which is a duplicate from cover letter.
        "name": "[duplicate tag] foo",
    },
    # A closed series, is a series that has at least 1 patch in an irrelevant state.
    "https://127.0.0.1/api/1.1/series/666/": {
        "id": 666,
        "name": "this series is closed!",
        "date": "2010-07-20T01:00:00",
        "patches": [{"id": 6661}, {"id": 6662}],
        "version": 0,
        "url": "https://example.com",
        "web_url": "https://example.com",
        "mbox": "https://example.com",
    },
    # Patch in an irrelevant state.
    "https://127.0.0.1/api/1.1/patches/6661/": {
        "id": 6661,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": get_dict_key(IRRELEVANT_STATES),
    },
    # Patch in a relevant state.
    "https://127.0.0.1/api/1.1/patches/6662/": {
        "id": 6662,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": get_dict_key(RELEVANT_STATES),
    },
    # Series with no cover letter and no patches.
    "https://127.0.0.1/api/1.1/series/667/": {
        "id": 667,
        "name": "this series has no cover letter!",
        "date": "2010-07-20T01:00:00",
        "patches": [],
        "cover_letter": None,
        "version": 1,
        "url": "https://example.com",
        "web_url": "https://example.com",
        "mbox": "https://example.com",
    },
    #### Expiration test cases
    # Series with expirable patches.
    "https://127.0.0.1/api/1.1/series/668/": {
        "id": 668,
        "name": "this series has no cover letter!",
        "date": "2010-07-20T01:00:00",
        "patches": [{"id": 6681}, {"id": 6682}, {"id": 6683}],
        "cover_letter": None,
        "version": 1,
        "url": "https://example.com",
        "web_url": "https://example.com",
        "mbox": "https://example.com",
    },
    # Patch in a non-expirable state.
    "https://127.0.0.1/api/1.1/patches/6681/": {
        "id": 6681,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": "new",
    },
    # Patch in an expirable state.
    "https://127.0.0.1/api/1.1/patches/6682/": {
        "id": 6682,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": get_dict_key(TTL),
        "date": DEFAULT_FREEZE_DATE,
    },
    # Patch in a non-expirable state.
    "https://127.0.0.1/api/1.1/patches/6683/": {
        "id": 6683,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": "new",
    },
    # Series with no expirable patches.
    "https://127.0.0.1/api/1.1/series/669/": {
        "id": 669,
        "name": "this series has no cover letter!",
        "date": "2010-07-20T01:00:00",
        "patches": [{"id": 6691}, {"id": 6692}, {"id": 6693}],
        "cover_letter": None,
        "version": 1,
        "url": "https://example.com",
        "web_url": "https://example.com",
        "mbox": "https://example.com",
    },
    # Patch in a non-expirable state.
    "https://127.0.0.1/api/1.1/patches/6691/": {
        "id": 6691,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": "new",
    },
    # Patch in a non-expirable state.
    "https://127.0.0.1/api/1.1/patches/6692/": {
        "id": 6692,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": "new",
    },
    # Patch in a non-expirable state.
    "https://127.0.0.1/api/1.1/patches/6693/": {
        "id": 6693,
        "project": {"id": PROJECT},
        "delegate": {"id": DELEGATE},
        "archived": True,
        "state": "new",
    },
}


class TestSeries(PatchworkTestCase):
    def test_series_closed(self) -> None:
        """
        If one of the patch is irrelevant, the series is closed.
        """
        self._pw_get_patcher.side_effect = pw_response_generator(DEFAULT_TEST_RESPONSES)
        series = self._pw.get_series_by_id(666)
        self.assertTrue(series.is_closed())

    def test_series_not_closed(self) -> None:
        """
        If none of the patches are irrelevant, the series is not closed.
        """
        self._pw_get_patcher.side_effect = pw_response_generator(DEFAULT_TEST_RESPONSES)
        series = self._pw.get_series_by_id(665)
        self.assertFalse(series.is_closed())

    def test_series_tags(self) -> None:
        """
        Series tags are extracted from the diffs/cover letter/serie's name. We extract the content
        from the square bracket content prefixing those names and filter some out.
        """
        self._pw_get_patcher.side_effect = pw_response_generator(DEFAULT_TEST_RESPONSES)
        series = self._pw.get_series_by_id(665)
        self.assertEqual(
            series.all_tags(),
            {
                # cover letter tags
                "cover letter tag",
                "duplicate tag",
                # relevant states
                get_dict_key(RELEVANT_STATES),
                get_dict_key(RELEVANT_STATES, 1),
                # series version
                "V4",
                # Series name
                "a/b",
                # patches
                "first patch tag",
                "second patch tag",
                "some stuff with spaces",
            },
        )

    def test_series_visible_tags(self) -> None:
        """
        Series' visible tags are only taken from patch states and series version.
        """
        self._pw_get_patcher.side_effect = pw_response_generator(DEFAULT_TEST_RESPONSES)
        series = self._pw.get_series_by_id(665)
        self.assertEqual(
            series.visible_tags(),
            {
                # relevant states
                get_dict_key(RELEVANT_STATES),
                get_dict_key(RELEVANT_STATES, 1),
                # series version
                "V4",
            },
        )

    def test_series_tags_handle_missing_values(self) -> None:
        """
        Test that we handle correctly series with empty cover_letter and/or no attaches patches.
        """
        self._pw_get_patcher.side_effect = pw_response_generator(DEFAULT_TEST_RESPONSES)
        series = self._pw.get_series_by_id(667)
        self.assertEqual(
            series.all_tags(),
            {
                "V1",
            },
        )

    def test_series_is_expired(self) -> None:
        """
        Test that when we are passed expiration date, the series is reported expired.
        """
        self._pw_get_patcher.side_effect = pw_response_generator(DEFAULT_TEST_RESPONSES)
        series = self._pw.get_series_by_id(668)
        ttl = TTL[get_dict_key(TTL)]
        # take the date of the patch and move to 1 second after that.
        now = datetime.datetime.fromisoformat(DEFAULT_FREEZE_DATE) + datetime.timedelta(
            seconds=ttl + 1
        )
        with freeze_time(now):
            self.assertTrue(series.is_expired())

    def test_series_is_not_expired(self) -> None:
        """
        Test that when we are not yet passed expiration date, the series is not reported expired.
        """
        self._pw_get_patcher.side_effect = pw_response_generator(DEFAULT_TEST_RESPONSES)
        series = self._pw.get_series_by_id(668)
        ttl = TTL[get_dict_key(TTL)]
        # take the date of the patch and move to 1 second before that.
        now = datetime.datetime.fromisoformat(DEFAULT_FREEZE_DATE) + datetime.timedelta(
            seconds=ttl - 1
        )
        with freeze_time(now):
            self.assertFalse(series.is_expired())

    def test_last_check_states_for_context(self) -> None:
        """
        Tests refactored `last_check_States_for_context` function,
        which now makes a GET request with additional queries to ensure that
        it's fetching single latest data with matching context value
        """
        # space intentional to test urlencoding
        CTX = "vmtest bpf-next-PR"
        # not using requests.utils.requote_uri() since that encodes space to %20, while requests.get() encodes space to +
        CTX_URLENCODED = "vmtest+bpf-next-PR"
        EXPECTED_ID = 42
        contexts_responses = {
            f"https://127.0.0.1/api/1.1/patches/12859379/checks/?context={CTX_URLENCODED}&order=-date&per_page=1": [
                {
                    "context": CTX,
                    "date": "2010-01-02T00:00:00",
                    "state": "pending",
                    "id": EXPECTED_ID,
                },
            ],
        }

        self._pw_get_patcher.side_effect = pw_response_generator(contexts_responses)
        check = self._pw.last_check_states_for_context(CTX, 12859379)
        self.assertEqual(check["id"], EXPECTED_ID)

    def test_series_checks_update_all_diffs(self) -> None:
        """
        Test that we update all the diffs in a serie if there is either no existing check
        or checks need update.
        """
        contexts_responses = copy.deepcopy(DEFAULT_TEST_RESPONSES)
        contexts_responses.update(
            {
                # Patch with existing context
                f"https://127.0.0.1/api/1.1/patches/6651/checks/{DEFAULT_CHECK_CTX_QUERY}": [
                    {
                        "context": DEFAULT_CHECK_CTX,
                        "date": "2010-01-01T00:00:00",
                        "state": "pending",
                    },
                ],
                # Patch without existing context
                f"https://127.0.0.1/api/1.1/patches/6652/checks/{DEFAULT_CHECK_CTX_QUERY}": [],
            }
        )

        self._pw_get_patcher.side_effect = pw_response_generator(contexts_responses)
        series = self._pw.get_series_by_id(665)
        series.set_check(
            context=DEFAULT_CHECK_CTX,
            state=None,
            target_url="https://127.0.0.1:0/target",
        )
        self.assertEqual(self._pw_post_patcher.call_count, 3)

    def test_series_checks_no_update_same_state_target(self) -> None:
        """
        Test that we don't update checks if the state and target_url have not changed.
        """
        TARGET_URL = "https://127.0.0.1:0/target"
        contexts_responses = copy.deepcopy(DEFAULT_TEST_RESPONSES)
        contexts_responses.update(
            {
                # Patch with existing context
                f"https://127.0.0.1/api/1.1/patches/6651/checks/{DEFAULT_CHECK_CTX_QUERY}": [
                    {
                        "context": DEFAULT_CHECK_CTX,
                        "date": "2010-01-01T00:00:00",
                        "state": "pending",
                        "target_url": TARGET_URL,
                    },
                ],
                # Patch without existing context
                f"https://127.0.0.1/api/1.1/patches/6652/checks/{DEFAULT_CHECK_CTX_QUERY}": [],
            }
        )

        self._pw_get_patcher.side_effect = pw_response_generator(contexts_responses)
        series = self._pw.get_series_by_id(665)
        series.set_check(context=DEFAULT_CHECK_CTX, state=None, target_url=TARGET_URL)
        # First patch is not updates
        self.assertEqual(self._pw_post_patcher.call_count, len(series.patches) - 1)

    def test_series_checks_update_same_state_diff_target(self) -> None:
        """
        Test that we update checks if the state is the same, but target_url has changed.
        """
        TARGET_URL = "https://127.0.0.1:0/target"
        contexts_responses = copy.deepcopy(DEFAULT_TEST_RESPONSES)
        contexts_responses.update(
            {
                # Patch with existing context
                f"https://127.0.0.1/api/1.1/patches/6651/checks/{DEFAULT_CHECK_CTX_QUERY}": [
                    {
                        "context": DEFAULT_CHECK_CTX,
                        "date": "2010-01-01T00:00:00",
                        "state": "pending",
                        # target_url not matching
                        "target_url": TARGET_URL + "something",
                    },
                ],
                # Patch without existing context
                f"https://127.0.0.1/api/1.1/patches/6652/checks/{DEFAULT_CHECK_CTX_QUERY}": [],
            }
        )

        self._pw_get_patcher.side_effect = pw_response_generator(contexts_responses)
        series = self._pw.get_series_by_id(665)
        series.set_check(context=DEFAULT_CHECK_CTX, state=None, target_url=TARGET_URL)
        # First patch is not updates
        self.assertEqual(self._pw_post_patcher.call_count, len(series.patches))

    def test_series_checks_update_diff_state_same_target(self) -> None:
        """
        Test that we update checks if the state is not the same, but target_url is.
        """
        TARGET_URL = "https://127.0.0.1:0/target"
        contexts_responses = copy.deepcopy(DEFAULT_TEST_RESPONSES)
        contexts_responses.update(
            {
                # Patch with existing context
                f"https://127.0.0.1/api/1.1/patches/6651/checks/{DEFAULT_CHECK_CTX_QUERY}": [
                    {
                        "context": DEFAULT_CHECK_CTX,
                        "date": "2010-01-01T00:00:00",
                        "state": "pending",
                        "target_url": TARGET_URL,
                    },
                    {
                        "context": "other context",
                    },
                ],
                # Patch without existing context
                f"https://127.0.0.1/api/1.1/patches/6652/checks/{DEFAULT_CHECK_CTX_QUERY}": [],
            }
        )

        self._pw_get_patcher.side_effect = pw_response_generator(contexts_responses)
        series = self._pw.get_series_by_id(665)
        # success is a conclusive (non-pending) state.
        series.set_check(
            context=DEFAULT_CHECK_CTX, state="success", target_url=TARGET_URL
        )
        # First patch is not updates
        self.assertEqual(self._pw_post_patcher.call_count, len(series.patches))

    def test_series_checks_no_update_diff_pending_state(self) -> None:
        """
        Test that we do not update checks if the new state is pending and we have an existing final state.
        """
        TARGET_URL = "https://127.0.0.1:0/target"
        contexts_responses = copy.deepcopy(DEFAULT_TEST_RESPONSES)
        contexts_responses.update(
            {
                # Patch with existing context
                f"https://127.0.0.1/api/1.1/patches/6651/checks/{DEFAULT_CHECK_CTX_QUERY}": [
                    {
                        "context": DEFAULT_CHECK_CTX,
                        "date": "2010-01-01T00:00:00",
                        # conclusive state.
                        "state": "success",
                        "target_url": TARGET_URL,
                    },
                ],
                # Patch without existing context
                f"https://127.0.0.1/api/1.1/patches/6652/checks/{DEFAULT_CHECK_CTX_QUERY}": [],
            }
        )

        self._pw_get_patcher.side_effect = pw_response_generator(contexts_responses)
        series = self._pw.get_series_by_id(665)
        series.set_check(context=DEFAULT_CHECK_CTX, state=None, target_url=TARGET_URL)
        # First patch is not updates
        self.assertEqual(self._pw_post_patcher.call_count, len(series.patches) - 1)


class TestSubject(PatchworkTestCase):
    @freeze_time(DEFAULT_FREEZE_DATE)
    def test_relevant_series(self) -> None:
        """
        Test that we find the relevant series for a given subject.
        """
        self._pw_get_patcher.side_effect = pw_response_generator(DEFAULT_TEST_RESPONSES)
        s = Subject("foo", self._pw)
        # There is 2 relevant series
        self.assertEqual(len(s.relevant_series), 4)
        # Series are ordered from oldest to newest
        series = s.relevant_series[0]
        self.assertEqual(series.id, FOO_SERIES_FIRST)
        # It has only 1 diff, diff 11
        self.assertEqual(len(series.get_patches()), 1)
        self.assertEqual([patch["id"] for patch in series.get_patches()], [11])

    @freeze_time(DEFAULT_FREEZE_DATE)
    def test_latest_series(self) -> None:
        """
        Test that latest_series only returns.... the latest serie.
        """
        self._pw_get_patcher.side_effect = pw_response_generator(DEFAULT_TEST_RESPONSES)
        s = Subject("foo", self._pw)
        latest_series = none_throws(s.latest_series)
        # It is Series with ID FOO_SERIES_LAST
        self.assertEqual(latest_series.id, FOO_SERIES_LAST)
        # and has 3 diffs
        self.assertEqual(len(latest_series.get_patches()), 3)

    @freeze_time(DEFAULT_FREEZE_DATE)
    def test_branch_name(self) -> None:
        """
        Test that the branch name is made using the first series ID in the list of relevant series.
        """
        self._pw_get_patcher.side_effect = pw_response_generator(DEFAULT_TEST_RESPONSES)
        s = Subject("foo", self._pw)
        branch = s.branch
        # It is Series with ID 4
        self.assertEqual(branch, f"series/{FOO_SERIES_FIRST}")
