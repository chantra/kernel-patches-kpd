#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import datetime
import unittest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import munch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from freezegun import freeze_time
from github import AppAuthentication, GithubException
from github.Requester import Requester

from sources.github_connector import (
    ACCESS_TOKEN_REFRESH_THRESHOLD_SECONDS,
    Github,
    GithubConnector,
)

DEFAULT_FREEZE_DATE = "2010-07-23T00:00:00"

TEST_ORG = "org"
TEST_REPO = "repo"
TEST_REPO_URL = f"https://user:pass@127.0.0.1:0/{TEST_ORG}/{TEST_REPO}"
TEST_APP_ID = 1
TEST_INSTALLATION_ID = 2
TEST_PRIV_KEY = (
    rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    .private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    .decode()
)


class GithubConnectorMock(GithubConnector):
    def __init__(self, *args, **kwargs) -> None:
        presets = {
            "repo_url": TEST_REPO_URL,
        }
        presets.update(kwargs)

        super().__init__(*args, **presets)


def get_default_gc_oauth_client() -> GithubConnectorMock:
    return GithubConnectorMock(github_oauth_token="random_gh_oauth_token")


def get_default_gc_app_auth_client(**kwargs) -> GithubConnectorMock:
    presets = {
        "app_auth": AppAuthentication(TEST_APP_ID, TEST_PRIV_KEY, TEST_INSTALLATION_ID),
    }
    presets.update(kwargs)
    return GithubConnectorMock(**presets)


class TestGithubConnector(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch("sources.github_connector.Github")
        self._gh_mock = patcher.start()
        self.addCleanup(patcher.stop)

        # Get app will make external calls. Mock it by default.
        get_app_patcher = patch("sources.github_connector.get_app")
        self._get_app_mock = get_app_patcher.start()
        self.addCleanup(get_app_patcher.stop)

    def test_oauth_get_repo_no_fallback(self) -> None:
        """
        We first try to get the repo from the logged user unless it throws an exception.
        """

        fake_user = "fake_user"
        # Our GH user mock
        user_mock = MagicMock(login=fake_user)
        # Is returned by `Github.get_user()` call
        self._gh_mock.return_value.get_user.return_value = user_mock

        gc = get_default_gc_oauth_client()
        # We do auth first
        gc.git.get_user.assert_called_once()
        # then try to get the repo as the user
        user_mock.get_repo.assert_called_once_with(TEST_REPO)
        # and we do not fallback to getting the org
        gc.git.get_organization.assert_not_called()

        # user_or_org is derived from the Github auth user....
        self.assertEqual(gc.user_or_org, fake_user)

    def test_oauth_get_repo_fallback_to_org_repo(self) -> None:
        """
        We first try to get the repo from the logged user, if it throw an exception, get it from the "organization" of the repo URL.
        """

        # Our GH user mock
        user_mock = MagicMock()
        # Is returned by `Github.get_user()` call
        self._gh_mock.return_value.get_user.return_value = user_mock

        # Force throwing an exception
        user_mock.get_repo.side_effect = GithubException(
            "gh exception", "data", "headers"
        )
        m = MagicMock()
        self._gh_mock.return_value.get_organization.return_value = m

        # and instanciate the Github connector instance again now that we have the right mock in place.

        gc = get_default_gc_oauth_client()
        # We do auth first
        gc.git.get_user.assert_called_once()
        # then try to get the repo as the user
        user_mock.get_repo.assert_called_once_with(TEST_REPO)
        # and fallback to getting the org
        gc.git.get_organization.assert_called_once_with(TEST_ORG)
        m.get_repo.assert_called_once_with(TEST_REPO)

        # user_or_org is set to org
        self.assertEqual(gc.user_or_org, TEST_ORG)

    def test_oauth_get_user_login(self) -> None:
        """
        Test that if git.get_user().login has a value, it is properly returned
        by `get_user_login`
        """
        fake_user = "fake_user"
        # We do auth first
        m = MagicMock(login=fake_user)
        self._gh_mock.return_value.get_user.return_value = m

        gc = get_default_gc_oauth_client()
        self.assertEqual(gc.get_user_login(), fake_user)

    def test_app_auth_get_user_login(self) -> None:
        """
        Test that if we authenticate as a bot, we get the correct user_login.
        """
        fake_app = "fake_app"
        self._get_app_mock.return_value = munch.munchify({"name": fake_app})
        gc = get_default_gc_app_auth_client()
        self.assertEqual(gc.get_user_login(), fake_app + "[bot]")

    def test_gc_connect_only_one_auth(self) -> None:
        """
        Make sure that only one github authentication mechanism is provided.
        """

        @dataclass
        class TestCase:
            name: str
            oauth_token: Optional[str] = "some oauth token"
            app_auth: Optional[AppAuthentication] = AppAuthentication(
                TEST_APP_ID, TEST_PRIV_KEY, TEST_INSTALLATION_ID
            )
            exception_expected: bool = True

        test_cases = [
            TestCase(
                name="no authentication mechanism",
                oauth_token=None,
                app_auth=None,
                exception_expected=True,
            ),
            TestCase(
                name="Too many authentication mechanism",
                exception_expected=True,
            ),
            TestCase(
                name="only oauth token mechanism",
                oauth_token=None,
                exception_expected=False,
            ),
            TestCase(
                name="only app_auth mechanism",
                app_auth=None,
                exception_expected=False,
            ),
        ]
        for case in test_cases:
            with self.subTest(msg=case.name):
                if case.exception_expected:
                    with self.assertRaises(AssertionError):
                        _ = GithubConnectorMock(
                            github_oauth_token=case.oauth_token, app_auth=case.app_auth
                        )
                else:
                    _ = GithubConnectorMock(
                        github_oauth_token=case.oauth_token, app_auth=case.app_auth
                    )


class TestGithubConnectorAuth(unittest.TestCase):
    def setUp(self) -> None:
        """
        Sets up github_connector so only some sub-parts of github api are mocked.
        This allows us to test the calls to the authentication backend and control
        its returned values via side_effects.
        """
        # Get app will make external calls. Mock it by default.
        get_app_patcher = patch("sources.github_connector.get_app")
        self._get_app_mock = get_app_patcher.start()
        self.addCleanup(get_app_patcher.stop)
        self._get_app_mock.return_value = munch.munchify({"name": "test_user"})

        get_repo_patcher = patch.object(Github, "get_repo")
        self._get_repo_mock = get_repo_patcher.start()
        self.addCleanup(get_repo_patcher.stop)

        get_user_patcher = patch.object(Github, "get_user")
        self._get_user_mock = get_user_patcher.start()
        self.addCleanup(get_user_patcher.stop)

    def test_renew_expired_token(self) -> None:
        """
        Verifies that `Requester._refresh_token_if_needed` does renew an expired token.
        """
        now = datetime.datetime.fromisoformat(DEFAULT_FREEZE_DATE)
        expired_at_date = now + datetime.timedelta(hours=2)
        expired_at_next = expired_at_date + datetime.timedelta(hours=2)
        side_effect = [
            munch.munchify({"token": "token1", "expires_at": expired_at_date}),
            munch.munchify({"token": "token2", "expires_at": expired_at_next}),
        ]
        with patch.object(
            Requester,
            "_get_installation_authorization",
            side_effect=side_effect,
        ) as p, freeze_time(DEFAULT_FREEZE_DATE) as frozen_datetime:
            gc = get_default_gc_app_auth_client()
            self.assertEqual(p.call_count, 1)
            # set time to 1 second after expiration so that we renew the token.
            frozen_datetime.tick(
                delta=datetime.timedelta(
                    seconds=datetime.timedelta(hours=2).seconds
                    - ACCESS_TOKEN_REFRESH_THRESHOLD_SECONDS
                    + 1
                )
            )
            # pyre-fixme[16]: `github.MainClass.Github` has no attribute `__requester`.
            gc.git._Github__requester._refresh_token_if_needed()
            self.assertEqual(p.call_count, 2)

    def test_donot_renew_non_expired_token(self) -> None:
        """
        Verifies that `Requester._refresh_token_if_needed` does not renew a token
        which is not expired yet.
        """
        now = datetime.datetime.fromisoformat(DEFAULT_FREEZE_DATE)
        expired_at_date = now + datetime.timedelta(hours=2)
        expired_at_next = expired_at_date + datetime.timedelta(hours=2)
        side_effect = [
            munch.munchify({"token": "token1", "expires_at": expired_at_date}),
            munch.munchify({"token": "token2", "expires_at": expired_at_next}),
        ]
        with patch.object(
            Requester,
            "_get_installation_authorization",
            side_effect=side_effect,
        ) as p, freeze_time(DEFAULT_FREEZE_DATE) as frozen_datetime:
            gc = get_default_gc_app_auth_client()
            self.assertEqual(p.call_count, 1)
            # Set time to 1 seconds before expiration so that we do not renew the token.
            frozen_datetime.tick(
                delta=datetime.timedelta(
                    seconds=datetime.timedelta(hours=2).seconds
                    - ACCESS_TOKEN_REFRESH_THRESHOLD_SECONDS
                    - 1
                )
            )
            # pyre-fixme[16]: `github.MainClass.Github` has no attribute `__requester`.
            gc.git._Github__requester._refresh_token_if_needed()
            self.assertEqual(p.call_count, 1)

    def test_repo_url(self) -> None:
        """
        Verifies that when using app authentication, our repo_url get updated to
        reflect the current token, while when using oauth authentication, the
        repo_url stays the same as provided by the caller upon initialization.
        """
        now = datetime.datetime.fromisoformat(DEFAULT_FREEZE_DATE)
        expired_at_date = now + datetime.timedelta(hours=2)
        expired_at_next = expired_at_date + datetime.timedelta(hours=2)
        side_effect = [
            munch.munchify({"token": "token1", "expires_at": expired_at_date}),
            munch.munchify({"token": "token2", "expires_at": expired_at_next}),
        ]
        with patch.object(
            Requester,
            "_get_installation_authorization",
            side_effect=side_effect,
        ) as p, freeze_time(DEFAULT_FREEZE_DATE) as frozen_datetime:
            gc_app_auth = get_default_gc_app_auth_client()
            gc_oauth = get_default_gc_oauth_client()

            self.assertEqual(p.call_count, 1)
            # Check that repo_url contains the first token.
            self.assertIn("test_user[bot]:token1", gc_app_auth.repo_url())
            self.assertEqual(TEST_REPO_URL, gc_oauth.repo_url())
            # Set time to 1 seconds before expiration so that we do not renew the token.
            frozen_datetime.tick(
                delta=datetime.timedelta(
                    seconds=datetime.timedelta(hours=2).seconds
                    - ACCESS_TOKEN_REFRESH_THRESHOLD_SECONDS
                    - 1
                )
            )
            # And still the same when the token is not renewed.
            self.assertIn("test_user[bot]:token1", gc_app_auth.repo_url())
            self.assertEqual(TEST_REPO_URL, gc_oauth.repo_url())

            self.assertEqual(p.call_count, 1)

            # move past renewal time
            frozen_datetime.tick(delta=datetime.timedelta(seconds=2))
            # And the token get renewed
            self.assertIn("test_user[bot]:token2", gc_app_auth.repo_url())
            self.assertEqual(TEST_REPO_URL, gc_oauth.repo_url())

            self.assertEqual(p.call_count, 2)

    def test_set_user_token_in_url_when_not_present(self) -> None:
        """
        Verifies that when user:token is not initially present in the `repo_url`,
        the user:token from the gh app is inserted into the url's netloc.
        """
        now = datetime.datetime.fromisoformat(DEFAULT_FREEZE_DATE)
        expired_at_date = now + datetime.timedelta(hours=2)
        side_effect = [
            munch.munchify({"token": "token1", "expires_at": expired_at_date}),
        ]

        with patch.object(
            Requester,
            "_get_installation_authorization",
            side_effect=side_effect,
        ) as p, freeze_time(DEFAULT_FREEZE_DATE):
            gc_app_auth = get_default_gc_app_auth_client(
                repo_url=f"https://127.0.0.1:0/{TEST_ORG}/{TEST_REPO}"
            )
            self.assertEqual(
                f"https://test_user[bot]:token1@127.0.0.1:0/{TEST_ORG}/{TEST_REPO}",
                gc_app_auth.repo_url(),
            )
            self.assertEqual(p.call_count, 1)

    def test_repo_url_port_handling(self) -> None:
        """
        Verifies that we properly handle whether or not to append the port number
        in the repo_url.
        """

        @dataclass
        class TestCase:
            name: str
            initial_url: str
            expected_url: str

        test_cases = [
            TestCase(
                name="port is 0",
                initial_url=f"https://127.0.0.1:0/{TEST_ORG}/{TEST_REPO}",
                expected_url=f"https://test_user[bot]:token1@127.0.0.1:0/{TEST_ORG}/{TEST_REPO}",
            ),
            TestCase(
                name="port is not 0",
                initial_url=f"https://127.0.0.1:1/{TEST_ORG}/{TEST_REPO}",
                expected_url=f"https://test_user[bot]:token1@127.0.0.1:1/{TEST_ORG}/{TEST_REPO}",
            ),
            TestCase(
                name="port is not present",
                initial_url=f"https://127.0.0.1/{TEST_ORG}/{TEST_REPO}",
                expected_url=f"https://test_user[bot]:token1@127.0.0.1/{TEST_ORG}/{TEST_REPO}",
            ),
        ]
        for case in test_cases:
            with self.subTest(msg=case.name):
                now = datetime.datetime.fromisoformat(DEFAULT_FREEZE_DATE)
                expired_at_date = now + datetime.timedelta(hours=2)
                side_effect = [
                    munch.munchify({"token": "token1", "expires_at": expired_at_date}),
                ]

                with patch.object(
                    Requester,
                    "_get_installation_authorization",
                    side_effect=side_effect,
                ) as p, freeze_time(DEFAULT_FREEZE_DATE):
                    gc_app_auth = get_default_gc_app_auth_client(
                        repo_url=case.initial_url
                    )
                    self.assertEqual(
                        case.expected_url,
                        gc_app_auth.repo_url(),
                    )
                    self.assertEqual(p.call_count, 1)
