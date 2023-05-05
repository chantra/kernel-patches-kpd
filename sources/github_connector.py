# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
from enum import Enum
from typing import Optional
from urllib.parse import urlparse

from github import AppAuthentication, Github, GithubApp, GithubException, Requester

# FIXME: monkey patch the refresh threshold of GH App access token to 30min.
# This monkey patching may silently fail if the variable was renamed so first
# let assert if that variable does not exist.
# Next, we change its value to 30min so we can ensure that whenever we have a token,
# it is valid for at least 30 min.
# The reason we are doing this is that when we use this token for git operations,
# we are essentially setting a URL with the token embedded in it when pulling/pushing
# to git.
# We could reset the token for every write operations (the repo is world readable), but there is
# chances we will miss some. Instead, it will be easier to set the origin URL on every sync,
# giving us `ACCESS_TOKEN_REFRESH_THRESHOLD_SECONDS` to perform the operations.
# While arbitrary, this should be enough. For the case where we do a full clone,
# this should only happen on fresh start, which comes with a token with a validity of
# 2 hours.

ACCESS_TOKEN_REFRESH_THRESHOLD_SECONDS = 30 * 60
assert hasattr(
    Requester, "ACCESS_TOKEN_REFRESH_THRESHOLD_SECONDS"
), "Could not monkey patch ACCESS_TOKEN_REFRESH_THRESHOLD_SECONDS, it may have changed upstream."
Requester.ACCESS_TOKEN_REFRESH_THRESHOLD_SECONDS = (
    ACCESS_TOKEN_REFRESH_THRESHOLD_SECONDS
)

BOT_USER_LOGIN_SUFFIX = "[bot]"

# Upstream get_app() call is broken currently.
# implement our own for now.
def get_app(app_auth: AppAuthentication) -> GithubApp.GithubApp:
    from github import GithubIntegration

    # pyre-fixme[16]: `AppAuthentication` has no attribute `app_id`.
    gi = GithubIntegration(app_auth.app_id, app_auth.private_key)
    headers, response = gi._GithubIntegration__requester.requestJsonAndCheck(
        "GET", "/app", headers=gi._get_headers()
    )
    return GithubApp.GithubApp(
        gi._GithubIntegration__requester, headers, response, completed=True
    )


class AuthType(Enum):
    APP_AUTH = 1
    OAUTH_TOKEN = 2
    UNKNOWN = 3


class GithubConnector:
    """
    Base class for fetching basic Github Repo information for
    fbcode.kernel.kernel_patches_daemon.github.source.github_sync and
    fbcode.kernel.kernel_patches_daemon.statcollector
    """

    def __init__(
        self,
        repo_url: str,
        github_oauth_token: Optional[str] = None,
        app_auth: Optional[AppAuthentication] = None,
        http_retries: Optional[int] = None,
    ) -> None:

        assert bool(github_oauth_token) ^ bool(
            app_auth
        ), "Only one of github_oauth_token or app_auth can be set"
        self.repo_name: str = os.path.basename(repo_url)
        self._repo_url: str = repo_url
        self.auth_type = AuthType.UNKNOWN

        self.git: Github = Github(
            login_or_token=github_oauth_token, app_auth=app_auth, retry=http_retries
        )
        gh_user = self.git.get_user()
        if app_auth is None:
            self.auth_type = AuthType.OAUTH_TOKEN
            self.user_login = gh_user.login
        else:
            self.auth_type = AuthType.APP_AUTH
            app = get_app(app_auth)
            # Github appends '[bot]' suffix to the NamedUser
            # >>> pull.user
            # NamedUser(login="kernel-patches-daemon-bpf[bot]")
            self.user_login = app.name + BOT_USER_LOGIN_SUFFIX

        self.user_or_org: str = self.user_login

        try:
            # When using app_auth, this will raise a GithubException
            self.repo = gh_user.get_repo(self.repo_name)
        except GithubException:
            # are we working under org repo?
            org = ""
            if "https://" not in repo_url and "ssh://" not in repo_url:
                org = repo_url.split(":")[-1].split("/")[0]
            else:
                org = repo_url.split("/")[-2]
            self.user_or_org = org
            self.repo = self.git.get_organization(org).get_repo(self.repo_name)

        assert (
            self.auth_type != AuthType.UNKNOWN
        ), "Auth type is still set to unknown... something is wrong."

    def get_user_login(self) -> str:
        return self.user_login

    def repo_url(self) -> str:
        # When using app_auth, the URL needs to be periodically updated as the
        # auth token expires.
        if self.auth_type == AuthType.APP_AUTH:
            # pyre-fixme[16]: `github.MainClass.Github` has no attribute `__requester`.
            # refresh token if needed
            self.git._Github__requester._refresh_token_if_needed()
            installation_auth = (
                self.git._Github__requester._Requester__installation_authorization
            )

            parsed_url = urlparse(self._repo_url)
            new_url = parsed_url._replace(
                netloc="{}:{}@{}{}".format(
                    self.get_user_login(),
                    installation_auth.token,
                    parsed_url.hostname,
                    # Case of port embedded in url
                    parsed_url.port is not None and f":{parsed_url.port}" or "",
                )
            )
            return new_url.geturl()

        return self._repo_url
