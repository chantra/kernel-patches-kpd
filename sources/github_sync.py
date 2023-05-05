#!/usr/bin/env python3

import copy
import hashlib
import logging
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import dateutil.parser
import git
from github import AppAuthentication, GithubException
from github.PullRequest import PullRequest
from github.Repository import Repository
from opentelemetry import metrics
from pyre_extensions import none_throws

# @manual=:github_connector
from .github_connector import GithubConnector

from .patchwork import Patchwork, Series, Subject  # @manual
from .stats import Stats  # @manual:=stats
from .utils import redact_url, remove_unsafe_chars  # @manual=:utils


logger: logging.Logger = logging.getLogger(__name__)
meter: metrics.Meter = metrics.get_meter("github_sync")

total_time = meter.create_histogram(name="total_time")
fetch_time = meter.create_histogram(name="fetch_time")
bug_occurence = meter.create_counter(name="bug_occurence")

HTTP_RETRIES = 3
BRANCH_TTL = 172800  # w
PULL_REQUEST_TTL = timedelta(days=7)
HEAD_BASE_SEPARATOR = "=>"
KNOWN_OK_COMMENT_EXCEPTIONS = {
    "Commenting is disabled on issues with more than 2500 comments"
}
CI_APP = 15368  # GithubApp(url="/apps/github-actions", id=15368)
CI_VMTEST_NAME = "VM_Test"

CI_DESCRIPTION = "vmtest"
MERGE_CONFLICT_LABEL = "merge-conflict"


class NewPRWithNoChangeException(Exception):
    def __init__(self, base_branch, target_branch, *args):
        super().__init__(args)
        self.base_branch = base_branch
        self.target_branch = target_branch

    def __str__(self):
        return f"No changes between {self.base_branch} and {self.target_branch}, cannot create PR if there is no change. Was this series/patches recently merged?"


def execute_command(cmd: str) -> None:
    logger.info(cmd)
    os.system(cmd)


def _uniq_tmp_folder(
    url: Optional[str], branch: Optional[str], base_directory: str
) -> str:
    # use same folder for multiple invocation to avoid cloning whole tree every time
    # but use different folder for different workers identified by url and branch name
    sha = hashlib.sha256()
    sha.update(f"{url}/{branch}".encode("utf-8"))
    # pyre-fixme[6]: For 1st argument expected `PathLike[Variable[AnyStr <: [str,
    #  bytes]]]` but got `Optional[str]`.
    repo_name = remove_unsafe_chars(os.path.basename(url))
    return os.path.join(base_directory, f"pw_sync_{repo_name}_{sha.hexdigest()}")


def create_color_labels(labels_cfg: Dict[str, str], repo: Repository) -> None:
    all_labels = {x.name.lower(): x for x in repo.get_labels()}
    labels_cfg = {k.lower(): v for k, v in labels_cfg.items()}
    for label in labels_cfg:
        if label in all_labels:
            if (
                all_labels[label].name != label
                or all_labels[label].color != labels_cfg[label]
            ):
                all_labels[label].edit(name=label, color=labels_cfg[label])
        else:
            repo.create_label(name=label, color=labels_cfg[label])


def _reset_repo(repo, branch: str) -> None:
    """
    Reset the repository into a known good state, with `branch` checked out.
    """
    try:
        repo.git.am("--abort")
    except git.exc.GitCommandError:
        pass

    repo.git.reset("--hard", branch)
    repo.git.checkout(branch)


def _is_outdated_pr(pr: PullRequest) -> bool:
    """
    Check if a pull request is outdated, i.e., whether it has not seen an update recently.
    """
    commits = pr.get_commits()
    if commits.totalCount == 0:
        return False

    # We only look at the most recent commit, which should be the last one. If
    # a user modified an earlier one, the magic of Git ensures that any commits
    # that have it as a parent will also be changed.
    commit = commits[commits.totalCount - 1]
    last_modified = dateutil.parser.parse(commit.stats.last_modified)
    age = datetime.now(timezone.utc) - last_modified
    logger.info(f"Pull request {pr} has age {age}")
    return age > PULL_REQUEST_TTL


class BranchWorker(GithubConnector):
    def __init__(
        self,
        parent: "GithubSync",
        labels_cfg: Dict[str, Any],
        repo_branch: str,
        repo_url: str,
        upstream_url: str,
        upstream_branch: str,
        base_directory: str,
        ci_repo_url: str,
        ci_branch: str,
        github_oauth_token: Optional[str] = None,
        app_auth: Optional[AppAuthentication] = None,
        http_retries: Optional[int] = None,
    ) -> None:
        super().__init__(
            repo_url=repo_url,
            github_oauth_token=github_oauth_token,
            app_auth=app_auth,
            http_retries=http_retries,
        )

        self.parent = parent

        self.ci_repo_url = ci_repo_url
        self.ci_repo_dir = _uniq_tmp_folder(ci_repo_url, ci_branch, base_directory)
        self.ci_branch = ci_branch
        # Set properly at a later time.
        self.ci_repo_local = None

        self.repo_dir = _uniq_tmp_folder(repo_url, repo_branch, base_directory)
        self.repo_branch = repo_branch
        self.repo_pr_base_branch = repo_branch + "_base"
        self.repo_local = None

        self.upstream_url = upstream_url
        self.upstream_remote = "upstream"
        self.upstream_branch = upstream_branch
        # Most recently used upstream SHA-1. Used to prevent unnecessary pushes
        # if upstream did not change.
        self.upstream_sha = None

        create_color_labels(labels_cfg, self.repo)
        # member variables
        self.branches = {}
        self.prs: Dict[str, PullRequest] = {}
        self.all_prs = {}
        self._closed_prs = None

    def _update_e2e_pr(
        self, title: str, base_branch: str, branch: str, has_codechange: bool
    ) -> Optional[PullRequest]:
        """Check if there is open PR on e2e branch, reopen if necessary."""
        pr = None

        message = f"branch: {branch}\nbase: {self.repo_branch}\nversion: {self.upstream_sha}\n"

        if title in self.prs:
            pr = self.prs[title]

        if pr:
            if pr.state == "closed":
                pr = None
            elif has_codechange:
                try:
                    pr.create_issue_comment(message)
                except GithubException as e:
                    # pyre-fixme[16]: `GithubException` has no attribute `__data`.
                    emsg = e._GithubException__data["message"]
                    if emsg in KNOWN_OK_COMMENT_EXCEPTIONS:
                        logger.info(f"Exception Ignored: {emsg}")
                    else:
                        raise e
        if not pr:
            # we creating new PR
            logger.info(f"Creating E2E test PR {title} : {branch}")
            pr = self.repo.create_pull(
                title=title, body=message, head=branch, base=base_branch
            )
            self.prs[title] = pr
            self.add_pr(pr)

        return pr

    def update_e2e_test_branch_and_update_pr(
        self, branch: str
    ) -> Optional[PullRequest]:
        base_branch = branch + "_base"
        branch_name = branch + "_test"

        self._update_pr_base_branch(base_branch)

        # Now that we have an updated base branch, create a dummy commit on top
        # so that we can actually create a pull request (this path tests
        # upstream directly, without any mailbox patches applied).
        self.repo_local.git.checkout("-B", branch_name)
        self.repo_local.git.commit("--allow-empty", "--message", "Dummy commit")

        title = f"[test] {branch_name}"

        # Force push only if there is no branch or code changes.
        pushed = False
        if branch_name not in self.branches or self.repo_local.git.diff(
            branch_name, f"remotes/origin/{branch_name}"
        ):
            self.repo_local.git.push("--force", "origin", branch_name)
            pushed = True

        self._update_e2e_pr(title, base_branch, branch_name, pushed)

    def do_sync(self) -> None:
        # fetch most recent upstream
        if self.upstream_remote in [x.name for x in self.repo_local.remotes]:
            urls = list(self.repo_local.remote(self.upstream_remote).urls)
            if urls != [self.upstream_url]:
                logger.warning(f"remote upstream set to track {urls}, re-creating")
                self.repo_local.delete_remote(self.upstream_remote)
                self.repo_local.create_remote(self.upstream_remote, self.upstream_url)
        else:
            self.repo_local.create_remote(self.upstream_remote, self.upstream_url)
        upstream_repo = self.repo_local.remote(self.upstream_remote)
        upstream_repo.fetch(self.upstream_branch)
        upstream_branch = getattr(upstream_repo.refs, self.upstream_branch)
        _reset_repo(self.repo_local, f"{self.upstream_remote}/{self.upstream_branch}")
        self.repo_local.git.push(
            "--force", "origin", f"{upstream_branch}:refs/heads/{self.repo_branch}"
        )
        self.upstream_sha = upstream_branch.object.hexsha

    def full_sync(self, path: str, url: str, branch: str) -> git.Repo:
        logging.info(f"Doing full clone from {redact_url(url)}, branch: {branch}")
        shutil.rmtree(path, ignore_errors=True)
        r = git.Repo.clone_from(url, path)
        _reset_repo(r, f"origin/{branch}")
        self.parent.increment_counter("full_clones")
        return r

    def fetch_repo(self, path: str, url: str, branch: str) -> git.Repo:
        logging.info(f"Checking local sync repo at {path}")

        if os.path.exists(f"{path}/.git"):
            # pyre-fixme[16]: `Repo` has no attribute `init`.
            repo = git.Repo.init(path)
            try:
                # Update origin URL to support GH app token refreshes
                repo.remote(name="origin").set_url(url)
                repo.git.fetch("--prune", "origin")
                _reset_repo(repo, f"origin/{branch}")
                self.parent.increment_counter("partial_clones")
            except git.exc.GitCommandError:
                # fall back to a full sync
                logger.exception("Exception fetching repo, falling back to full_sync")
                repo = self.full_sync(path, url, branch)
        else:
            repo = self.full_sync(path, url, branch)
        return repo

    def fetch_repo_branch(self) -> None:
        """
        Fetch the repository branch of interest only once
        """
        self.repo_local = self.fetch_repo(
            self.repo_dir, self.repo_url(), self.repo_branch
        )
        self.ci_repo_local = self.fetch_repo(
            self.ci_repo_dir,
            self.ci_repo_url,
            self.ci_branch,
        )
        self.ci_repo_local.git.checkout(f"origin/{self.ci_branch}")

    def _update_pr_base_branch(self, base_branch: str):
        """
        Update the pull request base branch by resetting it to upstream state
        and then adding CI files to it.
        """
        # Basically, on a detached head representing upstream state, add CI
        # files. This is the state that the base branch should have. Then see if
        # that is already the state that it has in the remote repository. If
        # not, push it. Lastly, always make sure to check out `base_branch`.
        _reset_repo(self.repo_local, f"{self.upstream_remote}/{self.upstream_branch}")
        self._add_ci_files()

        try:
            diff = self.repo_local.git.diff(f"remotes/origin/{base_branch}")
        except git.exc.GitCommandError:
            # The remote may not exist, in which case we want to push.
            diff = True

        if diff:
            self.repo_local.git.checkout("-B", base_branch)
            self.repo_local.git.push("--force", "origin", f"refs/heads/{base_branch}")
        else:
            self.repo_local.git.checkout("-B", base_branch, f"origin/{base_branch}")

    def _create_dummy_commit(self, branch_name: str) -> None:
        """
        Reset branch, create dummy commit
        """
        _reset_repo(self.repo_local, f"{self.upstream_remote}/{self.upstream_branch}")
        self.repo_local.git.checkout("-B", branch_name)
        self.repo_local.git.commit("--allow-empty", "--message", "Dummy commit")
        self.repo_local.git.push("--force", "origin", branch_name)

    def _unflag_pr(self, pr: PullRequest) -> None:
        pr.remove_from_labels(MERGE_CONFLICT_LABEL)

    def _is_pr_flagged(self, pr: PullRequest) -> bool:
        for label in pr.get_labels():
            if MERGE_CONFLICT_LABEL == label.name:
                return True
        return False

    def _close_pr(self, pr: PullRequest) -> None:
        pr.edit(state="closed")

    def _reopen_pr(self, pr: PullRequest) -> None:
        pr.edit(state="open")
        self.add_pr(pr)
        self.prs[pr.title] = pr

    def _sync_pr_tags(self, pr: PullRequest, tags: Iterable[str]) -> None:
        pr.set_labels(*tags)

    def _subject_count_series(self, subject: Subject) -> int:
        # This method is only for easy mocking
        return len(subject.relevant_series)

    def _guess_pr(
        self, series: Series, branch: Optional[str] = None
    ) -> Optional[PullRequest]:
        """
        Series could change name
        first series in a subject could be changed as well
        so we want to
        - try to guess based on name first
        - try to guess based on first series
        """
        title = f"{series.subject}"
        # try to find amond known relevant PRs
        if title in self.prs:
            return self.prs[title]
        else:
            if not branch:
                # resolve branch
                # series -> subject -> branch
                subject = Subject(series.subject, self.parent.pw)
                branch = self.subject_to_branch(subject)
            if branch in self.all_prs and self.repo_branch in self.all_prs[branch]:
                # we assuming only one PR can be active for one head->base
                return self.all_prs[branch][self.repo_branch][0]
        # we failed to find active PR, now let's try to guess closed PR
        # is:pr is:closed head:"series/358111=>bpf"
        if branch:
            return self.filter_closed_pr(branch)
        return None

    def _comment_series_pr(
        self,
        series: Series,
        branch_name: str,
        message: Optional[str] = None,
        can_create: bool = False,
        close: bool = False,
        flag: bool = False,
    ) -> Optional[PullRequest]:
        """
        Appends comment to a PR.
        """
        title = f"{series.subject}"
        pr_tags = copy.copy(series.visible_tags())
        pr_tags.add(self.repo_branch)

        if flag:
            pr_tags.add(MERGE_CONFLICT_LABEL)

        pr = self._guess_pr(series, branch=branch_name)

        if pr and pr.state == "closed":
            if can_create:
                try:
                    self._reopen_pr(pr)
                except GithubException:
                    logger.warning(
                        f"Error re-opening PR {pr.id}, treating PR as non-exists.",
                        exc_info=True,
                    )
                    pr = None
            elif close:
                # we closing PR and it's already closed
                return pr

        if not pr and can_create and not close:
            # If there is no merge conflict (flag is False) and no change, ignore the series
            if not flag and not self.repo_local.git.diff(
                self.repo_pr_base_branch, branch_name
            ):
                # raise an exception so it bubbles up to the caller.
                raise NewPRWithNoChangeException(self.repo_pr_base_branch, branch_name)
            # we creating new PR
            logger.info(f"Creating PR for '{series.subject}' with {series.age()} delay")
            self.parent.set_counter("pw_to_git_latency", int(series.age()))
            self.parent.increment_counter("prs_created")
            if flag:
                self.parent.increment_counter("initial_merge_conflicts")
                self._create_dummy_commit(branch_name)
            body = (
                f"Pull request for series with\nsubject: {title}\n"
                f"version: {series.version}\n"
                f"url: {series.web_url}\n"
            )
            pr = self.repo.create_pull(
                title=title, body=body, head=branch_name, base=self.repo_pr_base_branch
            )
            self.prs[title] = pr
            self.add_pr(pr)
        elif not pr and not can_create:
            # we closing PR and it's not found
            # how we get onto this state? expired and closed filtered on PW side
            # if we got here we already got series
            # this potentially indicates a bug in PR <-> series mapping
            # or some weird condition
            # this also may happen when we trying to add tags
            self.parent.increment_counter("bug_occurence")
            logger.error(f"BUG: Unable to find PR for {title} {series.web_url}")
            return None

        if (not flag) or (flag and not self._is_pr_flagged(pr)):
            if message:
                self.parent.increment_counter("prs_updated")
                try:
                    pr.create_issue_comment(message)
                except GithubException as e:
                    # pyre-fixme[16]: `GithubException` has no attribute `__data`.
                    emsg = e._GithubException__data["message"]
                    if emsg in KNOWN_OK_COMMENT_EXCEPTIONS:
                        logger.error(f"BUG: {emsg}")
                    else:
                        raise e

        self._sync_pr_tags(pr, pr_tags)

        if close:
            self.parent.increment_counter("prs_closed_total")
            if series.is_expired():
                self.parent.increment_counter("prs_closed_expired_reason")
            logger.warning(f"Closing PR {pr.number}: {pr.head.ref}")
            self._close_pr(pr)
        return pr

    def _pr_closed(self, branch_name: str, series: Series) -> bool:
        if (
            series.is_closed()
            or series.is_expired()
            or not series.has_matching_patches()
        ):
            if series.is_closed():
                comment = f"At least one diff in series {series.web_url} irrelevant now. Closing PR."
            elif not series.has_matching_patches():
                comment = f"At least one diff in series {series.web_url} irrelevant now for {self.parent.pw.search_patterns}"
            else:
                comment = (
                    f"At least one diff in series {series.web_url} expired. Closing PR."
                )
                logger.warning(comment)
            self._comment_series_pr(
                series, message=comment, close=True, branch_name=branch_name
            )
            # delete branch if there is no more PRs left from this branch
            if (
                series.is_closed()
                and branch_name in self.all_prs
                and len(self.all_prs[branch_name]) == 1
                and branch_name in self.branches
            ):
                self.delete_branch(branch_name)
            return True
        return False

    def delete_branch(self, branch_name: str) -> None:
        logger.warning(f"Removing branch {branch_name}")
        self.parent.increment_counter("branches_deleted")
        self.repo.get_git_ref(f"heads/{branch_name}").delete()

    def _add_ci_files(self) -> None:
        """
        Copy over and commit CI files (from the CI repository) to the current
        local repository's currently checked out branch.
        """
        if Path(f"{self.ci_repo_dir}/.github").exists():
            execute_command(f"cp --archive {self.ci_repo_dir}/.github {self.repo_dir}")
            self.repo_local.git.add("--force", ".github")
        execute_command(f"cp --archive {self.ci_repo_dir}/* {self.repo_dir}")
        self.repo_local.git.add("--all", "--force")
        self.repo_local.git.commit("--all", "--message", "adding ci files")

    def try_apply_mailbox_series(
        self, branch_name: str, series: Series
    ) -> Tuple[bool, Optional[Exception], Optional[Any]]:
        """Try to apply a mailbox series and return (True, None, None) if successful"""
        # The pull request will be created against `repo_pr_base_branch`. So
        # prepare it for that.
        self._update_pr_base_branch(self.repo_pr_base_branch)
        self.repo_local.git.checkout("-B", branch_name)

        # Apply series
        patch_filehandle = series.get_patch_blob()
        try:
            self.repo_local.git.am("--3way", istream=patch_filehandle)
        except git.exc.GitCommandError as e:
            conflict = self.repo_local.git.diff()
            return (False, e, conflict)
        return (True, None, None)

    def apply_push_comment(
        self, branch_name: str, series: Series
    ) -> Optional[PullRequest]:
        comment = (
            f"Upstream branch: {self.upstream_sha}\nseries: {series.web_url}\n"
            f"version: {series.version}\n"
        )
        success, e, conflict = self.try_apply_mailbox_series(branch_name, series)
        if not success:
            comment = (
                f"{comment}\nPull request is *NOT* updated. Failed to apply {series.web_url}\n"
                f"error message:\n```\n{e}\n```\n\n"
                f"conflict:\n```\n{conflict}\n```\n"
            )
            logger.warning(f"Failed to apply {series.url}")
            self.parent.increment_counter("merge_conflicts_total")
            return self._comment_series_pr(
                series,
                message=comment,
                branch_name=branch_name,
                flag=True,
                can_create=True,
            )
        # force push only if if's a new branch or there is code diffs between old and new branches
        # which could mean that we applied new set of patches or just rebased
        if branch_name in self.branches and (
            branch_name not in self.all_prs  # NO PR yet
            or self.repo_local.git.diff(
                branch_name, f"remotes/origin/{branch_name}"
            )  # have code changes
        ):
            # we have branch, but either NO PR or there is code changes, we must try to
            # re-open PR first, before doing force-push.
            pr = self._comment_series_pr(
                series,
                message=comment,
                branch_name=branch_name,
                can_create=True,
            )
            self.repo_local.git.push("--force", "origin", branch_name)
            return pr
        # we don't have a branch, also means no PR, push first then create PR
        elif branch_name not in self.branches:
            if not self.repo_local.git.diff(self.repo_pr_base_branch, branch_name):
                # raise an exception so it bubbles up to the caller.
                raise NewPRWithNoChangeException(self.repo_pr_base_branch, branch_name)
            self.repo_local.git.push("--force", "origin", branch_name)
            return self._comment_series_pr(
                series,
                message=comment,
                branch_name=branch_name,
                can_create=True,
            )
        else:
            # no code changes, just update tags
            return self._comment_series_pr(series, branch_name=branch_name)

    def checkout_and_patch(
        self, branch_name: str, series_to_apply: Series
    ) -> Optional[PullRequest]:
        """
        Patch in place and push.
        Returns true if whole series applied.
        Return None if at least one patch in series failed.
        If at least one patch in series failed nothing gets pushed.
        """
        self.parent.increment_counter("all_known_subjects")
        if self._pr_closed(branch_name, series_to_apply):
            return None
        return self.apply_push_comment(branch_name, series_to_apply)

    def add_pr(self, pr: PullRequest) -> None:
        self.all_prs.setdefault(pr.head.ref, {}).setdefault(pr.base.ref, [])
        self.all_prs[pr.head.ref][pr.base.ref].append(pr)

    def get_pulls(self) -> None:
        self.prs = {}
        for pr in self.repo.get_pulls(state="open", base=self.repo_pr_base_branch):
            if self._is_relevant_pr(pr):
                self.prs[pr.title] = pr

            # This check is probably redundant given that we are filtering for open PRs only already.
            if pr.state == "open":
                self.add_pr(pr)

    def _is_relevant_pr(self, pr: PullRequest) -> bool:
        """
        PR is relevant if it
        - coming from user
        - to same user
        - to branch {repo_branch}
        - is open
        """
        src_user = none_throws(pr.head.user).login
        tgt_user = none_throws(pr.base.user).login
        tgt_branch = pr.base.ref
        state = pr.state
        if (
            src_user == self.user_or_org
            and tgt_user == self.user_or_org
            and tgt_branch == self.repo_pr_base_branch
            and state == "open"
        ):
            return True
        return False

    def closed_prs(self) -> List[Any]:
        # GH api is not working: https://github.community/t/is-api-head-filter-even-working/135530
        # so i have to implement local cache
        # and local search
        # closed prs are last resort to re-open expired PRs
        # and also required for branch expiration
        if not self._closed_prs:
            self._closed_prs = list(
                self.repo.get_pulls(state="closed", base=self.repo_pr_base_branch)
            )
        return self._closed_prs

    def filter_closed_pr(self, head: str) -> Optional[PullRequest]:
        # this assumes only the most recent one closed PR per head
        res = None
        for pr in self.closed_prs():
            if pr.head.ref == head and (
                not res or res.updated_at.timestamp() < pr.updated_at.timestamp()
            ):
                res = pr
        return res

    def subject_to_branch(self, subject: Subject) -> str:
        return f"{subject.branch}{HEAD_BASE_SEPARATOR}{self.repo_branch}"

    def sync_checks(self, pr: PullRequest, series: Series) -> None:
        # if it's merge conflict - report failure
        ctx = f"{CI_DESCRIPTION}-{self.repo_branch}"
        if self._is_pr_flagged(pr):
            series.set_check(
                state="failure",
                target_url=pr.html_url,
                context=f"{ctx}-PR",
                description=MERGE_CONFLICT_LABEL,
            )
            return None
        logger.info(f"Fetching check suites for {pr.number}: {pr.head.ref}")
        # we use github actions, need to use check suite apis instead of combined status
        # https://docs.github.com/en/rest/reference/checks#check-suites
        cmt = self.repo.get_commit(pr.head.sha)
        conclusion = None
        # There is only 1 github-action check suite
        for suite in cmt.get_check_suites():
            if suite.app.id == CI_APP:
                conclusion = suite.conclusion
                break
        # We report the latest check-runs that belong to CI_APP suite ID.
        vmtests = []
        for run in cmt.get_check_runs(filter="latest"):
            if run.app.id == CI_APP:
                vmtests.append(run)
        vmtests_log = [
            f"{vmtest.conclusion} ({vmtest.details_url})" for vmtest in vmtests
        ]
        # in order to keep PW contexts somewhat deterministic, we sort the array of vmtests by name
        # and later use the index of the test in the array to generate the context name.
        vmtests = sorted(vmtests, key=lambda x: x.name)

        logger.info(
            f"Check suite status: overall: '{conclusion}', vmtests: '{vmtests_log}"
        )
        series.set_check(
            state=conclusion,
            target_url=pr.html_url,
            context=f"{ctx}-PR",
            description="PR summary",
        )
        if vmtests:
            for idx, vmtest in enumerate(vmtests):
                series.set_check(
                    state=vmtest.conclusion,
                    target_url=vmtest.details_url,
                    context=f"{ctx}-{CI_VMTEST_NAME}-{idx+1}",
                    description=f"Logs for {vmtest.name}",
                )

    def expire_branches(self) -> None:
        for branch in self.branches:
            # all branches
            if branch in self.all_prs:
                # that are not belong to any known open prs
                continue

            if HEAD_BASE_SEPARATOR in branch:
                split = branch.split(HEAD_BASE_SEPARATOR)
                if len(split) > 1 and split[1] == self.repo_branch:
                    # which have our repo_branch as target
                    # that doesn't have any closed PRs
                    # with last update within defined TTL
                    pr = self.filter_closed_pr(branch)
                    if not pr or time.time() - pr.updated_at.timestamp() > BRANCH_TTL:
                        self.delete_branch(branch)

    def expire_user_prs(self) -> None:
        """
        Close user-created (i.e., non KPD) pull requests that have not seen an update
        in a while.
        """
        for pr in self.repo.get_pulls(state="open"):
            # Anything not created by KPD is fair game for being closed.
            # FIXME(T148345498) : remove the reference to "kernel-patches-bot"
            # once the migration to GH app is over.
            if pr.user.login not in [
                self.get_user_login(),
                "kernel-patches-bot",
            ] and _is_outdated_pr(pr):
                logger.info(f"Pull request {pr} is found to be outdated")
                message = (
                    "Automatically cleaning up stale PR; feel free to reopen if needed"
                )
                try:
                    pr.create_issue_comment(message)
                except GithubException as e:
                    # pyre-fixme[16]: `GithubException` has no attribute `__data`.
                    emsg = e._GithubException__data["message"]
                    if emsg in KNOWN_OK_COMMENT_EXCEPTIONS:
                        logger.info(f"Exception Ignored: {emsg}")
                    else:
                        raise e

                self._close_pr(pr)


class GithubSync(Stats):
    def __init__(
        self,
        config: Dict[str, Any],
        labels_cfg: Dict[str, str],
    ) -> None:
        self.pw = Patchwork(
            # FIXME(yurinnick) remove this after migrating pw_url to pw_server
            server=config.get("pw_server", urlparse(config["pw_url"]).netloc),
            search_patterns=config["pw_search_patterns"],
            lookback_in_days=config.get("pw_lookback", 7),
            auth_token=config.get("pw_token"),
            http_retries=HTTP_RETRIES,
        )
        self.workers: Dict[str, BranchWorker] = {}
        self.tag_to_branch_mapping = config["tag_to_branch_mapping"]
        for branch in config["branches"]:
            data = config["branches"][branch]
            cfg = {}
            # Try to load both oauth and app_auth. GithubConnector will deal with
            # raising if the config is invalid.
            # legacy support for oauth token
            if "github_oauth_token" in data:
                cfg["github_oauth_token"] = data["github_oauth_token"]
            # If any of the GH app keys are provided, try to create the AppAuthentication.
            # Fail loudly of there is any missing keys.
            if any(
                [
                    k in data
                    for k in [
                        "github_app_id",
                        "github_installation_id",
                        "github_private_key",
                    ]
                ]
            ):
                github_private_key = None
                with open(data["github_private_key"], "r") as f:
                    github_private_key = f.read()
                cfg["app_auth"] = AppAuthentication(
                    data["github_app_id"],
                    github_private_key,
                    data["github_installation_id"],
                )

            worker = BranchWorker(
                parent=self,
                labels_cfg=labels_cfg,
                repo_branch=branch,
                repo_url=data["repo"],
                upstream_url=data["upstream"],
                upstream_branch=data.get("upstream_branch", "master"),
                ci_repo_url=data.get("ci_repo", None),
                ci_branch=data.get("ci_branch", None),
                base_directory=config.get("base_directory", "/tmp"),
                http_retries=HTTP_RETRIES,
                **cfg,
            )
            self.workers[branch] = worker

        # member variable initializations
        self.subjects: Sequence[Subject] = []
        super().__init__(
            {
                "prs_created",  # Only new PRs, including merge-conflicts
                "prs_updated",  # PRs which exist before and which got a forcepush
                "prs_closed_total",  # All PRs that was closed
                "prs_closed_expired_reason",  # All PRs that was closed as expired
                "branches_deleted",  # Branches that was deleted
                "full_cycle_duration",  # Duration of one sync cycle
                "mirror_duration",  # Duration of mirroring upstream
                "pw_fetch_duration",  # Duration of initial search in PW, exclusing time to map existing PRs to PW entities
                "patch_and_update_duration",  # Duration of git apply and git push
                "pw_to_git_latency",  # Average latency between patch created in PW and appear in GH
                "full_clones",  # Number of times we had to do a full clone instead of git fetch
                "partial_clones",  # Number of git fetches
                "merge_conflicts_total",  # All merge-conflicts
                "initial_merge_conflicts",  # Merge conflicts that happen for a first patch in a series known to us
                "existing_pr_merge_conflicts",  # Merge conflicts on PRs that was fast-forwardable before
                "all_known_subjects",  # All known subjects, including ones that was found in PW, GH. Also includes expired patches
                "prs_total",  # All prs within the scope of work for this worker
                "bug_occurence",  # Weird conditions which require investigation
                "empty_pr",  # Series that would result in an empty PR creation
            }
        )

    def sync_patches(self) -> None:
        """
        One subject = one branch
        creates branches when necessary
        apply patches where it's necessary
        delete branches where it's necessary
        version of series applies in the same branch
        as separate commit
        """

        # sync mirror and fetch current states of PRs
        self.drop_counters()
        self.pw.drop_counters()
        self.subjects = []

        sync_start = time.time()

        for branch, worker in self.workers.items():
            logging.info(f"Refreshing repo info for {branch}.")
            worker.fetch_repo_branch()
            worker.get_pulls()
            worker.do_sync()
            worker._closed_prs = None
            worker.branches = [x.name for x in worker.repo.get_branches()]

        mirror_done = time.time()

        for branch, worker in self.workers.items():
            worker.update_e2e_test_branch_and_update_pr(branch)

        # fetch recent subjects
        self.subjects = self.pw.get_relevant_subjects()
        pw_done = time.time()

        # 1. Get Subject's latest series
        # 2. Get series tags
        # 3. Map tags to a branches
        # 4. Start from first branch, try to apply and generate PR,
        #    if fails continue to next branch, if no more branches, generate a merge-conflict PR
        for subject in self.subjects:
            series = none_throws(subject.latest_series)
            logging.info(
                f"Processing {series.id}: {subject.subject} (tags: {series.all_tags()})"
            )

            mapped_branches = []
            for tag in self.tag_to_branch_mapping:
                if tag in series.all_tags():
                    mapped_branches = self.tag_to_branch_mapping[tag]
                    logging.info(
                        f"Tag '{tag}' mapped to branch order {mapped_branches}"
                    )
                    break
            else:
                mapped_branches = self.tag_to_branch_mapping["__DEFAULT__"]
                logging.info(f"Mapped to default branch order: {mapped_branches}")

            # series to apply - last known series
            last_branch = mapped_branches[-1]
            for branch in mapped_branches:
                worker = self.workers[branch]
                # PR branch name == sid of the first known series
                pr_branch_name = worker.subject_to_branch(subject)
                if not worker.try_apply_mailbox_series(pr_branch_name, series)[0]:
                    msg = f"Failed to apply series to {branch}, "
                    if branch != last_branch:
                        logging.info(msg + "moving to next.")
                        continue
                    else:
                        logging.info(msg + "no more next, staying.")
                logging.info(f"Choosing branch {branch} to create/update PR.")
                try:
                    pr = worker.checkout_and_patch(pr_branch_name, series)
                except NewPRWithNoChangeException:
                    self.increment_counter("empty_pr")
                    logger.exception("Could not create PR with no changes")

                    continue
                assert pr is not None
                logging.info(
                    f"PR created/updated: {pr.number}({pr.head.ref}): {pr.url}"
                )
                worker.sync_checks(pr, series)
                # Close out other PRs if exists
                correct_pr_hread_ref = pr.head.ref
                closed_pr_titles = []
                for _, other_worker in self.workers.items():
                    for _, other_pr in other_worker.prs.items():
                        if (
                            other_pr.head.ref.split(HEAD_BASE_SEPARATOR)[0]
                            == correct_pr_hread_ref.split(HEAD_BASE_SEPARATOR)[0]
                            and other_pr.head.ref != correct_pr_hread_ref
                        ):
                            logging.info(
                                f"Closing existing but incorrect PR: {other_pr.number}: {other_pr.head.ref}"
                            )
                            other_pr.edit(state="close")
                            closed_pr_titles.append(other_pr.title)
                # forget about closed prs
                for title in closed_pr_titles:
                    for _, other_worker in self.workers.items():
                        if title in other_worker.prs:
                            del other_worker.prs[title]
                break

        # sync old subjects
        subject_names = [x.subject for x in self.subjects]
        for _, worker in self.workers.items():
            for subject_name in worker.prs:
                pr = worker.prs[subject_name]
                if subject_name not in subject_names and worker._is_relevant_pr(pr):
                    branch_name_pr = worker.prs[subject_name].head.ref
                    # ignore unknown format branch/PRs.
                    if (
                        "/" not in branch_name_pr
                        or HEAD_BASE_SEPARATOR not in branch_name_pr
                    ):
                        continue
                    series_id = int(
                        branch_name_pr.split("/")[1].split(HEAD_BASE_SEPARATOR)[0]
                    )
                    series = self.pw.get_series_by_id(series_id)
                    subject = self.pw.get_subject_by_series(series)
                    if subject_name != subject.subject:
                        logger.warning(
                            f"Renaming PR {pr.number} from {subject_name} to {subject.subject} according to {series.id}"
                        )
                        pr.edit(title=subject.subject)
                    branch_name = (
                        f"{subject.branch}{HEAD_BASE_SEPARATOR}{worker.repo_branch}"
                    )
                    if not branch_name:
                        branch_name = branch_name_pr
                    latest_series = subject.latest_series
                    if not latest_series:
                        latest_series = series
                    worker.checkout_and_patch(branch_name, latest_series)
                    worker.sync_checks(pr, latest_series)

            worker.expire_branches()
            worker.expire_user_prs()

        patches_done = time.time()
        self.set_counter("full_cycle_duration", patches_done - sync_start)
        total_time.record(patches_done - sync_start)
        self.set_counter("mirror_duration", mirror_done - sync_start)
        self.set_counter("pw_fetch_duration", pw_done - mirror_done)
        fetch_time.record(pw_done - mirror_done)
        self.set_counter("patch_and_update_duration", patches_done - pw_done)
        self.set_counter("bug_occurence", self.pw.stats["bug_occurence"])
        bug_occurence.add(self.pw.stats["bug_occurence"])
        del self.pw.stats["bug_occurence"]
        for _, worker in self.workers.items():
            for p in worker.prs:
                pr = worker.prs[p]
                if worker._is_relevant_pr(pr):
                    self.increment_counter("prs_total")
        if self.stats["prs_created"] > 0:
            self.stats["pw_to_git_latency"] = (
                self.stats["pw_to_git_latency"] / self.stats["prs_created"]
            )
        self.stats.update(self.pw.stats)
