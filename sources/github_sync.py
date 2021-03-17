#!/usr/bin/env python3
from patchwork import Patchwork, Series, Subject
from github import Github, GithubException
import git
import re
import shutil
import os
import logging
import hashlib
import copy
import time

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

BRANCH_TTL = 172800  # w
HEAD_BASE_SEPARATOR = "=>"
KNOWN_OK_COMMENT_EXCEPTIONS = [
    "Commenting is disabled on issues with more than 2500 comments"
]


class GithubSync(object):
    DEFAULT_STAT_VALUES = {
        "prs_created": (0, "Only new PRs, including merge-conflicts"),
        "prs_updated": (0, "PRs which exist before and which got a forcepush"),
        "prs_closed_total": (0, "All PRs that was closed"),
        "prs_closed_expired_reason": (0, "All PRs that was closed as expired"),
        "branches_deleted": (0, "Branches that was deleted"),
        "full_cycle_duration": (0, "Duration of one sync cycle"),
        "mirror_duration": (0, "Duration of mirroring upstream"),
        "pw_fetch_duration": (
            0,
            "Duration of initial search in PW, exclusing time to map existing PRs to PW entities",
        ),
        "patch_and_update_duration": (0, "Duration of git apply and git push"),
        "pw_to_git_latency": (
            0,
            "Average latency between patch created in PW and appear in GH",
        ),
        "full_clones": (
            0,
            "Number of times we had to do a full clone instead of git fetch",
        ),
        "partial_clones": (0, "Number of git fetches"),
        "merge_conflicts_total": (0, "All merge-conflicts"),
        "initial_merge_conflicts": (
            0,
            "Merge conflicts that happen for a first patch in a series known to us",
        ),
        "existing_pr_merge_conflicts": (
            0,
            "Merge conflicts on PRs that was fast-forwardable before",
        ),
        "all_known_subjects": (
            0,
            "All known subjects, including ones that was found in PW, GH. Also includes expired patches.",
        ),
        "prs_total": (0, "All prs within the scope of work for this worker"),
        "bug_occurence": (0, "Weird conditions which require investigation"),
    }

    def __init__(
        self,
        pw_url,
        pw_search_patterns,
        master,
        repo_url,
        github_oauth_token,
        sync_from,
        source_master,
        ci_repo=None,
        ci_branch=None,
        merge_conflict_label="merge-conflict",
        pw_lookback=7,
        filter_tags=None,
        build_fixtures=False,
    ):
        self.downstream_branch = "downstream"
        self.build_fixtures = build_fixtures
        self.ci_repo = ci_repo
        if self.ci_repo:
            self.ci_branch = ci_branch
            self.ci_repo_dir = self._uniq_tmp_folder(ci_repo, ci_branch)
        self.repo_name = os.path.basename(repo_url)
        self.repo_url = repo_url
        self.sync_from = sync_from
        self.source_remote = "sync_from"
        self.master = master
        self.source_master = source_master
        self.git = Github(github_oauth_token)
        self.pw = Patchwork(
            pw_url,
            pw_search_patterns,
            pw_lookback=pw_lookback,
            filter_tags=filter_tags,
            build_fixtures=self.build_fixtures,
        )
        self.user = self.git.get_user()
        self.user_login = self.user.login
        try:
            self.repo = self.user.get_repo(self.repo_name)
        except GithubException:
            # are we working under org repo?
            org = os.path.split(repo_url)[0].split(":")[-1]
            self.user_login = org
            self.repo = self.git.get_organization(org).get_repo(self.repo_name)

        self.merge_conflict_label = merge_conflict_label
        # self.master = self.repo.get_branch(master)
        self.logger = logging.getLogger(__name__)
        self.repodir = self._uniq_tmp_folder(repo_url, master)

    def _uniq_tmp_folder(self, url, branch):
        # use same foder for multiple invocation to avoid cloning whole tree every time
        # but use different folder for different workers identified by url and branch name
        sha = hashlib.sha256()
        sha.update(f"{url}/{branch}".encode("utf-8"))
        return f"/tmp/pw_sync_{sha.hexdigest()}"

    def stat_update(self, key, increment=1):
        try:
            self.stats[key] += increment
        except:
            self.stat_update("bug_occurence")
            self.logger.error(f"BUG: Failed to update stats key: {key}, {increment}")

    def do_sync(self):
        # fetch upsteam and push to downstream
        if self.source_remote in self.local_repo.remotes:
            urls = list(self.local_repo.remote(self.source_remote).urls)
            if urls != [self.sync_from]:
                self.logger.warn(f"remote sync_from set to track {urls}, re-creating")
                self.local_repo.delete_remote(self.source_remote)
                self.local_repo.create_remote(self.source_remote, self.sync_from)
        else:
            self.local_repo.create_remote(self.source_remote, self.sync_from)
        self.source = self.local_repo.remote(self.source_remote)
        self.source.fetch(self.source_master)
        self.source_branch = getattr(self.source.refs, self.source_master)
        self._reset_repo(self.local_repo, f"{self.source_remote}/{self.source_master}")
        self.local_repo.git.push(
            "-f", "origin", f"{self.source_branch}:refs/heads/{self.master}"
        )
        self.master_sha = self.source_branch.object.hexsha

    def fetch_repo(self, path, url, branch):
        def full_sync(path, url, branch):
            shutil.rmtree(path, ignore_errors=True)
            r = git.Repo.clone_from(url, path)
            self._reset_repo(r, f"origin/{branch}")
            self.stat_update("full_clones")
            return r

        if os.path.exists(f"{path}/.git"):
            repo = git.Repo.init(path)
            try:
                repo.git.fetch("-p", "origin")
                self._reset_repo(repo, f"origin/{branch}")
                self.stat_update("partial_clones")
            except git.exc.GitCommandError as e:
                # fall back to a full sync
                repo = full_sync(path, url, branch)
        else:
            repo = full_sync(path, url, branch)
        return repo

    def fetch_master(self):
        """
            Fetch master only once
        """
        self.local_repo = self.fetch_repo(self.repodir, self.repo_url, self.master)
        self.ci_local_repo = self.fetch_repo(
            self.ci_repo_dir, self.ci_repo, self.ci_branch
        )
        self.ci_local_repo.git.checkout(f"origin/{self.ci_branch}")

    def _reset_repo(self, repo, branch):
        # wipe leftovers from previous cycle if any
        try:
            repo.git.am("--abort")
        except git.exc.GitCommandError:
            pass

        repo.git.reset("--hard", branch)
        repo.git.checkout(branch)

    def _create_dummy_commit(self, branch_name):
        """
            Reset branch, create dummy commit
        """
        self._reset_repo(self.local_repo, f"{self.source_remote}/{self.source_master}")
        if branch_name in self.local_repo.branches:
            self.local_repo.git.branch("-D", branch_name)
        self.local_repo.git.checkout("-b", branch_name)
        self.local_repo.git.commit("--allow-empty", "-m", "Dummy commit")
        self.local_repo.git.push("-f", "origin", branch_name)

    def _unflag_pr(self, pr):
        pr.remove_from_labels(self.merge_conflict_label)

    def _is_pr_flagged(self, pr):
        for label in pr.get_labels():
            if self.merge_conflict_label == label.name:
                return True
        return False

    def _close_pr(self, pr):
        pr.edit(state="closed")

    def _reopen_pr(self, pr):
        pr.edit(state="open")
        self.add_pr(pr)
        self.prs[pr.title] = pr

    def _sync_pr_tags(self, pr, tags):
        pr.set_labels(*tags)

    def _subject_count_series(self, subject):
        # This method is only for easy mocking
        return len(subject.relevant_series)

    def _guess_pr(self, series, branch=None):
        """
            Series could change name
            first series in a subject could be changed as well
            so we want to
            - try to guess based on name first
            - try to guess based on first series
        """
        title = f"{series.subject}"
        subject = None
        # try to find amond known relevant PRs
        if title in self.prs:
            return self.prs[title]
        else:
            if not branch:
                # resolve branch
                # series -> subject -> branch
                subject = Subject(series.subject, self.pw)
                branch = self.subject_to_branch(subject)
            if branch in self.all_prs and self.master in self.all_prs[branch]:
                # we assuming only one PR can be active for one head->base
                return self.all_prs[branch][self.master][0]
        # we failed to find active PR, now let's try to guess closed PR
        # is:pr is:closed head:"series/358111=>bpf"
        if branch and branch in self.branches and series._version() > 1:
            # no reason to look for closed expired PRs for V1
            # we assuming that series cannot be re-opened
            if not subject:
                subject = Subject(series.subject, self.pw)
            if self._subject_count_series(subject) > 1:
                # it's also irrelevant to look for closed expired PRs
                # if our series is first known series in a subject
                pr = self.filter_closed_pr(branch)
                if pr:
                    return pr
        return None

    def _comment_series_pr(
        self,
        series,
        message=None,
        branch_name=None,
        can_create=False,
        close=False,
        flag=False,
    ):
        """
            Appends comment to a PR.
        """
        title = f"{series.subject}"
        pr_tags = copy.copy(series.visible_tags)
        pr_tags.add(self.master)

        if flag:
            pr_tags.add(self.merge_conflict_label)

        pr = self._guess_pr(series, branch=branch_name)
        if not pr and can_create and not close:
            # we creating new PR
            self.logger.info(
                f"Creating PR for '{series.subject}' with {series.age} delay"
            )
            self.stat_update("pw_to_git_latency", series.age)
            self.stat_update("prs_created")
            if flag:
                self.stat_update("initial_merge_conflicts")
                self._create_dummy_commit(branch_name)
            body = (
                f"Pull request for series with\nsubject: {title}\n"
                f"version: {series.version}\n"
                f"url: {series.web_url}\n"
            )
            pr = self.repo.create_pull(
                title=title, body=body, head=branch_name, base=self.master
            )
            self.prs[title] = pr
            self.add_pr(pr)
        elif pr and pr.state == "closed" and can_create:
            # we need to re-open pr
            self._reopen_pr(pr)
        elif pr and pr.state == "closed" and close:
            # we closing PR and it's already closed
            return pr
        elif not pr and not can_create:
            # we closing PR and it's not found
            # how we get onto this state? expired and closed filtered on PW side
            # if we got here we already got series
            # this potentially indicates a bug in PR <-> series mapping
            # or some weird condition
            # this also may happen when we trying to add tags
            self.stat_update("bug_occurence")
            self.logger.error(f"BUG: Unable to find PR for {title} {series.web_url}")
            return False

        if (not flag) or (flag and not self._is_pr_flagged(pr)):
            if message:
                self.stat_update("prs_updated")
                try:
                    pr.create_issue_comment(message)
                except GithubException as e:
                    emsg = e._GithubException__data['message']
                    if emsg in KNOWN_OK_COMMENT_EXCEPTIONS:
                        self.logger.error(f"BUG: {emsg}")
                    else:
                        raise e

        self._sync_pr_tags(pr, pr_tags)

        if close:
            self.stat_update("prs_closed_total")
            if series.expired:
                self.stat_update("prs_closed_expired_reason")
            self.logger.warning(f"Closing PR {pr}")
            self._close_pr(pr)
        return pr

    def _pr_closed(self, branch_name, series):
        if series.closed or series.expired or not series.is_relevant_to_search():
            if series.closed:
                comment = f"At least one diff in series {series.web_url} irrelevant now. Closing PR."
            elif not series.is_relevant_to_search():
                comment = f"At least one diff in series {series.web_url} irrelevant now for {self.pw.pw_search_patterns}"
            else:
                comment = (
                    f"At least one diff in series {series.web_url} expired. Closing PR."
                )
                self.logger.warning(comment)
            self._comment_series_pr(
                series, message=comment, close=True, branch_name=branch_name
            )
            # delete branch if there is no more PRs left from this branch
            if (
                series.closed
                and branch_name in self.all_prs
                and len(self.all_prs[branch_name]) == 1
                and branch_name in self.branches
            ):
                self.delete_branch(branch_name)
            return True
        return False

    def delete_branch(self, branch_name):
        self.logger.warning(f"Removing branch {branch_name}")
        self.stat_update("branches_deleted")
        self.repo.get_git_ref(f"heads/{branch_name}").delete()

    def apply_mailbox_series(self, branch_name, series):
        # reset to upstream state
        self._reset_repo(self.local_repo, f"{self.source_remote}/{self.source_master}")
        if branch_name in self.local_repo.branches:
            self.local_repo.git.branch("-D", branch_name)
        self.local_repo.git.checkout("-b", branch_name)

        comment = (
            f"Master branch: {self.master_sha}\nseries: {series.web_url}\n"
            f"version: {series.version}\n"
        )

        # add CI commit
        if self.ci_repo:
            os.system(
                f"cp -a {self.ci_repo_dir}/* {self.ci_repo_dir}/.travis.yml {self.repodir}"
            )
            self.local_repo.git.add("-A")
            self.local_repo.git.add("-f", ".travis.yml")
            self.local_repo.git.commit("-a", "-m", "adding ci files")

        fh = series.patch_blob
        try:
            self.local_repo.git.am("-3", istream=fh)

        except git.exc.GitCommandError as e:
            conflict = self.local_repo.git.diff()
            comment = (
                f"{comment}\nPull request is *NOT* updated. Failed to apply {series.web_url}\n"
                f"error message:\n```\n{e}\n```\n\n"
                f"conflict:\n```\n{conflict}\n```\n"
            )
            self.logger.warn(f"Failed to apply {series.url}")
            self.stat_update("merge_conflicts_total")
            return self._comment_series_pr(
                series,
                message=comment,
                branch_name=branch_name,
                flag=True,
                can_create=True,
            )
        # force push only if if's a new branch or there is code diffs between old and new branches
        # which could mean that we applied new set of patches or just rebased
        if branch_name in self.branches and branch_name not in self.all_prs:
            # we have branch, but we don't have a PR, which mean we must try to
            # re-open PR first, before doing force-push
            pr = self._comment_series_pr(
                series, message=comment, branch_name=branch_name, can_create=True,
            )
            self.local_repo.git.push("-f", "origin", branch_name)
            return pr
        # we don't have branch, we must do force-push first
        # or it's normal update
        elif branch_name not in self.branches or self.local_repo.git.diff(
            branch_name, f"remotes/origin/{branch_name}"
        ):
            # we don't have a baseline to compare the diff
            # which means it wasn't in `git fetch`
            # which most likely mean we need to create new branch and thus PR
            self.local_repo.git.push("-f", "origin", branch_name)
            return self._comment_series_pr(
                series, message=comment, branch_name=branch_name, can_create=True,
            )
        else:
            # no code changes, just update tags
            return self._comment_series_pr(series, branch_name=branch_name)

    def checkout_and_patch(self, branch_name, series_to_apply):
        """
            Patch in place and push.
            Returns true if whole series applied.
            Return False if at least one patch in series failed.
            If at least one patch in series failed nothing gets pushed.
        """
        self.stat_update("all_known_subjects")
        if self._pr_closed(branch_name, series_to_apply):
            return False
        return self.apply_mailbox_series(branch_name, series_to_apply)

    def add_pr(self, pr):
        self.all_prs.setdefault(pr.head.ref, {}).setdefault(pr.base.ref, [])
        self.all_prs[pr.head.ref][pr.base.ref].append(pr)

    def _dump_pr(self, pr):
        if not self.build_fixtures:
            return None
        # this is only for fixture collection for unit tests
        import json

        with open("./gh_fixtures.json", "a+") as f:
            f.seek(0)
            try:
                fixtures = json.load(f)
            except json.decoder.JSONDecodeError:
                fixtures = {}
        fixtures[pr.number] = {
            "title": pr.title,
            "state": pr.state,
            "base": {"ref": pr.base.ref, "user": {"login": pr.base.user.login}},
            "head": {"ref": pr.head.ref, "user": {"login": pr.base.user.login}},
            "updated_at": pr.updated_at.timestamp(),
        }
        with open("./gh_fixtures.json", "w") as f:
            json.dump(fixtures, f)

    def get_pulls(self):
        for pr in self.repo.get_pulls(state="open", base=self.master):
            self._dump_pr(pr)
            if self._is_relevant_pr(pr):
                self.prs[pr.title] = pr

            if pr.state == "open":
                self.add_pr(pr)

    def _is_relevant_pr(self, pr):
        """
            PR is relevant if it
            - coming from user
            - to same user
            - to branch {master}
            - is open
        """
        src_user = pr.head.user.login
        src_branch = pr.head.ref
        tgt_user = pr.base.user.login
        tgt_branch = pr.base.ref
        state = pr.state
        if (
            src_user == self.user_login
            and tgt_user == self.user_login
            and tgt_branch == self.master
            and state == "open"
        ):
            return True
        return False

    def _reset_stats(self):
        self.stats = {}
        for k in GithubSync.DEFAULT_STAT_VALUES:
            self.stats[k] = GithubSync.DEFAULT_STAT_VALUES[k][0]

    def closed_prs(self):
        # GH api is not working: https://github.community/t/is-api-head-filter-even-working/135530
        # so i have to implement local cache
        # and local search
        # closed prs are last resort to re-open expired PRs
        # and also required for branch expiration
        if not self._closed_prs:
            self._closed_prs = [
                x for x in self.repo.get_pulls(state="closed", base=self.master)
            ]
            for pr in self._closed_prs:
                self._dump_pr(pr)
        return self._closed_prs

    def filter_closed_pr(self, head):
        # this assumes only the most recent one closed PR per head
        res = None
        for pr in self.closed_prs():
            if pr.head.ref == head and (
                not res or res.updated_at.timestamp() < pr.updated_at.timestamp()
            ):
                res = pr
        return res

    def subject_to_branch(self, subject):
        return f"{subject.branch}{HEAD_BASE_SEPARATOR}{self.master}"

    def expire_branches(self):
        for branch in self.branches:
            # all bracnhes
            if branch in self.all_prs:
                # that are not belong to any known open prs
                continue

            if HEAD_BASE_SEPARATOR in branch:
                split = branch.split(HEAD_BASE_SEPARATOR)
                if len(split) > 1 and split[1] == self.master:
                    # which have our master as base
                    # that doesn't have any closed PRs
                    # with last update within defined TTL
                    pr = self.filter_closed_pr(branch)
                    if not pr or time.time() - pr.updated_at.timestamp() > BRANCH_TTL:
                        self.delete_branch(branch)

    def sync_branches(self):
        """
            One subject = one branch
            creates branches when necessary
            apply patches where it's necessary
            delete branches where it's necessary
            version of series applies in the same branch
            as separate commit
        """

        # sync mirror and fetch current states of PRs
        self._reset_stats()
        self.pw.drop_counters()
        sync_start = time.time()
        self.subjects = {}
        self.prs = {}
        self.all_prs = {}
        self.fetch_master()
        self.get_pulls()
        self.do_sync()
        self._closed_prs = None
        self.branches = [x.name for x in self.repo.get_branches()]
        mirror_done = time.time()

        self.subjects = self.pw.get_relevant_subjects()
        pw_done = time.time()
        # fetch recent subjects
        for subject in self.subjects:
            series_id = subject.id
            # branch name == sid of the first known series
            branch_name = self.subject_to_branch(subject)
            # series to apply - last known series
            series = subject.latest_series
            self.checkout_and_patch(branch_name, series)
        # sync old subjects
        subject_names = [x.subject for x in self.subjects]
        for subject_name in self.prs:
            pr = self.prs[subject_name]
            if subject_name not in subject_names and self._is_relevant_pr(pr):
                branch_name_pr = self.prs[subject_name].head.ref
                series_id = branch_name_pr.split("/")[1].split(HEAD_BASE_SEPARATOR)[0]
                series = self.pw.get_series_by_id(series_id)
                subject = self.pw.get_subject_by_series(series)
                if subject_name != subject.subject:
                    self.logger.error(f"Renaming PR {pr.number} from {subject_name} to {subject.subject} according to {subject.latest_series.id}")
                    pr.edit(title=subject.subject)
                branch_name = f"{subject.branch}{HEAD_BASE_SEPARATOR}{self.master}"
                self.checkout_and_patch(branch_name, subject.latest_series)
        self.expire_branches()
        patches_done = time.time()
        self.stat_update("full_cycle_duration", patches_done - sync_start)
        self.stat_update("mirror_duration", mirror_done - sync_start)
        self.stat_update("pw_fetch_duration", pw_done - mirror_done)
        self.stat_update("patch_and_update_duration", patches_done - pw_done)
        self.stat_update("bug_occurence", self.pw.stats["bug_occurence"])
        del self.pw.stats["bug_occurence"]
        for p in self.prs:
            pr = self.prs[p]
            if self._is_relevant_pr(pr):
                self.stat_update("prs_total")
        if self.stats["prs_created"] > 0:
            self.stats["pw_to_git_latency"] = (
                self.stats["pw_to_git_latency"] / self.stats["prs_created"]
            )

        self.stats.update(self.pw.stats)
