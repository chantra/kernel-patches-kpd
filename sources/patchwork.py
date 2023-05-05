#!/usr/bin/env python3

import copy
import datetime
import json
import logging
import re
import tempfile
from typing import Any, AnyStr, Callable, Dict, IO, List, Optional, Sequence, Set
from urllib.parse import urljoin

import dateutil.parser as dateparser
import requests
from cachetools import cached, TTLCache
from opentelemetry import metrics
from pyre_extensions import none_throws
from requests.adapters import HTTPAdapter

from .stats import Stats  # @manual:=stats

# when we want to push this patch through CI
RELEVANT_STATES: Dict[str, int] = {
    "new": 1,
    "under-review": 2,
    "rfc": 5,
    "changes-requested": 7,
    "queued": 13,
    "needs_ack": 15,
}
RFC_TAG: str = "RFC"
# with these tags will be closed if no updates within TTL
TTL = {"changes-requested": 3600, "rfc": 3600}

# when we are not interested in this patch anymore
IRRELEVANT_STATES: Dict[str, int] = {
    "rejected": 4,
    "accepted": 3,
    "not-applicable": 6,
    "superseded": 9,
    "awaiting-upstream": 8,
    "deferred": 10,
    "mainlined": 11,
    "handled-elsewhere": 17,
}

PW_CHECK_PENDING_STATES: Dict[Optional[str], str] = {
    None: "pending",
    "cancelled": "pending",
}

PW_CHECK_CONCLUSIVE_STATES: Dict[str, str] = {
    "success": "success",
    "skipped": "success",
    "warning": "warning",
    "failure": "fail",
}


PW_CHECK_STATES: Dict[Optional[str], str] = {
    **PW_CHECK_PENDING_STATES,
    **PW_CHECK_CONCLUSIVE_STATES,
}

SUBJECT_REGEXP = re.compile(r"(?P<header>\[[^\]]*\])? *(?P<name>.+)")
IGNORE_TAGS_REGEX = re.compile(r"([0-9]+/[0-9]+|V[0-9]+)|patch", re.IGNORECASE)
TAG_REGEXP = re.compile(r"^(\[(?P<tags>[^]]*)\])*")
PATCH_FILTERING_PROPERTIES = {"project", "delegate"}

logger: logging.Logger = logging.getLogger(__name__)
meter: metrics.Meter = metrics.get_meter("patchwork")

api_get_requests = meter.create_counter(name="get_requests")
api_post_requests = meter.create_counter(name="post_requests")
tag_parsing_failures = meter.create_counter(name="tag_parsing_failures")


def log_response(func: Callable[..., requests.Response]):
    def wrapper(*args, **kwargs) -> requests.Response:
        resp = func(*args, **kwargs)
        log = logger.debug if resp.status_code in range(200, 210) else logger.error
        log("Response code: %s", resp.status_code)
        try:
            log("Response data: %s", resp.json())
        except json.decoder.JSONDecodeError:
            logger.exception("Failed to decode JSON response")
        return resp

    return wrapper


def time_since_secs(date: str) -> float:
    parsed_datetime = dateparser.parse(date)
    duration = datetime.datetime.utcnow() - parsed_datetime
    return duration.total_seconds()


def parse_tags(input: str) -> Set[str]:
    # "[tag1 ,tag2]title" -> "tag1,tags" -> ["tag1", "tag2"]
    try:
        parsed_tags = none_throws(re.match(TAG_REGEXP, input)).group("tags").split(",")
    except Exception:
        logger.warning(f"Can't parse tags from string '{input}'")
        tag_parsing_failures.add(1)
        return set()

    tags = [tag.strip() for tag in parsed_tags]
    return {tag for tag in tags if not re.match(IGNORE_TAGS_REGEX, tag)}


class Subject:
    def __init__(self, subject: str, pw_client: "Patchwork") -> None:
        self.pw_client = pw_client
        self.subject = subject

    @property
    def branch(self) -> Optional[str]:
        if len(self.relevant_series) == 0:
            return None
        return f"series/{self.relevant_series[0].id}"

    @property
    def latest_series(self) -> Optional["Series"]:
        if len(self.relevant_series) == 0:
            return None
        return self.relevant_series[-1]

    @property
    @cached(cache=TTLCache(maxsize=1, ttl=600))
    def relevant_series(self) -> List["Series"]:
        """
        cache and return sorted list of relevant series
        where first element is first known version of same subject
        and last is the most recent
        """
        series_list = self.pw_client.get_series(params={"q": self.subject})

        # we using full text search which could give ambiguous results
        # so we must filter out irrelevant results
        relevant_series = [
            series
            for series in series_list
            if series.subject == self.subject and series.has_matching_patches()
        ]
        # sort series by age desc,  so last series is the most recent one
        return sorted(relevant_series, key=lambda x: x.age(), reverse=True)


class Series:
    def __init__(self, pw_client: "Patchwork", data: Dict) -> None:
        self.pw_client = pw_client
        self.data = data
        self._patch_blob = None

        # We should be able to create object from a short version of series object from /patches/ endpoint
        # Docs: https://patchwork.readthedocs.io/en/latest/api/rest/schemas/v1.2/#get--api-1.2-patches-
        # {
        #     "id": 1,
        #     "url": "https://example.com",
        #     "web_url": "https://example.com",
        #     "name": "string",
        #     "date": "string",
        #     "version": 1,
        #     "mbox": "https://example.com"
        # }
        self.id = data["id"]
        self.name = data["name"]
        self.date = data["date"]
        self.url = data["url"]
        self.web_url = data["web_url"]
        self.version = data["version"]
        self.mbox = data["mbox"]
        self.patches = data.get("patches", [])
        self.cover_letter = data.get("cover_letter")

        try:
            self.subject = none_throws(re.match(SUBJECT_REGEXP, data["name"])).group(
                "name"
            )
        except Exception:
            raise ValueError(
                f"Failed to parse subject from series name '{data['name']}'"
            )

    def _is_patch_matching(self, patch: Dict[str, Any]) -> bool:
        for pattern in self.pw_client.search_patterns:
            for prop_name, expected_value in pattern.items():
                if prop_name in PATCH_FILTERING_PROPERTIES:
                    try:
                        # these values can be None so we need to filter them out first
                        if not patch[prop_name]:
                            return False
                        if patch[prop_name]["id"] != expected_value:
                            return False
                    except KeyError:
                        return False
                elif patch[prop_name] != expected_value:
                    return False
        return True

    def age(self) -> float:
        return time_since_secs(self.date)

    @cached(cache=TTLCache(maxsize=1, ttl=600))
    def get_patches(self) -> List[Dict]:
        """
        Returns patches preserving original order
        for the most recent relevant series
        """
        return [self.pw_client.get_patch_by_id(patch["id"]) for patch in self.patches]

    def is_closed(self) -> bool:
        """
        Series considered closed if at least one patch in this series
        is in irrelevant states
        """
        for patch in self.get_patches():
            if patch["state"] in IRRELEVANT_STATES:
                return True
        return False

    @cached(cache=TTLCache(maxsize=1, ttl=120))
    def all_tags(self) -> Set[str]:
        """
        Tags fetched from series name, diffs and cover letter
        for most relevant series
        """
        tags = {f"V{self.version}"}

        for patch in self.get_patches():
            tags |= parse_tags(patch["name"])
            tags.add(patch["state"])

        if self.cover_letter:
            tags |= parse_tags(self.cover_letter["name"])

        tags |= parse_tags(self.name)

        return tags

    def visible_tags(self) -> Set[str]:
        return {f"V{self.version}", *[diff["state"] for diff in self.get_patches()]}

    def is_expired(self) -> bool:
        for diff in self.get_patches():
            if diff["state"] in TTL:
                if time_since_secs(diff["date"]) >= TTL[diff["state"]]:
                    return True
        return False

    def get_patch_blob(self) -> IO:
        """Returns file-like object"""
        if not self._patch_blob:
            data = self.pw_client.get_blob(self.mbox)
            self._patch_blob = tempfile.NamedTemporaryFile(mode="r+b")
            self._patch_blob.write(data)

        self._patch_blob.seek(0)
        return self._patch_blob

    def has_matching_patches(self) -> bool:
        for patch in self.get_patches():
            if self._is_patch_matching(patch):
                return True

        return False

    def set_check(self, **kwargs) -> None:
        for diff in self.get_patches():
            self.pw_client.post_check(patch_id=diff["id"], orig_data=kwargs)


class Patchwork(Stats):
    def __init__(
        self,
        server: str,
        search_patterns: List[Dict[str, Any]],
        auth_token: Optional[str] = None,
        lookback_in_days: int = 7,
        api_version: str = "1.2",
        http_retries: Optional[int] = None,
    ) -> None:
        self.api_url = f"https://{server}/api/{api_version}/"
        self.auth_token = auth_token
        if not auth_token:
            logger.warning("Patchwork client runs in read-only mode")
        self.search_patterns = search_patterns
        self.since = self.format_since(lookback_in_days)
        # member variable initializations
        self.known_series = {}
        self.known_subjects = {}
        super().__init__(
            [
                "non_api_count",
                "non_api_time",
                "series_search_count",
                "series_search_time",
                "patches_search_count",
                "patches_search_time",
                "series_by_id_count",
                "series_by_id_time",
                "patches_by_id_count",
                "patches_by_id_time",
            ]
        )
        self.http_session = requests.Session()
        adapter = HTTPAdapter(max_retries=http_retries)

        self.http_session.mount("http://", adapter)
        self.http_session.mount("https://", adapter)

    def format_since(self, pw_lookback: int) -> str:
        today = datetime.datetime.utcnow().date()
        lookback = today - datetime.timedelta(days=pw_lookback)
        return lookback.strftime("%Y-%m-%dT%H:%M:%S")

    @log_response
    def __get(self, path: AnyStr, params: Optional[Dict] = None) -> requests.Response:
        logger.debug(f"Patchwork GET {path}, params: {params}")
        # pyre-ignore
        resp = self.http_session.get(url=urljoin(self.api_url, path), params=params)
        api_get_requests.add(1)
        return resp

    @Stats.metered("by_id")
    def __get_object_by_id(self, object_type: str, object_id: int) -> Dict:
        return self.__get(f"{object_type}/{object_id}/").json()

    @Stats.metered("search")
    def __get_objects_recursive(
        self, object_type: str, params: Optional[Dict] = None
    ) -> List[Dict]:
        items = []
        path = f"{object_type}/"
        while True:
            response = self.__get(path, params=params)
            items += response.json()

            if "next" not in response.links:
                break

            path = response.links["next"]["url"]
        return items

    @log_response
    def __post(self, path: AnyStr, data: Dict) -> requests.Response:
        logger.debug(f"Patchwork POST {path}, data: {data}")
        resp = self.http_session.post(
            # pyre-ignore
            url=urljoin(self.api_url, path),
            headers={"Authorization": f"Token {self.auth_token}"},
            data=data,
        )
        api_post_requests.add(1)
        return resp

    def __try_post(self, path: AnyStr, data: Dict) -> Optional[requests.Response]:
        if not self.auth_token:
            logger.warning(
                f"Ignoring POST {path} request: Patchwork client in read-only mode"
            )
            return None

        return self.__post(path, data)

    @Stats.metered("api", obj_type="non")
    def get_blob(self, url: AnyStr) -> bytes:
        return self.http_session.get(url, allow_redirects=True).content

    def last_check_states_for_context(
        self, context: str, patch_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Return dict of the latest check with matching context and patch_id if exists.
        Returns None if such check does not exist.
        """

        # get single latest one with matching context value
        # patchwork.kernel.org api ignores case for context query
        checks = self.__get(
            f"patches/{patch_id}/checks/",
            params={
                "context": context,
                "order": "-date",
                "per_page": "1",
            },
        ).json()

        if len(checks) > 0:
            return checks[0]

        return None

    def post_check(
        self, patch_id: int, orig_data: Dict[str, Any]
    ) -> Optional[requests.Response]:
        c_data = copy.copy(orig_data)
        context = c_data["context"]
        check = self.last_check_states_for_context(context, patch_id)

        if check is None:
            logger.debug(f"Got {c_data} for {patch_id}")
        else:
            logger.debug(f"Got {c_data} for {patch_id} with {check}")
        if (
            check is not None  # we posting not the first time
            and (c_data["state"] in PW_CHECK_PENDING_STATES)  # and new state is pending
            and (
                PW_CHECK_PENDING_STATES[c_data["state"]] != check["state"]
            )  # and old state not pending
        ):
            logger.info(
                f"Not posting state {orig_data} for patch {patch_id}"
                f" as it's pending and previous state"
                f"({check['state']}) is available."
            )
            return None

        # replace check state to one that valid for PW from one that in GH
        pw_state = PW_CHECK_STATES.get(c_data["state"], PW_CHECK_STATES.get(None))
        c_data["state"] = pw_state
        state = pw_state
        if (
            check is not None
            and check.get("state", None) == state
            and check.get("target_url", None) == c_data["target_url"]
        ):
            logger.info(
                f"Not posting state {c_data} for patch {patch_id} "
                f"as it's same as previous posted state({state}) and url is the same"
            )
            return None
        return self.__try_post(f"patches/{patch_id}/checks/", data=c_data)

    def get_series_by_id(self, series_id: int) -> Series:
        # fetches directly only if series is not available in local scope
        if series_id not in self.known_series:
            self.known_series[series_id] = Series(
                self, self.__get_object_by_id("series", series_id)
            )

        return self.known_series[series_id]

    def get_subject_by_series(self, series: Series) -> Subject:
        # local cache for subjects
        if series.subject not in self.known_subjects:
            subject = Subject(series.subject, self)
            self.known_subjects[series.subject] = subject

        return self.known_subjects[series.subject]

    def get_relevant_subjects(self) -> Sequence[Subject]:
        subjects = {}
        filtered_subjects = []
        self.known_series = {}
        self.known_subjects = {}

        for pattern in self.search_patterns:
            p = {
                "since": self.since,
                "state": RELEVANT_STATES.values(),
                "archived": False,
            }
            p.update(pattern)
            logger.info(f"Searching PW with: {p}")
            all_patches = self.__get_objects_recursive("patches", params=p)
            for patch in all_patches:
                patch_series = patch["series"]
                for series in patch_series:
                    if series["name"]:
                        s = Series(self, series)
                        self.known_series[str(s.id)] = s
                    else:
                        self.increment_counter("bug_occurence")
                        logger.error(f"Malformed series: {series}")
                        continue

                    if s.subject not in subjects:
                        subjects[s.subject] = Subject(s.subject, self)
                        self.known_subjects[s.subject] = subjects[s.subject]

            logger.info(f"Total subjects: {len(subjects)}")
            for subject_name, subject_obj in subjects.items():
                latest_series = subject_obj.latest_series
                if not latest_series:
                    logger.error(f"Subject '{subject_name}' doesn't have any series")
                    continue

                if latest_series.is_expired():
                    logger.info(
                        f"Expired: {latest_series.url} {subject_name}",
                    )
                    continue
                if latest_series.is_closed():
                    logger.info(
                        f"Closed: {latest_series.url} {subject_name}",
                    )
                    continue

                logger.info(f"Relevant: {latest_series.url} {subject_name}")
                filtered_subjects.append(subject_obj)
        return filtered_subjects

    def get_patch_by_id(self, id: int) -> Dict:
        return self.__get_object_by_id("patches", id)

    def get_series(self, params: Optional[Dict]) -> List[Series]:
        return [
            Series(self, json)
            for json in self.__get_objects_recursive("series", params=params)
        ]
