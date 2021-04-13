#!/usr/bin/env python3
import json
import requests
import datetime as DT
import re
import logging
import datetime
import time
import dateutil.parser as dp
import pytz
import functools
import tempfile

from tzlocal import get_localzone

# when we want to push this patch through CI
RELEVANT_STATES = {
    "new": 1,
    "rfc": 5,
    "changes-requested": 7,
}
RFC_TAG = "RFC"
RELEVANT_STATE_IDS = [RELEVANT_STATES[x] for x in RELEVANT_STATES]
# with these tags will be closed if no updates within TTL
TTL = {"changes-requested": 3600, "rfc": 3600}

# when we don't interested in this patch anymore
IRRELEVANT_STATES = {
    "accepted": 3,
    "rejected": 4,
    "not-applicable": 6,
    "superseded": 9,
    "under-review": 2,
    "awaiting-upstream": 8,
    "deferred": 10,
    "needs-review-ack": 11,
}

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Subject(object):
    def __init__(self, subject, pw_client):
        self.pw_client = pw_client
        self.subject = subject
        self._relevant_series = None

    @property
    def branch(self):
        if len(self.relevant_series) == 0:
            return None
        return f"series/{self.relevant_series[0].id}"

    def __getattr__(self, fn):
        return getattr(self.relevant_series[-1], fn)

    @property
    def latest_series(self):
        if len(self.relevant_series) == 0:
            return None
        return self.relevant_series[-1]

    @property
    def relevant_series(self):
        """
        cache and return sorted list of relevant series
        where first element is first known version of same subject
        and last is the most recent
        """
        if self._relevant_series:
            return self._relevant_series
        all_series = self.pw_client.get_all("series", filters={"q": self.subject})
        relevant_series = []
        for s in all_series:
            item = Series(s, self.pw_client)
            # we using full text search which could give ambigous results
            # so we must filter out irrelevant results
            if item.subject == self.subject:
                if item.is_relevant_to_search():
                    relevant_series.append(item)
        self._relevant_series = sorted(relevant_series, key=lambda k: k.version)
        rfcs = sorted(
            (s for s in relevant_series if RFC_TAG in s.tags), key=lambda k: k.version
        )
        non_rfcs = sorted(
            (s for s in relevant_series if RFC_TAG not in s.tags),
            key=lambda k: k.version,
        )
        self._relevant_series = rfcs + non_rfcs
        return self._relevant_series


class Series(object):
    def __init__(self, data, pw_client):
        self.pw_client = pw_client
        self.data = data
        self._relevant_series = None
        self._diffs = None
        self._tags = None
        self._patch_blob = None
        self._subject_regexp = re.compile(r"(?P<header>\[[^\]]*\])? *(?P<name>.+)")
        for key in data:
            setattr(self, key, data[key])
        self.subject = re.match(self._subject_regexp, data["name"]).group("name")
        self.ignore_tags = re.compile(r"([0-9]+/[0-9]+|V[0-9]+)|patch", re.IGNORECASE)
        self.tag_regexp = re.compile(r"^(\[(?P<tags>[^]]*)\])*")

    @property
    def diffs(self):
        # fetching patches
        """
        Returns patches preserving original order
        for the most recent relevant series
        """
        if self._diffs:
            return self._diffs
        self._diffs = []
        for patch in self.patches:
            p = self.pw_client.get("patches", patch["id"])
            self._diffs.append(p)
        return self._diffs

    def _closed(self):
        """
        Series considered closed if at least one patch in this series
        is in irrelevant states
        """
        for diff in self.diffs:
            if diff["state"] in IRRELEVANT_STATES:
                return True
        return False

    @property
    def closed(self):
        return self._closed()

    def _parse_for_tags(self, name):
        match = re.match(self.tag_regexp, name)
        if not match:
            return set()
        r = set()
        if match.groupdict()["tags"]:
            tags = match.groupdict()["tags"].split(",")
            for tag in tags:
                if not re.match(self.ignore_tags, tag):
                    r.add(tag)
        return r

    @property
    def tags(self):
        """
        Tags fetched from series name, diffs and cover letter
        for most relevant series
        """
        if self._tags:
            return self._tags
        self._tags = set()
        for diff in self.diffs:
            self._tags |= self._parse_for_tags(diff["name"])
            self._tags.add(diff["state"])
        if self.cover_letter:
            self._tags |= self._parse_for_tags(self.cover_letter["name"])
        self._tags |= self._parse_for_tags(self.name)
        self._tags.add(f"V{self.version}")

        return self._tags

    def _version(self):
        return self.version

    @property
    def visible_tags(self):
        self._visible_tags = set()
        self._visible_tags.add(f"V{self.version}")
        for diff in self.diffs:
            self._visible_tags.add(diff["state"])

        return self._visible_tags

    @property
    def expirable(self):
        for diff in self.diffs:
            if diff["state"] in TTL:
                return True
        return False

    def _expired(self):
        now = datetime.datetime.now()
        for diff in self.diffs:
            if diff["state"] in TTL:
                if self._get_age(diff["date"]) >= TTL[diff["state"]]:
                    return True
        return False

    @property
    def expired(self):
        return self._expired()

    def _get_age(self, date):
        now = datetime.datetime.now().astimezone(get_localzone())
        d = dp.parse(date + "Z").astimezone(get_localzone())
        return (now - d).total_seconds()

    @property
    def age(self):
        return self._get_age(self.date)

    @property
    def patch_blob(self):
        """ Returns file-like object """
        if self._patch_blob:
            self._patch_blob.seek(0)
            return self._patch_blob
        data = self.pw_client.get_blob(self.mbox)
        self._patch_blob = tempfile.NamedTemporaryFile(mode="r+b")
        self._patch_blob.write(data)
        self._patch_blob.seek(0)

        return self._patch_blob

    def _match_diff(self, pattern, diff):
        # no special handling for no key exceptions
        # right now i'm considering that if we wasn't able to find key we want to die
        for key, value in pattern.items():
            if key in ["project", "delegate"]:
                if key not in diff or not diff[key] or "id" not in diff[key]:
                    return False
                if diff[key]["id"] != value:
                    return False
            elif diff[key] != value:
                return False
        return True

    def is_relevant_to_search(self):
        for pattern in self.pw_client.pw_search_patterns:
            for diff in self.diffs:
                if self._match_diff(pattern, diff):
                    return True
        return False


class Patchwork(object):
    def __init__(
        self,
        url,
        pw_search_patterns,
        pw_lookback=7,
        filter_tags=None,
        build_fixtures=False,
    ):
        self.build_fixtures = build_fixtures
        self.server = url
        self.logger = logging.getLogger(__name__)

        self.since = self.format_since(pw_lookback)
        self.pw_search_patterns = pw_search_patterns
        self.filter_tags = set(filter_tags)

    def format_since(self, pw_lookback):
        today = DT.date.today()
        lookback = today - DT.timedelta(days=pw_lookback)
        return lookback.strftime("%Y-%m-%dT%H:%M:%S")

    def _request(self, url):
        self.logger.debug(f"Patchwork {self.server} request: {url}")
        ret = requests.get(url)
        self.logger.debug("Response", ret)
        try:
            self.logger.debug("Response data", ret.json())
        except json.decoder.JSONDecodeError:
            self.logger.debug("Response data", ret.text)

        return ret

    def drop_counters(self):
        self.stats = {"bug_occurence": 0}
        for obj in ["series", "patches", "projects"]:
            for query_type in ["by_id", "search"]:
                self.stats[f"{obj}_{query_type}_count"] = 0
                self.stats[f"{obj}_{query_type}_time"] = 0
        self.stats[f"non_api_count"] = 0
        self.stats[f"non_api_time"] = 0

    def stat_update(self, key, increment=1):
        try:
            self.stats[key] += increment
        except:
            self.stat_update("bug_occurence")
            self.logger.error(f"Failed to update stats key: {key}, {increment}")

    def metered(query_type, obj_type=None):
        def metered_decorator(func):
            @functools.wraps(func)
            def metered_wrapper(*args, **kwargs):
                self = args[0]
                if not obj_type:
                    obj = args[1]
                else:
                    obj = obj_type
                start = time.time()
                result = func(*args, **kwargs)
                t = time.time() - start
                self.stat_update(f"{obj}_{query_type}_time", t)
                self.stat_update(f"{obj}_{query_type}_count")
                return result

            return metered_wrapper

        return metered_decorator

    @metered("by_id")
    def get(self, object_type, identifier):
        return self._get(f"{object_type}/{identifier}/").json()

    @metered("api", obj_type="non")
    def get_blob(self, url):
        r = requests.get(url, allow_redirects=True)
        return r.content

    # this method only used for fixtures collection for unit tests
    def _get_w_fixtures(self, req):
        with open("./pw_fixtures.json", "a+") as f:
            f.seek(0)
            try:
                fixtures = json.load(f)
            except json.decoder.JSONDecodeError:
                fixtures = {}
            r = self._request(f"{self.server}/api/1.1/{req}")
            if req in fixtures:
                self.logger.warning(f"Repetitive call {req}")
            fixtures[req] = r.json()
        with open("./pw_fixtures.json", "w") as f:
            json.dump(fixtures, f)
            return r

    def _get(self, req):
        if self.build_fixtures:
            return self._get_w_fixtures(req)
        else:
            return self._request(f"{self.server}/api/1.1/{req}")

    @metered("search")
    def get_all(self, object_type, filters=None):
        if filters is None:
            filters = {}
        params = ""
        for key, val in filters.items():
            if val is not None:
                if isinstance(val, list):
                    for v in val:
                        params += f"{key}={v}&"
                else:
                    params += f"{key}={val}&"

        items = []

        response = self._get(f"{object_type}/?{params}")
        # Handle paging, by chasing the "Link" elements
        while response:
            for o in response.json():
                items.append(o)

            if "Link" not in response.headers:
                break

            # There are multiple links separated by commas
            links = response.headers["Link"].split(",")
            # And each link has the format of <url>; rel="type"
            response = None
            for link in links:
                info = link.split(";")
                if info[1].strip() == 'rel="next"':
                    response = self._request(info[0][1:-1])

        return items

    def get_project(self, name):
        all_projects = self.get_all("projects")
        for project in all_projects:
            if project["name"] == name:
                self.logger.debug(f"Found {project}")
                return project

    def get_series_by_id(self, sid):
        # fetches directly only if series is not available in local scope
        if sid not in self.known_series:
            series = Series(self.get("series", sid), self)
            self.known_series[sid] = series
        else:
            series = self.known_series[sid]
        return series

    def get_subject_by_series(self, series):
        # local cache for subjects
        if series.subject not in self.known_subjects:
            subject = Subject(series.subject, self)
            self.known_subjects[series.subject] = subject
        else:
            subject = self.known_subjects[series.subject]
        return subject

    def get_relevant_subjects(self, full=True):
        subjects = {}
        filtered_subjects = []
        self.known_series = {}
        self.known_subjects = {}

        for pattern in self.pw_search_patterns:
            p = {"since": self.since, "state": RELEVANT_STATE_IDS, "archived": False}
            p.update(pattern)
            self.logger.warning(p)
            all_patches = self.get_all("patches", filters=p)
            for patch in all_patches:
                patch_series = patch["series"]
                for series in patch_series:
                    if series["name"]:
                        s = Series(series, self)
                        self.known_series[str(s.id)] = s
                    else:
                        self.stat_update("bug_occurence")
                        self.logger.error(f"Malformed series: {series}")
                        continue
                    if s.subject not in subjects:
                        subjects[s.subject] = Subject(s.subject, self)
                        self.known_subjects[s.subject] = subjects[s.subject]
            for subject in subjects:
                excluded_tags = subjects[subject].tags & self.filter_tags
                if (
                    not excluded_tags
                    and not subjects[subject].expired
                    and not subjects[subject].closed
                ):
                    self.logger.warning(f"Found matching relevant subject {subject}")
                    filtered_subjects.append(subjects[subject])
                elif subjects[subject].expired:
                    self.logger.warning(
                        f"Filtered {subjects[subject].url} ( {subject} ) as expired",
                    )
                elif subjects[subject].closed:
                    self.logger.warning(
                        f"Filtered {subjects[subject].url} ( {subject} ) as closed",
                    )
                else:
                    self.logger.warning(
                        f"Filtered {subjects[subject].url} ( {subject} )  due to tags: %s",
                        excluded_tags,
                    )
        return filtered_subjects
