#!/usr/bin/env python3

import argparse
import json
import logging
import os
import re
import time
from subprocess import PIPE, Popen
from typing import Any, Callable, Dict, Final, Optional

from github import Github, GithubException
from github.Repository import Repository
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource

from sources.github_sync import GithubSync  # @manual

logger: logging.Logger = logging.getLogger(__name__)


LOOP_DELAY_SECS: Final[int] = 60


def get_repo(git: Github, project: str) -> Repository:
    repo_name = os.path.basename(project)
    try:
        user = git.get_user()
        repo = user.get_repo(repo_name)
    except GithubException:
        org = ""
        if "https://" not in project and "ssh://" not in project:
            org = project.split(":")[-1].split("/")[0]
        else:
            org = project.split("/")[-2]
        repo = git.get_organization(org).get_repo(repo_name)
    return repo


class PWDaemon:
    def __init__(
        self,
        cfg: Dict[str, Any],
        labels_cfg: Dict[str, str],
        metrics_logger: Optional[Callable] = None,
    ) -> None:
        config_version = cfg.get("version", 0)
        if config_version != 2:
            raise ValueError(
                f"KPD only supports version 2, got version {config_version} instead"
            )

        self.metrics_logger = metrics_logger
        self.project = cfg["project"]
        self.worker = GithubSync(config=cfg, labels_cfg=labels_cfg)

    def process_stats(self) -> None:
        if self.metrics_logger:
            self.metrics_logger(self.project, self.worker.stats)

    def loop(self) -> None:
        while True:
            logger.info("Sync loop starting...")
            self.worker.sync_patches()
            self.process_stats()
            time.sleep(LOOP_DELAY_SECS)


def purge(cfg) -> None:
    with open(cfg) as f:
        config = json.load(f)
    for project, project_cfg in config.items():
        git = Github(project_cfg["github_oauth_token"])
        repo = get_repo(git, project)
        refs_to_remove = [
            f"heads/{branch_name}"
            for branch_name in repo.get_branches()
            if re.match(r"series/[0-9]+.*", branch_name.name)
        ]
        logging.info(f"Removing references: {refs_to_remove}")
        for ref in refs_to_remove:
            repo.get_git_ref(ref).delete()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Starts kernel-patches daemon")
    parser.add_argument(
        "--config",
        default="~/.kernel-patches/config.json",
        help="Specify config location",
    )
    parser.add_argument(
        "--label-colors",
        default="~/.kernel-patches/labels.json",
        help="Specify label coloring config location.",
    )
    parser.add_argument(
        "--metric-logger",
        default="~/.kernel-patches/metric_logger.sh",
        help="Specify external scripts which stdin will be fed with metrics",
    )
    parser.add_argument(
        "--action",
        default="start",
        choices=["start", "purge"],
        help="Purge will kill all existing PRs and delete all branches",
    )
    args = parser.parse_args()
    return args


def log_through_script(script: str, metrics: Dict) -> None:
    if script and os.path.isfile(script) and os.access(script, os.X_OK):
        p = Popen([script], stdout=PIPE, stdin=PIPE, stderr=PIPE)
        p.communicate(input=json.dumps(metrics).encode())


if __name__ == "__main__":
    args: argparse.Namespace = parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level="INFO"
    )
    cfg_file = os.path.expanduser(args.config)
    labels_file = os.path.expanduser(args.label_colors)
    metrics_logger_script = os.path.expanduser(args.metric_logger)
    metrics_logger = lambda metrics: log_through_script(metrics_logger_script, metrics)

    meter_provider = MeterProvider(
        resource=Resource(attributes={"service_name": "kernel_patches_daemon"}),
        metric_readers=[PeriodicExportingMetricReader(ConsoleMetricExporter())],
    )
    metrics.set_meter_provider(meter_provider)

    with open(cfg_file) as f:
        cfg = json.load(f)

    with open(labels_file) as f:
        labels_cfg = json.load(f)

    if args.action == "purge":
        try:
            purge(cfg=cfg)
            exit(0)
        except Exception:
            logger.exception("Failed to purge")
            exit(1)

    PWDaemon(
        cfg=cfg,
        labels_cfg=labels_cfg,
        metrics_logger=metrics_logger,
    ).loop()
