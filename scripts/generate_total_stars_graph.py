"""Generate a 6-month total-stars graph across all public repositories."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import requests

matplotlib.use("Agg")

GITHUB_API_BASE_URL = "https://api.github.com"
STARGAZERS_ACCEPT_HEADER = "application/vnd.github.star+json"


def subtract_months(reference_date: dt.date, months: int) -> dt.date:
    """Return the date that is `months` calendar months before `reference_date`."""
    if months < 0:
        raise ValueError("months must be non-negative")

    year = reference_date.year
    month = reference_date.month - months
    while month <= 0:
        month += 12
        year -= 1

    # Clamp the day to the last valid day in the target month.
    if month in {1, 3, 5, 7, 8, 10, 12}:
        last_day = 31
    elif month in {4, 6, 9, 11}:
        last_day = 30
    else:
        is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        last_day = 29 if is_leap_year else 28

    day = min(reference_date.day, last_day)
    return dt.date(year, month, day)


def _request_json(
    session: requests.Session,
    url: str,
    token: str | None,
    params: dict[str, Any] | None = None,
    accept_header: str = "application/vnd.github+json",
) -> list[dict[str, Any]]:
    """Send a GitHub API request and return decoded JSON."""
    headers = {
        "Accept": accept_header,
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "francis1998-profile-stars-graph",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = session.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    json_payload = response.json()
    if isinstance(json_payload, list):
        return json_payload
    raise ValueError(f"Unexpected JSON payload type from {url}")


def list_public_owner_repositories(username: str, token: str | None) -> list[str]:
    """Return all non-fork public repositories owned by the user."""
    repository_names: list[str] = []
    page = 1
    with requests.Session() as session:
        while True:
            endpoint = f"{GITHUB_API_BASE_URL}/users/{username}/repos"
            repositories = _request_json(
                session=session,
                url=endpoint,
                token=token,
                params={"per_page": 100, "page": page, "type": "owner", "sort": "updated"},
            )
            if not repositories:
                break

            for repository in repositories:
                if repository.get("fork"):
                    continue
                repository_name = repository.get("name")
                if isinstance(repository_name, str) and repository_name:
                    repository_names.append(repository_name)

            if len(repositories) < 100:
                break
            page += 1

    return repository_names


def list_star_dates_for_repository(
    username: str,
    repository_name: str,
    token: str | None,
) -> list[dt.date]:
    """Return all stargazer dates for a repository."""
    star_dates: list[dt.date] = []
    page = 1
    with requests.Session() as session:
        while True:
            endpoint = f"{GITHUB_API_BASE_URL}/repos/{username}/{repository_name}/stargazers"
            stargazers = _request_json(
                session=session,
                url=endpoint,
                token=token,
                params={"per_page": 100, "page": page},
                accept_header=STARGAZERS_ACCEPT_HEADER,
            )
            if not stargazers:
                break

            for stargazer in stargazers:
                starred_at = stargazer.get("starred_at")
                if isinstance(starred_at, str):
                    star_dates.append(dt.datetime.fromisoformat(starred_at.replace("Z", "+00:00")).date())

            if len(stargazers) < 100:
                break
            page += 1

    return star_dates


def build_total_stars_series(
    all_star_dates: list[dt.date],
    start_date: dt.date,
    end_date: dt.date,
) -> tuple[list[dt.date], list[int]]:
    """Build cumulative daily totals from start_date to end_date."""
    stars_per_day = Counter(all_star_dates)
    baseline_total = sum(1 for star_date in all_star_dates if star_date < start_date)

    daily_dates: list[dt.date] = []
    cumulative_totals: list[int] = []

    running_total = baseline_total
    current_date = start_date
    while current_date <= end_date:
        running_total += stars_per_day.get(current_date, 0)
        daily_dates.append(current_date)
        cumulative_totals.append(running_total)
        current_date += dt.timedelta(days=1)

    return daily_dates, cumulative_totals


def render_graph(dates: list[dt.date], totals: list[int], output_path: Path) -> None:
    """Render the total-stars SVG graph."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("dark_background")
    figure, axis = plt.subplots(figsize=(12, 3.8), dpi=140)
    figure.patch.set_facecolor("#0D1117")
    axis.set_facecolor("#0D1117")

    axis.plot(dates, totals, color="#58A6FF", linewidth=2.3)
    axis.fill_between(dates, totals, [min(totals)] * len(totals), color="#58A6FF", alpha=0.20)

    axis.set_title("Total Stars Across All Repositories (Last 6 Months)", color="#C9D1D9", fontsize=12, pad=10)
    axis.set_xlabel("Date", color="#8B949E")
    axis.set_ylabel("Total Stars", color="#8B949E")
    axis.tick_params(axis="x", colors="#8B949E")
    axis.tick_params(axis="y", colors="#8B949E")
    axis.grid(True, linestyle="--", alpha=0.25, color="#30363D")

    axis.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    figure.autofmt_xdate()

    for spine in axis.spines.values():
        spine.set_color("#30363D")

    final_total = totals[-1] if totals else 0
    axis.text(
        0.99,
        0.93,
        f"Current total: {final_total}",
        transform=axis.transAxes,
        ha="right",
        va="center",
        color="#C9D1D9",
        fontsize=10,
    )

    figure.tight_layout()
    figure.savefig(output_path, format="svg")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate total-stars graph SVG.")
    parser.add_argument("--user", required=True, help="GitHub username")
    parser.add_argument("--months", type=int, default=6, help="Window size in months")
    parser.add_argument("--output", required=True, help="Output SVG path")
    return parser.parse_args()


def main() -> None:
    """Generate and save the stars graph."""
    arguments = parse_arguments()
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    end_date = dt.datetime.now(dt.timezone.utc).date()
    start_date = subtract_months(end_date, arguments.months)

    repository_names = list_public_owner_repositories(arguments.user, token)
    all_star_dates: list[dt.date] = []
    for repository_name in repository_names:
        all_star_dates.extend(list_star_dates_for_repository(arguments.user, repository_name, token))

    dates, totals = build_total_stars_series(all_star_dates, start_date, end_date)
    render_graph(dates, totals, Path(arguments.output))


if __name__ == "__main__":
    main()
