"""Generate stars graph and large total metrics for the profile README."""

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
from matplotlib.patches import FancyBboxPatch
import requests

matplotlib.use("Agg")

GITHUB_API_BASE_URL = "https://api.github.com"
GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
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


def _request_graphql(
    session: requests.Session,
    query: str,
    token: str | None,
    variables: dict[str, Any],
) -> dict[str, Any]:
    """Send a GitHub GraphQL request and return decoded JSON."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "francis1998-profile-stars-graph",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = session.post(
        GITHUB_GRAPHQL_URL,
        headers=headers,
        json={"query": query, "variables": variables},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise ValueError(f"GraphQL error: {payload['errors']}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("Unexpected GraphQL payload")
    return data


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


def get_total_commit_contributions(username: str, token: str | None) -> int:
    """Return all-time total commit contributions using yearly GraphQL windows."""
    user_query = """
    query($login: String!) {
      user(login: $login) {
        createdAt
      }
    }
    """
    contribution_query = """
    query($login: String!, $from: DateTime!, $to: DateTime!) {
      user(login: $login) {
        contributionsCollection(from: $from, to: $to) {
          totalCommitContributions
        }
      }
    }
    """

    with requests.Session() as session:
        user_data = _request_graphql(session, user_query, token, {"login": username})
        created_at = (
            user_data.get("user", {}).get("createdAt")
            if isinstance(user_data.get("user"), dict)
            else None
        )
        if not isinstance(created_at, str):
            raise ValueError("Failed to fetch user creation date")

        start_date = dt.datetime.fromisoformat(created_at.replace("Z", "+00:00")).date()
        end_date = dt.datetime.now(dt.timezone.utc).date()
        total_commits = 0

        window_start = start_date
        while window_start <= end_date:
            window_end = min(window_start + dt.timedelta(days=364), end_date)
            from_timestamp = dt.datetime.combine(
                window_start,
                dt.time.min,
                tzinfo=dt.timezone.utc,
            ).isoformat()
            to_timestamp = dt.datetime.combine(
                window_end,
                dt.time.max,
                tzinfo=dt.timezone.utc,
            ).isoformat()
            contribution_data = _request_graphql(
                session,
                contribution_query,
                token,
                {"login": username, "from": from_timestamp, "to": to_timestamp},
            )

            user_payload = contribution_data.get("user")
            if not isinstance(user_payload, dict):
                raise ValueError("Failed to fetch contributions payload")
            collection_payload = user_payload.get("contributionsCollection")
            if not isinstance(collection_payload, dict):
                raise ValueError("Missing contributions collection")
            commit_count = collection_payload.get("totalCommitContributions")
            if not isinstance(commit_count, int):
                raise ValueError("Invalid commit count from GraphQL")
            total_commits += commit_count
            window_start = window_end + dt.timedelta(days=1)

    return total_commits


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


def render_totals_card(
    total_stars: int,
    total_commits: int,
    output_path: Path,
) -> None:
    """Render large totals card with commits and stars."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 2.2), dpi=160)
    figure.patch.set_facecolor("#0D1117")
    axis.set_facecolor("#0D1117")
    axis.axis("off")
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)

    left_card = FancyBboxPatch(
        (3, 12),
        45,
        76,
        boxstyle="round,pad=0.8,rounding_size=7",
        linewidth=1.2,
        edgecolor="#30363D",
        facecolor="#11161D",
    )
    right_card = FancyBboxPatch(
        (52, 12),
        45,
        76,
        boxstyle="round,pad=0.8,rounding_size=7",
        linewidth=1.2,
        edgecolor="#30363D",
        facecolor="#11161D",
    )
    axis.add_patch(left_card)
    axis.add_patch(right_card)

    axis.text(25.5, 66, "TOTAL COMMITS", ha="center", va="center", color="#8B949E", fontsize=11, weight="bold")
    axis.text(25.5, 40, f"{total_commits:,}", ha="center", va="center", color="#58A6FF", fontsize=28, weight="bold")
    axis.text(74.5, 66, "TOTAL STARS", ha="center", va="center", color="#8B949E", fontsize=11, weight="bold")
    axis.text(74.5, 40, f"{total_stars:,}", ha="center", va="center", color="#F2CC60", fontsize=28, weight="bold")

    figure.tight_layout(pad=0.2)
    figure.savefig(output_path, format="svg")
    plt.close(figure)


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate total-stars graph SVG.")
    parser.add_argument("--user", required=True, help="GitHub username")
    parser.add_argument("--months", type=int, default=6, help="Window size in months")
    parser.add_argument("--output", help="Output SVG path for stars graph")
    parser.add_argument("--metrics-output", required=True, help="Output SVG path for metrics card")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip stars graph generation and only render totals card",
    )
    return parser.parse_args()


def main() -> None:
    """Generate and save the stars graph and totals card."""
    arguments = parse_arguments()
    token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    end_date = dt.datetime.now(dt.timezone.utc).date()
    start_date = subtract_months(end_date, arguments.months)

    repository_names = list_public_owner_repositories(arguments.user, token)
    all_star_dates: list[dt.date] = []
    for repository_name in repository_names:
        all_star_dates.extend(list_star_dates_for_repository(arguments.user, repository_name, token))

    total_stars = len(all_star_dates)
    total_commits = get_total_commit_contributions(arguments.user, token)
    render_totals_card(total_stars, total_commits, Path(arguments.metrics_output))

    if arguments.metrics_only:
        return

    if not arguments.output:
        raise ValueError("--output is required unless --metrics-only is set")

    dates, totals = build_total_stars_series(all_star_dates, start_date, end_date)
    render_graph(dates, totals, Path(arguments.output))


if __name__ == "__main__":
    main()
