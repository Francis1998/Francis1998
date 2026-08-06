"""Generate stars graph and large total metrics for the profile README."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
import requests

matplotlib.use("Agg")

GITHUB_API_BASE_URL = "https://api.github.com"
GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
HISTORY_SCHEMA_VERSION = 1


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


def get_total_public_owner_stars(username: str, token: str | None) -> int:
    """Return the sum of stargazer counts across non-fork public owner repositories.

    Uses repository metadata (`stargazers_count`) instead of listing stargazers.
    Listing stargazers is forbidden for GitHub Actions installation tokens.
    """
    total_stars = 0
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
                star_count = repository.get("stargazers_count")
                if isinstance(star_count, int):
                    total_stars += star_count
                elif isinstance(star_count, float):
                    total_stars += int(star_count)

            if len(repositories) < 100:
                break
            page += 1

    return total_stars


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


def _parse_sample_timestamp(raw_timestamp: str) -> dt.datetime:
    """Parse a stored ISO-8601 sample timestamp into an aware UTC datetime."""
    parsed = dt.datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def load_star_run_history(history_path: Path) -> list[tuple[dt.datetime, int]]:
    """Load persisted per-run star samples from disk."""
    if not history_path.exists():
        return []

    payload = json.loads(history_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid history payload in {history_path}")

    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise ValueError(f"Missing samples list in {history_path}")

    samples: list[tuple[dt.datetime, int]] = []
    for raw_sample in raw_samples:
        if not isinstance(raw_sample, dict):
            continue
        recorded_at = raw_sample.get("recorded_at")
        total_stars = raw_sample.get("total_stars")
        if not isinstance(recorded_at, str) or not isinstance(total_stars, int):
            continue
        samples.append((_parse_sample_timestamp(recorded_at), total_stars))

    samples.sort(key=lambda sample: sample[0])
    return samples


def save_star_run_history(history_path: Path, samples: list[tuple[dt.datetime, int]]) -> None:
    """Persist per-run star samples to disk."""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": HISTORY_SCHEMA_VERSION,
        "samples": [
            {
                "recorded_at": timestamp.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                "total_stars": total_stars,
            }
            for timestamp, total_stars in samples
        ],
    }
    history_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_star_run_sample(
    samples: list[tuple[dt.datetime, int]],
    recorded_at: dt.datetime,
    total_stars: int,
    weeks: int,
) -> list[tuple[dt.datetime, int]]:
    """Append the current run sample and keep only the last N weeks of points."""
    if weeks <= 0:
        raise ValueError("weeks must be positive")
    if total_stars < 0:
        raise ValueError("total_stars must be non-negative")

    recorded_at_utc = recorded_at.astimezone(dt.timezone.utc)
    updated_samples = list(samples)
    updated_samples.append((recorded_at_utc, total_stars))
    updated_samples.sort(key=lambda sample: sample[0])

    window_start = recorded_at_utc - dt.timedelta(weeks=weeks)
    return [sample for sample in updated_samples if sample[0] >= window_start]


def render_run_star_graph(
    samples: list[tuple[dt.datetime, int]],
    weeks: int,
    output_path: Path,
) -> None:
    """Render total stars graph using one point per action run."""
    if weeks <= 0:
        raise ValueError("weeks must be positive")
    if not samples:
        raise ValueError("samples must not be empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamps = [sample[0] for sample in samples]
    totals = [sample[1] for sample in samples]

    plt.style.use("dark_background")
    figure, axis = plt.subplots(figsize=(12, 3.8), dpi=140)
    figure.patch.set_facecolor("#0D1117")
    axis.set_facecolor("#0D1117")

    axis.plot(
        timestamps,
        totals,
        color="#58A6FF",
        linewidth=2.3,
        marker="o",
        markersize=4.5,
    )
    y_floor = min(totals)
    axis.fill_between(timestamps, totals, y_floor, color="#58A6FF", alpha=0.20)

    week_label = "1 Week" if weeks == 1 else f"{weeks} Weeks"
    axis.set_title(
        f"Total Stars Across All Repositories (Last {week_label}, Per Action Run)",
        color="#C9D1D9",
        fontsize=12,
        pad=10,
    )
    axis.set_xlabel("Date", color="#8B949E")
    axis.set_ylabel("Total Stars", color="#8B949E")
    axis.tick_params(axis="x", colors="#8B949E")
    axis.tick_params(axis="y", colors="#8B949E")
    axis.grid(True, axis="y", linestyle="--", alpha=0.25, color="#30363D")

    axis.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axis.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
    figure.autofmt_xdate()
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    y_min = min(totals)
    y_max = max(totals)
    axis.set_ylim(max(0, y_min - 1), max(1, y_max) + 1)

    latest_timestamp = timestamps[-1]
    axis.set_xlim(
        latest_timestamp - dt.timedelta(weeks=weeks),
        latest_timestamp + dt.timedelta(hours=3),
    )

    for spine in axis.spines.values():
        spine.set_color("#30363D")

    current_total = totals[-1]
    axis.text(
        0.99,
        0.93,
        f"Current total stars: {current_total}",
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
    parser.add_argument("--weeks", type=int, default=1, help="Window size in weeks")
    parser.add_argument("--weekly-output", help="Output SVG path for stars graph")
    parser.add_argument(
        "--history-output",
        default="assets/stars-run-history.json",
        help="Persisted per-run star history JSON path",
    )
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
    end_datetime = dt.datetime.now(dt.timezone.utc)

    total_stars = get_total_public_owner_stars(arguments.user, token)
    total_commits = get_total_commit_contributions(arguments.user, token)
    render_totals_card(total_stars, total_commits, Path(arguments.metrics_output))

    if arguments.metrics_only:
        return

    if not arguments.weekly_output:
        raise ValueError("--weekly-output is required unless --metrics-only is set")

    history_path = Path(arguments.history_output)
    existing_samples = load_star_run_history(history_path)
    updated_samples = append_star_run_sample(
        samples=existing_samples,
        recorded_at=end_datetime,
        total_stars=total_stars,
        weeks=arguments.weeks,
    )
    save_star_run_history(history_path, updated_samples)
    render_run_star_graph(
        samples=updated_samples,
        weeks=arguments.weeks,
        output_path=Path(arguments.weekly_output),
    )


if __name__ == "__main__":
    main()
