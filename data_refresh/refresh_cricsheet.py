"""
Cricsheet Data Refresh Tool

Downloads and stages T20 cricket match data from Cricsheet into a staging
directory for verification before deploying to the main data/ folder.

Usage:
    uv run python data_refresh/refresh_cricsheet.py --full
    uv run python data_refresh/refresh_cricsheet.py --full --competitions ipl bbl
    uv run python data_refresh/refresh_cricsheet.py --incremental
    uv run python data_refresh/refresh_cricsheet.py --verify
    uv run python data_refresh/refresh_cricsheet.py --deploy
    uv run python data_refresh/refresh_cricsheet.py --status
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Competition Registry
# ---------------------------------------------------------------------------

COMPETITIONS = [
    {
        "code": "ipl",
        "name": "Indian Premier League",
        "url": "https://cricsheet.org/downloads/ipl_male_json.zip",
        "target_dir": "ipl_json",
        "parent": "",
    },
    {
        "code": "bbl",
        "name": "Big Bash League",
        "url": "https://cricsheet.org/downloads/bbl_male_json.zip",
        "target_dir": "bbl_json",
        "parent": "other_t20_data",
    },
    {
        "code": "psl",
        "name": "Pakistan Super League",
        "url": "https://cricsheet.org/downloads/psl_male_json.zip",
        "target_dir": "psl_json",
        "parent": "other_t20_data",
    },
    {
        "code": "cpl",
        "name": "Caribbean Premier League",
        "url": "https://cricsheet.org/downloads/cpl_male_json.zip",
        "target_dir": "cpl_json",
        "parent": "other_t20_data",
    },
    {
        "code": "ntb",
        "name": "T20 Blast",
        "url": "https://cricsheet.org/downloads/ntb_male_json.zip",
        "target_dir": "ntb_json",
        "parent": "other_t20_data",
    },
    {
        "code": "bpl",
        "name": "Bangladesh Premier League",
        "url": "https://cricsheet.org/downloads/bpl_male_json.zip",
        "target_dir": "bpl_json",
        "parent": "other_t20_data",
    },
    {
        "code": "sma",
        "name": "Syed Mushtaq Ali Trophy",
        "url": "https://cricsheet.org/downloads/sma_male_json.zip",
        "target_dir": "sma_json",
        "parent": "other_t20_data",
    },
    {
        "code": "t20s",
        "name": "T20 Internationals",
        "url": "https://cricsheet.org/downloads/t20s_male_json.zip",
        "target_dir": "t20s_json",
        "parent": "other_t20_data",
    },
    {
        "code": "sat",
        "name": "SA20",
        "url": "https://cricsheet.org/downloads/sat_male_json.zip",
        "target_dir": "sat_json",
        "parent": "other_t20_data",
    },
    {
        "code": "ilt",
        "name": "International League T20",
        "url": "https://cricsheet.org/downloads/ilt_male_json.zip",
        "target_dir": "ilt_json",
        "parent": "other_t20_data",
    },
    {
        "code": "mlc",
        "name": "Major League Cricket",
        "url": "https://cricsheet.org/downloads/mlc_male_json.zip",
        "target_dir": "mlc_json",
        "parent": "other_t20_data",
    },
    {
        "code": "msl",
        "name": "Mzansi Super League",
        "url": "https://cricsheet.org/downloads/msl_male_json.zip",
        "target_dir": "msl_json",
        "parent": "other_t20_data",
    },
]

COMP_BY_CODE = {c["code"]: c for c in COMPETITIONS}
ALL_CODES = [c["code"] for c in COMPETITIONS]

INCREMENTAL_URL = "https://cricsheet.org/downloads/recently_added_30_male_json.zip"

# Event name patterns → competition code (for incremental classification)
EVENT_TO_CODE = {
    "Indian Premier League": "ipl",
    "Big Bash League": "bbl",
    "Pakistan Super League": "psl",
    "Caribbean Premier League": "cpl",
    "Vitality Blast": "ntb",
    "NatWest T20 Blast": "ntb",
    "T20 Blast": "ntb",
    "Bangladesh Premier League": "bpl",
    "Syed Mushtaq Ali Trophy": "sma",
    "SA20": "sat",
    "International League T20": "ilt",
    "Major League Cricket": "mlc",
    "Mzansi Super League": "msl",
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    project_root: Path
    data_dir: Path
    staging_dir: Path
    competitions: list
    dry_run: bool = False


def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")


def resolve_staging_dir(staging_root: Path, comp: dict) -> Path:
    if comp["parent"]:
        return staging_root / comp["parent"] / comp["target_dir"]
    return staging_root / comp["target_dir"]


def resolve_data_dir(data_root: Path, comp: dict) -> Path:
    if comp["parent"]:
        return data_root / comp["parent"] / comp["target_dir"]
    return data_root / comp["target_dir"]


def count_json_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return len(list(directory.glob("*.json")))


def count_json_files_with_suffix(data_root: Path, comp: dict) -> int:
    """Count JSON files, also checking for ' (1)' suffix variant."""
    direct = resolve_data_dir(data_root, comp)
    if direct.exists():
        return count_json_files(direct)
    # Check for " (1)" suffix variant
    if comp["parent"]:
        parent = data_root / comp["parent"]
        candidates = list(parent.glob(f"{comp['target_dir']}*"))
        for c in candidates:
            if c.is_dir():
                return count_json_files(c)
    return 0


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r    {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)
    else:
        mb = downloaded / (1024 * 1024)
        print(f"\r    {mb:.1f} MB downloaded", end="", flush=True)


def download_and_extract(url: str, target_dir: Path, name: str) -> dict:
    """Download a zip and extract JSON files into target_dir.

    Returns {"new_files": int, "total_files": int, "skipped": int}
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = {f.name for f in target_dir.glob("*.json")}

    tmp_path = Path(tempfile.mktemp(suffix=".zip"))
    try:
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(url, tmp_path, reporthook=progress_hook)
        print()  # newline after progress

        new_count = 0
        skipped = 0
        with zipfile.ZipFile(tmp_path, "r") as zf:
            json_members = [m for m in zf.namelist() if m.endswith(".json")]
            for member in json_members:
                filename = Path(member).name
                if filename in existing:
                    skipped += 1
                    continue
                data = zf.read(member)
                (target_dir / filename).write_bytes(data)
                new_count += 1

        total = count_json_files(target_dir)
        return {"new_files": new_count, "total_files": total, "skipped": skipped}

    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Full Refresh
# ---------------------------------------------------------------------------

def run_full(config: Config) -> dict:
    """Download complete zips for selected competitions into staging."""
    print("=" * 60)
    print("FULL REFRESH -> staging")
    print("=" * 60)

    results = {}
    for i, comp in enumerate(config.competitions):
        staging = resolve_staging_dir(config.staging_dir, comp)
        if config.dry_run:
            current_staging = count_json_files(staging)
            current_data = count_json_files_with_suffix(config.data_dir, comp)
            print(f"  [{comp['code']}] {comp['name']}")
            print(f"    URL: {comp['url']}")
            print(f"    Staging: {staging} ({current_staging} files)")
            print(f"    Data:    {resolve_data_dir(config.data_dir, comp)} ({current_data} files)")
            results[comp["code"]] = {"dry_run": True}
            continue

        try:
            stats = download_and_extract(comp["url"], staging, comp["name"])
            results[comp["code"]] = stats
            print(f"    {comp['name']}: {stats['new_files']} new, {stats['total_files']} total")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            print(f"    ERROR downloading {comp['name']}: {e}")
            results[comp["code"]] = {"error": str(e)}
        except zipfile.BadZipFile as e:
            print(f"    ERROR: corrupt zip for {comp['name']}: {e}")
            results[comp["code"]] = {"error": str(e)}

        # Courtesy delay between downloads
        if not config.dry_run and i < len(config.competitions) - 1:
            time.sleep(1)

    if not config.dry_run:
        print()
        run_verify(config)

    return results


# ---------------------------------------------------------------------------
# Incremental Refresh
# ---------------------------------------------------------------------------

def classify_match(match_data: dict) -> str | None:
    """Determine competition code from match JSON."""
    info = match_data.get("info", {})

    # Check event name
    event = info.get("event", {})
    event_name = event.get("name", "") if isinstance(event, dict) else str(event)

    for pattern, code in EVENT_TO_CODE.items():
        if pattern.lower() in event_name.lower():
            return code

    # T20I: has match_type_number or match_type == "T20I"
    if info.get("match_type_number") is not None or info.get("match_type") == "T20I":
        return "t20s"

    return None


def run_incremental(config: Config) -> dict:
    """Download recently added matches and sort into staging directories."""
    print("=" * 60)
    print("INCREMENTAL REFRESH (last 30 days) -> staging")
    print("=" * 60)

    if config.dry_run:
        print(f"  Would download: {INCREMENTAL_URL}")
        print(f"  Would classify and route files to staging/")
        return {"dry_run": True}

    selected_codes = {c["code"] for c in config.competitions}

    tmp_path = Path(tempfile.mktemp(suffix=".zip"))
    results = defaultdict(lambda: {"new": 0, "skipped": 0})

    try:
        print(f"  Downloading recently added matches...")
        urllib.request.urlretrieve(INCREMENTAL_URL, tmp_path, reporthook=progress_hook)
        print()

        with zipfile.ZipFile(tmp_path, "r") as zf:
            json_members = [m for m in zf.namelist() if m.endswith(".json")]
            print(f"  {len(json_members)} JSON files in recent update")

            for member in json_members:
                raw = zf.read(member)
                try:
                    match_data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                code = classify_match(match_data)
                if code is None or code not in selected_codes:
                    results["_skipped"]["skipped"] += 1
                    continue

                comp = COMP_BY_CODE.get(code)
                if comp is None:
                    continue

                staging = resolve_staging_dir(config.staging_dir, comp)
                staging.mkdir(parents=True, exist_ok=True)
                filename = Path(member).name
                dest = staging / filename

                if dest.exists():
                    results[code]["skipped"] += 1
                else:
                    dest.write_bytes(raw)
                    results[code]["new"] += 1
                    print(f"    + {code}: {filename}")

    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"  ERROR downloading recent matches: {e}")
        return {"error": str(e)}
    except zipfile.BadZipFile as e:
        print(f"  ERROR: corrupt zip: {e}")
        return {"error": str(e)}
    finally:
        tmp_path.unlink(missing_ok=True)

    # Summary
    print()
    for code in sorted(results):
        if code != "_skipped":
            s = results[code]
            print(f"  {code}: {s['new']} new, {s['skipped']} already in staging")

    skipped_other = results["_skipped"]["skipped"]
    if skipped_other:
        print(f"  ({skipped_other} files from other/unrecognized competitions skipped)")

    print()
    run_verify(config)
    return dict(results)


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

def run_verify(config: Config) -> bool:
    """Validate staged data: JSON integrity, structure, and comparison to data/."""
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    all_ok = True
    corrupt_files = []

    header = f"  {'Competition':<28} {'Staging':>8} {'data/':>8} {'Delta':>8}  Status"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for comp in config.competitions:
        staging = resolve_staging_dir(config.staging_dir, comp)
        staging_count = count_json_files(staging)

        if staging_count == 0:
            continue

        # Validate JSON integrity (sample up to 10 files)
        json_files = list(staging.glob("*.json"))
        sample = json_files[:10] if len(json_files) > 10 else json_files
        comp_corrupt = 0
        for f in sample:
            try:
                data = json.loads(f.read_text())
                if "info" not in data:
                    corrupt_files.append(str(f))
                    comp_corrupt += 1
            except (json.JSONDecodeError, UnicodeDecodeError):
                corrupt_files.append(str(f))
                comp_corrupt += 1

        data_count = count_json_files_with_suffix(config.data_dir, comp)
        delta = staging_count - data_count

        if comp_corrupt > 0:
            status = f"WARN: {comp_corrupt} corrupt in sample"
            all_ok = False
        elif delta > 0:
            status = f"+{delta} new matches"
        elif delta == 0:
            status = "up to date"
        else:
            status = f"{delta} (fewer than data/)"

        delta_str = f"+{delta}" if delta > 0 else str(delta)
        print(f"  {comp['name']:<28} {staging_count:>8,} {data_count:>8,} {delta_str:>8}  {status}")

    if corrupt_files:
        print(f"\n  Corrupt files found:")
        for f in corrupt_files:
            print(f"    {f}")

    print()
    if all_ok:
        print("  Verification: PASSED")
    else:
        print("  Verification: FAILED (see warnings above)")

    return all_ok


# ---------------------------------------------------------------------------
# Deploy
# ---------------------------------------------------------------------------

def normalize_folder_names(data_dir: Path) -> None:
    """Rename folders with ' (1)' suffix to clean names."""
    other_t20 = data_dir / "other_t20_data"
    if not other_t20.exists():
        return
    for folder in sorted(other_t20.iterdir()):
        if folder.is_dir() and " (1)" in folder.name:
            clean_name = folder.name.replace(" (1)", "")
            target = folder.parent / clean_name
            if target.exists():
                # Merge: move files from suffixed folder into clean folder
                moved = 0
                for f in folder.iterdir():
                    dest = target / f.name
                    if not dest.exists():
                        shutil.copy2(f, dest)
                        moved += 1
                # Remove the suffixed folder after merging
                shutil.rmtree(folder)
                print(f"  Merged '{folder.name}' into '{clean_name}' ({moved} files)")
            else:
                folder.rename(target)
                print(f"  Renamed '{folder.name}' -> '{clean_name}'")


def run_deploy(config: Config) -> bool:
    """Copy verified staging data to data/."""
    print("=" * 60)
    print("DEPLOY: staging -> data/")
    print("=" * 60)

    # Verify first
    ok = run_verify(config)
    if not ok:
        print("\n  Deploy ABORTED: verification failed. Fix issues first.")
        return False

    print()

    # Normalize existing folder names
    normalize_folder_names(config.data_dir)

    total_copied = 0
    for comp in config.competitions:
        staging = resolve_staging_dir(config.staging_dir, comp)
        if not staging.exists():
            continue

        staging_files = list(staging.glob("*.json"))
        if not staging_files:
            continue

        target = resolve_data_dir(config.data_dir, comp)
        target.mkdir(parents=True, exist_ok=True)
        existing = {f.name for f in target.glob("*.json")}

        copied = 0
        for f in staging_files:
            if f.name not in existing:
                shutil.copy2(f, target / f.name)
                copied += 1

        if copied > 0:
            total_copied += copied
            total = count_json_files(target)
            print(f"  {comp['name']}: copied {copied} new files ({total} total)")

    print(f"\n  Deploy complete: {total_copied} new files copied to data/")
    return True


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def run_status(config: Config) -> None:
    """Show comparison of staging vs data/."""
    print("=" * 60)
    print("STATUS: staging vs data/")
    print("=" * 60)

    header = f"  {'Competition':<28} {'Staging':>8} {'data/':>8} {'Delta':>8}  {'Pending'}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    any_staged = False
    for comp in config.competitions:
        staging = resolve_staging_dir(config.staging_dir, comp)
        staging_count = count_json_files(staging)
        data_count = count_json_files_with_suffix(config.data_dir, comp)

        if staging_count == 0 and data_count == 0:
            continue

        any_staged = any_staged or staging_count > 0

        delta = staging_count - data_count
        delta_str = f"+{delta}" if delta > 0 else str(delta)

        if staging_count == 0:
            pending = "not staged"
        elif delta > 0:
            pending = f"{delta} to deploy"
        elif delta == 0 and staging_count > 0:
            pending = "up to date"
        else:
            pending = "staging behind"

        print(f"  {comp['name']:<28} {staging_count:>8,} {data_count:>8,} {delta_str:>8}  {pending}")

    if not any_staged:
        print("\n  No data in staging. Run --full or --incremental first.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and manage Cricsheet T20 match data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --full                          Download all competitions to staging
  %(prog)s --full --competitions ipl bbl   Download only IPL and BBL
  %(prog)s --incremental                   Download last 30 days of new matches
  %(prog)s --verify                        Validate staged data
  %(prog)s --deploy                        Copy verified staging to data/
  %(prog)s --status                        Compare staging vs data/
  %(prog)s --full --dry-run                Preview without downloading
        """,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true", help="Download complete competition zips")
    mode.add_argument("--incremental", action="store_true", help="Download recently added matches (last 30 days)")
    mode.add_argument("--verify", action="store_true", help="Validate staged data")
    mode.add_argument("--deploy", action="store_true", help="Copy verified staging to data/")
    mode.add_argument("--status", action="store_true", help="Show staging vs data/ comparison")

    parser.add_argument(
        "--competitions",
        nargs="+",
        choices=ALL_CODES,
        default=None,
        help=f"Competitions to refresh (default: all). Choices: {', '.join(ALL_CODES)}",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview actions without downloading")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override data directory path")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = find_project_root()
    data_dir = args.data_dir or (project_root / "data")
    staging_dir = project_root / "data_refresh" / "staging"

    # Resolve competitions
    if args.competitions:
        comps = [COMP_BY_CODE[c] for c in args.competitions]
    else:
        comps = list(COMPETITIONS)

    config = Config(
        project_root=project_root,
        data_dir=data_dir,
        staging_dir=staging_dir,
        competitions=comps,
        dry_run=args.dry_run,
    )

    print(f"Project root: {project_root}")
    print(f"Data dir:     {data_dir}")
    print(f"Staging dir:  {staging_dir}")
    print(f"Competitions: {', '.join(c['code'] for c in comps)}")
    print()

    if args.full:
        run_full(config)
    elif args.incremental:
        run_incremental(config)
    elif args.verify:
        run_verify(config)
    elif args.deploy:
        run_deploy(config)
    elif args.status:
        run_status(config)


if __name__ == "__main__":
    main()
