import os
import re
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import List, TypedDict, Dict, Tuple
from dataclasses import dataclass
from benchmark import benchmark
from benchmark.evaluation import gen_confusion_matrix


from gen import create_bad_diff
from bugsinpy.api import (
    get_bug_info,
    checkout_bug,
    install_dependencies,
    coverage,
    BugInfo,
)


PROJECT_NAME = "black"
BUGSINPY_PATH = "/workspace/BugsInPy"


BugInfo = TypedDict(
    "BugInfo", {"bug_number": int, "correct_patch": str, "args": List[str]}
)


@dataclass
class PatchInfo:
    file_name: str
    rel_path: str


PATCHES = {
    "6": [
        PatchInfo(
            file_name="async_as_identifier.py",
            rel_path="tests/data/async_as_identifier.py",
        )
    ]
}


def main():
    args = parse_args()

    # Generate context in the output folder
    if args.mode == "output-context":
        gen_context(args)
    elif args.mode == "run-tool":
        output_dir = args.output if args.output is not None else "run-tool-output"
        benchmark(output_dir)
    elif args.mode == "evaluation":
        output_dir = args.output if args.output is not None else "evaluation-data"
        gen_confusion_matrix(output_dir, args.run_tool_output_dir)


def get_files_from_patch(patch_content: str) -> List[str]:
    """Extracts file paths from a git diff patch."""
    # Matches lines like '--- a/src/black/__init__.py'
    # and captures 'src/black/__init__.py'
    return re.findall(r"^\-\-\- a/(.+)$", patch_content, re.MULTILINE)


def gen_context(args):
    # Create output directory if it doesn't exist
    base_output_dir = Path("output").resolve()  # Use absolute path
    base_output_dir.mkdir(exist_ok=True)

    bugs_info = get_bug_info(PROJECT_NAME, BUGSINPY_PATH)

    single_run = -1
    if args.single is not None:
        if not (args.single >= "1" and args.single <= "23"):
            raise Exception("arg --single must be between 1 and 23 (bug number)")
        single_run = int(args.single)

    for info in bugs_info:
        if single_run != -1 and info["bug_number"] != single_run:
            continue
        print(f"=== Bug {info['bug_number']} ===")
        bug_output_dir = base_output_dir / str(info["bug_number"])
        base_install_dir = Path(
            BUGSINPY_PATH + "/framework/bin/temp/black-" + str(info["bug_number"])
        )

        good_install_dir = base_install_dir / "good"
        good_config = configure_and_setup(info, good_install_dir, base_output_dir, True)

        good_patch_files = get_files_from_patch(info["correct_patch"])
        good_output_path = bug_output_dir / "good"
        good_output_path.mkdir(exist_ok=True)

        for file in good_patch_files:
            src_file = good_install_dir / "black" / file
            dest_file = good_output_path / file.replace("/", "-")
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_file, dest_file)

        output_diff(info["correct_patch"], bug_output_dir, "good_patch.txt")

        # run on good
        run_tracer(info, good_config, "good", bug_output_dir)

        # Extract coverage for good version
        good_coverage = coverage(Path(BUGSINPY_PATH), good_install_dir / "black")
        good_coverage_path = bug_output_dir / "good_coverage.txt"
        with open(good_coverage_path, "w") as f:
            f.write(good_coverage)

        bad_install_dir = base_install_dir / "bad"
        bad_config = configure_and_setup(info, bad_install_dir, base_output_dir, False)

        bad_diff = get_bad_diff(info["correct_patch"], bug_output_dir / "bad_patch.txt")

        output_diff(bad_diff, bug_output_dir, "bad_patch.txt")

        rel_path_files = get_files_from_patch(bad_diff)
        bad_proj_dir = bad_install_dir / "black"
        for rel_path in rel_path_files:
            original_file = bad_proj_dir / (rel_path + ".original")
            if not original_file.exists():
                print(f"{original_file} doesn't exist... copying")
                shutil.copy(bad_proj_dir / rel_path, original_file)

        if args.re_apply_diff:
            for rel_path in rel_path_files:
                original_file = bad_proj_dir / (rel_path + ".original")
                shutil.copy(original_file, bad_proj_dir / rel_path)

        # patch missing files
        patches = PATCHES.get(str(info["bug_number"]), [])
        patched_file_dir = Path("patches") / str(info["bug_number"])
        for missing_file_info in patches:
            patched_file_path = patched_file_dir / Path(missing_file_info.file_name)
            shutil.copy(patched_file_path, bad_proj_dir / missing_file_info.rel_path)

        # Apply the bad patch to the buggy version
        apply_patch(bug_output_dir / "bad_patch.txt", bad_install_dir / "black")

        print("DEBUG: Copying patched files...")
        bad_output_path = bug_output_dir / "bad"
        bad_output_path.mkdir(exist_ok=True)

        for file in rel_path_files:
            src_file = bad_install_dir / "black" / file
            dest_file = bad_output_path / file.replace("/", "-")
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_file, dest_file)
        print("DEBUG: Finished copying patched files.")

        run_tracer(info, bad_config, "bad", bug_output_dir)

        # Extract coverage for bad version
        bad_coverage = coverage(Path(BUGSINPY_PATH), bad_install_dir / "black")
        bad_coverage_path = bug_output_dir / "bad_coverage.txt"
        with open(bad_coverage_path, "w") as f:
            f.write(bad_coverage)


def run_tracer(info: BugInfo, config: Dict, version: str, bug_output_dir: Path):
    """Runs the tracer for a given version (good or bad)."""
    callgraph_path = bug_output_dir / f"{version}_callgraph.txt"
    if not callgraph_path.exists() or len(callgraph_path.read_text()) == 0:
        print(
            f"Running tracer for bug {info['bug_number']} for {version}_callgraph.txt..."
        )
        result = subprocess.run(
            [
                config["python_path"],
                config["tracer_script_path"],
                "--bug-number",
                str(info["bug_number"]),
                "--version",
                version,
            ]
            + info["args"],
            cwd=config["temp_dir"] + "/black",
            env=config["env"],
            text=True,
        )

        print(f"Call graph for {version} version written to: {callgraph_path}")
        print(f"Return code: {result.returncode}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--re-apply-diff", action="store_true")
    parser.add_argument("--single", action="store")
    parser.add_argument(
        "--mode", required=True, choices=["output-context", "run-tool", "evaluation"]
    )
    parser.add_argument(
        "--output",
        help="Directory to store run tool output",
    )
    parser.add_argument(
        "--run-tool-output-dir",
        help="Directory of run-tool output, used as input for evaluation mode.",
        default="run-tool-output",
    )
    return parser.parse_args()


def get_bad_diff(good_diff: str, bad_diff_path: str) -> str:
    if not os.path.exists(bad_diff_path):
        bad_diff = create_bad_diff(good_diff)
        if bad_diff is None:
            raise Exception("Could not generate bad_diff from good diff")
    else:
        print("Using previous bad_patch.txt")
        with open(bad_diff_path) as f:
            bad_diff = f.read()

    return bad_diff


def output_diff(diff: str, output_dir: Path, filename: str):
    with open(output_dir / filename, "w") as f:
        f.write(diff)


def configure_and_setup(info, instal_dir: Path, output_dir: Path, isGood: bool):
    """Configure a bug environment and return necessary paths and environment."""
    bug_output_dir = output_dir / str(info["bug_number"])
    create_output_folder(bug_output_dir)

    project_dir_name = "black"

    if not os.path.exists(instal_dir / project_dir_name):
        checkout_bug(
            PROJECT_NAME,
            BUGSINPY_PATH,
            instal_dir.__str__(),
            str(info["bug_number"]),
            isGood,
        )

    if not os.path.exists(
        instal_dir.__str__() + f"/{project_dir_name}/env/bin/python3"
    ):
        install_dependencies(
            BUGSINPY_PATH, instal_dir.__str__() + f"/{project_dir_name}"
        )

    tracer_script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tracer.py"
    )
    python_path = os.path.join(
        instal_dir, f"{project_dir_name}/env/bin/python3"
    )  # proj always clones to /black :(

    black_project_dir = os.path.join(instal_dir, project_dir_name)

    env = {
        "PYTHONPATH": black_project_dir,
    }

    return {
        "temp_dir": instal_dir.__str__(),
        "tracer_script_path": tracer_script_path,
        "python_path": python_path,
        "env": env,
    }


def create_output_folder(bug_output_dir: Path):
    bug_output_dir.mkdir(exist_ok=True)


def apply_patch(patch_path: Path, target_dir: Path):
    """Applies a patch to a target directory."""
    if not patch_path.exists() or patch_path.stat().st_size == 0:
        raise Exception(f"Patch file is missing or empty: {patch_path}")

    # First, do a dry run to check if the patch would apply successfully
    dry_run_result = subprocess.run(
        [
            "patch",
            "-p1",
            "-N",
            "--verbose",
            "--dry-run",
            "-i",
            str(patch_path),
        ],
        cwd=str(target_dir),
        capture_output=True,
        text=True,
    )

    # Check if dry run failed
    if dry_run_result.returncode != 0:
        print(f"DRY RUN FAILED: Patch {patch_path} would not apply cleanly")
        print(f"STDOUT: {dry_run_result.stdout}")

        # Check if any hunks failed to apply
        if "Hunk #" in dry_run_result.stdout and "FAILED" in dry_run_result.stdout:
            print(f"STDERR: {dry_run_result.stderr}")
            raise Exception("Invalid patch: some hunks failed to apply")
        else:
            print("Patch already applied... skipping patching")
            return

    # If dry run succeeded, apply the patch for real
    print(f"DRY RUN SUCCESS: Patch {patch_path} would apply cleanly, applying now...")
    result = subprocess.run(
        [
            "patch",
            "-p1",
            "-N",
            "--no-backup",
            "-i",
            str(patch_path),
        ],
        cwd=str(target_dir),
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(f"SUCCESS: Patch {patch_path} applied successfully to {target_dir}")
    else:
        print(f"ERROR: Patch failed for {patch_path} in {target_dir}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise Exception(f"Failed to apply patch: {result.stderr}")


if __name__ == "__main__":
    main()
