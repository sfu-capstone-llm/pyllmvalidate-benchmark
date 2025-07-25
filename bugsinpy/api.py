import os
import shlex
import subprocess
from pathlib import Path
from typing import List, TypedDict
import re


BugInfo = TypedDict(
    "BugInfo", {"bug_number": int, "correct_patch": str, "args": List[str]}
)


def get_bug_info(project_name: str, bugsinpy_path: str) -> List[BugInfo]:
    path = Path(bugsinpy_path) / "projects" / project_name / "bugs"
    if not path.exists():
        raise Exception(f"Could not find path: {path}")

    bugs_info = []

    bug_folders = []
    for item in path.iterdir():
        if item.is_dir() and item.name.isdigit():
            bug_folders.append(item)

    bug_folders.sort(key=lambda x: int(x.name))

    for bug_folder in bug_folders:
        try:
            bug_data = _process_bug_folder(bug_folder)
            bugs_info.append(bug_data)
        except Exception as e:
            print(f"Error processing {bug_folder}: {e}")

    return bugs_info


def _process_bug_folder(bug_path: Path) -> BugInfo:
    bug_number = int(bug_path.name)

    # Read bug_patch.txt
    patch_file = bug_path / "bug_patch.txt"
    if patch_file.exists():
        with open(patch_file, "r", encoding="utf-8") as f:
            diff = f.read()
    else:
        diff = ""

    test_file = bug_path / "run_test.sh"
    if test_file.exists():
        with open(test_file, "r", encoding="utf-8") as f:
            test_content = f.readline().strip()
        args = _extract_unittest_args(test_content)
    else:
        args = []

    return BugInfo(bug_number=bug_number, correct_patch=diff, args=args)


def _extract_unittest_args(run_test_content: str) -> List[str]:
    """Extract unittest arguments from run_test.sh content"""
    try:
        cmd = run_test_content.strip().split("|")[0].split(">")[0].strip()
        tokens = shlex.split(cmd)

        if "unittest" in tokens:
            unittest_idx = tokens.index("unittest")
            return tokens[unittest_idx:]
        else:
            if len(tokens) >= 3 and tokens[0] == "python" and tokens[1] == "-m":
                return tokens[2:]
            return tokens
    except:
        return []


def checkout_bug(
    project_name: str, bugsinpy_path: str, workdir: str, bug_id: str, is_fixed: bool
):
    print(f"Checking out bug {bug_id} ({'fixed' if is_fixed else 'buggy'} version)...")
    subprocess.run(
        [
            str(Path(bugsinpy_path) / "framework/bin/bugsinpy-checkout"),
            "-p",
            project_name,
            "-i",
            bug_id,
            "-w",
            workdir,
            "-v",
            "1" if is_fixed else "0",
        ]
    )


def install_dependencies(bugsinpy_path: str, folder_path: str):
    print("Installing dependencies...")

    minimal_env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": os.environ.get("HOME", ""),
    }

    # Test what Python version will be used
    try:
        python_version_result = subprocess.run(
            ["python3", "--version"],
            env=minimal_env,
            capture_output=True,
            text=True,
        )
        print(
            f"DEBUG: Python version that will be used: {python_version_result.stdout.strip()}"
        )
    except Exception as e:
        print(f"DEBUG: Could not check Python version: {e}")

    res = subprocess.run(
        [str(Path(bugsinpy_path) / "framework/bin/bugsinpy-compile")],
        cwd=folder_path,
        env=minimal_env,
        text=True,
        capture_output=True,
    )
    if "This is not a checkout project folder" in res.stdout:
        raise Exception(
            f"unable to install {folder_path}. Try deleting the folder and rerunning the command"
        )


COVERAGE_TABLE_RE = re.compile(
    r"(Name\s+Stmts\s+Miss\s+Cover\s+Missing\n[-]+\n.*?\n[-]+\nTOTAL\s+.*)",
    re.DOTALL,
)


def coverage(bugsinpy_path: Path, work_dir: Path) -> str:
    res = subprocess.run(
        bugsinpy_path / "framework/bin/bugsinpy-coverage",
        cwd=work_dir,
        text=True,
        capture_output=True,
    )
    # print(res.stderr)
    match = COVERAGE_TABLE_RE.search(res.stdout)
    if not match:
        raise ValueError("Coverage table not found")
    return match.group(1)
