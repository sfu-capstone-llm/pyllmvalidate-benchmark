import os
import numbers
import re
import shlex
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import List, TypedDict, Dict, Tuple
from dataclasses import dataclass


from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KDY")


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


def get_files_from_patch(patch_content: str) -> List[str]:
    """Extracts file paths from a git diff patch."""
    # Matches lines like '--- a/src/black/__init__.py'
    # and captures 'src/black/__init__.py'
    return re.findall(r"^\-\-\- a/(.+)$", patch_content, re.MULTILINE)


def main():
    args = parse_args()
    # Create output directory if it doesn't exist
    base_output_dir = Path("output").resolve()  # Use absolute path
    base_output_dir.mkdir(exist_ok=True)

    bugs_dir = Path("/workspace/BugsInPy/projects/black/bugs")
    bugs_info = get_bug_info(bugs_dir)

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
            "/workspace/BugsInPy/framework/bin/temp/black-" + str(info["bug_number"])
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

        output_good_diff(info["correct_patch"], bug_output_dir)

        # run on good
        good_callgraph = Path(bug_output_dir / "good_callgraph.txt")
        if not good_callgraph.exists() or len(good_callgraph.read_text()) == 0:
            print(
                f"Running tracer for bug {info['bug_number']} for good_callgraph.txt..."
            )
            result = subprocess.run(
                [
                    good_config["python_path"],
                    good_config["tracer_script_path"],
                    "--bug-number",
                    str(info["bug_number"]),
                    "--version",
                    "good",
                ]
                + info["args"],
                cwd=good_config["temp_dir"] + "/black",
                env=good_config["env"],
                text=True,
            )

            print(f"Call graph written to: {bug_output_dir}")
            print(f"Return code: {result.returncode}")

        bad_install_dir = base_install_dir / "bad"
        bad_config = configure_and_setup(info, bad_install_dir, base_output_dir, False)

        bad_diff = get_bad_diff(info["correct_patch"], bug_output_dir / "bad_patch.txt")

        output_bad_diff(bad_diff, bug_output_dir)

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

        bad_callgraph = Path(bug_output_dir / "bad_callgraph.txt")
        if not bad_callgraph.exists() or len(bad_callgraph.read_text()) == 0:
            print(
                f"Running tracer for bug {info['bug_number']} for bad_callgraph.txt..."
            )
            result = subprocess.run(
                [
                    bad_config["python_path"],
                    bad_config["tracer_script_path"],
                    "--bug-number",
                    str(info["bug_number"]),
                    "--version",
                    "bad",
                ]
                + info["args"],
                cwd=bad_config["temp_dir"] + "/black",
                env=bad_config["env"],
                text=True,
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--re-apply-diff", action="store_true")
    parser.add_argument("--single", action="store")
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


def output_good_diff(diff: str, output_dir: Path):
    with open(output_dir / "good_patch.txt", "w") as f:
        f.write(diff)


def output_bad_diff(diff: str, output_dir: Path):
    with open(output_dir / "bad_patch.txt", "w") as f:
        f.write(diff)


def create_bad_diff(good_diff: str) -> str | None:
    system_prompt = """
# Identity

You are a program that generates a buggy diff by mutating a correct diff. The diff can contain multiple files.
You will be given a correct diff that fixes a bug. Your job is to use the correct diff as a reference to create a bad diff, but make it look like it might be correct.

# Instructions

- Output a valid diff in unified diff format (usable with `patch`)
- Do NOT change metadata (headers, index lines, filenames, hunk positions)
- Try to fool a human reviewer—your change should look plausible but be incorrect
- The diff can contain multiple files so do not remove the headers for the other files
- Do not introduce or fix unrelated code
- Ensure the hunk headers (`@@ -X,Y +Z,W @@`) remain accurate based on line count
- Do not include any extra explanation, markdown, or comments—only output the raw diff
- Some things you can do are remove statements, set default values, remove function calls, return empty array (pick a mutation operator randomly and apply it)
    """
    client = OpenAI(
        # base_url="http://host.docker.internal:1234/v1",
        api_key=OPENAI_API_KEY
    )
    res = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": good_diff},
        ],
        temperature=0.3,
    )
    return res.output_text


def configure_and_setup(info, instal_dir: Path, output_dir: Path, isGood: bool):
    """Configure a bug environment and return necessary paths and environment."""
    bug_output_dir = output_dir / str(info["bug_number"])
    create_output_folder(bug_output_dir)

    if not os.path.exists(instal_dir / "black"):
        checkout_bug(
            instal_dir.__str__(), str(info["bug_number"]), "1" if isGood else "0"
        )

    if not os.path.exists(instal_dir.__str__() + "/black/env/bin/python3"):
        install(instal_dir.__str__() + "/black")

    tracer_script_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tracer.py"
    )
    python_path = os.path.join(
        instal_dir, "black/env/bin/python3"
    )  # proj always clones to /black :(

    black_project_dir = os.path.join(instal_dir, "black")

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


def checkout_bug(path: str, bug_number: str, v: str):
    print(
        f"Checking out bug {bug_number} ({'fixed' if v == '1' else 'buggy'} version)..."
    )
    subprocess.run(
        [
            "/workspace/BugsInPy/framework/bin/bugsinpy-checkout",
            "-p",
            "black",
            "-i",
            bug_number,
            "-w",
            path,
            "-v",
            v,  # 0 forbuggy version and 1 for fixed
        ]
    )


def install(folder_path: str):
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
        ["/workspace/BugsInPy/framework/bin/bugsinpy-compile"],
        cwd=folder_path,
        env=minimal_env,
        text=True,
        capture_output=True,
    )
    if "This is not a checkout project folder" in res.stdout:
        raise Exception(
            f"unable to install {folder_path}. Try deleting the folder and rerunning the command"
        )


def get_bug_info(path: Path) -> List[BugInfo]:
    if not path.exists():
        raise Exception("could not find path")

    bugs_info = []

    bug_folders = []
    for item in path.iterdir():
        if item.is_dir() and item.name.isdigit():
            bug_folders.append(item)

    bug_folders.sort(key=lambda x: int(x.name))

    for bug_folder in bug_folders:
        try:
            bug_data = process_bug_folder(bug_folder)
            bugs_info.append(bug_data)
        except Exception as e:
            print(f"Error processing {bug_folder}: {e}")

    return bugs_info


def extract_unittest_args(run_test_content: str) -> List[str]:
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


def process_bug_folder(bug_path: Path) -> BugInfo:
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
        args = extract_unittest_args(test_content)
    else:
        args = []

    return BugInfo(bug_number=bug_number, correct_patch=diff, args=args)


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
