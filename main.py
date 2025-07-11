import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List, TypedDict, Dict, Tuple
from gen import gen_bad_file

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KDY")

from gen.gen import create_bad_diff

BugInfo = TypedDict(
    "BugInfo", {"bug_number": int, "correct_patch": str, "args": List[str]}
)


class DiffInfo(TypedDict):
    rel_file_path: str
    diff_content: str


def get_files_from_patch(patch_content: str) -> List[str]:
    """Extracts file paths from a git diff patch."""
    # Matches lines like '--- a/src/black/__init__.py'
    # and captures 'src/black/__init__.py'
    return re.findall(r"^\-\-\- a/(.+)$", patch_content, re.MULTILINE)


def main():
    # Create output directory if it doesn't exist
    base_output_dir = Path("output").resolve()  # Use absolute path
    base_output_dir.mkdir(exist_ok=True)

    bugs_dir = Path("/workspace/BugsInPy/projects/black/bugs")
    bugs_info = get_bug_info(bugs_dir)

    for info in bugs_info:
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
        if not os.path.exists(bug_output_dir / "good_callgraph.txt"):
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

        # For each good file, prompt the ai with the good file and the diff for that file
        bad_install_dir = base_install_dir / "bad"
        good_project_dir = good_install_dir / "black"
        diffs_info = process_diff(
            info["bug_number"], info["correct_patch"], good_project_dir
        )

        bad_diff = get_bad_diff(info["correct_patch"], bug_output_dir / "bad_patch.txt")
        bad_diffs_info = process_diff(
            info["bug_number"], bad_diff, bad_install_dir / "black"
        )
        output_bad_diff(bad_diff, bug_output_dir)

        for key, good_diff_info in diffs_info.items():
            bug_number, file_path_str = key
            file_path = Path(file_path_str)
            rel_file_path = good_diff_info["rel_file_path"]

            original_buggy_file_path = bad_install_dir / "black" / rel_file_path
            original_buggy_content = original_buggy_file_path.read_text()

            bad_diff_info = bad_diffs_info.get(key)

            print("gen bad file...")
            bad_file_str = gen_bad_file(
                bad_diff_info["diff_content"], original_buggy_content
            )
            print("finished gen bad file")

            # Create bad file from str
            bad_output_file = bug_output_dir / "bad" / file_path.name
            bad_output_file.parent.mkdir(exist_ok=True, parents=True)
            with open(bad_output_file, "w") as f:
                f.write(bad_file_str)

            # apply bad file to bad proj
            shutil.copy(
                bad_output_file,
                bad_install_dir / "black" / rel_file_path,
            )

        break

        bad_config = configure_and_setup(info, bad_install_dir, base_output_dir, False)

        output_bad_diff(bad_diff, bug_output_dir)

        # Apply the bad patch to the buggy version
        apply_patch(bug_output_dir / "bad_patch.txt", bad_install_dir / "black")

        print("DEBUG: Getting files from bad patch...")
        bad_patch_files = get_files_from_patch(bad_diff)
        print(f"DEBUG: Found {len(bad_patch_files)} files in bad patch.")

        print("DEBUG: Copying patched files...")
        bad_output_path = bug_output_dir / "bad"
        bad_output_path.mkdir(exist_ok=True)

        for file in bad_patch_files:
            src_file = bad_install_dir / "black" / file
            dest_file = bad_output_path / Path(file).name
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_file, dest_file)
        print("DEBUG: Finished copying patched files.")

        if not os.path.exists(bug_output_dir / "bad_callgraph.txt"):
            print(
                f"Running tracer for bug {info['bug_number']} for bad_callgraph.txt..."
            )
            try:
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
                    capture_output=True,
                    timeout=300,  # 5-minute timeout
                )
                print(f"Tracer finished with return code: {result.returncode}")
                if result.returncode != 0:
                    print(f"Tracer STDOUT: {result.stdout}")
                    print(f"Tracer STDERR: {result.stderr}")
            except subprocess.TimeoutExpired as e:
                print(f"ERROR: Tracer for bug {info['bug_number']} timed out.")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")

        # Add a break to only process one bug for now
        # break


def process_diff(
    bug_number: str, diff: str, base_install_path: Path
) -> Dict[Tuple[str, str], DiffInfo]:
    if not diff.strip():
        return []

    diffs_info: Dict[Tuple[str, str], DiffInfo] = {}
    diff_parts: List[str] = []

    if not diff.startswith("diff --git"):
        raise Exception("diff does not have diff --git in it")

    raw_parts = diff.split("diff --git ")[1:]
    diff_parts = ["diff --git " + part for part in raw_parts]

    for diff_content in diff_parts:
        # Matches lines like '--- a/src/black/__init__.py'
        match = re.search(r"^\-\-\- a/(.+)$", diff_content, re.MULTILINE)
        if not match:
            raise Exception("couldn't not get name from diff")

        file_name = match.group(1).strip()
        diffs_info[(str(bug_number), file_name)] = {
            "rel_file_path": file_name,
            "diff_content": diff_content,
        }

    return diffs_info


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


def configure_and_setup(info, instal_dir: Path, output_dir: Path, isGood: bool):
    """Configure a bug environment and return necessary paths and environment."""
    bug_output_dir = output_dir / str(info["bug_number"])
    create_output_folder(bug_output_dir)

    if not os.path.exists(instal_dir):
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

    # Create minimal environment with just PATH and HOME
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

    print("DEBUG: Installing with minimal environment")

    subprocess.run(
        ["/workspace/BugsInPy/framework/bin/bugsinpy-compile"],
        cwd=folder_path,
        env=minimal_env,
        text=True,
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

    # The -p1 option strips the 'a/' and 'b/' prefixes from file paths in the patch
    # The -N option ignores patches that seem to be reversed or already applied.
    result = subprocess.run(
        ["patch", "-p1", "-N", "-i", str(patch_path)],
        cwd=str(target_dir),
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        print(f"SUCCESS: Patch {patch_path} applied successfully to {target_dir}")
    elif "Reversed (or previously applied) patch detected!" in result.stdout:
        print(
            f"INFO: Patch at {patch_path} was already applied or is reversed. Continuing."
        )
    else:
        print(f"ERROR: Patch failed for {patch_path} in {target_dir}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise Exception(f"Failed to apply patch: {result.stderr}")


if __name__ == "__main__":
    main()
