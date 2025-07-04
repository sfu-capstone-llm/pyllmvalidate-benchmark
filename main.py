import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import List, TypedDict

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KDY")

from openai import OpenAI

BugInfo = TypedDict(
    "BugInfo", {"bug_number": int, "correct_patch": str, "args": List[str]}
)


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

        shutil.copy(
            good_install_dir / "black" / "black.py", bug_output_dir / "good_black.py"
        )
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

        bad_install_dir = base_install_dir / "bad"
        bad_config = configure_and_setup(info, bad_install_dir, base_output_dir, False)

        bad_diff = ""
        if not os.path.exists(bug_output_dir / "bad_patch.txt"):
            bad_diff = create_bad_diff(info["correct_patch"])
            if bad_diff is None:
                raise Exception("Could not generate bad_diff from good diff", info)
        else:
            print("Using previous bad_patch.txt")
            with open(bug_output_dir / "bad_patch.txt") as f:
                bad_diff = f.read()

        output_bad_diff(bad_diff, bug_output_dir)

        # Copy the BUGGY version first (since we want to create a "bad fix" from the buggy version)
        shutil.copy(
            bad_install_dir / "black" / "black.py", bug_output_dir / "bad_black.py"
        )
        apply_bad_patch_with_git_diff(bug_output_dir / "bad_patch.txt", bug_output_dir)

        if not os.path.exists(bug_output_dir / "bad_callgraph.txt"):
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
                    "--file-path",
                    str(bug_output_dir / "bad_black.py"),
                ]
                + info["args"],
                cwd=bad_config["temp_dir"] + "/black",
                env=bad_config["env"],
                text=True,
            )


def output_good_diff(diff: str, output_dir: Path):
    with open(output_dir / "good_patch.txt", "w") as f:
        f.write(diff)


def output_bad_diff(diff: str, output_dir: Path):
    with open(output_dir / "bad_patch.txt", "w") as f:
        f.write(diff)


def create_bad_diff(good_diff: str) -> str | None:
    system_prompt = """
    - Receive a GOOD patch that fixes a bug
    - Create a BAD patch that with a buggy fix
    - The output of this call will be passed directly to patch utility tool so do NOT include markdown or aditional text
    - Ensure the diff patch is the correct format
    - Ensure the ALL the diff offset and are correct (@@ -X,Y +Z,W @@)
    - Verify the offset numbers are correct for each patch

    Input: Good patch
    Output: Bad patch (same structure, only the actual changes are different)
    """

    client = OpenAI(
        # base_url="http://host.docker.internal:1234/v1",
        api_key=OPENAI_API_KEY
    )
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": good_diff},
        ],
        temperature=0.3,
    )
    return completion.choices[0].message.content


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


def apply_bad_patch_with_git_diff(bad_patch_path: Path, bug_output_dir: Path):
    bad_black_path = bug_output_dir / "bad_black.py"

    # Check if patch file exists and is not empty
    if not bad_patch_path.exists():
        raise Exception(f"Bad patch file does not exist: {bad_patch_path}")

    # Read and validate patch content
    with open(bad_patch_path, "r", encoding="utf-8") as f:
        patch_content = f.read()

    if not patch_content.strip():
        raise Exception(f"Bad patch file is empty: {bad_patch_path}")

    print(f"DEBUG: Patch file size: {len(patch_content)} characters")
    print(f"DEBUG: Patch ends with newline: {patch_content.endswith('\n')}")

    # Ensure patch ends with newline
    if not patch_content.endswith("\n"):
        print("WARNING: Patch doesn't end with newline, adding one")
        with open(bad_patch_path, "w", encoding="utf-8") as f:
            f.write(patch_content + "\n")

    # Apply patch with better error handling
    result = subprocess.run(
        ["patch", str(bad_black_path), str(bad_patch_path)],
        cwd=str(bug_output_dir),
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"ERROR: Patch failed with return code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")

        # Show patch content for debugging
        print("DEBUG: Patch content preview:")
        print(
            patch_content[:500] + "..." if len(patch_content) > 500 else patch_content
        )

        raise Exception(f"Failed to apply patch: {result.stderr}")

    print("SUCCESS: Patch applied successfully")
    print(f"Patch output: {result.stdout}")


if __name__ == "__main__":
    main()
