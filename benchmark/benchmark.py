from enum import Enum
from typing import List
from pathlib import Path
from dataclasses import dataclass
import subprocess
import json
from sklearn.metrics import classification_report


class Version(Enum):
    GOOD = 1
    BAD = 2


@dataclass
class FileData:
    rel_path: Path
    data: str


@dataclass
class BugData:
    bug_id: int
    version: Version
    callgraph: str
    diff: str
    description: str
    files: List[FileData]

    def __str__(self):
        file_strs = "".join(
            f"path: {f.rel_path}\ncontent:\n{f.data}\n\n" for f in self.files
        )
        return f"""
# Description

{self.description}

# Diff

{self.diff}

# Method Trace

{self.callgraph}

# Files

{file_strs}
"""


def benchmark(output_dir: str):
    output_path = Path(output_dir)
    y_true = []
    y_pred = []
    patches = get_patches()
    for bug_data in patches:
        ver_str = "good" if bug_data.version == Version.GOOD else "bad"
        bug_input = str(bug_data)
        input_file_path = output_path / f"{bug_data.bug_id}-{ver_str}-input.txt"
        input_file_path.parent.mkdir(parents=True, exist_ok=True)
        input_file_path.write_text(bug_input)

        result = subprocess.run(
            ["uv", "run", "main.py"],
            text=True,
            capture_output=True,
            cwd="/workspace/pyllmvalidate-cli",
            input=bug_input,
        )
        if result.returncode == 2:
            print(result.stdout)
            print(result.stderr)
            raise Exception(
                "There was a problem with the openai call. Did you include the OPEN_API_key in the .env"
            )

        stdout_file_path = output_path / f"{bug_data.bug_id}-{ver_str}-stdout.txt"
        stdout_file_path.write_text(result.stdout)

        exit_code = result.returncode
        current_y_pred = 1 if exit_code == 0 else 0
        current_y_true = 1 if bug_data.version == Version.GOOD else 0
        y_pred.append(current_y_pred)
        y_true.append(current_y_true)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["bad", "good"],
        zero_division=0,
    )
    report_path = output_path / "classification-report.txt"
    report_path.write_text(report)

    confusion_matrix_path = output_path / "confusion-matrx.json"
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with open(confusion_matrix_path, "w") as f:
        json.dump({"y_true": y_true, "y_pred": y_pred}, f)


def get_patches() -> List[BugData]:
    output_dir = Path("output")
    patches: List[BugData] = []
    for path in output_dir.iterdir():
        if not path.is_dir() or not path.name.isdigit():
            continue
        good_bug_data = get_bug_data(
            int(path.name), path, Path(f"descriptions/{path.name}.txt"), Version.GOOD
        )
        patches.append(good_bug_data)
        bad_bug_data = get_bug_data(
            int(path.name), path, Path(f"descriptions/{path.name}.txt"), Version.BAD
        )
        patches.append(bad_bug_data)
    return patches


def get_bug_data(
    bug_id: int, bug_dir: Path, bug_description_path: Path, version: Version
) -> BugData:
    ver = "good" if version.value == 1 else "bad"
    callgraph = (bug_dir / f"{ver}_callgraph.txt").read_text()
    diff = (bug_dir / f"{ver}_patch.txt").read_text()
    description = bug_description_path.read_text()

    files: List[FileData] = []
    for path in (bug_dir / ver).iterdir():
        content = path.read_text()
        files.append(FileData(rel_path=str(path), data=content))

    return BugData(
        bug_id=bug_id,
        version=version,
        callgraph=callgraph,
        diff=diff,
        description=description,
        files=files,
    )


def _run_all():
    pass


def run():
    pass
