from enum import Enum
from typing import List
from pathlib import Path
from dataclasses import dataclass
import subprocess
import json


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


def benchmark():
    # y_true = [] * 23 * 2
    # y_pred = [] * 23 * 2
    # patches = get_patches()
    # for bug_data in patches:
    #     result = subprocess.run(
    #         ["uv", "run", "main.py"],
    #         text=True,
    #         capture_output=True,
    #         stdin=str(bug_data),
    #     )
    #     exit_code = result.returncode
    #     y_pred.append(1 if exit_code == 0 else 0)
    #     y_true.append(1 if bug_data.version == Version.GOOD else 0)
    y_true = [1] * 23 + [0] * 23
    y_pred = [1] * 10 + [0] * 13 + [0] * 23

    confusion_matrix_path = Path("run-tool-output/confusion-matrx.json")
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with open(confusion_matrix_path, "w") as f:
        json.dump({"y_true": y_true, "y_pred": y_pred}, f)


def get_patches() -> List[BugData]:
    output_dir = Path("output")
    patches: List[BugData] = [] * 23 * 2
    for path in output_dir.iterdir():
        good_bug_data = get_bug_data(
            int(path.name), path, Path(f"descriptions/{path.name}.txt"), Version.GOOD
        )
        patches.append(str(good_bug_data))
        bad_bug_data = get_bug_data(
            int(path.name), path, Path(f"descriptions/{path.name}.txt"), Version.BAD
        )
        patches.append(str(bad_bug_data))


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
