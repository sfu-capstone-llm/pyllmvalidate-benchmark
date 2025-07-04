import ast
import os
import runpy
import sys
from collections import defaultdict
from pathlib import Path
from typing import List


def main():
    print("Running with:", sys.executable)

    # Extract bug number and version from sys.argv
    bug_number = None
    version = None
    file_path = None
    filtered_argv = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--bug-number" and i + 1 < len(sys.argv):
            bug_number = sys.argv[i + 1]
            i += 2
            continue
        elif sys.argv[i] == "--version" and i + 1 < len(sys.argv):
            version = sys.argv[i + 1]
            i += 2
            continue
        elif sys.argv[i] == "--file-path" and i + 1 < len(sys.argv):
            file_path = sys.argv[i + 1]
            i += 2
            continue
        else:
            filtered_argv.append(sys.argv[i])
            i += 1

    sys.argv = filtered_argv
    print(f"DEBUG: bug_number = {bug_number}")
    print(f"DEBUG: version = {version}")
    print(f"DEBUG: file_path = {file_path}")
    print(f"DEBUG: filtered sys.argv = {sys.argv}")

    if len(sys.argv) < 2:
        print("DEBUG: Not enough arguments, returning early")
        return

    entry = sys.argv[1]  # The module name (e.g., 'unittest')
    sys.argv = sys.argv[1:]  # Remove script path, keep module and args

    print(f"DEBUG: entry = {entry}")
    print("DEBUG: About to call build_call_graph")

    call_graph_result = build_call_graph(entry, "./", ["tests.test_black"], file_path)

    print(f"DEBUG: call_graph_result length = {len(call_graph_result)}")

    # Construct output file path based on script location
    script_dir = Path(__file__).parent.resolve()
    output_dir = script_dir / "output"
    bug_output_dir = output_dir / str(bug_number)
    bug_output_dir.mkdir(parents=True, exist_ok=True)

    # Use version to determine output filename
    if version == "good":
        output_file = bug_output_dir / "good_callgraph.txt"
    elif version == "bad":
        output_file = bug_output_dir / "bad_callgraph.txt"
    else:
        raise ValueError(
            f"Invalid or missing version parameter: {version}. Must be 'good' or 'bad'."
        )

    print(f"DEBUG: Writing to {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(call_graph_result)
    print(f"Call graph written to {output_file}")


def build_call_graph(
    entry: str, project_dir: str, exlude_list: List[str], file_path: str = None
):
    call_graph = defaultdict(set)
    call_stack = []

    project_functions = set(get_all_function_names_from_project(project_dir))

    def is_project_function(func_name):
        return func_name in project_functions

    def tracefunc(frame, event, arg):
        if event == "call":
            code = frame.f_code
            func_name = code.co_name
            module = frame.f_globals.get("__name__", "")

            # Get the actual module name from the entry point
            if module == "__main__":
                module = entry

            full_name = f"{module}.{func_name}"

            for item in exlude_list:
                if full_name.startswith(item):
                    return tracefunc

            if call_stack:
                caller = call_stack[-1]
                if is_project_function(caller):
                    call_graph[caller].add(full_name)

            call_stack.append(full_name)

        elif event == "return" and call_stack:
            call_stack.pop()

        return tracefunc

    sys.settrace(tracefunc)
    try:
        if file_path:
            # Import the bad_black.py file and replace the black module with it
            import importlib.util

            spec = importlib.util.spec_from_file_location("black", file_path)
            bad_black_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bad_black_module)

            # Replace black module in sys.modules so unittest imports the bad version
            sys.modules["black"] = bad_black_module

        runpy.run_module(entry, run_name="__main__")
    except SystemExit:
        # unittest calls sys.exit(), catch it to continue execution
        pass
    finally:
        sys.settrace(None)

    call_graph_str = ""
    for caller, callees in call_graph.items():
        for callee in callees:
            call_graph_str = f"{call_graph_str}\n{caller}->{callee}"

    return call_graph_str.lstrip("\n")  # Remove leading newline


def get_all_function_names_from_project(project_dir="."):
    defined_funcs = set()

    for dirpath, dirnames, filenames in os.walk(project_dir):
        # Skip unwanted folders
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".")
            and d not in ("__pycache__", "venv", ".venv", "env")
        ]

        for filename in filenames:
            if filename.endswith(".py"):
                path = os.path.join(dirpath, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read(), filename=path)
                # Skip python 2 files
                except Exception:
                    continue

                # Get module name from file path
                rel_path = os.path.relpath(path, project_dir)
                module_name = os.path.splitext(rel_path)[0].replace(os.sep, ".")

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        defined_funcs.add(f"{module_name}.{node.name}")

    return defined_funcs


if __name__ == "__main__":
    main()
