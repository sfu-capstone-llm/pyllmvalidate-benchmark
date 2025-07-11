import os
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KDY")


def gen_bad_file(bad_diff: str, original_buggy_file: str) -> str:
    print("creating steps...")
    steps = _convert_diff_to_steps(bad_diff)
    print(steps)

    print("applying to file")
    modified_file = _apply_steps_to_file(steps, original_buggy_file)
    print("applied steps to original file")
    return modified_file


def _convert_diff_to_steps(diff_content: str) -> str:
    system_prompt = """
# Identity

You are a program that converts a diff file into a simple, numbered list of step-by-step instructions for a programmer to follow.

# Instructions

* Each step must clearly state the line number, the action (delete or insert), and the exact code content
* Use the hunk headers (e.g., `@@ -A,B +C,D @@`) to determine the starting line number for changes in the original file
* Represent a line change as a "delete" followed by an "insert"
* Your output must ONLY be the numbered list of instructions
"""

    user_prompt = f"""
Please convert the following diff into a numbered list of instructions:

{diff_content}
"""

    client = OpenAI(api_key=OPENAI_API_KEY)
    res = client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return res.output_text


def _apply_steps_to_file(steps: str, original_content: str) -> str:
    system_prompt = """
# Identity

You are an expert programmer. You will be given the full source code of a file and a numbered list of instructions to modify it.

# Instructions

* Follow the instructions provided in the user's prompt precisely and apply the changes to the file
* Your output must be ONLY the complete, modified file content
* Do not add any comments, markdown, or explanation
"""

    user_prompt = f"""
Instructions to follow:
{steps}
Original File Content:
{original_content}
"""

    client = OpenAI(api_key=OPENAI_API_KEY)
    res = client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return res.output_text


def create_bad_diff(good_diff: str) -> str | None:
    # system_prompt = """
    # You are a program that takes a correct patch that fixes a bug and outputs a incorrect patch with a buggy fix.

    # Rules:
    # - The output must be a valid diff in patch format (usable with `patch`).
    # - The output must preserve diff metadata (headers, offsets, index lines).
    # - Only modify the lines with '-' and '+'.
    # - Keep diff offsets (`@@ -X,Y +Z,W @@`) accurate after changes.
    # - Do NOT include any markdown, comments, or explanation—only the raw diff.
    # - Do NOT add or remove lines outside the modified lines.
    # - The output will be parsed and validated automatically.
    # """
    system_prompt = """
    # Identity

    You are a program that takes a correct patch that fixes a bug and outputs a incorrect patch with a buggy fix.

    # Instructions
    * The output must be a valid diff in patch format (usable with `patch`)
    * The output must preserve diff metadata (headers, offsets, index lines)
    * Only modify the lines with '-' and '+'
    * Ensure the diff hunk ranges (`@@ -X,Y +Z,W @@`) are accurate after changes
    * Do NOT include any markdown, comments, or explanation—only the raw diff.
    * The output will be parsed and validated automatically.
    """
    # system_prompt = """
    # You are a program that takes a correct patch that fixes a bug and outputs a incorrect patch with a buggy fix.

    # Rules:
    # - The output must be a valid diff in patch format (usable with `patch`).
    # - The output must preserve diff metadata (headers, offsets, index lines).
    # - Only modify the lines with '-' and '+'.
    # - Keep diff hunk ranges (`@@ -X,Y +Z,W @@`) are accurate after changes.
    # - To have a valid diff, modify the add and removed lines instead of changing the offset numbers
    # - Do NOT include any markdown, comments, or explanation—only the raw diff.
    # - The output will be parsed and validated automatically.
    # """

    client = OpenAI(
        # base_url="http://host.docker.internal:1234/v1",
        api_key=OPENAI_API_KEY
    )
    res = client.responses.create(
        model="gpt-4.1-nano",
        input=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": good_diff},
        ],
        temperature=0.3,
    )
    return res.output_text
