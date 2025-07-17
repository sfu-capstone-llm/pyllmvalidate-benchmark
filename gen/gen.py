import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


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
        model="gpt-4.1-2025-04-14",
        input=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": good_diff},
        ],
        temperature=0.3,
    )
    return res.output_text
