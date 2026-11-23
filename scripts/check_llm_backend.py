#!/usr/bin/env python
"""Send one short request to an LLM backend to check the API key and routing.

Example:
    BIPOP_VERBOSE=1 python scripts/check_llm_backend.py --model_name deepseek-r1
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bipop.agents import VALID_MODELS  # noqa: E402
from bipop.llm.client import build_client, chat  # noqa: E402
from bipop.llm.llm_pddl import extract_code_string  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test of an LLM backend.")
    parser.add_argument("--model_name", choices=VALID_MODELS, default="deepseek-r1")
    args = parser.parse_args()

    messages = [
        {
            "role": "user",
            "content": "Reply with a single short Python code block that prints 'hello'. No explanation.",
        }
    ]
    try:
        client = build_client(args.model_name)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Sending a test request to {args.model_name} ...")
    start = time.time()
    content = chat(client, args.model_name, messages, temperature=0.0)
    print(f"\n--- Response received in {time.time() - start:.1f}s ({len(content)} chars) ---")
    print(content[:2000])

    code = extract_code_string(content)
    print("\n--- extracted code ---")
    print(code if code is not None else "<no python code block found>")

    if not content.strip():
        print("\nFAIL: empty response.")
        return 1
    print("\nPASS: got a non-empty response.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
