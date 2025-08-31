import os
import sys


def _load_prompt() -> str:
    path = os.path.join(os.path.dirname(__file__), "prompt.md")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[prompts] WARNING: Failed to read prompt file at: {path}", file=sys.stderr)
        print(f"[prompts] Reason: {e}", file=sys.stderr)
        print("[prompts] Exiting early with code 0 (no prompt loaded)", file=sys.stderr)
        sys.exit(0)


BUG_CATEGORIZATION_PROMPT = _load_prompt()
