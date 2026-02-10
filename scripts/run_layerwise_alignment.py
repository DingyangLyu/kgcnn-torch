#!/usr/bin/env python3
"""Run all implemented layerwise alignment scripts from manifest."""
import json
import os
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "scripts", "layerwise_alignment_manifest.json")


def main():
    with open(MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    items = manifest["models"]
    implemented = [x for x in items if x.get("status") == "implemented" and "script" in x]
    pending = [x for x in items if x.get("status") != "implemented"]

    print(f"Total models in manifest: {len(items)}")
    print(f"Implemented aligners: {len(implemented)}")
    print(f"Pending aligners: {len(pending)}")

    failures = []
    for item in implemented:
        name = item["name"]
        script = os.path.join(ROOT, item["script"])
        print(f"\n=== Running {name}: {item['script']} ===")
        env = dict(os.environ)
        env.setdefault("KERAS_BACKEND", "torch")
        proc = subprocess.run([sys.executable, script], cwd=ROOT, env=env, text=True)
        if proc.returncode != 0:
            failures.append(name)

    print("\n=== Summary ===")
    if failures:
        print(f"FAILED aligners: {failures}")
        raise SystemExit(1)
    print("All implemented aligners passed.")

    if pending:
        pending_names = [x["name"] for x in pending]
        print(f"Pending ({len(pending_names)}): {', '.join(pending_names)}")


if __name__ == "__main__":
    main()
