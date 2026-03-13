import json
from pathlib import Path

results = []

with open("results.jsonl") as f:
    for line in f:
        results.append(json.loads(line))

for r in results:
    try:
        parsed_output = json.loads(r["output"].removeprefix("```json").removesuffix("```"))
    except json.decoder.JSONDecodeError:
        continue
    label = "yes" if "/val/fire/" in r["image"] else "no"
    print(label, parsed_output["recommended_action"], r["image"], sep="\t")
