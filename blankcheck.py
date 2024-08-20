import json

with open("pyg3.jsonl", "r", encoding="utf-8") as f:
    with open("pyg3_noblanks.jsonl", "w", encoding="utf-8") as g:
        for line in f:
            cutoff = None
            entry = json.loads(line)
            for idx, msg in enumerate(entry["conversations"]):
                # Check if any message is blank.
                if msg["value"].strip() == "":
                    cutoff = idx if msg["from"] == "human" else idx - 1
                    entry["conversations"] = entry["conversations"][:cutoff]
                    break
            # Check following conditions:
            # 1. If the conversation has less than three messages, don't write it.
            if len(entry["conversations"]) < 3 \
                and len(set([msg["from"] for msg in entry["conversations"]])) == 3:
                continue
            g.write(json.dumps(entry) + "\n")
