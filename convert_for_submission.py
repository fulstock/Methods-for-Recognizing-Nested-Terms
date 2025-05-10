import os
import json

datasets = ["./RuTermEval"]

export_path = "./tests/exports"
submissions_path = "./tests/submissions"

for root, dirs, files in os.walk(export_path):
    for file in files:
        if not os.path.exists(os.path.join(submissions_path, '\\'.join(root.split('\\')[1:]))):
            os.makedirs(os.path.join(submissions_path, '\\'.join(root.split('\\')[1:])))
        subf = open(os.path.join(submissions_path, '\\'.join(root.split('\\')[1:]), file), "w", encoding = "UTF-8")
        with open(os.path.join(root, file), "r", encoding = "UTF-8") as oldf:
            for line in oldf:
                d = json.loads(line)
                nd = {
                    "entities" : [[e[0], e[1], e[2]] for e in d["pred_ner"]],
                    "id" : d["id"],
                    "text" : d["text"]
                }
                subf.write(json.dumps(nd, ensure_ascii = False) + "\n")
        subf.close()