import json
import os

tags = set()

def scores(true, pred):

    tp = len([p for p in pred if p in true])
    fp = len([p for p in pred if p not in true])
    fn = len([t for t in true if t not in pred])

    return tp, fp, fn

def f1_score(tp, fp, fn):

    return (2 * tp) / (2 * tp + fp + fn + 1e-10)

def count_metrics(filename):

    docs = []

    with open(filename, "r", encoding = "UTF-8") as file:
        for line in file:
            docs.append(json.loads(line))

    full_scores = []
    inner_scores = []
    outer_scores = []
    tagged_scores = {}

    for doc in docs:
        
        gold_entities = doc["gold_ner"]
        pred_entities = doc["pred_ner"]

        pred_entities = [[p[0], p[1], p[2], p[3]] for p in pred_entities]

        inner_gold_entities = [v for v in gold_entities if len([s for s in gold_entities if v is not s and s[0] <= v[0] and v[1] <= s[1]]) > 0]
        outer_gold_entities = [v for v in gold_entities if v not in inner_gold_entities]

        inner_pred_entities = [v for v in pred_entities if len([s for s in pred_entities if v is not s and s[0] <= v[0] and v[1] <= s[1]]) > 0]
        outer_pred_entities = [v for v in pred_entities if v not in inner_pred_entities]

        tags.update([g[2] for g in gold_entities])

        full_scores.append(scores(gold_entities, pred_entities))
        inner_scores.append(scores(inner_gold_entities, inner_pred_entities))
        outer_scores.append(scores(outer_gold_entities, outer_pred_entities))
        for tag in tags:
            tagged_gold_entities = [v for v in gold_entities if v[2] == tag]
            tagged_pred_entities = [v for v in pred_entities if v[2] == tag]
            if tag not in tagged_scores.keys():
                tagged_scores[tag] = [scores(tagged_gold_entities, tagged_pred_entities)]
            else:
                tagged_scores[tag].append(scores(tagged_gold_entities, tagged_pred_entities))
        
    full_scores_tp = sum([s[0] for s in full_scores])
    full_scores_fp = sum([s[1] for s in full_scores])
    full_scores_fn = sum([s[2] for s in full_scores])

    inner_scores_tp = sum([s[0] for s in inner_scores])
    inner_scores_fp = sum([s[1] for s in inner_scores])
    inner_scores_fn = sum([s[2] for s in inner_scores])

    outer_scores_tp = sum([s[0] for s in outer_scores])
    outer_scores_fp = sum([s[1] for s in outer_scores])
    outer_scores_fn = sum([s[2] for s in outer_scores])

    full_micro_f1 = f1_score(full_scores_tp, full_scores_fp, full_scores_fn)
    inner_micro_f1 = f1_score(inner_scores_tp, inner_scores_fp, inner_scores_fn)
    outer_micro_f1 = f1_score(outer_scores_tp, outer_scores_fp, outer_scores_fn)

    # full_f1_scores = [f1_score(tp, fp, fn) for tp, fp, fn in 
    #         zip([s[0] for s in full_scores], [s[1] for s in full_scores], [s[2] for s in full_scores])
    #     ]
    # inner_f1_scores = [f1_score(tp, fp, fn) for tp, fp, fn in 
    #         zip([s[0] for s in inner_scores], [s[1] for s in inner_scores], [s[2] for s in inner_scores])
    #     ]
    # outer_f1_scores = [f1_score(tp, fp, fn) for tp, fp, fn in 
    #         zip([s[0] for s in outer_scores], [s[1] for s in outer_scores], [s[2] for s in outer_scores])
    #     ]

    # print(full_f1_scores)

    # full_macro_f1 = sum(full_f1_scores) / len(full_f1_scores)
    # inner_macro_f1 = sum(inner_f1_scores) / len(inner_f1_scores)
    # outer_macro_f1 = sum(outer_f1_scores) / len(outer_f1_scores)

    full_f1_scores = [[full] for tag in tags]

    return full_macro_f1, full_micro_f1, inner_macro_f1, inner_micro_f1, outer_macro_f1, outer_micro_f1


for root, dirs, files in os.walk("./exports"):
    for file in files:
        fma, fmi, ima, imi, oma, omi = count_metrics(os.path.join(root, file))
        print("======================")
        print("For file", os.path.join(root, file))
        print("Micro f1 and Macro f1:")
        print("Overall score:", fmi, fma)
        print("Inner score:", imi, ima)
        print("Outer score:", omi, oma)
        print("======================")