import os
import json

total_parts = 5

method = "middle"
mask = "diglets"

original_data = "S:/HRCode/data/RuTermEval/track2/flat/pure/train.json" 
damaged_data = "S:/HRCode/data/RuTermEval/track2/flat/damage/" + method + "/" + mask + "/train.json"
parted_data = "S:/HRCode/data/RuTermEval/track2/flat/parted/" + method + "/" + mask

with open(original_data, "r", encoding = "UTF-8") as origfile:
    origlines = origfile.readlines()
with open(damaged_data, "r", encoding = "UTF-8") as dmgfile:
    dmglines = dmgfile.readlines()

part_delims = [int(len(origlines) * part / total_parts) for part in range(total_parts)]
part_delims.append(len(origlines))
part_paired_delims = [(p1, p2 - 1) for p1, p2 in zip(part_delims, part_delims[1:])]

for part in range(total_parts):
    train_lines = [d for d_idx, d in enumerate(dmglines) if d_idx < part_paired_delims[part][0] or d_idx > part_paired_delims[part][1]]
    predict_lines = [o for o_idx, o in enumerate(origlines) if o_idx >= part_paired_delims[part][0] and o_idx <= part_paired_delims[part][1]]
    if not os.path.exists(parted_data + "/part" + str(part + 1)):
        os.makedirs(parted_data + "/part" + str(part + 1))
    with open(parted_data + "/part" + str(part + 1) + "/train.json", "w", encoding = "UTF-8") as partfile:
        partfile.writelines(train_lines)
    with open(parted_data + "/part" + str(part + 1) + "/dev.json", "w", encoding = "UTF-8") as partfile:
        partfile.writelines(predict_lines)