import json
import os
import random
import re

from nltk.data import load
from nltk.tokenize import NLTKWordTokenizer

## Comment out after first download. 
import nltk
nltk.download('punkt')
##

from tqdm.auto import tqdm

train_data = []
dev_data = []
test_data = []

# =========================

track = "track2"

# =========================


if track in ["track2", "track3"]:
    train_data_file = "train_t23_v1.jsonl"
else:
    train_data_file = "train_t1_v1.jsonl"

if track in ["track1", "track2"]:
    dev_data_file = "test1_t12_full_v2.jsonl"
    test_data_file = "test2_t12_v2.jsonl"
else:
    dev_data_file = "test1_t3_full_v2.jsonl"
    test_data_file = "test2_t3_v2.jsonl"

with open(os.path.join("S:/HRCode/data/RuTermEval", track, train_data_file), "r", encoding = "UTF-8") as train_file:
    for line in train_file:
        train_data.append(json.loads(line))

with open(os.path.join("S:/HRCode/data/RuTermEval", track, dev_data_file), "r", encoding = "UTF-8") as dev_file:
    for line in dev_file:
        dev_data.append(json.loads(line))

with open(os.path.join("S:/HRCode/data/RuTermEval", track, test_data_file), "r", encoding = "UTF-8") as test_file:
    for line in test_file:
        test_data.append(json.loads(line))

ru_tokenizer = load("tokenizers/punkt/russian.pickle") 
word_tokenizer = NLTKWordTokenizer()

# tags = ["COMMON", "NOMEN", "SPECIFIC"]

entity_count = 0
inc_count = 0
inclusions_by_type = dict()

for d_idx, data in enumerate(tqdm(train_data)):

    tid = data["id"]
    txtdata = data["text"]
    labels = data["label"]

    entity_types = []
    entity_start_chars = []
    entity_end_chars = []

    for label in labels:

        if track != "track1":
            entity_type = label[2].upper()
        else:
            entity_type = "ANY"
        start_char = label[0]
        end_char = label[1]

        entity_types.append(entity_type)
        entity_start_chars.append(start_char)
        entity_end_chars.append(end_char)



    file_entities = list(zip(entity_start_chars, entity_end_chars, entity_types))

    outermost_entities = set([(su, eu, tu) for su, eu, tu in file_entities if len(list(filter(lambda ea: ea[0] < su and ea[1] >= eu or ea[0] <= su and ea[1] > eu, file_entities))) == 0])
    outermost_entities = sorted(list(outermost_entities), key = lambda x: x[0])

    entities = outermost_entities.copy()

    inclusion_spans = set()
    inclusion_span_types = dict()

    for e_idx, e in enumerate(outermost_entities):
        entities_with_this_inclusion = [v for v in outermost_entities if (v[1] < e[0] or e[1] < v[0]) \
            and txtdata[e[0] : e[1]] in txtdata[v[0] : v[1]] and txtdata[e[0] : e[1]] != txtdata[v[0] : v[1]]]
        if len(entities_with_this_inclusion) > 0:
            inclusion_spans.add(txtdata[e[0] : e[1]])
            inclusion_span_types[txtdata[e[0] : e[1]]] = e[2]
    
    inclusions = []
    for inclusion_span in inclusion_spans:
        probable_inclusions = [(m.start(), m.end()) for m in re.finditer(inclusion_span, txtdata)]
        for p in probable_inclusions:
            for o in outermost_entities:
                if (o[0] < p[0] and p[1] <= o[1]) or (o[0] <= p[0] and p[1] < o[1]):
                    inclusions.append((p[0], p[1], inclusion_span_types[inclusion_span]))
                    if inclusion_span_types[inclusion_span] in inclusions_by_type.keys():
                        inclusions_by_type[inclusion_span_types[inclusion_span]] += 1
                    else:
                        inclusions_by_type[inclusion_span_types[inclusion_span]] = 1
                    
    # print([(s, e, t, txtdata[s : e]) for s, e, t in file_entities])
    entities.extend(inclusions)
    # print([(s, e, t, txtdata[s : e]) for s, e, t in file_entities])
    inc_count += len(inclusions)

    entities = sorted(list(entities), key = lambda x: x[0])

    entity_start_chars, entity_end_chars, entity_types = zip(*entities)

    entity_count += len(entity_start_chars)

    offset_mapping = []

    sentence_spans = ru_tokenizer.span_tokenize(txtdata)
    
    for span in sentence_spans:

        start, end = span
        context = txtdata[start : end]

        word_spans = word_tokenizer.span_tokenize(context)
        offset_mapping.extend([(s + start, e + start) for s, e in word_spans])

    try:
        assert len(entity_types) == len(entity_start_chars) == len(entity_end_chars)
    except AssertionError:
        print(f[:-4])
        print(txtdata)
        print(entity_types)
        print(len(entity_types))
        print(entity_start_chars)
        print(len(entity_start_chars))
        print(entity_end_chars)
        print(len(entity_end_chars))

        for s, e, t in zip(entity_start_chars, entity_end_chars, entity_types):
            print(t, txtdata[s : e])

        raise AssertionError

    start_words, end_words = zip(*offset_mapping)

    doc_entities = {
        'text': txtdata,
        'entity_types': entity_types,
        'entity_start_chars': entity_start_chars,
        'entity_end_chars': entity_end_chars,
        'id': tid,
        'word_start_chars': start_words,
        'word_end_chars': end_words
    }

    train_data[d_idx] = doc_entities

print("Train subset total entities:", entity_count)
print("Train subset total inclusions:", inc_count)
entity_count = 0

for d_idx, data in enumerate(tqdm(dev_data)):

    tid = data["id"]
    txtdata = data["text"]
    labels = data["label"]

    entity_types = []
    entity_start_chars = []
    entity_end_chars = []

    for label in labels:

        if track != "track1":
            entity_type = label[2].upper()
        else:
            entity_type = "ANY"
        start_char = label[0]
        end_char = label[1]

        entity_types.append(entity_type)
        entity_start_chars.append(start_char)
        entity_end_chars.append(end_char)
    entity_count += len(entity_start_chars)

    offset_mapping = []

    sentence_spans = ru_tokenizer.span_tokenize(txtdata)
    
    for span in sentence_spans:

        start, end = span
        context = txtdata[start : end]

        word_spans = word_tokenizer.span_tokenize(context)
        offset_mapping.extend([(s + start, e + start) for s, e in word_spans])

    try:
        assert len(entity_types) == len(entity_start_chars) == len(entity_end_chars)
    except AssertionError:
        print(f[:-4])
        print(txtdata)
        print(entity_types)
        print(len(entity_types))
        print(entity_start_chars)
        print(len(entity_start_chars))
        print(entity_end_chars)
        print(len(entity_end_chars))

        for s, e, t in zip(entity_start_chars, entity_end_chars, entity_types):
            print(t, txtdata[s : e])

        raise AssertionError

    start_words, end_words = zip(*offset_mapping)

    doc_entities = {
        'text': txtdata,
        'entity_types': entity_types,
        'entity_start_chars': entity_start_chars,
        'entity_end_chars': entity_end_chars,
        'id': tid,
        'word_start_chars': start_words,
        'word_end_chars': end_words
    }

    dev_data[d_idx] = doc_entities

print("Dev subset total entities:", entity_count)

for d_idx, data in enumerate(tqdm(test_data)):

    tid = data["id"]
    txtdata = data["text"]

    offset_mapping = []

    sentence_spans = ru_tokenizer.span_tokenize(txtdata)
    
    for span in sentence_spans:

        start, end = span
        context = txtdata[start : end]

        word_spans = word_tokenizer.span_tokenize(context)
        offset_mapping.extend([(s + start, e + start) for s, e in word_spans])

    start_words, end_words = zip(*offset_mapping)

    doc_entities = {
        'text': txtdata,
        'entity_types': [],
        'entity_start_chars': [],
        'entity_end_chars': [],
        'id': tid,
        'word_start_chars': start_words,
        'word_end_chars': end_words
    }

    test_data[d_idx] = doc_entities

if not os.path.exists(os.path.join("S:/HRCode/data/RuTermEval", track, "flat/winc")):
    os.makedirs(os.path.join("S:/HRCode/data/RuTermEval", track, "flat/winc"), exist_ok = True)

with open(os.path.join("S:/HRCode/data/RuTermEval", track, "flat/winc", "train.json"), "w", encoding = "UTF-8") as train_file:
    for data in train_data:
        train_file.write(json.dumps(data, ensure_ascii = False) + "\n")

with open(os.path.join("S:/HRCode/data/RuTermEval", track, "flat/winc", "dev.json"), "w", encoding = "UTF-8") as dev_file:
    for data in dev_data:
        dev_file.write(json.dumps(data, ensure_ascii = False) + "\n")

with open(os.path.join("S:/HRCode/data/RuTermEval", track, "flat/winc", "test.json"), "w", encoding = "UTF-8") as test_file:
    for data in test_data:
        test_file.write(json.dumps(data, ensure_ascii = False) + "\n")