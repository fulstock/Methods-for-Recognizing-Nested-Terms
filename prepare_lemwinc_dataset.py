import json
import os
import random
import re

from nltk.data import load
from nltk.tokenize import NLTKWordTokenizer
import pymorphy2

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

ru_tokenizer = load("tokenizers/punkt/russian.pickle") # Загрузка токенизатора для русского языка
word_tokenizer = NLTKWordTokenizer()
morph = pymorphy2.MorphAnalyzer()

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

    offset_mapping = []

    sentence_spans = ru_tokenizer.span_tokenize(txtdata)
    
    for span in sentence_spans:

        start, end = span
        context = txtdata[start : end]

        word_spans = word_tokenizer.span_tokenize(context)
        offset_mapping.extend([(s + start, e + start) for s, e in word_spans])

    start_words, end_words = zip(*offset_mapping)

    file_entities = list(zip(entity_start_chars, entity_end_chars, entity_types))

    outermost_entities = set([(su, eu, tu) for su, eu, tu in file_entities if len(list(filter(lambda ea: ea[0] < su and ea[1] >= eu or ea[0] <= su and ea[1] > eu, file_entities))) == 0])
    outermost_entities = sorted(list(outermost_entities), key = lambda x: x[0])
    entities = outermost_entities.copy()

    outermost_entities = []
    for sc, ec, t in entities:
        try:
            sw0 = [s_idx for s_idx, sw in enumerate(start_words) if sw == sc][0]
            ew0 = [e_idx for e_idx, ew in enumerate(end_words) if ew == ec][0]
        except IndexError:
            continue
        outermost_entities.append((sc, ec, sw0, ew0, t))
    

    inclusion_spans = set()
    inclusion_span_types = dict()

    words = [(s, e, txtdata[s : e]) for s, e in zip(start_words, end_words)]
    morphed_words = [(morph.parse(w[2])[0].normal_form, w[0], w[1]) for w in words]
    lemm_words = [l[0] for l in morphed_words]
    lemm_entities = []
    for sc, ec, sw, ew, t in outermost_entities:
        lemm_entities.append((sc, ec, sw, ew, set(lemm_words[sw : ew + 1]), t))


    # print(lemm_entities)

    for e_idx, e in enumerate(lemm_entities):
        if any(["орфографический" in s in s for s in e[4]]) and any(["словарь" in s in s for s in e[4]]):
            print(e)
        entities_with_this_inclusion = [v for v in lemm_entities if (v[1] < e[0] or e[1] < v[0]) \
            and e[4].issubset(v[4]) and e[4] != v[4]]
        if len(entities_with_this_inclusion) > 0:
            if any(["орфографический" in s in s for s in e[4]]) and any(["словарь" in s in s for s in e[4]]):
                print("ewti", entities_with_this_inclusion)
            inclusion_spans.add("||".join(sorted(list(e[4]))))
            inclusion_span_types["||".join(sorted(list(e[4])))] = e[5]
    
    inclusions = []    
    for inclusion_span in inclusion_spans:
        inclusion_span = set(inclusion_span.split('||'))
        print(inclusion_span)
        probable_inclusions = []
        for w1_idx, word1 in enumerate(morphed_words):
            for w2_idx, word2 in enumerate(morphed_words):
                word_span_set = set(lemm_words[w1_idx : w2_idx + 1])
                # print(inclusion_span)
                # print(word_span_set)
                if inclusion_span.issubset(word_span_set):
                    if any(["словарь" in s in s for s in inclusion_span]):
                        print(inclusion_span)
                        print(word_span_set)
                    probable_inclusions.append((morphed_words[w1_idx][1], morphed_words[w2_idx][2], w1_idx, w2_idx))
        # print(inclusion_span)
        # print(morphed_words)
        for p in probable_inclusions:
            for o in outermost_entities:
                if (o[0] < p[0] and p[1] <= o[1]) or (o[0] <= p[0] and p[1] < o[1]):
                    inclusions.append((p[0], p[1], inclusion_span_types["||".join(sorted(list(inclusion_span)))]))
                    if "орфограф" in txtdata[p[0] : p[1]]:
                        print(txtdata[p[0] : p[1]])
                        print(p)
                        exit(1)
                    if inclusion_span_types["||".join(sorted(list(inclusion_span)))] in inclusions_by_type.keys():
                        inclusions_by_type[inclusion_span_types["||".join(sorted(list(inclusion_span)))]] += 1
                    else:
                        inclusions_by_type[inclusion_span_types["||".join(sorted(list(inclusion_span)))]] = 1
    # print(inclusions)                
    # print([(s, e, t, txtdata[s : e]) for s, e, t in file_entities])
    entities.extend(inclusions)
    # print([(s, e, t, txtdata[s : e]) for s, e, t in file_entities])
    inc_count += len(inclusions)

    entities = sorted(list(entities), key = lambda x: x[0])

    entity_start_chars, entity_end_chars, entity_types = zip(*entities)

    entity_count += len(entity_start_chars)

    

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

if not os.path.exists(os.path.join("S:/HRCode/data/RuTermEval", track, "flat/lemwinc")):
    os.makedirs(os.path.join("S:/HRCode/data/RuTermEval", track, "flat/lemwinc"), exist_ok = True)

with open(os.path.join("S:/HRCode/data/RuTermEval", track, "flat/lemwinc", "train.json"), "w", encoding = "UTF-8") as train_file:
    for data in train_data:
        train_file.write(json.dumps(data, ensure_ascii = False) + "\n")

with open(os.path.join("S:/HRCode/data/RuTermEval", track, "flat/lemwinc", "dev.json"), "w", encoding = "UTF-8") as dev_file:
    for data in dev_data:
        dev_file.write(json.dumps(data, ensure_ascii = False) + "\n")

with open(os.path.join("S:/HRCode/data/RuTermEval", track, "flat/lemwinc", "test.json"), "w", encoding = "UTF-8") as test_file:
    for data in test_data:
        test_file.write(json.dumps(data, ensure_ascii = False) + "\n")