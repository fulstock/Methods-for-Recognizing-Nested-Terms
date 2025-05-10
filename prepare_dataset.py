import json
import os
import random

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

track = "track1"
train_data_file = "train_t1_v1.jsonl"
dev_data_file = "test1_t12_full_v2.jsonl"
test_data_file = "test2_t12_v2.jsonl"

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

# tags = ["COMMON", "NOMEN", "SPECIFIC"]

entity_count = 0

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

with open(os.path.join("S:/HRCode/data/RuTermEval", track, "train.json"), "w", encoding = "UTF-8") as train_file:
    for data in train_data:
        train_file.write(json.dumps(data, ensure_ascii = False) + "\n")

with open(os.path.join("S:/HRCode/data/RuTermEval", track, "dev.json"), "w", encoding = "UTF-8") as dev_file:
    for data in dev_data:
        dev_file.write(json.dumps(data, ensure_ascii = False) + "\n")

with open(os.path.join("S:/HRCode/data/RuTermEval", track, "test.json"), "w", encoding = "UTF-8") as test_file:
    for data in test_data:
        test_file.write(json.dumps(data, ensure_ascii = False) + "\n")