import os
import json
from tqdm.auto import tqdm

from random import choice, choices, randrange, randint
from string import ascii_lowercase

from natasha import Doc, NewsEmbedding, NewsSyntaxParser, Segmenter

def damage(text, long_entities, kept_entities, method = "random", mask = "digits"):

    new_text = text
    updated_kept_entities = kept_entities.copy()

    offset = 0
    for le_idx, (s, e, t, token_spans) in enumerate(long_entities):

        token_spans_offset = token_spans[0][0]

        if method == "random":
            damaged_token_idx = choice(list(range(len(token_spans))))
        elif method == "start":
            damaged_token_idx = 0
        elif method == "end":
            damaged_token_idx = len(token_spans) - 1
        elif method == "middle":
            damaged_token_idx = len(token_spans) // 2
        elif method == "syntax":
            try:
                doc = Doc(text[s : e])
                parser = NewsSyntaxParser(NewsEmbedding())
                doc.segment(Segmenter())
                doc.parse_syntax(parser)
                root_tokens = [token for token in doc.tokens if token.rel == 'root']
                if len(root_tokens) == 0:
                    root_tokens = [token for token in doc.tokens if token.head_id == token.id]
                    if len(root_tokens) == 0:
                        root_tokens = [token for token in doc.tokens if 'mod' not in token.rel]
                        if len(root_tokens) == 0:
                            root_tokens = [choice(doc.tokens)]

                root_token = root_tokens[0]

                try:
                    root_token_start = root_token.start
                    damaged_token_idx = [ts_idx for ts_idx, ts in enumerate(token_spans) if ts[0] == root_token_start + token_spans_offset][0]

                except IndexError:
                    print("======")
                    print(token_spans)
                    print([text[t[0] : t[1]] for t in token_spans])
                    print(root_token.text)
                    print(root_token_start)
                    print(root_token_start + token_spans_offset)
                    print("======")
                    damaged_token_idx = len(token_spans) // 2
            except UnicodeDecodeError:
                damaged_token_idx = len(token_spans) // 2

        #print(text[token_spans[damaged_token_idx][0] : token_spans[damaged_token_idx][1]])
        damaged_token_length = len(text[token_spans[damaged_token_idx][0] : token_spans[damaged_token_idx][1]])
        russian_letters = [chr(i) for i in range(ord('а'), ord('я') + 1)]

        damaged_token_digits = str(randrange(10 ** (damaged_token_length - 1), 10 ** damaged_token_length))
        damaged_token_letters = "".join(choices(russian_letters, k = damaged_token_length))

        if mask == "digits":
            damaged_token = damaged_token_digits
        elif mask == "letters":
            damaged_token = damaged_token_letters
        elif mask == "diglets":
            damaged_token = choice([damaged_token_digits, damaged_token_letters])
        else:
            if mask == "semicolon":
                damaged_token = ";"
            
            elif mask == "comma":
                damaged_token = ","

            elif mask == 'onedigit':
                damaged_token = str(randint(0, 9))
            elif mask == 'oneletter':
                damaged_token = choice(russian_letters)

            curr_offset = 1 - damaged_token_length
            offset += curr_offset

            for ke_idx, (sk, ek, tk) in enumerate(updated_kept_entities):
                if sk > e:
                    updated_kept_entities[ke_idx] = (sk + curr_offset, ek + curr_offset, tk)
            for le1_idx, (s1, e1, t1, token_spans1) in enumerate(long_entities[le_idx + 1:]):
                token_spans1 = [(s1 + curr_offset, e1 + curr_offset) for s1, e1 in token_spans1]
                long_entities[le_idx + 1 + le1_idx] = (s1 + curr_offset, e1 + curr_offset, t1, token_spans1)

        new_text = new_text[:token_spans[damaged_token_idx][0]] + damaged_token + new_text[token_spans[damaged_token_idx][1]:]
        # print(damaged_token)

    
    try:
        assert len(text) + offset == len(new_text) 
    except AssertionError:
        print(len(text))
        print(len(new_text))
        print(len(new_text) + offset)
        print(text)
        print(new_text)
        raise AssertionError

    assert len(updated_kept_entities) == len(kept_entities)

    return new_text, updated_kept_entities

method = "middle"
mask = "diglets"

original_flatdata = "S:/HRCode/data/RuTermEval/track3/flat/pure"
new_data = "S:/HRCode/data/RuTermEval/track3/flat/damage2/" + method + "/" + mask

datasets = ["train", "dev", "test"]

for dataset in datasets:
    filename = dataset + ".json"
    print(os.path.join(original_flatdata, filename))
    origfile = open(os.path.join(original_flatdata, filename), "r", encoding = "UTF-8")
    if not os.path.exists(new_data):
        os.makedirs(new_data)
    newfile = open(os.path.join(new_data, filename), "w", encoding = "UTF-8")

    for line in tqdm(origfile.readlines()):
        if dataset != "train":
            newfile.write(line.strip() + "\n")
            continue
        doc_dict = json.loads(line)
        text = doc_dict["text"]
        entities = list(zip(doc_dict["entity_start_chars"], doc_dict["entity_end_chars"], doc_dict["entity_types"]))
        word_spans = list(zip(doc_dict["word_start_chars"], doc_dict["word_end_chars"]))

        # 1. Find all entities of length >= 3

        long_entities = []
        kept_entities = []
        for s, e, t in entities:
            start_words = [(w_s, w_e) for w_s, w_e in word_spans if w_s >= s]
            within_words = [(w_s, w_e) for w_s, w_e in start_words if w_e <= e]      
            if len(within_words) >= 3:
                long_entities.append((s, e, t, within_words))
            else:
                kept_entities.append((s, e, t))

        # 2. Damage entities

        # methods: start, end, middle, random, syntax
        # masks: digits, letters

        new_text, updated_kept_entities = damage(text, long_entities, kept_entities, method = method, mask = mask)

        #print(new_text)
        #print(kept_entities)

        new_entity_start_chars, new_entity_end_chars, new_entity_types = [list(t) for t in zip(*updated_kept_entities)]
        new_doc_dict = {
            "text" : new_text,
            "entity_start_chars" : new_entity_start_chars,
            "entity_end_chars" : new_entity_end_chars,
            "entity_types" : new_entity_types,
            "word_start_chars" : doc_dict["word_start_chars"],
            "word_end_chars" : doc_dict["word_end_chars"],
            "id" : doc_dict["id"]
        }
        newfile.write(json.dumps(new_doc_dict, ensure_ascii = False) + '\n')

    origfile.close()
    newfile.close()