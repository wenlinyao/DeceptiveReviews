#!/usr/bin/env python
import gzip
import json

def advertising_lexicon_load(filename):
    advertising_phrases = set()
    input_file = open(filename, 'r')
    for line in input_file:
        if not line.strip():
            continue
        words = line.split()
        if words[0] == "#":
            continue
        phrases = line.split("*")
        for phrase in phrases:
            words = " ".join(phrase.split())
            if words == "":
                continue
            advertising_phrases.add(words.lower())
    return advertising_phrases