#import sys
#sys.path.append("../MaxEnt/")
#from LogisticRegression import *
import nltk
import os
import pdb
import numpy as np
import argparse
import pickle
import re, string
from collections import Counter
from coreNLP_xml import *
from LIWC_load import *
from generate_rerun_file import *
import random
import time

def string_standardize(myString): # remove punctuation and return lower case
    return myString.lower().translate(None, string.punctuation)

def obligation_lexicon_load(file_name):
    oblig_lexicon_file = open(file_name, 'r')
    oblig_lexicon = set()
    for line in oblig_lexicon_file:
        if not line.strip():
            continue
        item = " ".join(line.split())
        oblig_lexicon.add(item)
    return oblig_lexicon

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


def emotion_lexicon_load(file_name):
    
    emotion_lexicon_file = open(file_name, 'r')
    emotion_lexicon = {}
    for line in emotion_lexicon_file:
        words = line.split()
        if words[0] not in emotion_lexicon:
            emotion_lexicon[words[0]] = []
        if words[2] != '0':
            emotion_lexicon[words[0]].append(words[1])
    return emotion_lexicon


def effective_lexicon_load(file_name):
    effective_lexicon_file = open(file_name, 'r')
    effective_lexicon = {}
    for line in effective_lexicon_file:
        if not line.strip():
            continue
        words = line.split(",")
        effective_lexicon[words[0]] = float(words[1])
    return effective_lexicon

def tokenize(words, grams):
    #words = sentence.split()
    tokens = []
    for gram in grams:
        for i in range(len(words) - gram + 1):
            tokens += ["_*_".join(words[i:i+gram])]
    
    return tokens


def LIWC_search(word, LIWC_dic):
    new_word = word
    if word in LIWC_dic:
        return LIWC_dic[new_word]
    for i in range(1, len(word)):
        new_word = word[0:-i] + '*'
        if new_word in LIWC_dic:
            return LIWC_dic[new_word]
    return []

def build_dict(f, invalid_POS_tags, emotion, emotion_lexicon, LIWC_dic, valid_LIWC_cate, advertising_lexicon, effective_lexicon, parameter_setting):
    #src_folder = "../real_data/preprocess_1_onefile/"
    src_folder = "../real_data/preprocess_merge/"

    with open(src_folder + "productId2name.p", 'rb') as file:
        productId2name = pickle.load(file)
    with open(src_folder + "productId2cate.p", 'rb') as file:
        productId2cate = pickle.load(file)

    track_lines = open(f.replace(".txt", "_track.txt"), 'r').readlines()

    dic = Counter()
    line_count = 0
    
    

    for line in open(f).xreadlines():

        for c in string.punctuation:
            line = line.replace(c, " ")
        line = line.lower()

        line_count += 1
        sentenceList, sentence_parseList, sentence_dependencyList = parse_document(f.replace(".txt", "_folder_coreNLP/" + str(line_count) + ".txt.out"))
        track_line = track_lines[line_count - 1]
        if "ngram" in parameter_setting and parameter_setting["ngram"] != 0:
            ngram = str(parameter_setting["ngram"])
            ngram = [int(i) for i in ngram]
            words = []
            for sentence in sentenceList:
                for token in sentence:
                    if len(invalid_POS_tags) != 0 and token.POS in invalid_POS_tags:
                        continue
                    words.append(token.word)
                    #words.append(token.lemma)
            dic.update(tokenize(words, ngram))

        if "product_name_ngram" in parameter_setting and parameter_setting["product_name_ngram"] != 0:
            product_name_ngram = str(parameter_setting["product_name_ngram"])
            product_name_ngram = [int(i) for i in product_name_ngram]
            words = []
            productId = track_line.split()[1]
            if productId in productId2name:
                for word in productId2name[productId].split():
                    words.append(word)
                dic.update(tokenize(words, product_name_ngram))

        if "POS" in parameter_setting and parameter_setting["POS"] != 0:
            POS_ngram = str(parameter_setting["POS"])
            POS_ngram = [int(i) for i in POS_ngram]
            POS_tags = []
            for sentence in sentenceList:
                for token in sentence:
                    POS_tags.append(token.POS)
            dic.update(tokenize(POS_tags, POS_ngram))


        if "LIWC" in parameter_setting and parameter_setting["LIWC"] == 1:
            for sentence in sentenceList:
                for token in sentence:
                    LIWC_labels = LIWC_search(token.word, LIWC_dic)
                    for label in LIWC_labels:
                        if label in valid_LIWC_cate:
                            dic.update(['LIWC_' + label])
        """
        if "obligation" in parameter_setting and parameter_setting["obligation"] == 1:
            for word in oblig_lexicon:
                if word in line:
                    dic.update("_".join(word.split()))
        """

        if "advertising_phrases" in parameter_setting and parameter_setting["advertising_phrases"] == 1:
            for phrase in advertising_lexicon:
                if phrase in line:
                    dic.update(["_".join(phrase.split())])
        
        if "product_name_overlap" in parameter_setting and parameter_setting["product_name_overlap"] != 0:
            overlap_ngram = str(parameter_setting["product_name_overlap"])
            overlap_ngram = [int(i) for i in overlap_ngram]
            
            productId = track_line.split()[1]
            if productId in productId2name:
                for grams in overlap_ngram:
                    review_tokens = tokenize(string_standardize(line).split(), [grams])
                    product_name_tokens = tokenize(string_standardize(productId2name[productId]).split(), [grams])

                    for i in range(0, len (set(review_tokens) & set(product_name_tokens) )):
                        dic.update(["product_name_overlap_" + str(grams) + "gram"])


        if "syntax_production" in parameter_setting and parameter_setting["syntax_production"] == 1:
            for sentence_parse in sentence_parseList:
                tree = nltk.tree.Tree.fromstring(sentence_parse)
                dic.update(tree.productions())

        if "unlexicalized_production" in parameter_setting and parameter_setting["unlexicalized_production"] == 1:
            for sentence_parse in sentence_parseList:
                tree = nltk.tree.Tree.fromstring(sentence_parse)
                for prod in tree.productions():
                    if str(prod)[-1] == "'":
                        continue
                    dic.update([prod])



        #if parameter_setting["syntactic_complexity"] == 1: 
        if "passive_voice" in parameter_setting and parameter_setting["passive_voice"] == 1:
            for sentence_dep in sentence_dependencyList:
                for dep in sentence_dep:
                    if dep.type == "nsubjpass":
                        dic.update(["nsubjpass"])

        if "dependencies" in parameter_setting and parameter_setting["dependencies"] == 1:
            for sentence_dep in sentence_dependencyList:
                for dep in sentence_dep:
                    dic.update([dep.type])

        if "dependency_pair" in parameter_setting and parameter_setting["dependency_pair"] == 1:
            for sentence_dep in sentence_dependencyList:
                for dep in sentence_dep:
                    pair = dep.gov + " " + dep.type + " " + dep.dep
                    dic.update([pair])

        if "emotion_trans" in parameter_setting and parameter_setting["emotion_trans"] == 1:
            last_sentence_emotion = []
            for sentence in sentenceList:
                sentence_emotion = []
                for token in sentence:
                    if token.word in emotion_lexicon:
                        sentence_emotion += emotion_lexicon[token.word]
                for emo1 in last_sentence_emotion:
                    for emo2 in sentence_emotion:
                        pair = emo1 + ' --> ' + emo2
                        dic.update([pair])
                last_sentence_emotion = sentence_emotion

        if "token_emotion" in parameter_setting and parameter_setting["token_emotion"] == 1:
            token_emotionList = []
            for sentence in sentenceList:
                for token in sentence:
                    if token.word in emotion_lexicon:
                        token_emotionList.append(emotion_lexicon[token.word])
            for i in range(0, len(token_emotionList)):
                for emo in token_emotionList[i]:
                    dic.update(['$' + emo + '$'])

        if "token_emotion_trans" in parameter_setting and parameter_setting["token_emotion_trans"] == 1:
            token_emotionList = []
            for sentence in sentenceList:
                for token in sentence:
                    if token.word in emotion_lexicon:
                        token_emotionList.append(emotion_lexicon[token.word])
            for i in range(0, len(token_emotionList) - 1):
                for emo1 in token_emotionList[i]:
                    for emo2 in token_emotionList[i + 1]:
                        pair = emo1 + ' >>> ' + emo2
                        dic.update([pair])

        if "emotion_trans_lexical" in parameter_setting and parameter_setting["emotion_trans_lexical"] == 1:
            last_sentence_emotion_words = []
            for sentence in sentenceList:
                sentence_emotion_words = []
                for token in sentence:
                    if token.word in emotion_lexicon:
                        sentence_emotion_words.append(token.word)
                for word1 in last_sentence_emotion_words:
                    for word2 in sentence_emotion_words:
                        pair = word1 + ' --> ' + word2
                        dic.update([pair])
                last_sentence_emotion_words = sentence_emotion_words

        if "token_emotion_trans_lexical" in parameter_setting and parameter_setting["token_emotion_trans_lexical"] == 1:
            token_emotion_lexicalList = []
            for sentence in sentenceList:
                for token in sentence:
                    if token.word in emotion_lexicon:
                        token_emotion_lexicalList.append(token.word)
            for i in range(0, len(token_emotion_lexicalList) - 1):
                pair = token_emotion_lexicalList[i] + ' >>> ' + token_emotion_lexicalList[i + 1]
                dic.update([pair])

    return dic

def process_files(file_pos, file_neg, dic, r, invalid_POS_tags, emotion, emotion_lexicon, LIWC_dic, valid_LIWC_cate, advertising_lexicon, effective_lexicon, outfn, parameter_setting, feature_value_type):
    #src_folder = "../real_data/preprocess_1_onefile/"
    src_folder = "../real_data/preprocess_merge/"

    with open(src_folder + "productId2name.p", 'rb') as file:
        productId2name = pickle.load(file)
    with open(src_folder + "productId2cate.p", 'rb') as file:
        productId2cate = pickle.load(file)

    output = []
    
    #SynComp_scale = [2189.0,  106.0,  230.0,  172.0,  100.0,  103.0,  53.0,  90.0,  280.0,  192.0,  165.0,  53.0,  25.0,  22.0,  20.0,  1.0,  18.0,  12.0,  1.0,  7.0,  3.0,  19.0,  9.0]
    SynComp_scale = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    current_feature_length = 0


    for beg_line, f in zip(["1", "-1"], [file_pos, file_neg]):
        track_lines = open(f.replace(".txt", "_track.txt"), 'r').readlines()

        line_count = 0
        SynComp = open(f.replace(".txt", "_SynComp.txt"), 'r').readlines()

        for line in open(f).xreadlines():

            for c in string.punctuation:
                line = line.replace(c, " ")
            line = line.lower()


            line_count += 1
            indexes = []
            sentenceList, sentence_parseList, sentence_dependencyList = parse_document(f.replace(".txt", "_folder_coreNLP/" + str(line_count) + ".txt.out"))
            track_line = track_lines[line_count - 1]
            if "ngram" in parameter_setting and parameter_setting["ngram"] != 0:
                ngram = str(parameter_setting["ngram"])
                ngram = [int(i) for i in ngram]

                words = []
                for sentence in sentenceList:
                    for token in sentence:
                        if len(invalid_POS_tags) != 0 and token.POS in invalid_POS_tags:
                            continue
                        words.append(token.word)
                        #words.append(token.lemma)
                tokens = tokenize(words, ngram)
                for t in tokens:
                    try:
                        indexes += [dic[t]]
                    except KeyError:
                        pass
            if "product_name_ngram" in parameter_setting and parameter_setting["product_name_ngram"] != 0:
                product_name_ngram = str(parameter_setting["product_name_ngram"])
                product_name_ngram = [int(i) for i in product_name_ngram]
                words = []
                productId = track_line.split()[1]
                if productId not in productId2name:
                    for word in productId2name[productId].split():
                        words.append(word)
                    product_name_tokens = tokenize(words, product_name_ngram)
                    for t in product_name_tokens:
                        try:
                            indexes += [dic[t]]
                        except KeyError:
                            pass

            if "POS" in parameter_setting and parameter_setting["POS"] != 0:
                POS_ngram = str(parameter_setting["POS"])
                POS_ngram = [int(i) for i in POS_ngram]
                POS_tags = []
                for sentence in sentenceList:
                    for token in sentence:
                        POS_tags.append(token.POS)
                POS_tokens = tokenize(POS_tags, POS_ngram)
                for t in POS_tokens:
                    try:
                        indexes += [dic[t]]
                    except KeyError:
                        pass

            if "LIWC" in parameter_setting and parameter_setting["LIWC"] == 1:
                LIWC_labelList = []
                for sentence in sentenceList:
                    for token in sentence:
                        LIWC_labels = LIWC_search(token.word, LIWC_dic)
                        for label in LIWC_labels:
                            if label in valid_LIWC_cate:
                                LIWC_labelList.append('LIWC_' + label)
                #print LIWC_labelList
                #raw_input("continue?")
                for t in LIWC_labelList:
                    try:
                        indexes += [dic[t]]
                    except KeyError:
                        pass
            """
            if "obligation" in parameter_setting and parameter_setting["obligation"] == 1:
                obligList = []
                for word in oblig_lexicon:
                    if word in line:
                        obligList.append("_".join(word.split()))
                for t in obligList:
                    try:
                        indexes += [dic[t]]
                    except KeyError:
                        pass
            """

            if "advertising_phrases" in parameter_setting and parameter_setting["advertising_phrases"] == 1:
                advertisingList = []
                for phrase in advertising_lexicon:
                    if phrase in line:
                        advertisingList.append("_".join(phrase.split()))
                for t in advertisingList:
                    try:
                        indexes += [dic[t]]
                        #r[dic[t]] = advertising_lexicon[t.replace("_", " ")]
                    except KeyError:
                        pass

            if "product_name_overlap" in parameter_setting and parameter_setting["product_name_overlap"] != 0:
                product_name_overlapList = []
                overlap_ngram = str(parameter_setting["product_name_overlap"])
                overlap_ngram = [int(i) for i in overlap_ngram]
                
                productId = track_line.split()[1]
                if productId in productId2name:
                    for grams in overlap_ngram:
                        review_tokens = tokenize(string_standardize(line).split(), [grams])
                        product_name_tokens = tokenize(string_standardize(productId2name[productId]).split(), [grams])

                        for i in range(0, len (set(review_tokens) & set(product_name_tokens) )):
                            product_name_overlapList.append("product_name_overlap_" + str(grams) + "gram")
                    for t in product_name_overlapList:
                        try:
                            indexes += [dic[t]]
                        except KeyError:
                            pass


            if "syntax_production" in parameter_setting and parameter_setting["syntax_production"] == 1:
                for sentence_parse in sentence_parseList:
                    tree = nltk.tree.Tree.fromstring(sentence_parse)
                    for t in tree.productions():
                        try:
                            indexes += [dic[t]]
                        except KeyError:
                            pass

            if "unlexicalized_production" in parameter_setting and parameter_setting["unlexicalized_production"] == 1:
                productionList = []
                for sentence_parse in sentence_parseList:
                    tree = nltk.tree.Tree.fromstring(sentence_parse)
                    for prod in tree.productions():
                        if str(prod)[-1] == "'":
                            continue
                        productionList.append(prod)
                for t in productionList:
                    try:
                        indexes += [dic[t]]
                    except KeyError:
                        pass
            
            if "passive_voice" in parameter_setting and parameter_setting["passive_voice"] == 1:
                depList = []
                for sentence_dep in sentence_dependencyList:
                    for dep in sentence_dep:
                        if dep.type == "nsubjpass":
                            depList.append(dep.type)
                for t in depList:
                    try:
                        indexes += [dic[t]]
                    except KeyError:
                        pass
            if "dependencies" in parameter_setting and parameter_setting["dependencies"] == 1:
                depList = []
                for sentence_dep in sentence_dependencyList:
                    for dep in sentence_dep:
                        depList.append(dep.type)
                for t in depList:
                    try:
                        indexes += [dic[t]]
                    except KeyError:
                        pass

            if "dependency_pair" in parameter_setting and parameter_setting["dependency_pair"] == 1:
                pairList = []
                for sentence_dep in sentence_dependencyList:
                    for dep in sentence_dep:
                        pair = dep.gov + " " + dep.type + " " + dep.dep
                        pairList.append(pair)
                for t in pairList:
                    try:
                        indexes += [dic[i]]
                    except KeyError:
                        pass


            if "emotion_trans" in parameter_setting and parameter_setting["emotion_trans"] == 1:
                emotion_transList = []

                last_sentence_emotion = []
                for sentence in sentenceList:
                    sentence_emotion = []
                    for token in sentence:
                        if token.word in emotion_lexicon:
                            sentence_emotion += emotion_lexicon[token.word]
                    for emo1 in last_sentence_emotion:
                        for emo2 in sentence_emotion:
                            pair = emo1 + ' --> ' + emo2
                            emotion_transList.append(pair)
                    last_sentence_emotion = sentence_emotion
                for t in emotion_transList:
                    try:
                        indexes += [dic[t]]
                    except KeyError:
                        pass

            if "token_emotion" in parameter_setting and parameter_setting["token_emotion"] == 1:
                all_emotionList = []
                token_emotionList = []
                for sentence in sentenceList:
                    for token in sentence:
                        if token.word in emotion_lexicon:
                            token_emotionList.append(emotion_lexicon[token.word])
                for i in range(0, len(token_emotionList)):
                    for emo in token_emotionList[i]:
                        all_emotionList.append('$' + emo + '$')
                for t in all_emotionList:
                    try:
                        indexes += [dic[t]]
                    except KeyError:
                        pass

            if "token_emotion_trans" in parameter_setting and parameter_setting["token_emotion_trans"] == 1:
                token_emotion_transList = []

                token_emotionList = []
                for sentence in sentenceList:
                    for token in sentence:
                        if token.word in emotion_lexicon:
                            token_emotionList.append(emotion_lexicon[token.word])
                for i in range(0, len(token_emotionList) - 1):
                    for emo1 in token_emotionList[i]:
                        for emo2 in token_emotionList[i + 1]:
                            pair = emo1 + ' >>> ' + emo2
                            token_emotion_transList.append(pair)

                for t in token_emotion_transList:
                    try:
                        indexes += [dic[t]]
                    except KeyError:
                        pass
                    

            if "emotion_trans_lexical" in parameter_setting and parameter_setting["emotion_trans_lexical"] == 1:
                emotion_trans_lexicalList = []

                last_sentence_emotion_words = []
                for sentence in sentenceList:
                    sentence_emotion_words = []
                    for token in sentence:
                        if token.word in emotion_lexicon:
                            sentence_emotion_words.append(token.word)
                    for word1 in last_sentence_emotion_words:
                        for word2 in sentence_emotion_words:
                            pair = word1 + ' --> ' + word2
                            emotion_trans_lexicalList.append(pair)
                    last_sentence_emotion_words = sentence_emotion_words
                for t in emotion_trans_lexicalList:
                    try:
                        indexes += [dic[t]]
                    except KeyError:
                        pass

            if "token_emotion_trans_lexical" in parameter_setting and parameter_setting["token_emotion_trans_lexical"] == 1:
                token_emotion_trans_lexicalList = []

                token_emotion_lexicalList = []
                for sentence in sentenceList:
                    for token in sentence:
                        if token.word in emotion_lexicon:
                            token_emotion_lexicalList.append(token.word)
                for i in range(0, len(token_emotion_lexicalList) - 1):
                    pair = token_emotion_lexicalList[i] + ' >>> ' + token_emotion_lexicalList[i + 1]
                    token_emotion_trans_lexicalList.append(pair)
                for t in token_emotion_trans_lexicalList:
                    try:
                        indexes += [dic[t]]
                    except KeyError:
                        pass

            new_indexes = list(set(indexes))
            new_indexes.sort()
            line = [beg_line]
            if feature_value_type == "nb":
                for i in new_indexes:
                    line += ["%i:%f" % (i + 1, r[i])]
            elif feature_value_type == "bool":
                for i in new_indexes:
                    line += ["%i:%f" % (i + 1, 1.0)]
            elif feature_value_type == "freq":
                freq_Counter = Counter(indexes)
                for item in sorted(freq_Counter):
                    line += ["%i:%f" % (item + 1, float(freq_Counter[item]))]


            current_feature_length = len(dic)
            if "syntactic_complexity" in parameter_setting:
                words = SynComp[line_count - 1].replace('\n','').split(',')
                for i, word in enumerate(words[1:]):
                    if i not in parameter_setting["syntactic_complexity"]:
                        continue
                    current_feature_length += 1
                    value = float(word)
                    value = value / SynComp_scale[i]
                    line += ["%i:%f" % (current_feature_length, value)]
            

            output += [" ".join(line)]
    output = "\n".join(output)
    f = open(outfn, "w")
    f.writelines(output)
    f.close()
    #outfn_track.close()


def compute_ratio(poscounts, negcounts, alpha=1):
    alltokens = list(set(poscounts.keys() + negcounts.keys()))
    
    dic = dict((t, i) for i, t in enumerate(alltokens))

    d = len(dic)
    print "computing r..."
    p, q = np.ones(d) * alpha , np.ones(d) * alpha
    for t in alltokens:
        p[dic[t]] += poscounts[t]
        q[dic[t]] += negcounts[t]
    p /= abs(p).sum()
    q /= abs(q).sum()
    r = np.log(p/q)

    # save all features
    # id to feature
    feature_dic = dict((i, t) for i, t in enumerate(alltokens))

    return dic, feature_dic, r

# def main(ptrain, ntrain, ptest, ntest, out, liblinear, ngram):
def classifier_main(classifier, folder, ptrain, ntrain, ptest, ntest, out, liblinear, parameter_setting, invalid_POS_tags, top_f_num, feature_value_type):
    ptrain = folder + ptrain
    ntrain = folder + ntrain
    ptest = folder + ptest
    ntest = folder + ntest
    random.seed(1)
    emotion_lexicon = emotion_lexicon_load("emotion_lexicon")
    emotion = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]


    LIWC_dic = LIWC_load("../dic/LIWC2015_English.dic")
    
    valid_LIWC_cate = set([str(i) for i in range(1, 126)]) # consider all LIWC labels
    #valid_LIWC_cate = set(["4", "5", "6", "7", "8", "31", "32", "51", "52", "53", "54", "55", "56", "90", "91", "92", "101", "102", "103"])

    #oblig_lexicon = obligation_lexicon_load("../dic/obligation_words")

    advertising_lexicon = advertising_lexicon_load("../dic/effective_advertising_phrases")
    effective_lexicon = effective_lexicon_load("../dic/effective_ngram")
    
    #print advertising_lexicon
    
    
    #print emotion_lexicon
    #ngram = str(parameter_setting["ngram"])
    #ngram = [int(i) for i in ngram]
    #print "ngram:", ngram
    print parameter_setting
    print "counting..."
    poscounts = build_dict(ptrain, invalid_POS_tags, emotion, emotion_lexicon, LIWC_dic, valid_LIWC_cate, advertising_lexicon, effective_lexicon, parameter_setting) # pos dict to map features to frequency
    negcounts = build_dict(ntrain, invalid_POS_tags, emotion, emotion_lexicon, LIWC_dic, valid_LIWC_cate, advertising_lexicon, effective_lexicon, parameter_setting) # neg dict to map features to frequency

    print "feature set extracting..."
    dic, feature_dic, r = compute_ratio(poscounts, negcounts)


    with open(folder + "feature_dic.p", 'wb') as file:
        pickle.dump(feature_dic, file)

    # save r value dic from nbsvm paper
    with open(folder + "nbsvm_r.p", 'wb') as file:
        pickle.dump(r, file)

    with open(folder + "poscounts.p", 'wb') as file:
        pickle.dump(poscounts, file)

    #with open(folder + "negcounts.p", 'wb') as file:
    #    pickle.dump(negcounts, file)



    print "processing files..."
    process_files(ptrain, ntrain, dic, r, invalid_POS_tags, emotion, emotion_lexicon, LIWC_dic, valid_LIWC_cate, advertising_lexicon, effective_lexicon, folder + "train-file.txt", parameter_setting, feature_value_type)
    process_files(ptest, ntest, dic, r, invalid_POS_tags, emotion, emotion_lexicon, LIWC_dic, valid_LIWC_cate, advertising_lexicon, effective_lexicon, folder + "test-file.txt", parameter_setting, feature_value_type)
    
    print "process_files finished."

    train = os.path.join(liblinear, "train") 
    predict = os.path.join(liblinear, "predict") 

    # -s type : set type of solver (default 1)
    # for multi-class classification
    #  0 -- L2-regularized logistic regression (primal)
    #  1 -- L2-regularized L2-loss support vector classification (dual)
    #  2 -- L2-regularized L2-loss support vector classification (primal)
    
    if classifier == "SVM":
        os.system(train + " -s 2 " + folder + "train-file.txt " + folder + "model.logreg")
        os.system(predict + " -b 0 " + folder + "test-file.txt " + folder + "model.logreg " + folder + out)
    elif classifier == "MaxEnt":
        os.system(train + " -s 0 " + folder + "train-file.txt " + folder + "model.logreg")
        os.system(predict + " -b 1 " + folder + "test-file.txt " + folder + "model.logreg " + folder + out)

    if top_f_num != None:
        print "*************  TOP", top_f_num, "feature  ******************"

        new_train_file = "train-file" + "_TOP" + str(top_f_num) + ".txt"
        new_test_file = "test-file" + "_TOP" + str(top_f_num) + ".txt"

        new_out = out + "_TOP" + str(top_f_num)

        generate_rerun_file_main(top_f_num, folder, "model.logreg", "train-file.txt", "test-file.txt", new_train_file, new_test_file)
        if classifier == "SVM":
            os.system(train + " -s 2 " + folder + new_train_file + " " + folder + "model" + "_TOP" + str(top_f_num) + ".logreg")
            os.system(predict + " -b 0 " + folder + new_test_file + " " + folder + "model" + "_TOP" + str(top_f_num) + ".logreg " + folder + new_out)
        elif classifier == "MaxEnt":
            os.system(train + " -s 0 " + folder + new_train_file + " " + folder + "model" + "_TOP" + str(top_f_num) + ".logreg")
            os.system(predict + " -b 1 " + folder + new_test_file + " " + folder + "model" + "_TOP" + str(top_f_num) + ".logreg " + folder + new_out)
    


    # sklearn implement (performance is worse than LIBLINEAR)
    #if classifier == "MaxEnt":
    #    MaxEnt_main(folder, "train-file.txt", "test-file.txt", "model.logreg", out)
    
    #os.system("rm model.logreg train-nbsvm.txt test-nbsvm.txt")
    
    
    
    
    
