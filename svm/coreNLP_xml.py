from StringIO import StringIO
from sets import Set
import glob
import timeit
import gc
import gzip
from multiprocessing import Process
import io

class WordToken:
    def __init__ (self, id, word, lemma, POS, NER):
        self.id = id
        self.word = word
        self.lemma = lemma
        self.POS = POS
        self.NER = NER
        
class BasicDependency:
    def __init__ (self, type, gov, dep):
        self.type = type
        self.gov = gov
        self.dep = dep

class CollapsedDependency:
    def __init__ (self, type, gov, dep):
        self.type = type
        self.gov = gov
        self.dep = dep


def parse_document(xmlFile):
    #print "parse_document: ", xmlFile
    lines = io.open(xmlFile, 'r', encoding='utf8')
    sentences_flag = False
    sentence_flag = False
    tokens_flag = False
    token_flag = False
    collapsed_dependencies_flag = False
    collapsed_dep_flag = False
    tokenList = []
    collapsed_dependenciesList = []
    paragraph = [] # store all beginning sentence ids of each paragraph (a paragraph is one review)
    word = ""
    lemma = ""
    POS = ""
    NER = ""
    sentence = ""
    relation_flag = ""
    collapsed_dep_type = ""
    sentence_id = -1
    sentenceList = []
    sentence_dependencyList = []
    sentence_parseList = []

    for each_line in lines:        
        
        words = each_line.split()

        if (len(words) == 0):
            continue
        #structure start
        if (words[0] == '<sentences>'):
            sentences_flag = True #sentences structure start
            continue # process next line
            
        if (sentences_flag==True and words[0] == '<sentence'):
            sentence_id = words[1].split("\"")[1]
            tokenList = []
            collapsed_dependenciesList = []
            relation_flag = ""
            sentence_flag = True #sentences structure start
            continue # process next line

        if (sentence_flag == True and words[0] == '<tokens>'):
            tokens_flag = True #tokens structure start
            continue
        if (tokens_flag == True and words[0] == '<token' and len(words) >= 2):
            token_flag = True
            token_id = int (words[1].replace("id=\"", "").replace("\">", ""))
            continue
        
        if (sentence_flag == True and words[0] == '<dependencies' and words[1] == 'type=\"collapsed-ccprocessed-dependencies\">'):
            collapsed_dependencies_flag = True
            continue
        if (collapsed_dependencies_flag == True and words[0] == '<dep' and len(words) >= 2):
            collapsed_dep_flag = True
            collapsed_dep_type = words[1].replace("type=\"", "").replace("\">", "")
            continue

        if (collapsed_dep_flag == True):
            if (words[0] == '<governor'):
                collapsed_gov = words[1]
                continue
            if (words[0] == '<dependent'):
                collapsed_dep = words[1]
                continue
        if (sentence_flag == True and '<parse>' in each_line and '</parse>' in each_line):
            sentence_parseList.append(each_line.replace('<parse>', '').replace('</parse>', '').replace('\n', ''))
            continue
        #structure end
        if (token_flag == True and words[0] == '</token>'):
            # reminder: token list start with index 0, but token id start with 1
            tokenList.append(WordToken(str(token_id), word, lemma, POS, NER))
            token_flag = False
            continue
        if (tokens_flag == True and words[0] == '</tokens>'):
            tokens_flag = False
            continue
            
        if (sentence_flag == True and words[0] == '</sentence>'):
            # output all event words index in one sentence
            #if tokenList[0].word == "-LRB-" and tokenList[1].word == "-RRB-":
            #    paragraph.append(sentence_id)
            sentenceList.append(tokenList)
            sentence_dependencyList.append(collapsed_dependenciesList)
            sentence_flag = False
            continue
        if (sentences_flag == True and words[0] == '</sentences>'):
            sentences_flag = False
            continue
        if (collapsed_dependencies_flag == True and words[0] == '</dependencies>'):
            collapsed_dependencies_flag = False
            continue
        if (collapsed_dep_flag == True and words[0] == '</dep>'):
            collapsed_dependenciesList.append(CollapsedDependency(collapsed_dep_type, collapsed_gov, collapsed_dep))
            collapsed_dep_flag = False
            collapsed_dep_type = ""
            continue
        if (token_flag == True):
            if (words[0].find('<word>') != -1):
                word = words[0].replace("<word>", "").replace("</word>", "")
                continue
            if (words[0].find('<lemma>') != -1):
                lemma = words[0].replace("<lemma>", "").replace("</lemma>", "")
                #sentence_dic[token_id] = word
                continue
            if (words[0].find('<POS>') != -1):
                POS = words[0].replace("<POS>", "").replace("</POS>", "")
                continue
            if (words[0].find('<NER>') != -1):
                NER = words[0].replace("<NER>", "").replace("</NER>", "")
                continue
    lines.close()
    return sentenceList, sentence_parseList, sentence_dependencyList
"""
if __name__ == "__main__":
    
    sentenceList, paragraph = parse_document("../real_data/preprocess/val-dec.txt.out")
    for sentence in sentenceList:
        for token in sentence:
            print token.word,
        
    print paragraph
"""