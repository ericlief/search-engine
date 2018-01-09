'''
Author: Arden Dertat
Contact: ardendertat@gmail.com
License: MIT License
'''

#!/usr/bin/env python

import sys
import re
import gzip
from glob import glob
import io
from os.path import expanduser
# from porterStemmer import PorterStemmer
from collections import defaultdict
from array import array
import gc
from tables.tests.test_indexvalues import BuffersizeMultipleChunksize
from collections import Counter
# porter=PorterStemmer()


class InvertedSimple:

    def __init__(self, wdir):
#         self.index = defaultdict(list)  # the inverted index
        
        #self.home = expanduser("~")
        #self.out = self.home + out + ".gz"
        #self.doclist = self.home + doclist
        self.lexicon = {}
        
    def stopwords(self):
        '''get stopwords from the stopwords file'''
        f = open(self.stopwords, 'r')
        stopwords = [line.rstrip() for line in f]
        self.stopwords = dict.fromkeys(stopwords)
        f.close()

    def parseWords(self, line):
        '''given a stream of text, get the terms from the text'''
        #line = line.lower()
        print(line)
#         terms = re.findall("^[0-9]+\b+([a-zA-Z])*[0-9]*\b+([a-zA-Z]+)[0-9]*[-_]?.*\b+([A-Z0-9-]+)\b+([0-9])+\b([a-zA-Z]+)$", line, re.DOTALL)
#         terms = re.findall("^[0-9]+\s+([a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+)[0-9]*\s+([a-zěščřžýáíéůA-ZĚŠČŘŽÝÁÍÉŮ]+)[0-9]*[-_]?.*\s+([A-ZĚŠČŘŽÝÁÍÉŮ0-9-=]+)\s+([0-9])+\s+([a-zěščřžýáíéůA-ZĚŠČŘŽÝÁÍÉŮ]+)$", line, re.DOTALL)
#         terms = re.findall("^([0-9]+)\s+([a-zA-Z]*)[0-9]*\\s+)",line, re.DOTALL)
        
        pattern = (r"^[0-9]+\s+"  # word number
                   "([a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+)[0-9]*\s+" # form
                   "([a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+)[0-9]*[-_]?.*\s+"    # lemma 
                   "[A-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ0-9-=]+\s+"
                   "[a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+$")
      
# #         to update lemma
#         pattern = (r"^[0-9]+\s+"  # word number
#                    "([a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+)[0-9]*\s+" # form
#                    "(.+)\s+"    # lemma 
#                    "[A-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ0-9-=]+\s+"
#                    "[a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+$")
      
#         print(pattern)
        
        terms = re.findall(pattern, line)
         
        print(terms)
        
        return terms
#         form = terms.group(2) i 
#         lemma = terms.group(3)
#         tag = terms.group(4)
#         parent = terms.group(5)
#         deprel = terms.group(6)
        
#         lines = text.split("\n")
#             for line in line:
#                 parseWords()
#         
        
#         line = re.sub(r'[^a-z0-9 ]', ' ', line)  # put spaces instead of non-alphanumeric characters
#         line = line.split()
#         line = [w for w in line if w not in self.stopwords]  # eliminate the stopwords
# #         line=[ porter.stem(word, 0, len(word)-1) for word in line]
#         return form, lemma

    def parseDoc(self, f):
        ''' returns the id, title and text of the next page in the collection '''
#         doc = []
#         for line in doc:
#             if line == '</DOC>\n':
#                 break
#             doc.append(line)
# 
#         curPage = ''.join(doc)
        
        doc = f.readlines()
        doc = ''.join(doc)
                
#          
#         print("orig doc")
#         print(doc)

        docid = re.search('<DOCID>(.*)</DOCID>', doc, re.DOTALL)
        title = re.search('<TITLE>(.*)</TITLE>', doc, re.DOTALL)
        head = re.search('<HEADING>(.*)</HEADING>', doc, re.DOTALL)
        text = re.search('<TEXT>(.*)</TEXT>', doc, re.DOTALL)
#         print(docid.group(1))
#         print(title.group(1))
        if docid == None or (title == None and head == None and text == None):
            raise IOError("document file does not conform to format")

        parsedDoc = {}
        parsedDoc['docid'] = docid.group(1) 
        parsedDoc['title'] = title.group(1) if title else None
        parsedDoc['head'] = head.group(1) if head else None 
        parsedDoc['text'] = text.group(1) if text else None

        return parsedDoc

    def writeIndex(self):
 
        with gzip.open(wdir + '/output/simple-index.gz', 'wt') as f:
            print("writing index")
            for token, postings in sorted(self.index.items()):
                df = len(postings)
                postings = [self.splitInts(post) for post in postings]
                postings = [str(self.expandDocid(x)) + ',' + str(y) for x, y in postings]
#                 print(post)
#                 print(list(map(self.splitInts, postings)))
#                 print(token + '\t' + ' '.join(map(str, map(self.splitInts, postings))))
#                 print(token + '\t' + str(df) + '\t' + ' '.join(postings))

                offset = f.tell()
                f.write(token + '\t' + str(df) + '\t' + ' '.join(postings) + '\n')
                self.index[token] = offset

    def writeOffsets(self):
 
        with gzip.open(wdir + '/output/offsets.gz', 'wt') as f:
            print("writing offsets")
            for token, offset in sorted(self.index.items()):
#                 print(token + '\t' + str(offset) + '\n')
                f.write(token + '\t' + str(offset) + '\n')
                
    
    def mergeIndices(self):
        files = glob(self.index + "*")
        handles = map(open, files)

    def getParams(self):
        '''get the parameters stopwords file, collection file, and the output index file'''
        param = sys.argv
        self.stopwordsFile = param[1]
        self.collectionFile = param[2]
        self.indexFile = param[3]

    def combineInts(self, x, y):
        hi = x << 32
        lo = y & 0x0000FFFF
        return hi | lo

    def splitInts(self, z):
        x = z >> 32
        y = z & 0x0000FFFF
#         return str(x) + ',' + str(y)
        return x, y

    def truncateDocid(self, docid):
        if docid[0] == 'L':
            docid = '1' + docid[7:]  # begins with LN
        else:
            docid = '2' + docid[7:]  # begins with MF

        return int(docid)

    def expandDocid(self, docid):
        docid = str(docid)
        if docid[0] == '1':
            docid = 'LN-2002' + docid[1:]  # begins with LN
        else:
            docid = 'MF-2002' + docid[1:]  # begins with MF

        return docid

    def buildIndex(self):
        '''main of the program, creates the index'''
        
        gc.enable()
        self.index = defaultdict(lambda: array('L'))  # main dict
        for doc in open(wdir + 'documents.list', 'rt'):
            fname = doc.rstrip()  # documents/LN-20020102023.vert
            path = wdir + fname
            f = gzip.open(path + '.gz', 'rt')

            # Parse file into sections and append text
            parsedDoc = self.parseDoc(f)  # returns a dictionary of parsed xml sections
            text = ''.join([v for k, v in parsedDoc.items() if v is not None and k != "docid"])
            docid = parsedDoc["docid"]
#             if docid[0] == 'L':
#                 docid = '1' + docid[7:]     # begins with LN
#             else:
#                 docid = '2' + docid[7:]     # begins with MF
#
#             docid = int(docid)
            docid = self.truncateDocid(docid)
#             print("processing doc " + str(docid))

            pattern = (r"^[0-9]+\s+"  # word number
               "([a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+)[0-9]*\s+"  # form
               "[a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+[0-9]*[-_]?.*\s+"  # lemma
               "[A-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ0-9-=]+\s+"
               "[a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+$")

            tokens = re.findall(pattern, text, re.MULTILINE)
            counts = Counter(tokens)
            
#             print(counts)

            for token, cnt in counts.items():
                idPlusTf = self.combineInts(docid, cnt)
#                 if token not in self.index:
                self.index[token].append(idPlusTf)  # append a new entry and postings list
                    #self.lexicon[token].append(token) 

#                 else:
#                     self.index[token].append(idPlusTf)
#                   if docid not in postings:
#                     postings.append(idPlusTf)
#                     del postings
            gc.collect()

        self.writeIndex()
        self.writeOffsets()

if __name__ == "__main__":
    home = expanduser("~")

    #doclist = "/data/test/documents.list"
    #out = "/data/test/output/index-simple"
    wdir = home + "/data/"
    index = InvertedSimple(wdir)
    index.buildIndex()
     

