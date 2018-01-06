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

# porter=PorterStemmer()


class InvertedIndex:

    def __init__(self, doclist, out):
#         self.index = defaultdict(list)  # the inverted index
        
        self.home = expanduser("~")
        self.index = self.home + out
        self.doclist = self.home + doclist
        
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

    def writeBlock(self, dict, blockNo=1):
 
        with gzip.open(self.index + str(blockNo), 'wb') as f:
            print("writing")
            for token, postings in sorted(dict.items()):
    #             postinglist = []
#                 for posting in postings:
                print(token + '\t' + ' '.join(map(str, postings)))
                f.write(token + '\t' + ' '.join(map(str, postings)) + '\n')
#                     docID = p[0]
#                     positions = p[1]
#                     postinglist.append(':'.join([str(docID), ','.join(map(str, positions))]))
#                 print >> f, ''.join((term, '|', ';'.join(postinglist)))

#         f.close()

    def mergeIndices(self):
        files = glob(self.index + "*")
        handles = map(open, files)

    def getParams(self):
        '''get the parameters stopwords file, collection file, and the output index file'''
        param = sys.argv
        self.stopwordsFile = param[1]
        self.collectionFile = param[2]
        self.indexFile = param[3]

#     def nextDocument(self):
#                 
#         for doc in open(home+"/data/documents.list",  'r'):
#             yield doc

    def buildIndex(self, blockSize=10000):
        '''main of the program, creates the index'''
        
#         files = []
        # docsList = files.readall(files) 
#         /code/search-engine/input/testdoc.txt
        stream = io.TextIOWrapper(sys.stdin.buffer, encoding='iso-8859-2')
        index = {}
#         home = expanduser("~")
        gc.enable()  
        dict = defaultdict(lambda: array('I'))    # init dict
        bufferSize = 248   # size of empty dict of type array of unsigned ints

#         for doc in open(home+"/data/documents.list",  'r'):
#             
#             while i < len(tokenStream):

                # Process next block
#                 bufferSize = 0   # total words processed per block
                     
        docs = open(self.doclist, 'rt').readlines()
#              open(home+"/data/documents.list",  'r'):
        n = 0
        blockNo = 1
        while (n < len(docs)):
            
            dict = defaultdict(lambda: array('I'))    # current dict

            while bufferSize < blockSize:
#                     token = tokenStream[i]
#                     if token not in dict:
#                         dict[token].append(docid)
#                         dict[token] 
#                     
#                     
#                     n += 1
    #                     i += 1
                if n < len(docs):
                    nextDoc = docs[n]
                    # Get next file from documents.list
                    fname = nextDoc.rstrip()  # documents/LN-20020102023.vert
                    path = self.home + "/data/" + fname + ".gz"
        #             files.append(f)  
                    f = gzip.open(path, 'rt')
                    
                    # Parse file into sections and append text
                    parsedDoc = self.parseDoc(f)     # returns a dictionary of parsed xml sections
                    text = ''.join([v for k, v in parsedDoc.items() if v is not None and k != "docid"])
#                     docid = int(parsedDoc["docid"])
                    docid = n
        #             print(text)
        #             print(docid)
        
                    pattern = (r"^[0-9]+\s+"  # word number
                       "([a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+)[0-9]*\s+" # form
                       "[a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+[0-9]*[-_]?.*\s+"    # lemma 
                       "[A-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ0-9-=]+\s+"
                       "[a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+$")
          
#                     print(text)
                    tokens = re.findall(pattern, text, re.MULTILINE)
                    i = 0  # pos in stream
        
                    for token in tokens:
         
                        # Process next block
#                         n = 0   # total words processed per block
#                         dict = defaultdict(lambda: array('I'))    # current dict
#                         while n < blockSize:
#                         token = tokenStream[i]
                        if token not in dict:
                            dict[token].append(docid)   # append a new entry and postings list
                            bufferSize += 80            # 64 for new array, 16 for int
                        else:
                            postings = dict[token]
                            if docid not in postings:
                                postings.append(docid)
                                bufferSize += 4     # augment 4 bytes per unsigned int in array()
                                                    # this is the amortized size which depends on implementation
    #                         
                            
                    
                n += 1      # next doc
#                     i += 1
        
            self.writeBlock(dict, blockNo)
            print(blockNo, bufferSize)
            bufferSize = 0
            blockNo += 1
            gc.collect()
        

#             for doc in text.split("\n"):
#                 if doc:
#                     parsed = self.parseWords(doc)
#                     if parsed:
#                         form = parsed[0]
#                         lemma = parsed[1]
#                     print(parsed)
#                     
#             
#             break
        
#             text = ''
#             for v in parsedDoc:
#                 if v is not None: 
#                     text = text + v
            
#         print(files[0])
#         print(len(files))
        
#         self.getParams()
        






# 
# 
#         self.collFile = open(self.collectionFile, 'r')
#         self.stopwords()
        

        # bug in python garbage collector!
        # appending to list becomes O(N) instead of O(1) as the size grows if gc is enabled.
#         gc.disable()
# 
#         pagedict = {}
#         pagedict = self.parseCollection()
#         # main loop creating the index
#         while pagedict != {}:
#             lines = '\n'.join((pagedict['title'], pagedict['text']))
#             pageid = int(pagedict['id'])
#             terms = self.getTerms(lines)
#             prin(dd)
#             # build the index for the current page
#             termdictPage = {}
#             for position, term in enumerate(terms):
#                 try:
#                     termdictPage[term][1].append(position)
#                 except:
#                     termdictPage[term] = [pageid, array('I', [position])]
# 
#             # merge the current page index with the main index
#             for termpage, postingpage in termdictPage.iteritems():
#                 self.index[termpage].append(postingpage)
# 
#             pagedict = self.parseCollection()
# 
#         gc.enable()
# 
#         self.writeIndexToFile()


if __name__ == "__main__":
#     home = expanduser("~")
    doclist = "/data/documents.list"
    out = "/data/output/index"
    index = InvertedIndex(doclist, out)
    index.buildIndex()
     

