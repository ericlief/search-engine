#!/usr/bin/env python

import math
import sys
import re
import gzip
from os.path import expanduser
from os.path import expanduser
from collections import defaultdict
from array import array
import gc
from collections import Counter
import heapq


class InvertedIndex:

    def __init__(self, wdir):
#         self.index = defaultdict(list)  # the inverted index
        
        # self.home = expanduser("~")
        # self.out = self.home + out + ".gz"
        # self.doclist = self.home + doclist
        self.lexicon = {}
        self.wdir = wdir
        self.n = 0
        
    def stopwords(self):
        '''Read stopwords from the stopwords file'''
        with open(self.wdir + 'stoplist.txt', 'rt') as f:
            self.stopwords = {line.rstrip() for line in f}

    def parseWords(self, text):
        '''given a stream of text, get the terms from the text'''
 
        # Use regex to extract wordforms, or lemmas, etc.
        # Using lemma here
        pattern = (r"^[0-9]+\s+"  # word number
           "[a-zěščřžťďňńáéíýóůúA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮÚ]+[0-9]*\s+"  # form
           "([a-zěščřžťďňńáéíýóůúA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮÚ]+)[0-9]*[-_]?.*\s+"  # lemma
           "[A-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮÚ0-9-=]+\s+"
           "[a-zěščřžťďňńáéíýóůúA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮÚ]+$")

        tokens = re.findall(pattern, text, re.MULTILINE)

        # Normalize: lowercase, filter stopwords
        tokens = [token.lower() for token in tokens if token not in self.stopwords]

        return tokens

    def parseDoc(self, f):
        ''' returns the id, title, heading and text of a document '''

#
        text = f.read()
#         print(text)
        docidMatch = re.search('<DOCID>(.*)</DOCID>', text, re.DOTALL)
        docid = docidMatch.group(1)
        titleMatch = re.search('<TITLE>(.*)</TITLE>', text, re.DOTALL)
        if titleMatch is not None:
            title = titleMatch.group(1)
        else:
            title = None

#             Uncomment for text
#         textMatch = re.findall('<TEXT>(.*)</TEXT>', text, re.DOTALL)
#         textMatch = re.search('<TEXT>(.*)</TEXT>', text, re.DOTALL)
#         text = textMatch.group(1)
#         if len(textMatch) > 1:
#             raise EnvironmentError()
# 
#         text = ''.join(textMatch)
#         print(text)
#         head = re.search('<HEADING>(.*)</HEADING>', doc, re.DOTALL)
#         text = re.search('<TEXT>(.*)</TEXT>', doc, re.DOTALL)
#         print(docid.group(1))
#         print(title.group(1))
#         if docid is None or (title is None and head is None and text is None):
#             raise IOError("document file does not conform to format")

#         parsedDoc = {}
#         parsedDoc['docid'] = docid.group(1)
#         parsedDoc['title'] = title.group(1) if title else None
#         parsedDoc['head'] = head.group(1) if head else None
#         parsedDoc['text'] = text.group(1) if text else None

        return docid, title

    def parseQuery(self, topic):
        ''' returns the title, description and narrative of the topic/query'''
        
        with open(wdir + topic, 'rt') as f:

            topic = f.read()

            # Perform a regex search on text
            qid = re.search('<num>(.*)</num>', topic, re.DOTALL)
            title = re.search('<title>(.*)</title>', topic, re.DOTALL)
            desc = re.search('<desc>(.*)</desc>', topic, re.DOTALL)
            narr = re.search('<narr>(.*)</narr>', topic, re.DOTALL)
    #         print(docid.group(1))
    #         print(title.group(1))
            if qid is None or (title is None and desc is None and narr is None):
                raise IOError("topic file does not conform to format")

            # Capture and store groups in a dict
            parsedTopic = {}
            parsedTopic['qid'] = qid.group(1)
            parsedTopic['title'] = title.group(1)
            parsedTopic['desc'] = desc.group(1) if desc else None
            parsedTopic['narr'] = narr.group(1) if narr else None

            return parsedTopic

    def writeIndex(self):
 
        with gzip.open(wdir + run + '/index.gz', 'wt') as f:
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

        with gzip.open(wdir + run + '/offsets.gz', 'wt') as f:
            print("writing offsets")
            for token, offset in sorted(self.index.items()):
#                 print(token + '\t' + str(offset) + '\n')
                f.write(token + '\t' + str(offset) + '\n')


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

    def getPostings(self, offset, index):
        with gzip.open(index, 'rt') as f:
            f.seek(offset)
            line = f.readline()
#             print(line)
            return line
        
    def calculateWeightOfTerm(self, term, tf, df, scheme=None, doc=None):
        if not scheme:
            return 1  # boolean method
        
        # Get term freq scheme
        if scheme[0] == 'n':  # natural
            tf = float(tf)
        elif scheme[0] == 'l':  # logarithmic
            tf = 1.0 + math.log10(tf)
        elif scheme[0] == 'a' and doc is not None:  # augmented (scale by max tf)
                tf = .4 + .6 * tf / maxTF[doc]
        elif scheme[0] == 'b':  # boolean
            tf = 1.0 if tf > 0 else 0.0
        elif scheme[0] == 'L':  # boolean
            tf = (1.0 + math.log10(tf) / 1.0 + math.log10(aveTF[doc]))
        else:
            raise AttributeError("Illegal scheme for tf.")

        # Get doc freq scheme  
        if scheme[1] == 'n':    # natural    
            df = 1.0
        elif scheme[1] == 't':  # idf
            df = math.log10(float(self.n)/df)
        elif scheme[1] == 'p':  # prob idf
            df = max(0, math.log10((self.n - df)/float(df)))
#         print("tf={}\tdf={}".format(tf, df))
        else:
            raise AttributeError("Illegal scheme for df.")
        return tf * df

    def cosineScore(self, query, docScheme, queryScheme, k):
        
        # Parse query from title
        query = query.rstrip()
        parsedQuery = index.parseQuery(query)
        qid = parsedQuery['qid'].strip()  # number <num> of topic doc
        title = parsedQuery['title']  # use for query terms
        
        # Get terms and normalize length if necessary
        terms = index.parseWords(title)     # lowercase text, filter stopwords
        terms = Counter(terms)              # get counts and store in dict/hash
    
        # For each query term calculate weight
        scores = defaultdict(lambda: 0.0)
        for term, tf in terms.items():
            if term in offsets:
                offset = offsets[term]
#                 print("getting {} at {}".format(term, offset))
                indexFile.seek(offset)
                line = indexFile.readline().rstrip().split()
                df = float(line[1])
                wQuery = self.calculateWeightOfTerm(term, tf, df, queryScheme)   # weight of query 
#                 print("df", df)
#                 print("w query=", wQuery)
                postings = line[2:]
                postings = [posting.split(',') for posting in postings]
                postings = [(doc, int(tf)) for doc, tf in postings]
#                             print(postings)
                for doc, tf in postings:
#                     print(doc, tf)
                    wDoc = self.calculateWeightOfTerm(term, tf, df, docScheme, doc)   # weight of doc 
                    score = wQuery * wDoc
#                     print('score', score)
                    scores[doc] += wQuery * wDoc
#                     score += float(tf)
#                     print(doc, scores[doc])

            else:
                print("Word {} not in index. Skipping...".format(term))
        
        lengthQuery = 0.0
        if queryScheme[-1] == 'c':             # cosine norm
            for term, tf in terms.items():
                lengthQuery += tf * tf
                # print(token, tf, length)
            lengthQuery = math.sqrt(lengthQuery)
        else:
            lengthQuery = 1.0  # no norm

#         print("len query=", lengthQuery)

        # Normalize scores
        if docScheme[-1] == 'c':
            for doc, score in scores.items():
#                 print(doc, score)
                lengthDoc = docLengths[doc]
#                 print("len=", lengthDoc, lengthQuery)
                scores[doc] /= lengthDoc * lengthQuery  # normalize (cosine)
#                 print("norm", scores[doc])
        elif docScheme[-1] == 'u':
            a = .5  # slope normally between .25-.4
            pivot = 2730  # ave bytes in disks 1-2 of TREC
            for doc, score in scores.items():
#                 print(doc, score)
                pivotedLengthDoc = a * uniq[doc] + (1 - a) * pivot
#                 print("len=", lengthDoc, lengthQuery)
                scores[doc] /= pivotedLengthDoc * lengthQuery  # normalize (cosine)
#
        # Get top k scores
        topK = heapq.nlargest(k, scores.items(), key=lambda x: x[1])

        # Write to disk
        for i, (doc, score) in enumerate(topK):
            res = str(qid) + ' ' + '0 ' + doc + ' ' + str(i) + ' ' + str(score) + ' ' + run + '\n'
            results.write(res)

        return topK
    
    def buildIndex(self):
        '''main of the program, creates the index'''
        
        gc.enable()
        self.index = defaultdict(lambda: array('L'))  # main index
        lengths = {}  # for calculating and storing document (cosine) lengths
        cnt = 0
        
        # Parse all documents
        for doc in open(wdir + 'documents.list', 'rt'):
            fname = doc.rstrip()  # documents/LN-20020102023.vert
            path = wdir + fname
            f = gzip.open(path + '.gz', 'rt')

            # Parse file into sections and append text
#             parsedDoc = self.parseDoc(f)  # returns a dictionary of parsed xml sections
#             text = ''.join([v for k, v in parsedDoc.items() if v is not None and k != "docid"])
#             docid = parsedDoc["docid"]
#             if docid[0] == 'L':
#                 docid = '1' + docid[7:]     # begins with LN
#             else:
#                 docid = '2' + docid[7:]     # begins with MF
#
#             docid = int(docid)

            # Truncate prefix of doc id in order to save space: will expand later
            did, text = self.parseDoc(f)
            if text is None:
                continue
            docid = self.truncateDocid(did)
#             print("processing doc " + str(docid))

#             # Use regex to extract wordforms, or lemmas, etc.
#             # Using lemma here
#             pattern = (r"^[0-9]+\s+"  # word number
#                "[a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+[0-9]*\s+"  # form
#                "([a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+)[0-9]*[-_]?.*\s+"  # lemma
#                "[A-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ0-9-=]+\s+"
#                "[a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+$")
# 
#             tokens = re.findall(pattern, text, re.MULTILINE)
# 
#             # Normalize: lowercase, filter stopwords
#             tokens = [token.lower() for token in tokens if token not in self.stopwords]
            
            tokens = self.parseWords(text)
            if not tokens:
                continue
            counts = Counter(tokens)
#             print(counts)

            # Calculate normalized document length
            length = 0.0
            for token, tf in counts.items():

                # Combine 32 bit docid and tf into a 64-bit long to save space (recover later)
                idPlusTf = self.combineInts(docid, tf)
                length += tf * tf  # add sqrd components
                self.index[token].append(idPlusTf)  # append a new entry and postings list
            lengths[docid] = math.sqrt(length)  # take sqrt
            cnt += 1
            f.close()
            
        self.writeIndex()
        self.writeOffsets()
        
        # Write doc lengths to disk
        with gzip.open(wdir + run + '/lengths.gz', 'wt') as f:
            print("writing doc length")
            for docid, length in lengths.items():
#                 print(self.expandDocid(docid) + '\t' + str(length) + '\n')
                # f.write(docid + '\t' + str(length) + '\n')
                f.write(self.expandDocid(docid) + '\t' + str(length) + '\n')

        print("Wrote {} doc lengths".format(cnt))

        # Get rid of garbage
        del tokens
        del counts
        del lengths
        gc.collect()

    def calculateNumberUniqTerms(self):
        '''main of the program, creates the index'''

        gc.enable()
        uniq = {}  # main dict
        cnt = 0
        for doc in open(wdir + 'documents.list', 'rt'):
            fname = doc.rstrip()  # documents/LN-20020102023.vert
            path = wdir + fname
            f = gzip.open(path + '.gz', 'rt')
            docid, text = self.parseDoc(f)
            if text is None:
                continue
            docid = self.truncateDocid(docid)
            tokens = self.parseWords(text)
            if not tokens:
                continue
            counts = Counter(tokens)
            uniq[docid] = len(counts)
#             print(docid, uniq[docid])
#             print(counts)
            del tokens
            del counts
            gc.collect()
            cnt += 1
            
        with gzip.open(wdir + run + '/uniq.gz', 'wt') as f:
            print("writing uniq terms")
            for docid, c in uniq.items():
                f.write(self.expandDocid(docid) + '\t' + str(c) + '\n')

        print("Wrote {} uniq words per doc".format(cnt))
        del uniq
        gc.collect()
        
    def parseWeightingScheme(self, scheme):
        pass

    def calculateDocLen(self):
        '''main of the program, creates the index'''

        gc.enable()
        lengths = {}  # main dict
        for doc in open(wdir + 'documents.list', 'rt'):
            fname = doc.rstrip()  # documents/LN-20020102023.vert
            path = wdir + fname
            f = gzip.open(path + '.gz', 'rt')
            docid, text = self.parseDoc(f)
            if text is None:
                continue
            docid = self.truncateDocid(docid)
            tokens = self.parseWords(text)
            if not tokens:
                continue
            counts = Counter(tokens)
#             print(counts)
            length = 0
            for token, cnt in counts.items():
                length += cnt * cnt
            lengths[docid] = math.sqrt(length)
            del tokens
            del counts
            gc.collect()

        with gzip.open(wdir + run + '/lengths.gz', 'wt') as f:
            print("writing doc length")
            for docid, length in lengths.items():
                print(self.expandDocid(docid) + '\t' + str(length) + '\n')
                f.write(self.expandDocid(docid) + '\t' + str(length) + '\n')

    def calculateMaxTermFreq(self):
        '''main of the program, creates the index'''

        gc.enable()
        max = {}  # main dict
        for doc in open(wdir + 'documents.list', 'rt'):
            fname = doc.rstrip()  # documents/LN-20020102023.vert
            path = wdir + fname
            f = gzip.open(path + '.gz', 'rt')
            docid, text = self.parseDoc(f)

#             print(docid, text)

            if text is None:
                continue
            docid = self.truncateDocid(docid)
            tokens = self.parseWords(text)
            if not tokens:
                continue
#             print(tokens)

            counts = Counter(tokens)
            max[docid] = counts.most_common(1)[0][1]  # tf of most common element
#             print(docid, counts.most_common(1))
            del tokens
            del counts
            gc.collect()

        with gzip.open(wdir + run + '/max-tf.gz', 'wt') as f:
            print("writing max tf")
            for docid, tf in max.items():
#                 print(self.expandDocid(docid) + '\t' + str(length) + '\n')
                f.write(self.expandDocid(docid) + '\t' + str(tf) + '\n')

    def calculateAveTermFreq(self):
        '''main of the program, creates the index'''

        gc.enable()
        aveTF = {}  # ave tfs per doc
        for doc in open(wdir + 'documents.list', 'rt'):
            fname = doc.rstrip()  # documents/LN-20020102023.vert
            path = wdir + fname
            f = gzip.open(path + '.gz', 'rt')
            docid, text = self.parseDoc(f)
            if text is None:
                continue
            docid = self.truncateDocid(docid)
            tokens = self.parseWords(text)
            if not tokens:
                continue
            counts = Counter(tokens)
            s = sum([c for t, c in counts.items()])  # sum tfs
            aveTF[docid] = s / float(len(counts))
            del tokens
            del counts
            del s
            gc.collect()

        with gzip.open(wdir + run + '/ave-tf.gz', 'wt') as f:
            print("writing ave tf")
            for docid, ave in aveTF.items():
#                 print(self.expandDocid(docid) + '\t' + str(length) + '\n')
                f.write(self.expandDocid(docid) + '\t' + str(ave) + '\n')


if __name__ == "__main__":
    home = expanduser("~")
    wdir = home + "/data/"
    index = InvertedIndex(wdir)
    index.stopwords()
    index.n = 81735 # no documents
    
    # Build index
    if len(sys.argv) < 2:
        raise ValueError('Must provide run type')

    run = sys.argv[1]  # baseline, etc.

    if len(sys.argv) == 6:
 
        # Train with topics list
        if  sys.argv[2] == '-t':
            
            print("training topics")

            docScheme = sys.argv[3]     # ddd triplet
            queryScheme = sys.argv[4]   # qqq triplet
            k = int(sys.argv[5])

            # Get word offsets list
            offsets = {}
            with gzip.open(wdir + run + '/offsets.gz', 'rt') as f:
                for line in f:
                    word, offset = line.rstrip().split()
                    offsets[word] = int(offset)

            docLengths = {}
            with gzip.open(wdir + run + '/lengths.gz', 'rt') as f:
                for line in f:
                    doc, length = line.rstrip().split()
                    docLengths[doc] = float(length)
            
            maxTF = {}
            with gzip.open(wdir + run + '/max-tf.gz', 'rt') as f:
                for line in f:
                    doc, tf = line.rstrip().split()
                    maxTF[doc] = int(tf)

            aveTF = {}
            with gzip.open(wdir + run + '/ave-tf.gz', 'rt') as f:
                for line in f:
                    doc, tf = line.rstrip().split()
                    aveTF[doc] = float(tf)

            uniq = {}
            with gzip.open(wdir + run + '/uniq.gz', 'rt') as f:
                for line in f:
                    doc, u = line.rstrip().split()
                    uniq[doc] = int(u)

            with open(wdir + 'train-topics.list', 'rt') as topicsList:
                with gzip.open(wdir + run + '/index.gz', 'rt') as indexFile:
                    with open(wdir + 'results' + '-' + run + '.dat', 'wt') as results:
    
                        # Get query terms for `title` field in topics list
                        scores = defaultdict(lambda: 0.0)       # store all topic scores here
                        for query in topicsList:

#                             print(query)
                            score = index.cosineScore(query, docScheme, queryScheme, k)

    else:

        # Fetch posting
        if 'f' in sys.argv[2]:
            offset = int(sys.argv[2])
            if len(sys.argv) == 4:
                f = sys.argv[3]
                index.getPostings(offset, f)
            else:
                f = wdir + run + '/index.gz'
                index.getPostings(offset, f)

        # Build index
        if 'b' in sys.argv[2]:
            index.buildIndex()
    
        # Calculate doc lengths
        if 'l' in sys.argv[2]:
            index.calculateDocLen()
    
        # Calculate num unique terms
        if 'u' in sys.argv[2]:
            index.calculateNumberUniqTerms()

        # Calculate max lengh for augmented tf
        if 'm' in sys.argv[2]:
            index.calculateMaxTermFreq()

        # Calculate average term frequency
        if 'a' in sys.argv[2]:
            index.calculateAveTermFreq()

#                             break
#                             scores = defaultdict(lambda: 0.0)
#                             query = query.rstrip()
#                             parsedQuery = index.parseQuery(query)
#     
#                             qid = parsedQuery['qid'].strip()  # number <num> of topic doc
#                             title = parsedQuery['title']  # use for query terms
#     
#                             terms = index.parseWords(title)
#     #                         pattern = (r"^[0-9]+\s+"  # word number
#     #                            "([a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+)[0-9]*\s+"  # form
#     #                            "[a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+[0-9]*[-_]?.*\s+"  # lemma
#     #                            "[A-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ0-9-=]+\s+"
#     #                            "[a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+$")
#     #
#     #                         terms = re.findall(pattern, title, re.MULTILINE)
#     
#     #                         # Normalize query terms: lowercase and filter stopwords
#     #                         terms = [term.lower() for term in terms if term not in index.stopwords]
#     
#         #                     print("topic terms ", terms)
#     
#                             # For each query term calculate weight
#         #                     scores = array('f')
#                             for term in terms:
#                                 wt = index.calculateWeightQuery(term)
#                                 if term in offsets:
#                                     offset = offsets[term]
#         #                             print("getting {} at {}".format(term, offset))
#                                     indexFile.seek(offset)
#                                     line = indexFile.readline().rstrip().split()
#         #                             print(line)
#                                     df = line[1]
#                                     postings = line[2:]
#                                     postings = [posting.split(',') for posting in postings]
#         #                             print(postings)
#                                     for doc, tf in postings:
#         #                                 print(doc, tf)
#                                         scores[doc] += float(tf)
#         #                                 print(doc, scores[doc])
#                                 else:
#                                     print("Word {} not in index. Skipping...".format(term))
    
#                             # Normalize scores and insert in priority queue
#                             heap = []
#                             for doc, score in scores.items():
#         #                         print(doc, score)
#                                 length = docLengths[doc]
#         #                         print("len=", length)
#                                 scores[doc] = score / docLengths[doc]  # normalize (cosine)
#         #                         print("norm", scores[doc])
#                     #             heapq.heappush(heap, (score, doc))
#
#                             # Write top k scores
#
#                     #         k = 1000
#                             topk = heapq.nlargest(1000, scores.items(), key=lambda x: x[1])
#         #                     print(topk)
#         #                     with gzip.open(wdir + '/output/top-' + topic[-14:] + '.gz', 'wt') as f:
#     #                         with open(wdir + 'results.dat', 'wt') as f:
#
#                 #           for i in range(k):
#                             for i, (doc, score) in enumerate(topk):
#                 #                 score, doc = heapq.heappop(heap)
#                                 res = str(qid) + ' ' + '0 ' + doc + ' ' + str(i) + ' ' + str(score) + ' ' + run + '\n'
#     #                             print(repr(res))
#                                 results.write(res)
                            
#         path = wdir + "/documents/MF-20020619203.vert"
#         f = gzip.open(path + '.gz', 'rt')
#         docid, text = index.parseDoc(f)
#         print(text)
