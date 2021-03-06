#!/usr/bin/env python

import math
import sys
import re
import gzip
from os.path import expanduser
from collections import defaultdict
from array import array
import gc
from collections import Counter
import heapq

class InvertedSimple:

    def __init__(self, wdir):
        self.wdir = wdir  # working directory for data
        self.n = 0  #

    def stopwords(self):
        """Read stopwords from the stopwords file"""
        with open(self.wdir + 'stoplist.txt', 'rt') as f:
            self.stopwords = {line.rstrip() for line in f}

    def parseWords(self, text):
        """Parse all terms in a text, and then preprocess:
        perform lowercasing and remove stopwords from list"""

        # Use regex to extract wordforms, or lemmas, etc.
        # Using lemma here
        pattern = (r"^[0-9]+\s+"  # word number
                   "([a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+)[0-9]*\s+" # form
                   "([a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+)[0-9]*[-_]?.*\s+"    # lemma
                   "[A-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ0-9-=]+\s+"
                   "[a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+$")

        tokens = re.findall(pattern, text, re.MULTILINE)

        return tokens

    def parseDoc(self, f):
        """Parse the docid, title, heading and text of a document as
        necessary. I used title to compile the title index (tier 1) and
        text to compile the text index (tier 2)"""

        text = f.read()
        docidMatch = re.search('<DOCID>(.*)</DOCID>', text, re.DOTALL)
        docid = docidMatch.group(1)

        return docid, text

    def parseQuery(self, topic):
        """Parse the title, description and narrative of the topic/query"""

        with open(wdir + topic, 'rt') as f:

            topic = f.read()

            # Perform a regex search on text
            qid = re.search('<num>(.*)</num>', topic, re.DOTALL)
            title = re.search('<title>(.*)</title>', topic, re.DOTALL)
            desc = re.search('<desc>(.*)</desc>', topic, re.DOTALL)
            narr = re.search('<narr>(.*)</narr>', topic, re.DOTALL)

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
        """Helper method to write index to disk"""

        with gzip.open(wdir + run + '/index.gz', 'wt') as f:
            print("writing index")
            for token, postings in sorted(self.index.items()):
                df = len(postings)
                postings = [self.splitInts(post) for post in postings]
                postings = [str(self.expandDocid(x)) + ',' + str(y) for x, y in postings]
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

    def getPostings(self, offset, index):
        with gzip.open(index, 'rt') as f:
            f.seek(offset)
            line = f.readline()
#             print(line)
            return line

    def calculateWeightQuery(self, term, method=None):
        if not method:
            return 1    # boolean method
        else:
            pass

    def cosineScore(self, query):

        pass


    def buildIndex(self):
        '''main of the program, creates the index'''

        gc.enable()
        self.index = defaultdict(lambda: array('L'))  # main index
        lengths = {}  # for calculating and storing document (cosine) lengths
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

            docid, text = self.parseDoc(f)
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

            length = 0
            for token, cnt in counts.items():
                idPlusTf = self.combineInts(docid, cnt)
                length += cnt * cnt  # add sqrd components

#                 if token not in self.index:
                self.index[token].append(idPlusTf)  # append a new entry and postings list
                    #self.lexicon[token].append(token)
            lengths[docid] = math.sqrt(length)  # sqrt

#                 else:
#                     self.index[token].append(idPlusTf)
#                   if docid not in postings:
#                     postings.append(idPlusTf)
#                     del postings
            del tokens
            del counts

            gc.collect()

        self.writeIndex()
        self.writeOffsets()
#
#                   length = 0
#             for token, cnt in counts.items():
#                 length += cnt * cnt
#             lengths[docid] = math.sqrt(length)
#             del tokens
#             del counts
#             gc.collect()
#
        with gzip.open(wdir + '/output/lengths.gz', 'wt') as f:
            print("writing doc length")
            for docid, length in lengths.items():
#                 print(self.expandDocid(docid) + '\t' + str(length) + '\n')
                f.write(self.expandDocid(docid) + '\t' + str(length) + '\n')

    def calculateDocLen(self):
        '''main of the program, creates the index'''

        gc.enable()
        lengths = {}  # main dict
        for doc in open(wdir + 'documents.list', 'rt'):
            fname = doc.rstrip()  # documents/LN-20020102023.vert
            path = wdir + fname
            f = gzip.open(path + '.gz', 'rt')
#             f = gzip.open(path + '.gz', 'rt')

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

            docid, text = self.parseDoc(f)
            docid = self.truncateDocid(docid)
#             print(docid)
#             print(text)
#             print("processing doc " + str(docid))

            pattern = (r"^[0-9]+\s+"  # word number
               "([a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+)[0-9]*\s+"  # form
               "[a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+[0-9]*[-_]?.*\s+"  # lemma
               "[A-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ0-9-=]+\s+"
               "[a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+$")

            tokens = re.findall(pattern, text, re.MULTILINE)
            counts = Counter(tokens)
            print(counts)

#             print(counts)
            length = 0
            for token, cnt in counts.items():
                length += cnt * cnt
            lengths[docid] = math.sqrt(length)
            del tokens
            del counts
            gc.collect()

        with gzip.open(wdir + '/output/lengths.gz', 'wt') as f:
            print("writing doc length")
            for docid, length in lengths.items():
                print(self.expandDocid(docid) + '\t' + str(length) + '\n')
                f.write(self.expandDocid(docid) + '\t' + str(length) + '\n')


if __name__ == "__main__":
    home = expanduser("~")

    # doclist = "/data/test/documents.list"
    # out = "/data/test/output/index-simple"
    wdir = home + "/data/"
    index = InvertedSimple(wdir)
#     print(sys.argv)

    # Build index
    if sys.argv[1] == '-b':
        index.buildIndex()

    if sys.argv[1] == '-r' and sys.argv[2]:
        offset = int(sys.argv[2])
        if len(sys.argv) == 4:
            f = sys.argv[3]
            index.getPostings(offset, f)
        else:
            f = wdir + 'output/index.gz'
            index.getPostings(offset, f)

    if sys.argv[1] == '-l':
        index.calculateDocLen()

    # Train with topics list
    if  sys.argv[1] == '-t':

        runid = 'baseline'
        # Get word offsets list
        offsets = {}
        with gzip.open(wdir + 'output/offsets.gz', 'rt') as f:
            for line in f:
                word, offset = line.rstrip().split()
                offsets[word] = int(offset)

        docLengths = {}
        with gzip.open(wdir + 'output/lengths.gz', 'rt') as f:
            for line in f:
                doc, length = line.rstrip().split()
                docLengths[doc] = float(length)

        with open(wdir + 'train-topics.list', 'rt') as topicsList:
            with gzip.open(wdir + 'output/index.gz', 'rt') as indexFile:
                with open(wdir + 'results.dat', 'wt') as results:

                    # Get query terms for a topics list
                    for query in topicsList:
                        scores = defaultdict(lambda: 0.0)
                        query = query.rstrip()
                        parsedQuery = index.parseQuery(query)

                        qid = parsedQuery['qid'].strip()  # number <num> of topic doc
                        title = parsedQuery['title']  # use for query terms

                        pattern = (r"^[0-9]+\s+"  # word number
                           "([a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+)[0-9]*\s+"  # form
                           "[a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+[0-9]*[-_]?.*\s+"  # lemma
                           "[A-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ0-9-=]+\s+"
                           "[a-zěščřžťďňńáéíýóůA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮ]+$")

                        terms = re.findall(pattern, title, re.MULTILINE)
    #                     print("topic terms ", terms)

                        # For each query term calculate weight
    #                     scores = array('f')
                        for term in terms:
                            wt = index.calculateWeightQuery(term)
                            if term in offsets:
                                offset = offsets[term]
    #                             print("getting {} at {}".format(term, offset))
                                indexFile.seek(offset)
                                line = indexFile.readline().rstrip().split()
    #                             print(line)
                                df = line[1]
                                postings = line[2:]
                                postings = [posting.split(',') for posting in postings]
    #                             print(postings)
                                for doc, tf in postings:
    #                                 print(doc, tf)
                                    scores[doc] += float(tf)
    #                                 print(doc, scores[doc])
                            else:
                                print("Word {} not in index. Skipping...".format(term))

                        # Normalize scores and insert in priority queue
                        heap = []
                        for doc, score in scores.items():
    #                         print(doc, score)
                            length = docLengths[doc]
    #                         print("len=", length)
                            scores[doc] = score / docLengths[doc]  # normalize (cosine)
    #                         print("norm", scores[doc])
                #             heapq.heappush(heap, (score, doc))

                        # Write top k scores

                #         k = 1000
                        topk = heapq.nlargest(250, scores.items(), key=lambda x: x[1])
    #                     print(topk)
    #                     with gzip.open(wdir + '/output/top-' + topic[-14:] + '.gz', 'wt') as f:
#                         with open(wdir + 'results.dat', 'wt') as f:

            #           for i in range(k):
                        for i, (doc, score) in enumerate(topk):
            #                 score, doc = heapq.heappop(heap)
                            res = str(qid) + ' ' + '0 ' + doc + ' ' + str(i) + ' ' + str(score) + ' ' + runid + '\n'
#                             print(repr(res))
                            results.write(res)
                        break
#         path = wdir + "/documents/MF-20020619203.vert"
#         f = gzip.open(path + '.gz', 'rt')
#         docid, text = index.parseDoc(f)
#         print(text)
