#!/usr/bin/env python

import math
import sys
import os
import re
import gzip
from os.path import expanduser
from collections import defaultdict
from array import array
import gc
from collections import Counter
import heapq


class InvertedIndex:
    """
    Class for representing an inverted index, containing methods
    for building the index and storing its files, as well as related
    functionality such as document lengths, computing scores (tiered and
    untiered"""

    def __init__(self, n=0, isTiered=False):
        # self.cwd = cwd  # working directory for data
        self.n = n  # initial size (here static)
        self.isTiered = isTiered  # is this a two-tiered index or not
        if isTiered:
            nTiers = 2
            self.indexes = []
            for i in range(nTiers):
                self.indexes.append(defaultdict(lambda: array('L')))
            assert (self.indexes[0] is not self.indexes[1])
            # assert (self.indexes[0] != self.indexes[1])


            print("tiered", self.indexes)
        else:
            self.index = defaultdict(lambda: array('L'))  # main index


    def stopwords(self):
        """Read stopwords from the stopwords file"""
        par = os.path.dirname(os.getcwd())
        with open(par + '/search-engine/input/stopwords.txt', 'rt') as f:
            self.stopwords = {line.rstrip() for line in f}

    def parseWords(self, text, pp=True):
        """Parse all terms in a text, and then preprocess if
        the pp parameter is passed as True; pp includes:
        lowercasing and stopword removal"""

        # Use regex to extract wordforms, or lemmas, etc.
        # Using lemma here depending on the preprocessing flag pp
        if pp:
            pattern = (r"^[0-9]+\s+"  # word number
           "[a-zěščřžťďňńáéíýóůúA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮÚ]+[0-9]*\s+"
           "([a-zěščřžťďňńáéíýóůúA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮÚ]+)[0-9]*[-_]?.*\s+"  # use lemma
           "[A-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮÚ0-9-=]+\s+"
           "[a-zěščřžťďňńáéíýóůúA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮÚ]+$")

        else:
            pattern = (r"^[0-9]+\s+"  # word number
           "([a-zěščřžťďňńáéíýóůúA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮÚ]+)[0-9]*\s+"  # use form
           "[a-zěščřžťďňńáéíýóůúA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮÚ]+[0-9]*[-_]?.*\s+"
           "[A-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮÚ0-9-=]+\s+"
           "[a-zěščřžťďňńáéíýóůúA-ZĚŠČŘŽŤĎŇŃÁÉÍÝÓŮÚ]+$")

        tokens = re.findall(pattern, text, re.MULTILINE)

        # Normalize: lowercase, filter stopwords
        if pp:
            #tokens = [token.lower() for token in tokens if token not in self.stopwords]
            tokens = [token.lower() for token in tokens]

        # print(tokens)
        return tokens

    def parseDoc(self, f, title=False):
        """Parse the docid, title, heading and text of a document as
        necessary. I used title to compile the title index (tier 1) and
        text to compile the text index (tier 2)"""

        text = f.read()
        docidMatch = re.search('<DOCID>(.*)</DOCID>', text, re.DOTALL)
        docid = docidMatch.group(1)

        # Get title and return for tier 1 index
        if title:
            titleMatch = re.search('<TITLE>(.*)</TITLE>', text, re.DOTALL)
            if titleMatch is not None:
                text = titleMatch.group(1)
            else:
                text = None

        # Otherwise, parse and return <TEXT>
        else:
            textMatch = re.search('<TEXT>(.*)</TEXT>', text, re.DOTALL)
            text = textMatch.group(1)
#             if len(textMatch) > 1:
#                 raise EnvironmentError()

        #if docid is None or text is None:
        #    raise IOError("document file does not conform to format")

        return docid, text  # change for text

    def parseQuery(self, topic):
        """Parse the title, description and narrative of the topic/query"""

        with open(topic, 'rt') as f:

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

    def combineInts(self, x, y):
        """Helper method to combine a docid and tf integers, so as
        to speed up indexing and memory usage."""

        hi = x << 32
        lo = y & 0x0000FFFF
        return hi | lo

    def splitInts(self, z):
        """Helper method to recover original docid and tf resulting
        from above CombineInts method."""

        x = z >> 32
        y = z & 0x0000FFFF
        return x, y

    def truncateDocid(self, docid):
        """Helper method to shorten redundancies in
        docids for this task, converting to an int, thereby saving memory"""

        if docid[0] == 'L':
            docid = '1' + docid[7:]  # begins with LN
        else:
            docid = '2' + docid[7:]  # begins with MF

        return int(docid)

    def expandDocid(self, docid):
        """Helper method to recover full docid in string form"""

        docid = str(docid)
        if docid[0] == '1':
            docid = 'LN-2002' + docid[1:]  # begins with LN
        else:
            docid = 'MF-2002' + docid[1:]  # begins with MF

        return docid

    def getPostings(self, offset, tier):
        """Given a word offset from list, get its posting list"""

        with gzip.open(cwd + tier + '/index.gz', 'rt') as f:
            f.seek(offset)
            line = f.readline()
            return line

    def calculateWeightOfTerm(self, term, tf, df, scheme=None, doc=None, tier=0):
        """Given a term, its tf, df, a scheme list (doc, query), a
        document (unless we're dealing with a query, and a tier index
        calculate its unnormalized weight (tf * df)."""

        if not scheme:
            return 1  # boolean method

        # Get term freq scheme
        if scheme[0] == 'n':  # natural
            tf = float(tf)
        elif scheme[0] == 'l':  # logarithmic
            tf = 1.0 + math.log10(tf)
        elif scheme[0] == 'a' and doc is not None:  # augmented (scale by max tf)
                tf = .4 + .6 * tf / maxTF[tier][doc]
        elif scheme[0] == 'b':  # boolean
            tf = 1.0 if tf > 0 else 0.0
        elif scheme[0] == 'L':  # boolean
            tf = (1.0 + math.log10(tf) / 1.0 + math.log10(aveTF[tier][doc]))
        else:
            raise AttributeError("Illegal scheme for tf.")

        # Get doc freq scheme
        if scheme[1] == 'n':    # natural
            df = 1.0
        elif scheme[1] == 't':  # idf
            df = math.log10(float(self.n)/df)
        elif scheme[1] == 'p':  # prob idf
            df = max(0, math.log10((self.n - df)/float(df)))
        else:
            raise AttributeError("Illegal scheme for df.")
        return tf * df

    def atEndOfLists(self, lists):
        """Helper method for DAAT processing of indices"""
        for pl in lists:
            if pl.current() != -1:
                print("not at end of list")
                return False
        return True

    def cosineScoreTiered(self, query, docScheme, queryScheme, k, pp=True):
        """Method for computing the cosine score of a
        query using a tiered index. The algorithm works as follows:
        first level 1 scores are computed. If all k docs are
        not found, the next level is searched and merged with
        previous level results."""

        # Parse query from title
        query = query.rstrip()
        parsedQuery = index.parseQuery(query)
        qid = parsedQuery['qid'].strip()  # number <num> of topic doc
        title = parsedQuery['title']  # use for query terms

        # Get terms and normalize length if necessary
        terms = index.parseWords(title, pp)     # lowercase text, filter stopwords
        terms = Counter(terms)              # get counts and store in dict/hash

        # For each query term calculate weight
        scores = defaultdict(lambda: 0.0)
        numTiers = 2  # first (base/titles) and second tier (text)
        n = 0  # num elements in heap
        tier = 0  # start here at base tier
        while tier < numTiers and n <= k:
            print('Tier', tier)

            # First get all lists for query terms and query term weights for this tier
            for term, tf in terms.items():
                if term in offsets[tier]:
                    offset = offsets[tier][term]  # byte offsets
                    indexes[tier].seek(offset)    # get postings

                    # Process postings
                    # term     10      LN-20020114114,1 LN-20020121109,2 LN-20020123062,3
                    line = indexes[tier].readline().rstrip().split()
                    df = float(line[1])
                    queryWeight = self.calculateWeightOfTerm(term, tf, df,
                                                             queryScheme, doc=None, tier=tier)  # weight of query
                    postings = line[2:]
                    postings = [posting.split(',') for posting in postings]
                    postings = [(doc, int(tf)) for doc, tf in postings]

                    # Get scores TAAT for docs in postings
                    for doc, tf in postings:
                        docWeight = self.calculateWeightOfTerm(term, tf, df,
                                        docScheme, doc, tier)   # weight of doc
                        score = queryWeight * docWeight
                        scores[doc] += score

                else:
                    print("Word {} not in index. Skipping...".format(term))

            # Normalize query if specified (only cosine normalization here)
            lengthQuery = 0.0
            if queryScheme[-1] == 'c':             # cosine norm
                for term, tf in terms.items():
                    lengthQuery += tf * tf
                    # print(token, tf, length)
                lengthQuery = math.sqrt(lengthQuery)
            else:
                lengthQuery = 1.0  # no norm

            # Normalize scores
            if docScheme[-1] == 'c':
                for doc, score in scores.items():
                    lengthDoc = docLengths[tier][doc]
                    scores[doc] /= lengthDoc * lengthQuery  # normalize (cosine)
            elif docScheme[-1] == 'u':
                a = .65  # slope normally between .25-.4, but .65 seemed optimal
                pivot = 2630  # 2630 = ave bytes in disks 1-2 of TREC
                for doc, score in scores.items():
                    pivotedLengthDoc = a * uniq[tier][doc] + (1 - a) * pivot
                    scores[doc] /= pivotedLengthDoc * lengthQuery  # normalize (cosine)

            # Get top k scores using a heap method
            topKScores = heapq.nlargest(k, scores.items(), key=lambda x: x[1])
            n = len(topKScores)

            # Have k scores, done, write to disk, return
            if n == k or tier == numTiers - 1:
                print("more than k scores. Or just done")

                # Write to disk
                for i, (doc, score) in enumerate(topKScores):
                    res = str(qid) + ' ' + '0 ' + doc + ' ' + str(i) + ' ' + str(score) + ' ' + runId + '\n'
                    results.write(res)
                return topKScores

            # Need more scores so go to next tier
            else:
                print("less than k scores moving to next tier")
                tier += 1


    def cosineScoreDAAT(self, query, docScheme, queryScheme, k):
        """
        Document-at-a-time method for computing cosine score.

        Unfinished, as I couldn't figure out how to get
        the iterator to work. Also need to pre-compute the
        upper/max threshold for documents. """

        class PostingsList:

            def __init__(self, l):
                self.list = l
                self.n = len(l)
                self.ptr = 0
                self.peakPtr = 0

            def peak(self):
                if self.peakPtr < self.n:
                    self.peakPtr += 1
                    return self.list[self.peakPtr - 1][0]   # return next doc
                else:
                    return -1

            def next(self):
                if self.ptr < self.n:
                    self.ptr += 1
                    return self.list[self.ptr - 1][0]   # return next doc
                else:
                    return -1

            def resetPeakPtr(self):
                peakPtr = ptr

            def current(self):
                if self.ptr < self.n:
                    return self.list[self.ptr][0]  # return current doc
                return -1

        # Parse query from title
        query = query.rstrip()
        parsedQuery = index.parseQuery(query)
        qid = parsedQuery['qid'].strip()  # number <num> of topic doc
        title = parsedQuery['title']  # use for query terms

        # Get terms and normalize length if necessary
        terms = index.parseWords(title)     # lowercase text, filter stopwords
        terms = Counter(terms)              # get counts and store in dict/hash

        # For each query term calculate weight
        #         scores = defaultdict(lambda: 0.0)
        scores = []  # heap
        numTiers = 2  # first (base/titles) and second tier (text)
        n = 0  # num elements in heap
        tier = 0  # start here at base tier
        while tier < numTiers and n <= k:
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

            # First get all lists for query terms and query term weights for this tier
            lists = []      # tier indices
            qWeights = []   # tier query w
            for i, (term, tf) in enumerate(terms.items()):
                if term in offsets[tier]:
                    offset = offsets[tier][term]
                    #                 print("getting {} at {}".format(term, offset))
                    indexes[tier].seek(offset)
                    line = indexes[tier].readline().rstrip().split()
                    df = float(line[1])
                    qWeight = self.calculateWeightOfTerm(term, tf, df,
                                                         queryScheme, tier)  # weight of query
                    #                 print("df", df)
                    #                 print("w query=", wQuery)
                    qWeights.append(qWeight)
                    postings = line[2:]
                    postings = [posting.split(',') for posting in postings]
                    postings = [(doc, int(tf)) for doc, tf in postings]
                    pl = PostingsList(postings)  # instantiate posting class
                    lists.append(pl)

                    print('added posting for: ', term)
                    print(postings)

            # for pl in lists:
                #print(pl.list)
                #print(pl.ptr)
                #print(pl.next())

            # Now calculate scores for tier, doc at a time:
            # if above threshold equal to current min, add to heap
            #             print(postings)


            while not self.atEndOfLists(lists):
                print("getting scores for lists of len", len(lists))
                for pl in lists:
                    thisDoc = pl.current()
                    print("this", thisDoc)
                    # Search for doc in other lists by advancing ptrs
                    for otherPL in lists:
                        if otherPL is not pl:

                            otherDoc = otherPL.current()
                            print("init other", otherDoc)
                            while otherDoc < thisDoc:  # advance next ptr
                                otherDoc = otherPL.peak()
                                print("next other", otherDoc)
                                if otherDoc == -1:  # reached end of list
                                    print("end of list")
                                    break
                            if otherDoc == thisDoc:
                                print("same doc")

                    # Advance ptr for all lists
                    for pl in lists:
                        pl.next()
            break
                #     wDoc = self.calculateWeightOfTerm(term, tf, df, docScheme, doc)  # weight of doc
        #             score = wQuery * wDoc
        #             #                     print('score', score)
        #             scores[doc] += wQuery * wDoc
        #             #                     score += float(tf)
        #             #                     print(doc, scores[doc])

        #             else:
        #                 print("Word {} not in index. Skipping...".format(term))

        # lengthQuery = 0.0
        # if queryScheme[-1] == 'c':  # cosine norm
        #     for term, tf in terms.items():
        #         lengthQuery += tf * tf
        #         # print(token, tf, length)
        #     lengthQuery = math.sqrt(lengthQuery)
        # else:
        #     lengthQuery = 1.0  # no norm

        #     #         print("len query=", lengthQuery)

        # # Normalize scores
        # if docScheme[-1] == 'c':
        #     for doc, score in scores.items():
        #         #                 print(doc, score)
        #         lengthDoc = docLengths[doc]
        #         #                 print("len=", lengthDoc, lengthQuery)
        #         scores[doc] /= lengthDoc * lengthQuery  # normalize (cosine)
        #         #                 print("norm", scores[doc])
        # elif docScheme[-1] == 'u':
        #     a = .5  # slope normally between .25-.4
        #     pivot = 2730  # ave bytes in disks 1-2 of TREC
        #     for doc, score in scores.items():
        #         #                 print(doc, score)
        #         pivotedLengthDoc = a * uniq[doc] + (1 - a) * pivot
        #         #                 print("len=", lengthDoc, lengthQuery)
        #         scores[doc] /= pivotedLengthDoc * lengthQuery  # normalize (cosine)

        # # Get top k scores
        # topK = heapq.nlargest(k, scores.items(), key=lambda x: x[1])

        # # Write to disk
        # for i, (doc, score) in enumerate(topK):
        #     res = str(qid) + ' ' + '0 ' + doc + ' ' + str(i) + ' ' + str(score) + ' ' + run + '\n'
        #     results.write(res)

        # return topK

    def cosineScoreTAAT(self, query, docScheme, queryScheme, k, pp=True):
        """
        Original non-tiered method for computing the cosine
        score"""

        tier = 0  # no tier here, so only one tier in list of index files [indexFile]

        # Parse query from title
        query = query.rstrip()
        parsedQuery = index.parseQuery(query)
        qid = parsedQuery['qid'].strip()  # number <num> of topic doc
        title = parsedQuery['title']  # use for query terms

        # Get terms and normalize length if necessary
        terms = index.parseWords(title, pp)  # lowercase text, filter stopwords
        terms = Counter(terms)  # get counts and store in dict/hash

        # For each query term calculate weight
        scores = defaultdict(lambda: 0.0)
        for term, tf in terms.items():
            if term in offsets[tier]:
                offset = offsets[tier][term]
#                 print("getting {} at {}".format(term, offset))
                indexes[tier].seek(offset)
                line = indexes[tier].readline().rstrip().split()
                df = float(line[1])
                wQuery = self.calculateWeightOfTerm(term, tf, df, queryScheme, doc=None)  # weight of query
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
                lengthDoc = docLengths[tier][doc]
#                 print("len=", lengthDoc, lengthQuery)
                scores[doc] /= lengthDoc * lengthQuery  # normalize (cosine)
#                 print("norm", scores[doc])
        elif docScheme[-1] == 'u':
            a = .65  # slope normally between .25-.4
            pivot = 2630  # ave bytes in disks 1-2 of TREC
            for doc, score in scores.items():
#                 print(doc, score)
                pivotedLengthDoc = a * uniq[tier][doc] + (1 - a) * pivot
#                 print("len=", lengthDoc, lengthQuery)
                scores[doc] /= pivotedLengthDoc * lengthQuery  # normalize (cosine)
#
        # Get top k scores
        topKScores = heapq.nlargest(k, scores.items(), key=lambda x: x[1])

        # Write to disk
        for i, (doc, score) in enumerate(topKScores):
            res = str(qid) + ' ' + '0 ' + doc + ' ' + str(i) + ' ' + str(score) + ' ' + runId + '\n'
            results.write(res)

        return topKScores

    def buildIndex(self, tier=None, tierName=None, pp=True):
        """Build index (indices if tiered)"""

        gc.enable()
        if self.isTiered:
            curIndex = self.indexes[tier]
        else:
            curIndex = self.index

        lengths = {}  # for calculating and storing document (cosine) lengths
        uniq = {}
        aveTF = {}
        maxTF = {}

        # Parse all documents
        for doc in open('documents.list', 'rt'):

#             print(os.getcwd())
#             fname = doc.rstrip()  # documents/LN-20020102023.vert
#             path = cwd + fname
#             f = gzip.open(path + '.gz', 'rt')
#
#             # Truncate prefix of doc id in order to save space: will expand later
#             did, text = self.parseDoc(f)
#             if text is None:
#                 continue
#             docid = self.truncateDocid(did)
#
#             # Parse words in doc and preprocess text
#             tokens = self.parseWords(text, pp)
#             if not tokens:
#                 continue
#             counts = Counter(tokens)

            # First parse and get token counts for each doc
            docid, counts = self.getTokenCounts(doc, tier, pp)
            if docid is None or counts is None:
                continue
            # Add terms to index
            self.addToIndex(docid, counts, tier)

            # Calculate normalized doc length
            lengths[docid] = self.calculateDocLen(counts)

            # Calculate unique terms in doc
            uniq[docid] = self.calculateNumberUniqTerms(counts)

            # Calculate ave tf in doc
            aveTF[docid] = self.calculateAveTermFreq(counts)

            # Calculate max tf in doc
            maxTF[docid] = self.calculateMaxTermFreq(counts)


        # Write data to disk
        self.writeIndex(tier, tierName)  # write index, creating offsets
        self.writeOffsets(tier, tierName)  # write offsets
        self.write(lengths, 'lengths', tierName)
        del lengths
        self.write(uniq, 'uniq', tierName)
        del uniq
        self.write(aveTF, 'ave-tf', tierName)
        del aveTF
        self.write(maxTF, 'max-tf', tierName)
        del maxTF

#         # Write doc lengths to disk
#         with gzip.open(cwd + tier + '/lengths.gz', 'wt') as f:
#             print("writing doc length")
#             for docid, length in lengths.items():
#                 f.write(self.expandDocid(docid) + '\t' + str(length) + '\n')
#
#         print("Wrote {} doc lengths".format(cnt))

        # Get rid of garbage
        gc.collect()

    def getTokenCounts(self, doc, tier=None, pp=True):
        """Helper method for getting the tokens and their counts
        for a document.
        Returns (docid, counts), where docid is compressed/abbreviated
        form. """

        # Get token counts
        fname = doc.rstrip()  # documents/LN-20020102023.vert
#         path = cwd + fname
        f = gzip.open(fname + '.gz', 'rt')

        # Only parse terms in <TITLE> of doc if tier 0, else parse all of <TEXT>
        if tier == 0:
            docid, text = self.parseDoc(f, title=True)
        else:
            docid, text = self.parseDoc(f)


        # Truncate prefix of doc id in order to save space: will expand later
        docid = self.truncateDocid(docid)

        # Most likely no title in text
        if text is None:
            # print('no text')
            return docid, None

        # Parse words in doc and preprocess text
        tokens = self.parseWords(text, pp)
        if not tokens:
            return docid, None

        return docid, Counter(tokens)

    def addToIndex(self, docid, counts, tier=None):
        """Add docid and tfs of a doc in compressed form to index.
        If tiered, add to current tier index."""

        for token, tf in counts.items():

            # Combine 32 bit docid and tf into a 64-bit long to save space (recover later)
            idPlusTf = self.combineInts(docid, tf)
            if self.isTiered:
                self.indexes[tier][token].append(idPlusTf)
                # for i in range(len(self.indexes)):
                assert self.indexes[0] is not self.indexes[1]
                assert self.indexes[0] != self.indexes[1]

            else:
                self.index[token].append(idPlusTf)  # append a new entry and postings list

    def calculateDocLen(self, counts):
        """Computer and write doc length for scoring"""

#         gc.enable()
#         # Get token counts if none have already been parsed and counted
#         if counts is None:
#             lengths = {}  # main dict
#             for doc in open(cwd + 'documents.list', 'rt'):
#                 fname = doc.rstrip()  # documents/LN-20020102023.vert
#                 path = cwd + fname
#                 f = gzip.open(path + '.gz', 'rt')
#                 docid, text = self.parseDoc(f)
#                 if text is None:
#                     continue
#                 docid = self.truncateDocid(docid)
#                 tokens = self.parseWords(text, pp)
#                 if not tokens:
#                     continue
#                 counts = Counter(tokens)

        # Calculate length of doc
        length = 0
        for token, cnt in counts.items():
            length += cnt * cnt
        length = math.sqrt(length)
#         if counts is None:
#             del tokens
#             del counts
#         gc.collect()

        return length

#         if tier:
#             f = gzip.open(cwd + run + '/' + tier + '/lengths.gz', 'wt')
#         else:
#             f = gzip.open(cwd + run + '/lengths.gz', 'wt')
#
#         print("writing doc length")
#         for docid, length in lengths.items():
#             # print(self.expandDocid(docid) + '\t' + str(length) + '\n')
#             f.write(self.expandDocid(docid) + '\t' + str(length) + '\n')
#
#         # Clean up
#         del lengths
#         f.close()
#         gc.collect()


    def calculateNumberUniqTerms(self, counts):
        """Compute number of unique terms in a doc, used in cosine score
        calculation."""

        return len(counts)
#         gc.enable()
#         uniq = {}  # main dict
#         cnt = 0
#         for doc in open(cwd + 'documents.list', 'rt'):
#             fname = doc.rstrip()  # documents/LN-20020102023.vert
#             path = cwd + fname
#             f = gzip.open(path + '.gz', 'rt')
#             docid, text = self.parseDoc(f)
#             if text is None:
#                 continue
#             docid = self.truncateDocid(docid)
#             tokens = self.parseWords(text, pp)
#             if not tokens:
#                 continue
#             counts = Counter(tokens)
#             uniq[docid] = len(counts)
#             print(docid, uniq[docid])
#             print(counts)

#             del tokens
#             del counts
#             gc.collect()
#             cnt += 1
#
#         if tier:
#             f = gzip.open(cwd + run + '/' + tier + '/uniq.gz', 'wt')
#         else:
#             f = gzip.open(cwd + run + '/uniq.gz', 'wt')
#
#         print("writing uniq terms")
#         for docid, c in uniq.items():
#             f.write(self.expandDocid(docid) + '\t' + str(c) + '\n')
#
#         # Clean up
#         del uniq
#         f.close()
#         gc.collect()

    def calculateMaxTermFreq(self, counts):
        """Compute and write max term frequency for scoring."""

        return counts.most_common(1)[0][1]  # tf of most common element
#         gc.enable()
#         max = {}  # main dict
#         for doc in open(cwd + 'documents.list', 'rt'):
#             fname = doc.rstrip()  # documents/LN-20020102023.vert
#             path = cwd + fname
#             f = gzip.open(path + '.gz', 'rt')
#             docid, text = self.parseDoc(f)
#
#             if text is None:
#                 continue
#             docid = self.truncateDocid(docid)
#             tokens = self.parseWords(text, pp)
#             if not tokens:
#                 continue
#
#             counts = Counter(tokens)
#             max[docid] = counts.most_common(1)[0][1]  # tf of most common element
#             del tokens
#             del counts
#             gc.collect()
#
#         if tier:
#             f = gzip.open(cwd + run + '/' + tier + '/max-tf.gz', 'wt')
#         else:
#             f = gzip.open(cwd + run + '/max-tf.gz', 'wt')
#
#         print("writing max tf")
#         for docid, tf in max.items():
# #           print(self.expandDocid(docid) + '\t' + str(length) + '\n')
#             f.write(self.expandDocid(docid) + '\t' + str(tf) + '\n')
#
#          # Clean up
#         del max
#         f.close()
#         gc.collect()

    def calculateAveTermFreq(self, counts):
        """Compute average term frequency in doc, used for scoring."""

        s = sum([c for t, c in counts.items()])  # sum tfs
        return s / float(len(counts))
#         gc.enable()
#         aveTF = {}  # ave tfs per doc
#         for doc in open(cwd + 'documents.list', 'rt'):
#             fname = doc.rstrip()  # documents/LN-20020102023.vert
#             path = cwd + fname
#             f = gzip.open(path + '.gz', 'rt')
#             docid, text = self.parseDoc(f)
#             if text is None:
#                 continue
#             docid = self.truncateDocid(docid)
#             tokens = self.parseWords(text, pp)
#             if not tokens:
#                 continue
#             counts = Counter(tokens)
#             s = sum([c for t, c in counts.items()])  # sum tfs
#             aveTF[docid] = s / float(len(counts))
#             del tokens
#             del counts
#             del s
#             gc.collect()
#
#         if tier:
#             f = gzip.open(cwd + run + '/' + tier + '/ave-tf.gz', 'wt')
#         else:
#             f = gzip.open(cwd + run + '/ave-tf.gz', 'wt')
#
#         print("writing ave tf")
#         for docid, ave in aveTF.items():
#             f.write(self.expandDocid(docid) + '\t' + str(ave) + '\n')
#
#         # Clean up
#         del aveTF
#         f.close()
#         gc.collect()

    def writeIndex(self, tier=None, tierName=None):
        """Helper method to write index to disk. Once written
        offsets are written to index, later to be written also"""

        gc.enable()

        if self.isTiered:
            f = gzip.open(cwd + run + '/' + tierName + '/index.gz', 'wt')
            curIndex = self.indexes[tier]  # get ref to cur tier index
        else:
            f = gzip.open(cwd + run + '/index.gz', 'wt')
            curIndex = self.index  # no tier, just index

        print("writing index")
        for token, postings in sorted(curIndex.items()):
            df = len(postings)
            postings = [self.splitInts(post) for post in postings]
            postings = [str(self.expandDocid(x)) + ',' + str(y) for x, y in postings]
            offset = f.tell()
            f.write(token + '\t' + str(df) + '\t' + ' '.join(postings) + '\n')
            curIndex[token] = offset  # replace posting with offset

        # Clean up
        f.close()
        del curIndex
        del postings
        gc.collect()

    def writeOffsets(self, tier=None, tierName=None):
        """Helper method to write byte offsets of term to disk"""

        gc.enable()

        if self.isTiered:
            f = gzip.open(cwd + run + '/' + tierName + '/offsets.gz', 'wt')
            curIndex = self.indexes[tier]  # get ref to cur tier index
        else:
            f = gzip.open(cwd + run + '/offsets.gz', 'wt')
            curIndex = self.index  # no tier, just index

        print("writing offsets")
        for token, offset in sorted(curIndex.items()):
            f.write(token + '\t' + str(offset) + '\n')

        # Clean up
        f.close()
        del curIndex

    def write(self, data, file, tierName=None):
        """Helper method to write data in the form of a dict
        with docid keys to disk"""

        if self.isTiered:
            f = gzip.open(cwd + run + '/' + tierName + '/' + file + '.gz', 'wt')
        else:
            f = gzip.open(cwd + run + '/' + file + '.gz', 'wt')

        print("writing ", file)
        for docid, val in data.items():
            f.write(self.expandDocid(docid) + '\t' + str(val) + '\n')

#     def writeDocLengths(self, tier=None):
#         """Helper method to write doc lengths to disk"""
#
#         if tier:
#             f = gzip.open(cwd + run + '/' + tier + '/lengths.gz', 'wt')
#         else:
#             f = gzip.open(cwd + run + '/lengths.gz', 'wt')
#
#         print("writing doc length")
#         for docid, length in self.lengths.items():
#             f.write(self.expandDocid(docid) + '\t' + str(length) + '\n')

if __name__ == "__main__":

    os.chdir('..')      # should be in search-engine/
    cwd = os.getcwd() + '/output/'  # working dir for you
    os.chdir('../A1')  # data in here

    # Get sys args
    if len(sys.argv) < 2:
        raise ValueError('Must provide run type')

    runId = sys.argv[1]  # test-run-0      baseline, etc.
    run = runId[-5:]       # trim to run-0

    if not os.path.exists(cwd + run):
        os.makedirs(cwd + run)  # .../output/run-0/...

    # Instantiate index class
    if 't' in sys.argv[2]:
        isTiered = True  # this is a two-tiered index
    else:
        isTiered = False
    index = InvertedIndex(81735, isTiered)   # instantiate index with size n
    index.stopwords()  # read stop word list

    # Train/Test query/topics
    if 'q' in sys.argv[2]:

        print("query/topics test/train")
        if len(sys.argv) >= 9:
            docScheme = sys.argv[3]     # ddd triplet
            queryScheme = sys.argv[4]   # qqq triplet
            k = int(sys.argv[5])
            topicsList = sys.argv[6]  # 'test-topics.list'
            docsList = sys.argv[7]  # 'documents.list'
            out = cwd + sys.argv[8]  # .../output/.dat file
            if len(sys.argv) == 10 and 'pp' in sys.argv[9]:  # pre-process (pp)
                pp = True
            else:
                pp = False

        # Load data - tier one and tier two
        offsets = []
        if isTiered:
            dirs = ['/tier0/offsets.gz', '/tier1/offsets.gz']
        else:
            dirs = ['/offsets.gz']
        for d in dirs:
            f = gzip.open(cwd + run + d, 'rt')
            off = {}
            for line in f:
                word, offset = line.rstrip().split()
                off[word] = int(offset)
            offsets.append(off)

        docLengths = []
        if isTiered:
            dirs = ['/tier0/lengths.gz', '/tier1/lengths.gz']
        else:
            dirs = ['/lengths.gz']
        for d in dirs:
            f = gzip.open(cwd + run + d, 'rt')
            mtf = {}
            dl = {}
            for line in f:
                doc, length = line.rstrip().split()
                dl[doc] = float(length)
            docLengths.append(dl)

        maxTF = []
        if isTiered:
            dirs = ['/tier0/max-tf.gz', '/tier1/max-tf.gz']
        else:
            dirs = ['/max-tf.gz']
        for d in dirs:
            f = gzip.open(cwd + run + d, 'rt')
            for line in f:
                doc, tf = line.rstrip().split()
                mtf[doc] = int(tf)
            maxTF.append(mtf)

        aveTF = []
        if isTiered:
            dirs = ['/tier0/ave-tf.gz', '/tier1/ave-tf.gz']
        else:
            dirs = ['/ave-tf.gz']
        for d in dirs:
            f = gzip.open(cwd + run + d, 'rt')
            atf = {}
            for line in f:
                doc, tf = line.rstrip().split()
                atf[doc] = float(tf)
            aveTF.append(atf)

        uniq = []
        if isTiered:
            dirs = ['/tier0/uniq.gz', '/tier1/uniq.gz']
        else:
            dirs = ['/uniq.gz']
        for d in dirs:
            f = gzip.open(cwd + run + d, 'rt')
            un = {}
            for line in f:
                doc, u = line.rstrip().split()
                un[doc] = int(u)
            uniq.append(un)

        # Compute scores for all documents in list
        indexes = []
        with open(topicsList, 'rt') as topicsListFile:
            if isTiered:
                dirs = ['/tier0/index.gz', '/tier1/index.gz']
            else:
                dirs = ['/index.gz']
            for d in dirs:
                f = gzip.open(cwd + run + d, 'rt')
#             with gzip.open(run + tier + '/index.gz', 'rt') as indexFile1:
#                 if isTiered:
#                     indexFile2 = gzip.open(run + tier + '/index2.gz', 'rt')
                indexes.append(f)
#                 indexes.append(indexFile1)
            with open(out, 'wt') as results:

                # Get query terms for `title` field in topics list
                if isTiered:
                    for query in topicsListFile:
                        # This is score for tiered set up
                        score = index.cosineScoreTiered(query, docScheme, queryScheme, k, pp)
                else:
                    for query in topicsListFile:
                        # This is score for tiered set up
                        score = index.cosineScoreTAAT(query, docScheme, queryScheme, k, pp)

            # Close files
            for f in indexes:
                f.close()

#     # Fetch posting
#     if 'f' in sys.argv[2]:
#         offset = int(sys.argv[2])
#         if len(sys.argv) == 4:
#             f = sys.argv[3]
#             index.getPostings(offset, f)
#         else:
#             f = cwd + run + '/index.gz'
#             index.getPostings(offset, f)

    # Build index
    # USAGE: python3 invertedIndex.py {-b, -bt} [-pp]
    if 'b' in sys.argv[2]:
        if len(sys.argv) == 4 and 'pp' in sys.argv[3]:  # pre-process text
            pp = True
        else:
            pp = False
#         if len(sys.argv == 5) and sys.argv[4] == 'tiered':
        if isTiered:  # tiered index:  [-bt]
            if not os.path.exists(cwd + run + '/tier0'):
                os.makedirs(cwd + run + '/tier0')  # .../output/run0/tier0/...
            if not os.path.exists(cwd + run + '/tier1'):
                os.makedirs(cwd + run + '/tier1')  # .../output/run0/tier0/...
            index.buildIndex(0, 'tier0', pp)  # build tier 0    (titles)
            index.buildIndex(1, 'tier1', pp)  # build tier 1    (text)
#             index.calculateDocLen(0, pp)
#             index.calculateDocLen(0, pp)

        else:
            index.buildIndex(pp=pp)

#     # Calculate doc lengths
#     if 'l' in sys.argv[2]:
#         if len(sys.argv == 4) and 'pp' in sys.argv[3]:  # pre-process text
#             pp = True
#         else:
#             pp = False
#         if len(sys.argv == 5) and sys.argv[4] == 'tiered':
#             isTiered = True
#         index.calculateDocLen(tier, pp)
#
#     # Calculate num unique terms
#     if 'u' in sys.argv[2]:
#         index.calculateNumberUniqTerms(tier, pp)
#
#     # Calculate max lengh for augmented tf
#     if 'm' in sys.argv[2]:
#         index.calculateMaxTermFreq(tier, pp)
#
#     # Calculate average term frequency
#     if 'a' in sys.argv[2]:
#         index.calculateAveTermFreq(tier, pp)

#
#         offsets = []
#         with gzip.open(cwd + run + '/offsets1.gz', 'rt') as f:
#             off = {}
#             for line in f:
#                 word, offset = line.rstrip().split()
#                 off[word] = int(offset)
#             offsets.append(off)
#
#         with gzip.open(cwd + run + '/offsets2.gz', 'rt') as f:
#             off = {}
#             for line in f:
#                 word, offset = line.rstrip().split()
#                 off[word] = int(offset)
#             offsets.append(off)
#
#         docLengths = []
#         with gzip.open(cwd + run + '/lengths1.gz', 'rt') as f:
#             dl = {}
#             for line in f:
#                 doc, length = line.rstrip().split()
#                 dl[doc] = float(length)
#             docLengths.append(dl)
#
#         with gzip.open(cwd + run + '/lengths2.gz', 'rt') as f:
#             dl = {}
#             for line in f:
#                 doc, length = line.rstrip().split()
#                 dl[doc] = float(length)
#             docLengths.append(dl)
#
#         maxTF = []
#         with gzip.open(cwd + run + '/max-tf1.gz', 'rt') as f:
#             mtf = {}
#             for line in f:
#                 doc, tf = line.rstrip().split()
#                 mtf[doc] = int(tf)
#             maxTF.append(mtf)
#
#         with gzip.open(cwd + run + '/max-tf2.gz', 'rt') as f:
#             mtf = {}
#             for line in f:
#                 doc, tf = line.rstrip().split()
#                 mtf[doc] = int(tf)
#             maxTF.append(mtf)
#
#         aveTF = []
#         with gzip.open(cwd + run + '/ave-tf1.gz', 'rt') as f:
#             atf = {}
#             for line in f:
#                 doc, tf = line.rstrip().split()
#                 atf[doc] = float(tf)
#             aveTF.append(atf)
#
#         with gzip.open(cwd + run + '/ave-tf2.gz', 'rt') as f:
#             atf = {}
#             for line in f:
#                 doc, tf = line.rstrip().split()
#                 atf[doc] = float(tf)
#             aveTF.append(atf)
#
#         uniq = []
#         with gzip.open(cwd + run + '/uniq1.gz', 'rt') as f:
#             un = {}
#             for line in f:
#                 doc, u = line.rstrip().split()
#                 un[doc] = int(u)
#             uniq.append(un)
#
#         with gzip.open(cwd + run + '/uniq2.gz', 'rt') as f:
#             un = {}
#             for line in f:
#                 doc, u = line.rstrip().split()
#                 un[doc] = int(u)
#             uniq.append(un)
