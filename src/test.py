from os.path import expanduser
import re

home = expanduser('~')
with open(home + "/Desktop/test.txt", 'rt') as f:
    text = f.read()
# print(text)

docidMatch = re.search('<DOCID>(.*)</DOCID>', text, re.DOTALL)
docid = docidMatch.group(1)
#        titleMatch = re.search('<TITLE>(.*)</TITLE>', text, re.DOTALL)
#        if titleMatch is not None:
#            title = titleMatch.group(1)
#        else:
#            title = None
textMatch = re.search('<TEXT>(.*)</TEXT>', text, re.DOTALL)
text = textMatch.group(1)
# text = ''.join(textMatch)
print(text)
# print(len(textMatch))
