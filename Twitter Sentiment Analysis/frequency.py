from __future__ import division
from collections import Counter
import sys
import json

def frequency(tweet_file):
    "string -> dict"
    with open(tweet_file) as f:
         tweets = (json.loads(line).get('text','').split() for line in f)
         return Counter(word for tweet in tweets for word in tweet)

if __name__ == '__main__':
    freq = frequency(tweet_file=sys.argv[1])
    total = sum(freq.values())
    sys.stdout.writelines('{0}{1}\n'.format(word.encode('utf-8'), freq[word]/total)
    for word in freq)

