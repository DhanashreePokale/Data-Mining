
from __future__ import division
import sys
import json
from collections import defaultdict

def read_scores(sent_file):
    with open(sent_file) as f:
         return{line.split('\t')[0]: int(line.split('\t')[1]) for line in f}
def tweet_score(tweet,scores):
    return sum(scores.get(word,0) for word in tweet.split())

def parse(tweet):
    try:
       country = tweet['place']['country_code']
       state = tweet['place']['full_name'].split(", ")[1]
       text = tweet['text']
       return country, state, text
    except (KeyError, TypeError, IndexError):
       return None

def happiest_state(tweet_file, sent_scores):
    n_tweets, scores, happiness = [defaultdict(float) for _ in range(3)]
    with open(tweet_file) as f:
         tweets = (json.loads(line) for line in f)
         parsed_tweets = (parse(tweet) for tweet in tweets if parse(tweet))
         states_scores = ((state, tweet_score(text, sent_scores))
                         for country, state, text in parsed_tweets
                         if len(state) == 2 and country=='US')
         for state, score in states_scores:
             n_tweets[state] +=1
             scores[state] +=score
             happiness[state] = scores[state] / n_tweets[state]
    return max(happiness, key = happiness.get)

if __name__ == '__main__':
    scores = read_scores(sent_file=sys.argv[1])
    print (happiest_state(tweet_file=sys.argv[2], sent_scores = scores))
