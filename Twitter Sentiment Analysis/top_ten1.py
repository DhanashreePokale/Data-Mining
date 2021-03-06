import sys
import json
import operator

def allTweets():
    
    tweetFile = open(sys.argv[1])
    tweetHashtags = {}
    
    for line in tweetFile:
        tweet = json.loads(line)
        if 'entities' in tweet and 'hashtags' in tweet['entities']:
            currentTweetHashtags = tweet["entities"]["hashtags"]
            for currentTweetHashtag in currentTweetHashtags:
                tweetHashtagText = currentTweetHashtag["text"].encode('utf-8')
                if not tweetHashtagText in tweetHashtags:
                    tweetHashtags[tweetHashtagText] = 1
                else:
                    tweetHashtags[tweetHashtagText] += 1
    
    sortHashtags = sorted(tweetHashtags.items(), key=operator.itemgetter(1), reverse=True)
    counter = 0;
    for hashtag, hashtagFrequency in sortHashtags:
        counter += 1;
        print (hashtag + " " + str(hashtagFrequency));
        if counter == 10:
            break
        
            

def main():
    allTweets()

if __name__ == '__main__':
    main()
