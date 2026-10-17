
import tweepy
import numpy as np
import os
access_token = os.environ["TWITTER_ACCESS_TOKEN"]
access_token_secret = os.environ["TWITTER_ACCESS_TOKEN_SECRET"]
consumer_key = os.environ["TWITTER_CONSUMER_KEY"]
consumer_secret = os.environ["TWITTER_CONSUMER_SECRET"]
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True,  wait_on_rate_limit_notify=True)
#api = tweepy.API(auth)
t_list = []
credentials = {}
count = 0

for tweet in tweepy.Cursor(api.search, q="airfrance ", lang="en").items(40):
    credentials['text'] = tweet.text

    Time = tweet.created_at
    print(Time)
    Time = int(Time.strftime("%Y%m%d%H%M%S"))
    t_list.append(Time)
    ad = tweet.id
    count+=1

i=0
ct=0
x = []
y = []
while(i < count):
    str1 = str(t_list[i])
    str2 = str1[6:8]
    print(str2)
    if(i==0):
        x.append(str2)
    for a in range(len(x)):
        if(a == str2):
            ct+=1
            break
        else:
            ct = 0
            x.append(str2)
            ct+=1
            break
    i +=1

print(x)
