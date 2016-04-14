#Import the necessary methods from tweepy library
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json

access_token = "80282681-xCPKdvowdCTQblkemAKvf8uIa5WJelhxw6sOIkg1m"
access_token_secret = "BPdFtPDrvUxcBD3hmBeo0g5RM0usaB3p6cZIBZa6Xk7fy"
consumer_key = "2xxSiw42Ia3jSddvNzqaBVOv7"
consumer_secret = "UWDP3MtEfalQf7AAauuxRMasDKPjUUHsB6CxR3TDNitEavLVvG"


class StdOutListener(StreamListener):
    def __init__(self, limit=2000000):
        self.limit = limit

    def on_data(self, data):
        tweet = False

        try:
            tweet = json.loads(data)
        except:
            print "JSON parse failed"

        try:
            print tweet['text']
        except:
            pass

        if self.limit < 0:
            return False

        self.limit = self.limit - 1
        return True

    def on_error(self, status):
        # Kill it
        print status
        return False


if __name__ == '__main__':

    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    stream.sample()
