import praw
from classifier import classifier_engine
from keys import reddit_api_keys

k = reddit_api_keys()
reddit = praw.Reddit(
    client_id = k.client_id,  
    client_secret = k.client_secret,
    user_agent = 'bot' # name it anything
)

if(reddit.read_only):
    print("Connected to reddit!")

new_comments = reddit.subreddit('science').comments(limit=100)
queries = [comment.body for comment in new_comments]

engine = classifier_engine()
engine.load_corpus('./data/final_labelled_data.pkl', 'tweet', 'class')

engine.train_using_bow()
print(engine.evaluate())
print(engine.predict(queries))

engine.train_using_tfidf()
print(engine.evaluate())
print(engine.predict(queries))


engine.load_lexicon('hate-words')
engine.load_lexicon('neg-words')

engine.train_using_custom()
print(engine.evaluate())
print(engine.predict(queries))
