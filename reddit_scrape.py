import praw
import pandas as pd

# Reddit API credentials
reddit = praw.Reddit(
    client_id="bKUQYyqxMsQwBmECpl8guw",
    client_secret="IIzAvemnMFq22oD3a84-JAqRtveHmQ",
    user_agent="sentiment_project by /u/Legal-Vanilla8068"
)

# Choose subreddit
subreddit = reddit.subreddit("movies")

data = []

for submission in subreddit.hot(limit=10):
    submission.comments.replace_more(limit=0)  # Expand all comments

    for comment in submission.comments.list():
        data.append({
            "post_id": submission.id,
            "post_title": submission.title,
            "comment_id": comment.id,
            "comment_body": comment.body,
            "comment_score": comment.score,
            "created_utc": comment.created_utc
        })

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("reddit_movies_comments.csv", index=False)
print("Saved all comments to reddit_comments.csv")
