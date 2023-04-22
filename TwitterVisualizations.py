import pandas
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def read_csv():
    df = pandas.read_csv("Twitter.csv", encoding='latin-1', nrows=200, header=None, names=['target', 'id', 'date', 'flag', 'user', 'text'])
    sentiment_analysis(df)


def sentiment_analysis(df):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for index, row in df.iterrows():
        compound = analyzer.polarity_scores(row['text'])['compound']
        sentiments.append(compound)

    # Create a new column in the dataframe for the sentiment scores
    df['sentiment'] = sentiments

    # Generate visualizations
    generate_visualizations(df)


def generate_visualizations(df):
    if not os.path.exists('TwitterVisualizations'):
        os.mkdir('TwitterVisualizations')

    # Distribution plot of sentiment scores
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))
    sns.histplot(df, x='sentiment', bins=20)
    plt.title('Distribution of Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join('TwitterVisualizations', 'sentiment_distribution.png'))
    plt.close()

    # Bar graph of average sentiment score per user
    user_sentiments = df.groupby('user')['sentiment'].mean().reset_index()
    user_sentiments = user_sentiments.sort_values(by='sentiment', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=user_sentiments, x='sentiment', y='user')
    plt.title('Average Sentiment Score per User')
    plt.xlabel('Sentiment Score')
    plt.ylabel('User')
    plt.savefig(os.path.join('TwitterVisualizations', 'average_sentiment_per_user.png'))
    plt.close()

    # Word cloud of most frequent words in positive tweets
    positive_tweets = df[df['sentiment'] > 0]
    positive_tweet_text = ' '.join(positive_tweets['text'])
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate(positive_tweet_text)
    plt.figure(figsize=(8, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Frequent Words in Positive Tweets')
    plt.savefig(os.path.join('TwitterVisualizations', 'positive_word_cloud.png'))
    plt.close()

    # Word cloud of most frequent words in negative tweets
    negative_tweets = df[df['sentiment'] < 0]
    negative_tweet_text = ' '.join(negative_tweets['text'])
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate(negative_tweet_text)
    plt.figure(figsize=(8, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Frequent Words in Negative Tweets')
    plt.savefig(os.path.join('TwitterVisualizations', 'negative_word_cloud.png'))
    plt.close()


read_csv()
