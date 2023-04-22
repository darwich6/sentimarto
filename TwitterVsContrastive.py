import nltk
import pandas
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import scipy.stats as stats
from tabulate import tabulate


def read_csv():
    contrastive_df = pandas.read_csv("Contrastive.csv")

    # group the dataframe by art_style and create a dictionary of dataframes
    art_style_groups = contrastive_df.groupby('art_style')
    contrastive_dfs_dict = {}

    for art_style, group in art_style_groups:
        contrastive_dfs_dict[art_style] = group

    twitter_df = pandas.read_csv("Twitter.csv", encoding='latin-1', nrows=200, header=None,
                                 names=['target', 'id', 'date', 'flag', 'user', 'text'])

    sentiment_analysis(contrastive_dfs_dict, twitter_df)


def sentiment_analysis(dfs_dict, twitter_df):
    nltk.download('vader_lexicon')  # download VADER lexicon

    # create sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    constrastive_sentiments = {}

    for art_style, df in dfs_dict.items():
        art_style_sentiments = []
        for index, row in df.iterrows():
            utterance = row['utterance']
            anchor_painting = row['anchor_painting']
            sentiment_score = analyzer.polarity_scores(utterance)['compound']
            art_style_sentiments.append({'art_style': art_style, 'utterance': utterance,
                                         'anchor_painting': anchor_painting, 'sentiment_score': sentiment_score})
        # sort the art_style_sentiments list by sentiment_score
        art_style_sentiments = sorted(art_style_sentiments, key=lambda x: x['sentiment_score'], reverse=True)
        # select the top 50 positive and top 50 negative sentiments
        top_sentiments = art_style_sentiments[:50] + art_style_sentiments[-50:]
        constrastive_sentiments[art_style] = top_sentiments

    twitter_df_sentiments = []
    for index, row in twitter_df.iterrows():
        compound = analyzer.polarity_scores(row['text'])['compound']
        twitter_df_sentiments.append(compound)

    # Create a new column in the dataframe for the sentiment scores
    twitter_df['sentiment'] = twitter_df_sentiments
    sentiment_statistics(constrastive_sentiments, twitter_df)


def sentiment_statistics(constrastive_sentiments, twitter_df):
    # Create a list to store sentiment scores for each art style
    sentiment_scores = []
    for art_style, sentiments in constrastive_sentiments.items():
        art_style_scores = [sentiment['sentiment_score'] for sentiment in sentiments]
        sentiment_scores.append(art_style_scores)

    # Perform descriptive statistics and save to file
    descriptive_stats_table = []
    for i, art_style in enumerate(constrastive_sentiments.keys()):
        stats_row = [art_style, np.mean(sentiment_scores[i]), np.std(sentiment_scores[i], ddof=1),
                     np.min(sentiment_scores[i]), np.max(sentiment_scores[i])]
        descriptive_stats_table.append(stats_row)

    headers = ["Art Style", "Mean", "Standard Deviation", "Minimum", "Maximum"]
    with open("descriptive_stats_table.txt", "w") as f:
        f.write(tabulate(descriptive_stats_table, headers=headers))

    # Perform hypothesis testing and save to file
    hypothesis_table = []
    twitter_df_mean = np.mean(twitter_df['sentiment'])
    for art_style, sentiments in constrastive_sentiments.items():
        art_style_scores = [sentiment['sentiment_score'] for sentiment in sentiments]
        t_statistic, p_value = stats.ttest_1samp(art_style_scores, twitter_df_mean)
        hypothesis_row = [art_style, np.mean(art_style_scores), t_statistic, p_value]
        hypothesis_table.append(hypothesis_row)

    headers = ["Art Style", "Sample Mean", "t-statistic", "p-value"]
    with open("hypothesis_table.txt", "w") as f:
        f.write(tabulate(hypothesis_table, headers=headers))

read_csv()
