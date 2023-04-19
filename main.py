import pandas
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def read_csv():
    df = pandas.read_csv("Contrastive.csv")

    # group the dataframe by art_style and create a dictionary of dataframes
    art_style_groups = df.groupby('art_style')
    dfs_dict = {}
    for art_style, group in art_style_groups:
        dfs_dict[art_style] = group

    # print(df['emotion'].unique().tolist())
    # print the new dataframes
    # for key, value in dfs_dict.items():
    #    print(f"Dataframe for {key}:")
    #    print(value)
    #    print()

    sentiment_analysis(dfs_dict)


def sentiment_analysis(dfs_dict):
    nltk.download('vader_lexicon')  # download VADER lexicon

    # create sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    sentiments = {}

    for art_style, df in dfs_dict.items():
        art_style_sentiments = []
        for index, row in df.iterrows():
            utterance = row['utterance']
            anchor_painting = row['anchor_painting']
            sentiment_score = analyzer.polarity_scores(utterance)['compound']
            art_style_sentiments.append({'art_style': art_style, 'utterance': utterance,
                                         'anchor_painting': anchor_painting, 'sentiment_score': sentiment_score})
        sentiments[art_style] = art_style_sentiments

    print(sentiments)

read_csv()
