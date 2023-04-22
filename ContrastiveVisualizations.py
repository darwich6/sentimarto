import pandas
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import plotly.graph_objs as go
import os


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
        # sort the art_style_sentiments list by sentiment_score
        art_style_sentiments = sorted(art_style_sentiments, key=lambda x: x['sentiment_score'], reverse=True)
        # select the top 50 positive and top 50 negative sentiments
        top_sentiments = art_style_sentiments[:50] + art_style_sentiments[-50:]
        sentiments[art_style] = top_sentiments

    generate_visualizations(sentiments)


def generate_visualizations(sentiments):
    # create a directory for saving the visualizations
    if not os.path.exists("ContrastiveVisualizations"):
        os.makedirs("ContrastiveVisualizations")

    # Generate a bar chart for sentiment scores
    for art_style, sentiment_list in sentiments.items():
        sentiment_scores = [sentiment['sentiment_score'] for sentiment in sentiment_list]
        plt.bar(range(len(sentiment_scores)), sentiment_scores)
        plt.title('Sentiment Scores for ' + art_style)
        plt.xlabel('Utterance')
        plt.ylabel('Sentiment Score')
        plt.savefig(os.path.join("ContrastiveVisualizations", art_style + "_bargraph_sentiment_scores.png"))
        plt.clf()  # clear the figure after saving
        print("Finished creating Bar Plot for: " + art_style)

    # Generate a scatter plot for sentiment scores vs. utterance length
    for art_style, sentiment_list in sentiments.items():
        utterance_lengths = [len(sentiment['utterance']) for sentiment in sentiment_list]
        sentiment_scores = [sentiment['sentiment_score'] for sentiment in sentiment_list]
        # Select top 50 positive and top 50 negative sentiments
        positive_sentiments = [sentiment for sentiment in sentiment_list if sentiment['sentiment_score'] > 0][:50]
        negative_sentiments = [sentiment for sentiment in sentiment_list if sentiment['sentiment_score'] < 0][:50]
        positive_lengths = np.array([len(sentiment['utterance']) for sentiment in positive_sentiments])
        positive_scores = np.array([sentiment['sentiment_score'] for sentiment in positive_sentiments])
        negative_lengths = np.array([len(sentiment['utterance']) for sentiment in negative_sentiments])
        negative_scores = np.array([sentiment['sentiment_score'] for sentiment in negative_sentiments])
        # Create a scatter plot with marker size
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=positive_lengths, y=positive_scores,
                                 mode='markers', marker=dict(size=10 * np.ones(len(positive_sentiments)))))
        fig.add_trace(go.Scatter(x=negative_lengths, y=negative_scores,
                                 mode='markers', marker=dict(size=10 * np.ones(len(negative_sentiments)))))
        fig.update_layout(title='Sentiment Scores vs. Utterance Length for ' + art_style,
                          xaxis_title='Utterance Length',
                          yaxis_title='Sentiment Score')
        fig.write_image(os.path.join("ContrastiveVisualizations", art_style + "_scatterplot_sentiment_vs_length.png"))
        fig.write_html(os.path.join("ContrastiveVisualizations", art_style + "_scatterplot_sentiment_vs_length.html"))
        print("Finished creating Scatter Plot for: " + art_style)

    # Generate a word cloud for the most common words in positive and negative sentiment
    for art_style, sentiment_list in sentiments.items():
        positive_words = []
        negative_words = []
        for sentiment in sentiment_list:
            if sentiment['sentiment_score'] > 0:
                positive_words.extend(sentiment['utterance'].split())
            elif sentiment['sentiment_score'] < 0:
                negative_words.extend(sentiment['utterance'].split())

        positive_wordcloud = WordCloud(background_color='white').generate(' '.join(positive_words))
        plt.imshow(positive_wordcloud, interpolation='bilinear')
        plt.title('Most Common Words in Positive Sentiment for ' + art_style)
        plt.axis('off')
        plt.savefig(os.path.join("ContrastiveVisualizations", art_style + "_positive_wordcloud.png"))
        plt.clf()

        negative_wordcloud = WordCloud(background_color='white').generate(' '.join(negative_words))
        plt.imshow(negative_wordcloud, interpolation='bilinear')
        plt.title('Most Common Words in Negative Sentiment for ' + art_style)
        plt.axis('off')
        plt.savefig(os.path.join("ContrastiveVisualizations", art_style + "_negative_wordcloud.png"))
        plt.clf()
        print("Finished creating Word Cloud for: " + art_style)


# Call the function to generate the visualizations for the sentiment data
read_csv()
print("done")
