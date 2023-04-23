import os

import nltk
import requests
import pandas
import tabulate
from dotenv import load_dotenv
from textwrap import wrap
from PIL import Image
from io import BytesIO
from nltk.sentiment.vader import SentimentIntensityAnalyzer

load_dotenv()


def generate_image(prompt, model, api_key):
    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {
        "model": model,
        "prompt": prompt,
        "num_images": 1,
        "size": "1024x1024",
        "response_format": "url",
    }
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    return response_json['data'][0]['url']


def process_csv():
    # create the OpenAIGenerations folder if it doesn't exist
    if not os.path.exists('OpenAIGenerations'):
        os.mkdir('OpenAIGenerations')

    df = pandas.read_csv("Contrastive.csv")

    # group the dataframe by art_style and create a dictionary of dataframes
    art_style_groups = df.groupby('art_style')
    dfs_dict = {}
    for art_style, group in art_style_groups:
        dfs_dict[art_style] = group

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
        # select the top positive and top negative sentiments
        top_sentiments = art_style_sentiments[:1] + art_style_sentiments[-1:]
        sentiments[art_style] = top_sentiments

    print("Finished Sentiment Analysis.")

    with open("OpenAIGenerations/sentiments_table.txt", "w") as f:
        f.write("Art Style\tPrompt\tSentiment Score\n")
        for art_style, top_sentiments in sentiments.items():
            for sentiment in top_sentiments:
                prompt_text = "\n".join(wrap(sentiment['utterance'], width=70))
                f.write(f"{art_style}\t{prompt_text}\t{sentiment['sentiment_score']}\n")
        print("Sentiments table saved.")

    for art_style, top_sentiments in sentiments.items():
        for i, sentiment in enumerate(top_sentiments):
            prompt_text = "\n".join(wrap(sentiment['utterance'], width=70))
            image_url = generate_image(prompt_text, 'image-alpha-001', os.getenv('OPENAI_API_KEY'))
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            image.save(f"OpenAIGenerations/{art_style}_{i + 1}.jpg")
            print(f"Image saved: {art_style}_{i + 1}.jpg")


process_csv()
