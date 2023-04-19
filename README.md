# sentimarto
Quantifying Human Sentiment in AI Generated Art

## Project Proposal
The ArtEmis 2.0 Dataset provides a unique opportunity to analyze sentiment in human responses to AI generated art
across multiple emotions and styles of art. With a less biased and more balanced distribution of emotions linked
to artwork, this dataset represents a complete approach to the wide range of artwork that could be generated.
Utilizing Python’s text2emotion library, I plan on validating the dominant emotion associated with each generated
image with its given explanation. Although text2emotion provides some limitation as it only provides functionality
for five basic emotions: happy, angry, sad, surprise, and fear this will provide an opportunity to validate the
dominant emotions in images and remove outliers from the dataset. Once removed, I plan to further analyze the
remaining prompts and images by performing a sentiment analysis. Utilizing VADER( Valence Aware Dictionary and
sEntiment Reasoner), a rule/lexicon-based, open-source sentiment analyzer built within the NLTK library,
we can effectively quantify emotional responses to AI Generated art. VADER is a tool specifically designed for sentiments
expressed in social media, and it can calculate the text sentiment and returns the probability of a given input sentence
to be positive, negative or neutral. From there, I will remove utterances according to specified threshold given by 
the sentiment analysis and then feed the remaining utterances into an Open AI model to construct images of a similar
natures. Future work will include an image-to-emotion analysis. 
