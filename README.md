# sentimarto
Validation of ArtEmis 2.0 Dataset Sentiment and Creation of Dall-E Images

## Project Proposal
The ArtEmis 2.0 Dataset provides a unique opportunity to analyze sentiment in human responses to AI generated art
across multiple emotions and styles of art. With a less biased and more balanced distribution of emotions linked
to artwork, this dataset represents a complete approach to the wide range of artwork that could be generated.
Utilizing VADER( Valence Aware Dictionary and sEntiment Reasoner), a rule/lexicon-based, open-source sentiment analyzer built within the NLTK library,
we can effectively quantify emotional textual responses to art. VADER is a tool specifically designed for sentiments
expressed in social media, and it can calculate the text sentiment and returns the probability of a given input sentence
to be positive, negative or neutral. From there, a selection of the top sentiments were used to run a One Sample
T-Test against a Twitter dataset storing 1.6 million tweets. Since Twitter is an accurate representation
of human emotional response to events, this dataset helped determine the validity of the response to the 
ArtEmis 2.0 dataset. Upon validation using a P-Value of 0.05, Open AI's Dall-E was used to create new AI Generated
Art using the style and the prompt that from the ArtEmis 2.0 dataset.

## Future Work
For further comparison and validation, the art pieces generated by Dall-E must be sampled and sent to users for emotional responses. Sentiments within these responses will then be calculated utilizing VADER and a One-Sample T-test will be run against the Sentiment Scores of the emotional responses to ArtEmis 2.0. Conclusive results that Dall-E can generate Art of the same emotional response will only be conclusive if there are no differences found in the sentiments between the responses found by the generation of the Art via Dall-E vs the ArtEmis 2.0 Dataset
