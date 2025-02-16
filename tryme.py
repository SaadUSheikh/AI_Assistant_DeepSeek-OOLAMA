# Import the pipeline function from the transformers library
from transformers import pipeline

# Initialize the sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis")

# Define a list of sentences to analyze
sentences = [
    "I love the new design of your website!",
    "I'm not satisfied with the customer service.",
    "The product quality is outstanding.",
    "I had a terrible experience with the delivery."
]

# Analyze the sentiment of each sentence
results = classifier(sentences)

# Display the results
for sentence, result in zip(sentences, results):
    print(f"Sentence: {sentence}")
    print(f"Sentiment: {result['label']}, Confidence: {result['score']:.2f}\n")
