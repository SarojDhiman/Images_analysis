from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from torch.nn.functional import softmax
from googletrans import Translator
from PIL import Image
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import json
import pytesseract

# Load pre-trained RoBERTa model and tokenizer for sentiment analysis
roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_model = AutoModelForSequenceClassification.from_pretrained(roberta_model_name)
sentiment_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)

# Load pre-trained section identification model
section_model_name = "facebook/bart-large-mnli"
section_classifier = pipeline("zero-shot-classification", model=section_model_name, tokenizer=section_model_name)

# Load sections data
with open("label_data_og.json", "r") as file:
    sections_data = json.load(file)

# Initialize translator
translator = Translator()

def extract_text_from_image(image_path, language):
    # Open the image using PIL (Python Imaging Library)
    image = Image.open(image_path)
    # Use pytesseract to extract text from the image
    extracted_text = pytesseract.image_to_string(image, lang=language)
    # Preprocess extracted text
    preprocessed_text = preprocess_text(extracted_text)
    return preprocessed_text

# Function to preprocess text
def preprocess_text(text):
    # Remove artifacts
    cleaned_text = re.sub(r'\n+', ' ', text)  # Remove multiple consecutive line breaks
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove multiple consecutive whitespaces
    return cleaned_text.strip()

# Function to translate text to English with support for multiple languages
def translate_text(text, source_language='auto'):
    translator = Translator()
    translated_text = translator.translate(text, src=source_language, dest='en').text
    return translated_text

# Function to process text
def process_text(text):
    # Tokenize and encode the text
    inputs = sentiment_tokenizer(text, return_tensors='pt', truncation=True)

    # Forward pass through the model
    outputs = sentiment_model(**inputs)

    # Get probabilities for each class (e.g., positive, negative)
    probs = softmax(outputs.logits, dim=1).detach().numpy()[0]

    # Interpret results
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    predicted_sentiment = sentiment_labels[probs.argmax()]

    # Map sentiment labels to emojis
    emoji_mapping = {
        'Negative': '😠',
        'Neutral': '😐',
        'Positive': '😊'
    }

    # Get the corresponding emoji for the predicted sentiment
    predicted_emoji = emoji_mapping.get(predicted_sentiment, '')

    return predicted_sentiment, predicted_emoji, probs

# Function to extract keywords from text
def extract_keywords(text, num_keywords=5):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalnum()]

    # Calculate keyword frequencies
    fdist = FreqDist(filtered_tokens)

    # Get the top N keywords
    top_keywords = fdist.most_common(num_keywords)

    # Extract keywords
    keywords = [keyword for keyword, _ in top_keywords]

    return keywords

# Function to identify sections in the text
def identify_sections(text):
    classification = section_classifier(text, list(sections_data.keys()))
    top_sections = classification['labels'][:3]
    top_probabilities = classification['scores'][:3]
    sections = [{"section": section, "probability": probability} for section, probability in zip(top_sections, top_probabilities)]
    return sections

# Example usage:
image_path = "eng.jpg"
language = "eng"  # Language code for English
text_from_image = extract_text_from_image(image_path, language)
english_text = translate_text(text_from_image)
sentiment, emoji, probs = process_text(english_text)
keywords = extract_keywords(english_text)
sections = identify_sections(english_text)

print("Text from image:", text_from_image)
print("Translated text:", english_text)
print("Sentiment:", sentiment)
print("Emoji:", emoji)
print("Probabilities:", probs)
print("Keywords:", keywords)
print("Sections:", sections)
