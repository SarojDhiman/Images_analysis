import os
import csv
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
#main_model =  siebert/sentiment-roberta-large-english

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

def analyze_images_in_directory(directory_path):
    results = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(directory_path, filename)
            language = "eng"  # Language code for English
            text_from_image = extract_text_from_image(image_path, language)
            english_text = translate_text(text_from_image)
            sentiment, emoji, probs = process_text(english_text)
            keywords = extract_keywords(english_text)
            # Check if english_text is not empty
            if english_text.strip():
                sections = identify_sections(english_text)
            else:
                sections = []  # No sections found if text is empty
            image_result = {
                'image_name': filename,
                'image_text': text_from_image,
                'translated_text': english_text,
                'sentiment': sentiment,
                'emoji': emoji,
                'probs': probs.tolist(),
                'keywords': keywords,
                'sections': sections
            }
            results.append(image_result)
    return results


def write_results_to_csv(results, output_csv):
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['image_name', 'image_text', 'translated_text', 'sentiment', 'emoji', 'probs', 'keywords', 'sections']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            # Decode byte literals to regular strings using UTF-8
            decoded_result = {key: value.decode('utf-8') if isinstance(value, bytes) else value for key, value in result.items()}
            writer.writerow(decoded_result)


if __name__ == "__main__":
    directory_path = "imagess"
    results = analyze_images_in_directory(directory_path)
    output_csv = "image_analysis_results.csv"
    write_results_to_csv(results, output_csv)
    print("Analysis results saved to:", output_csv)




