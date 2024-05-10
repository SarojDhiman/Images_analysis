# tweet_classifier_module.py

import pandas as pd
from transformers import pipeline
import json

class ZeroShotClassifierManager:
    def __init__(self, query_language_map, sections_data):
        self.pipelines = {}
        self.sections_data = sections_data

        for lang, model_name in query_language_map.items():
            try:
                self.pipelines[lang] = pipeline("zero-shot-classification", model=model_name, tokenizer=model_name)
            except Exception as e:
                print(f"Error loading pipeline for language '{lang}': {str(e)}")

    def process_queries(self, queries):
        results = {}
        for lang, lang_queries in queries.items():
            lang_results = []
            for query in lang_queries:
                lang_results.append(self.process_query(lang, query))
            results[lang] = lang_results
        return results

    def process_query(self, lang, query):
        classification = self.pipelines[lang](query, list(self.sections_data.keys()))
        top_sections = classification['labels'][:3]
        top_probabilities = classification['scores'][:3]
        return [{"section": section, "probability": probability} for section, probability in zip(top_sections, top_probabilities)]

    def process_csv(self, csv_path, tweet_column):
        df = pd.read_csv(csv_path)
        lang = "en"  # Assuming English language for now
        results = []
        for tweet in df[tweet_column]:
            classification = self.pipelines[lang](tweet, list(self.sections_data.keys()))
            top_sections = classification['labels'][:3]
            top_probabilities = classification['scores'][:3]
            results.append([{"section": section, "probability": probability} for section, probability in zip(top_sections, top_probabilities)])
        return results

def load_sections_data(json_path):
    with open(json_path, "r") as file:
        return json.load(file)

