import os
import json

import random
from sklearn.model_selection import train_test_split
import numpy as np


def load_json_files(folder_path):
    data_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    data_list.append((file, json.load(f)))
    return data_list

annotation_sources = {}

def prepare_article_data(articles_path):
    article_data = {}
    for year_folder in os.listdir(articles_path):
        year_path = os.path.join(articles_path, year_folder)
        if os.path.isdir(year_path):
            article_data[year_folder] = {}
            articles = load_json_files(year_path)
            for file_name, article in articles:
                #add a source to the annotations file using the article title before truncation
                annotation_sources[file_name[:-5] + '_ann' + '.json'] = article['source']
                article_name = file_name[:-7]  # Removing last two characters and .json
                news_provider = article.get('source', 'unknown').lower()
                # including the title in the sentences
                title = article['title']
                sentences = [sentence for paragraph in article['body-paragraphs'] for sentence in paragraph]
                if article_name not in article_data[year_folder]:
                    article_data[year_folder][article_name] = {}
                if news_provider not in article_data[year_folder][article_name]:
                    article_data[year_folder][article_name][news_provider] = {}

                article_data[year_folder][article_name][news_provider]= {'sentences': sentences, 'title': title}
    return article_data

def prepare_annotation_data(annotations_path):
    annotation_data = {}
    for year_folder in os.listdir(annotations_path):
        year_path = os.path.join(annotations_path, year_folder)
        if os.path.isdir(year_path):
            annotation_data[year_folder] = {}
            annotations = load_json_files(year_path)
            for file_name, annotation in annotations:
                news_provider = annotation_sources[file_name].lower()
                annotation_name = file_name[:-11]  # Removing last two characters and .json

                article_level_annotations = annotation.get('article-level-annotations', {})
                phrase_level_annotations = annotation.get('phrase-level-annotations', [])

                if annotation_name not in annotation_data[year_folder]:
                    annotation_data[year_folder][annotation_name] = {}
                if news_provider not in annotation_data[year_folder][annotation_name]:
                    annotation_data[year_folder][annotation_name][news_provider] = {}

                annotation_data[year_folder][annotation_name][news_provider]['article_level_annotations'] = article_level_annotations
                annotation_data[year_folder][annotation_name][news_provider]['phrase_level_annotations'] = phrase_level_annotations
    return annotation_data



def split_event_overlapping(articles, annotations, test_size=0.2):
    all_articles = []
    all_annotations = []

    for year, articles_dict in articles.items():
        for article_name, providers_dict in articles_dict.items():
            for provider, article_data in providers_dict.items():
                all_articles.append({
                    'title': article_data['title'],
                    'sentences': article_data['sentences']
                })
                all_annotations.append({
                    'article_level_annotations': annotations[year][article_name][provider]['article_level_annotations'],
                    'phrase_level_annotations': annotations[year][article_name][provider]['phrase_level_annotations']
                })

    train_articles, test_articles, train_annotations, test_annotations = train_test_split(
        all_articles, all_annotations, test_size=test_size, random_state=42
    )

    return np.array(train_articles), np.array(test_articles), np.array(train_annotations), np.array(test_annotations)

