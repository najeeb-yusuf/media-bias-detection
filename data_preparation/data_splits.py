import random
import numpy as np

def split_event_non_overlapping(articles, annotations, test_size=0.2):
    # Group articles and annotations by event
    train_articles = []
    train_annotations = []
    test_articles = []
    test_annotations = []

    grouped_articles = []
    grouped_annotations = []


    for year, articles_dict in articles.items():
      for article_name, providers_dict in articles_dict.items():
        grouped_articles.append(providers_dict)
        grouped_annotations.append(annotations[year][article_name])

    group_size = len(grouped_articles)
    test_size = int(group_size * test_size)
    train_size = group_size - test_size

    indices = list(range(group_size))
    random.shuffle(indices)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    for i in train_indices:
      for provider, article_data in grouped_articles[i].items():
        train_articles.append({
          'title': article_data['title'],
          'sentences': article_data['sentences']
        })
        train_annotations.append(grouped_annotations[i][provider])

    for i in test_indices:
      for provider, article_data in grouped_articles[i].items():
        test_articles.append({
          'title': article_data['title'],
          'sentences': article_data['sentences']
        })
        test_annotations.append(grouped_annotations[i][provider])
    return np.array(train_articles), np.array(test_articles), np.array(train_annotations), np.array(test_annotations)

