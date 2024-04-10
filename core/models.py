from django.db import models
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from pyclustering.cluster.gmeans import gmeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import numpy as np


class Recipe(models.Model):
    recipe_name = models.CharField(max_length=100)
    ingredients = models.TextField()
    dish_name = models.CharField(max_length=100)

    def tokenize_ingredients(self):
        # Split the ingredients string into a list
        ingredient_list = self.ingredients.split(',')

        # Create a dictionary with unique ingredients as keys and values as 1
        ingredient_dict = {ingredient: 1 for ingredient in ingredient_list}

        return ingredient_dict


    @staticmethod
    def cluster_recipes(recipes):
        # Tokenize ingredients for all recipes
        ingredients_list = [recipe.tokenize_ingredients() for recipe in recipes]
        # Use DictVectorizer to convert the list of dictionaries into a feature matrix
        dict_vectorizer = DictVectorizer(sparse=False)
        ingredients_matrix = dict_vectorizer.fit_transform(ingredients_list)

        # Determine the optimal number of clusters
        k_optimal = Recipe.optimal_k(ingredients_matrix, max_k=5)
        print("k_optimal: ", k_optimal)
        # Cluster the recipes using K-means
        kmeans = KMeans(n_clusters=k_optimal, init='k-means++')
        cluster_labels = kmeans.fit_predict(ingredients_matrix)

        # Dimensionality reduction using t-SNE
        tsne = TSNE(n_components=2)
        reduced_features = tsne.fit_transform(ingredients_matrix)

        return cluster_labels, reduced_features
