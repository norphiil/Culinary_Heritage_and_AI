from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from django.shortcuts import render
from collections import Counter
from .models import Recipe
import numpy as np
import csv

COLOR = {
    0: {
        "hex": "#FF0000",
        "name": "Red"
        },   # Red
    1: {
        "hex": "#00FF00",
        "name": "Lime"
        },   # Lime
    2: {
        "hex": "#0000FF",
        "name": "Blue"
        },   # Blue
    3: {
        "hex": "#FFFF00",
        "name": "Yellow"
        },   # Yellow
    4: {
        "hex": "#FF00FF",
        "name": "Fuchsia"
        },   # Fuchsia
    5: {
        "hex": "#00FFFF",
        "name": "Cyan / Aqua"
        },   # Cyan / Aqua
    6: {
        "hex": "#FFA500",
        "name": "Orange"
        },   # Orange
    7: {
        "hex": "#800080",
        "name": "Purple"
        },   # Purple
    8: {
        "hex": "#808080",
        "name": "Gray"
        },   # Gray
    9: {
        "hex": "#000000",
        "name": "Black"
        },   # Black
    10: {
        "hex": "#FFFFFF",
        "name": "White"
        },  # White
    11: {
        "hex": "#FFC0CB",
        "name": "Pink"
        },  # Pink
    12: {
        "hex": "#800000",
        "name": "Maroon"
        },  # Maroon
    13: {
        "hex": "#008000",
        "name": "Green"
        },  # Green
    14: {
        "hex": "#008080",
        "name": "Teal"
        },  # Teal
    15: {
        "hex": "#000080",
        "name": "Navy"
        },  # Navy
    16: {
        "hex": "#FFD700",
        "name": "Gold"
        },  # Gold
    17: {
        "hex": "#FF4500",
        "name": "OrangeRed"
        },  # OrangeRed
    18: {
        "hex": "#DC143C",
        "name": "Crimson"
        },  # Crimson
    19: {
        "hex": "#FF6347",
        "name": "Tomato"
        },  # Tomato
    20: {
        "hex": "#FF8C00",
        "name": "DarkOrange"
        },  # DarkOrange
    21: {
        "hex": "#FFA07A",
        "name": "LightSalmon"
        },  # LightSalmon
    22: {
        "hex": "#FFA500",
        "name": "Orange"
        },  # Orange
    23: {
        "hex": "#C71585",
        "name": "Deep Pink"
        },  # Deep Pink
    24: {
        "hex": "#00CED1",
        "name": "Dark Turquoise"
        },  # Dark Turquoise
    25: {
        "hex": "#2F4F4F",
        "name": "Dark Gray"
        },  # Dark Gray
}


def read_csv_and_save_to_db(file_path):
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            ingredient = row['Ingredients'].replace(",&", "").replace("&", "")
            if ingredient != "":
                Recipe.objects.create(
                    recipe_name=row['Recipe_Name'],
                    ingredients=row['Ingredients'],
                    dish_name=row['Dish_Name']
                )


def clean_all_recipes():
    Recipe.objects.all().delete()


def tokenize_and_cluster(recipes, selected_cluster: int = None):
    # Tokenize ingredients into binary vectors
    ingredients_list = [recipe.ingredients.split(',') for recipe in recipes]
    ingredients_binary = []
    for ingredients in ingredients_list:
        binary_vector = {ingredient: 1 for ingredient in ingredients}
        ingredients_binary.append(binary_vector)

    vectorizer: DictVectorizer = DictVectorizer(sparse=False)
    X: np.ndarray = vectorizer.fit_transform(ingredients_binary)

    hca = AgglomerativeClustering(n_clusters=None, distance_threshold=14)
    clusters = hca.fit_predict(X)

    pca: PCA = PCA(n_components=2)
    X_reduced: np.ndarray = pca.fit_transform(X)

    x_coords = X_reduced[:, 0]
    y_coords = X_reduced[:, 1]

    data: list = []
    new_recipes: list = []
    for i in range(len(recipes)):
        if selected_cluster is None or clusters[i] == selected_cluster:
            data.append({
                "x": x_coords[i],
                "y": y_coords[i],
                "cluster": clusters[i],
                "color": COLOR[clusters[i]]['hex'],
                "recipe_name": recipes[i].recipe_name,
            })
            new_recipes.append(recipes[i])

    return data, clusters, new_recipes


def get_similarity_recipes(recipes):
    # Tokenize ingredients
    ingredients_list = [recipe.ingredients.split(',') for recipe in recipes]
    ingredients_binary = []
    for ingredients in ingredients_list:
        binary_vector = {ingredient: 1 for ingredient in ingredients}
        ingredients_binary.append(binary_vector)

    vectorizer: DictVectorizer = DictVectorizer(sparse=False)
    X: np.ndarray = vectorizer.fit_transform(ingredients_binary)
    # Use TF-IDF to vectorize ingredients

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(X)

    return similarity_matrix


def find_similar_recipes(recipes, similarity_matrix):
    similar_recipes = []
    recipes_list = list(recipes)
    nb = 5
    for i, recipe in enumerate(recipes_list):
        similar_indices: int = np.argsort(similarity_matrix[i])[::-1][1:nb+1]
        similarities = [{"recipe": recipes_list[index], "similarity": similarity_matrix[i][index]} for index in similar_indices if recipes_list[index].recipe_name != recipe.recipe_name]
        avg_similarity = sum(similarity["similarity"] for similarity in similarities) / nb
        similar_recipes.append({"recipe": recipe, "avg_similarity": avg_similarity, "data": similarities})

    similar_recipes.sort(key=lambda x: x['avg_similarity'], reverse=True)
    return similar_recipes


def find_dissimilar_recipes(recipes, similarity_matrix):
    dissimilar_recipes = []
    recipes_list = list(recipes)
    nb = 5
    for i, recipe in enumerate(recipes):
        dissimilar_indices: int = np.argsort(similarity_matrix[i])[::-1][-nb:]
        dissimilarities = [{"recipe": recipes_list[index], "similarity": similarity_matrix[i][index]} for index in dissimilar_indices if recipes_list[index].recipe_name != recipe.recipe_name]
        avg_dissimilarity = sum(dissimilarity["similarity"] for dissimilarity in dissimilarities) / nb
        dissimilar_recipes.append({"recipe": recipe, "avg_dissimilarity": avg_dissimilarity, "data": dissimilarities})

    dissimilar_recipes.sort(key=lambda x: x['avg_dissimilarity'], reverse=False)
    return dissimilar_recipes


def get_bar_data(recipes):
    ingredients_count = Counter()
    for recipe in recipes:
        ingredients = recipe.ingredients.split(',')
        ingredients_count.update(ingredients)

    data = [{"label": ingredient, "y": count} for ingredient, count in ingredients_count.most_common()]
    return data


def index(request):
    # clean_all_recipes()
    if Recipe.objects.count() == 0:
        print("Reading CSV and saving to DB")
        read_csv_and_save_to_db('clean_data.csv')

    print("Reading from DB")
    recipes = Recipe.objects.all()
    selected_recipe: str = request.GET.get('recipe', None)
    selected_dish: str = request.GET.get('dish', None)
    selected_cluster: str = request.GET.get('clusters', None)
    if selected_cluster and selected_cluster.isdigit():
        selected_cluster = int(selected_cluster)
    else:
        selected_cluster = None
    if selected_dish:
        recipes = Recipe.objects.filter(dish_name=selected_dish)
        recipes_selector = recipes

        similarity_matrix = get_similarity_recipes(recipes)
        similar_recipes = find_similar_recipes(recipes, similarity_matrix)
        dissimilar_recipes = find_dissimilar_recipes(recipes, similarity_matrix)
        if selected_recipe:
            recipes = recipes.filter(recipe_name=selected_recipe)
            similar_recipes = [recipe for recipe in similar_recipes if recipe['recipe_name'] == selected_recipe]
            dissimilar_recipes = [recipe for recipe in dissimilar_recipes if recipe['recipe_name'] == selected_recipe]
    else:
        recipes_selector = recipes
        similar_recipes = []
        dissimilar_recipes = []
    bar_chart_graph_title = "Most Popular Ingredients"
    bar_chart_graph_subtitle = "Based on selected filters"
    bar_chart_x_title = "Ingredients"
    bar_chart_y_title = "Frequency"

    scatter_chart_graph_title = "Recipe Clustering"
    scatter_chart_graph_subtitle = "Based on selected filters"
    scatter_chart_x_title = ""
    scatter_chart_y_title = ""
    data, clusters, new_recipes = tokenize_and_cluster(recipes_selector, selected_cluster=selected_cluster)
    print("Similar Recipes", similar_recipes)
    print(len(similar_recipes))
    print(len(recipes))
    bar_data = get_bar_data(new_recipes)
    return render(request, 'index.html', {
        "bar_chart_title": bar_chart_graph_title,
        "bar_chart_subtitle": bar_chart_graph_subtitle,
        "bar_chart_x_title": bar_chart_x_title,
        "bar_chart_y_title": bar_chart_y_title,
        "bar_chart_data": bar_data,

        "scatter_chart_title": scatter_chart_graph_title,
        "scatter_chart_subtitle": scatter_chart_graph_subtitle,
        "scatter_chart_x_title": scatter_chart_x_title,
        "scatter_chart_y_title": scatter_chart_y_title,
        "scatter_chart_data": data,

        "similar_recipes": similar_recipes,
        "dissimilar_recipes": dissimilar_recipes,

        "clusters": [{"cluster": int(cluster), "color": COLOR[int(cluster)]["name"]} for cluster in range(max(clusters)+1)],
        "selected_cluster": selected_cluster,

        "dishes": Recipe.objects.values_list('dish_name', flat=True).distinct(),
        "recipes": recipes_selector,
        "selected_recipe": selected_recipe,
        "selected_dish": selected_dish,
    })
