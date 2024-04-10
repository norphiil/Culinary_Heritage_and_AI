from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA
from django.shortcuts import render
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from collections import Counter
from .models import Recipe
import csv


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


def optimal_k(data, max_k=10):
    if len(data) == 1:
        return 1
    distortions = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)

    # Find the optimal k
    delta_distortions = [distortions[i] - distortions[i+1] for i in range(len(distortions)-1)]
    k_optimal = delta_distortions.index(max(delta_distortions)) + 2  # Add 2 because we start from k=1
    return k_optimal


def tokenize_and_cluster(recipes):
    # Tokenize ingredients into binary vectors
    ingredients_list = [recipe.ingredients.split(',') for recipe in recipes]
    ingredients_binary = []
    for ingredients in ingredients_list:
        binary_vector = {ingredient: 1 for ingredient in ingredients}
        ingredients_binary.append(binary_vector)

    # Use DictVectorizer to convert binary vectors into a matrix
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(ingredients_binary)

    # Cluster similar dishes
    opt_k = optimal_k(X, max_k=5)
    kmeans = KMeans(n_clusters=opt_k)  # You can adjust the number of clusters as needed
    clusters = kmeans.fit_predict(X)

    # Apply dimensionality reduction using PCA for visualization
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    # Extract coordinates for plotting
    x_coords = X_reduced[:, 0]
    y_coords = X_reduced[:, 1]

    color_map = {
        0: "red",
        1: "blue",
        2: "green",
        3: "yellow",
        4: "purple",
        5: "orange",
        6: "pink",
        7: "brown",
        8: "black",
        9: "grey"
    }

    # Prepare data for plotting
    data = []
    for i in range(len(recipes)):
        data.append({
            "x": x_coords[i],
            "y": y_coords[i],
            "color": color_map[clusters[i]],
            "recipe_name": recipes[i].recipe_name
        })

    return data


def get_similarity_recipes(recipes):

    # Tokenize ingredients into binary vectors
    ingredients_list = [recipe.ingredients.split(',') for recipe in recipes]
    ingredients_binary = []
    for ingredients in ingredients_list:
        binary_vector = {ingredient: 1 for ingredient in ingredients}
        ingredients_binary.append(binary_vector)

    # Use DictVectorizer to convert binary vectors into a matrix
    vectorizer = DictVectorizer(sparse=False)
    X = vectorizer.fit_transform(ingredients_binary)

    # Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(X)
    
    # Apply dimensionality reduction using PCA for visualization
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    # Get indices of recipes sorted by similarity (excluding self-similarity)
    sorted_indices = similarity_matrix.argsort(axis=1)[:, ::-1][:, 1:]
    return sorted_indices, similarity_matrix, X_reduced


def find_similar_recipes(recipes, sorted_indices, similarity_matrix, X_reduced):
    similarity_scores = similarity_matrix.max(axis=1)

    x_coords = X_reduced[:, 0]
    y_coords = X_reduced[:, 1]
    scaled_x_coords = x_coords * similarity_scores
    scaled_y_coords = y_coords * similarity_scores
    # Extract most similar recipes
    similar_recipes = []
    for i, recipe_index in enumerate(sorted_indices):
        most_similar_index = recipe_index[0]
        # similar_recipes.append({
        #     "recipe_name": recipes[i].recipe_name,
        #     "similar_recipe_name": recipes[int(most_similar_index)].recipe_name,
        #     "similarity_score": similarity_matrix[i, most_similar_index]
        # })
        similar_recipes.append({
            "x": scaled_x_coords[i],
            "y": scaled_y_coords[i],
            "similarity_score": similarity_matrix[i, most_similar_index],
            "recipe_name": recipes[i].recipe_name
        })

    return similar_recipes


def find_dissimilar_recipes(recipes, sorted_indices, similarity_matrix, X_reduced):
    similarity_scores = 1 - similarity_matrix.max(axis=1)

    x_coords = X_reduced[:, 0]
    y_coords = X_reduced[:, 1]
    scaled_x_coords = x_coords * similarity_scores
    scaled_y_coords = y_coords * similarity_scores
    # Extract most dissimilar recipes
    dissimilar_recipes = []
    for i, recipe_index in enumerate(sorted_indices):
        most_dissimilar_index = recipe_index[-1]
        # dissimilar_recipes.append({
        #     "recipe_name": recipes[i].recipe_name,
        #     "dissimilar_recipe_name": recipes[int(most_dissimilar_index)].recipe_name,
        #     "dissimilarity_score": similarity_matrix[i, most_dissimilar_index]
        # })

        dissimilar_recipes.append({
            "x": scaled_x_coords[i],
            "y": scaled_y_coords[i],
            "similarity_score": similarity_matrix[i, most_dissimilar_index],
            "recipe_name": recipes[i].recipe_name
        })

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
    selected_recipe = request.GET.get('recipe', None)
    selected_dish = request.GET.get('dish', None)

    if selected_dish:
        recipes = Recipe.objects.filter(dish_name=selected_dish)
        recipes_selector = recipes

        sorted_indices, similarity_matrix, X_reduced = get_similarity_recipes(recipes)
        similar_recipes = find_similar_recipes(recipes, sorted_indices, similarity_matrix, X_reduced)
        dissimilar_recipes = find_dissimilar_recipes(recipes, sorted_indices, similarity_matrix, X_reduced)
        if selected_recipe:
            recipes = recipes.filter(recipe_name=selected_recipe)
    else:
        recipes_selector = recipes
        similar_recipes = []
        dissimilar_recipes = []
    bar_chart_graph_title = "Most Popular Ingredients"
    bar_chart_graph_subtitle = "Based on selected filters"
    bar_chart_x_title = "Ingredients"
    bar_chart_y_title = "Frequency"

    scatter_chart_graph_title = "Most Popular Ingredients"
    scatter_chart_graph_subtitle = "Based on selected filters"
    scatter_chart_x_title = "Ingredients"
    scatter_chart_y_title = "Frequency"

    similarity_bar_chart_graph_title = "Recipe Similarity"
    similarity_bar_chart_graph_subtitle = "Based on selected filters"
    similarity_bar_chart_x_title = "Recipe"
    similarity_bar_chart_y_title = "Similarity Score"

    dissimilarity_bar_chart_graph_title = "Recipe Dissimilarity"
    dissimilarity_bar_chart_graph_subtitle = "Based on selected filters"
    dissimilarity_bar_chart_x_title = "Recipe"
    dissimilarity_bar_chart_y_title = "Dissimilarity Score"

    return render(request, 'index.html', {
        "bar_chart_title": bar_chart_graph_title,
        "bar_chart_subtitle": bar_chart_graph_subtitle,
        "bar_chart_x_title": bar_chart_x_title,
        "bar_chart_y_title": bar_chart_y_title,
        "bar_chart_data": get_bar_data(recipes),

        "scatter_chart_title": scatter_chart_graph_title,
        "scatter_chart_subtitle": scatter_chart_graph_subtitle,
        "scatter_chart_x_title": scatter_chart_x_title,
        "scatter_chart_y_title": scatter_chart_y_title,
        "scatter_chart_data": tokenize_and_cluster(recipes_selector),

        "similarity_bar_chart_title": similarity_bar_chart_graph_title,
        "similarity_bar_chart_subtitle": similarity_bar_chart_graph_subtitle,
        "similarity_bar_chart_x_title": similarity_bar_chart_x_title,
        "similarity_bar_chart_y_title": similarity_bar_chart_y_title,
        "similarity_bar_chart_data": similar_recipes,

        "dissimilarity_bar_chart_title": dissimilarity_bar_chart_graph_title,
        "dissimilarity_bar_chart_subtitle": dissimilarity_bar_chart_graph_subtitle,
        "dissimilarity_bar_chart_x_title": dissimilarity_bar_chart_x_title,
        "dissimilarity_bar_chart_y_title": dissimilarity_bar_chart_y_title,
        "dissimilarity_bar_chart_data": dissimilar_recipes,

        "dishes": Recipe.objects.values_list('dish_name', flat=True).distinct(),
        "recipes": recipes_selector,
        "selected_recipe": selected_recipe,
        "selected_dish": selected_dish,
    })
