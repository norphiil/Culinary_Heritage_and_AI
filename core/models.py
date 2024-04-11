from django.db import models


class Recipe(models.Model):
    recipe_name = models.CharField(max_length=100)
    ingredients = models.TextField()
    dish_name = models.CharField(max_length=100)
