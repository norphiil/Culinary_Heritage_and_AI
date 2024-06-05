# Culinary_Heritage_and_AI

## Installation

(used with Python 3.11.5, but should work with other versions as well)

1. Clone the repository

```bash
git clone https://github.com/norphiil/Culinary_Heritage_and_AI.git
```

2. Install the required packages

```bash
pip install -r requirements.txt
```

3. Run the application

```bash
py .\manage.py runserver
```

## Usage

1. Open the application in your browser (Default `http://127.0.0.1:8000/`)
2. The database is created with the `clean_data.csv` file one time, you can click on rest database in the web interface to refresh and update the application database with the file data `clean_data.csv`.
3. You can zoom in the chart by selecting a specific area with your mouse
4. You can select a specific dish, recipe and cluster by clicking on the corresponding dropdown menu
   1. When you select a dish, the recipe and cluster dropdown menu will be updated with the corresponding data
   2. When you select a dish, the similar and dissimilar recipes section appears at the bottom of the page.
      1. You can select a specific recipe from the dropdown to only see the similarity between the selected recipe and the other recipes of the same dish.
      2. You can also select a specific recipe from the similar and dissimilar recipes section to see more details.
