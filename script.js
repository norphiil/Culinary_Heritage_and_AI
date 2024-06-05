document.getElementById('csvFileInput').addEventListener('change', handleFileSelect);

function handleFileSelect(event) {
    const file = event.target.files[0];

    Papa.parse(file, {
        complete: function(results) {
            var data = results.data;
            data.shift();
            data.pop();
            data = data.map(row => ({
                id: row[0],
                recipeName: row[1],
                ingredients: row[2],
                dishName: row[3],
            }));
            console.log('Parsed CSV data:', data);
            createStaticGraph(data);
            createDynamicGraph(data);
            createVisualization(data);
        }
    });

    document.getElementById('csvFileInput').remove();
}

function createStaticGraph(data) {
    const allIngredients = getAllIngredients(data);
    createGraph(allIngredients, 'Static Ingredients Graph');
}

function createDynamicGraph(data) {
    const dishSelector = document.createElement('select');
    const dishOptions = getDishOptions(data);
    dishSelector.innerHTML = '<option value="all" selected>All</option>' + dishOptions;
    document.body.appendChild(dishSelector);

    const recipeSelector = document.createElement('select');
    recipeSelector.id = 'recipeSelector';
    recipeSelector.innerHTML = '<option value="all" selected>All</option>';
    document.body.appendChild(recipeSelector);


    dishSelector.addEventListener('change', function() {
        const selectedDish = this.value;

        if (selectedDish === 'all') {
            recipeSelector.innerHTML = '<option value="all" selected>All</option>';
            const allIngredients = getAllIngredients(data);
            createGraph(allIngredients, 'All Ingredients Graph');
        } else {
            const recipeOptions = getRecipeOptions(data, selectedDish);
            recipeSelector.innerHTML = '<option value="all" >All</option>' + recipeOptions;
            const ingredients = getIngredientsForDish(data, selectedDish);
            createGraph(ingredients, `Dynamic Ingredients Graph for ${this.value} `);
        }
        return;
    });

    recipeSelector.addEventListener('change', function () {
        const selectedRecipe = this.value;
        const selectedDish = dishSelector.value;
        if (selectedRecipe === 'all') {
            const ingredients = getIngredientsForDish(data, selectedDish);
            createGraph(ingredients, `Dynamic Ingredients Graph for ${dishSelector.value} `);
        } else {
            const ingredients = getIngredientsForDishAndRecipe(data, selectedDish, selectedRecipe);
            createGraph(ingredients, `Dynamic Ingredients Graph for ${dishSelector.value} and ${this.value}`);
        }
        return;
    });

}

function getAllIngredients(data) {
    const allIngredients = data.reduce((acc, row) => {
        if (!row.id) return acc;
        const ingredients = row.ingredients.split(',');
        return acc.concat(ingredients);
    }, []);

    return countOccurrences(allIngredients);
}

function getDishOptions(data) {
    const uniqueDishes = [...new Set(data.map(row => row.dishName))];
    return uniqueDishes.map(dish => `<option value="${dish}">${dish}</option>`).join('');
}

function getRecipeOptions(data, dish) {
    const uniqueRecipes = [...new Set(data.filter(row => row.dishName === dish).map(row => row.recipeName))];
    return uniqueRecipes.map(recipe => `<option value="${recipe}">${recipe}</option>`).join('');
}

function getIngredientsForDish(data, dish) {
    const dishIngredients = data
        .filter(row => row.dishName === dish)
        .reduce((acc, row) => {
            const ingredients = row.ingredients.split(',');
            return acc.concat(ingredients);
        }, []);

    return countOccurrences(dishIngredients);
}

// Tokenize ingredients using natural library
function tokenizeIngredients(ingredients) {
    const tokenizer = new natural.WordTokenizer();
    return tokenizer.tokenize(ingredients);
}


function createVisualization(data) {

        // Function to convert ingredients to binary vector
    function ingredientsToBinaryVector(ingredients) {
        if (!ingredients) return;
        const uniqueIngredients = Array.from(new Set(ingredients.split(',')));
        return data.map(dish => uniqueIngredients.includes(dish) ? 1 : 0);
    }

    // Convert ingredients to binary vectors
  const binaryVectors = data.map(item => ingredientsToBinaryVector(item.ingredients));
function kmeans(data, k, maxIterations) {
  // Initialize centroids randomly
  const centroids = [];
  for (let i = 0; i < k; i++) {
    const randomIndex = Math.floor(Math.random() * data.length);
    centroids.push(data[randomIndex]);
  }

  // Iterate until convergence (centroids don't change significantly)
  let hasConverged = false;
  let iterations = 0;
  while (!hasConverged && iterations < maxIterations) {
    iterations++;

    // Assign data points to closest centroids
    const assignments = data.map(point => {
      let minDistance = Infinity;
      let closestCentroidIndex = -1;
      for (let j = 0; j < k; j++) {
        const distance = euclideanDistance(point, centroids[j]);
        if (distance < minDistance) {
          minDistance = distance;
          closestCentroidIndex = j;
        }
      }
      return closestCentroidIndex;
    });

    // Update centroids based on assigned points
    const newCentroids = [];
    for (let i = 0; i < k; i++) {
      const assignedPoints = data.filter((point, index) => assignments[index] === i);
      const newCentroid = assignedPoints.reduce((acc, point) => {
        return {
          ...acc,
          // Average the values of each dimension across assigned points
          ...(Object.keys(point).forEach(key => (acc[key] = (acc[key] || 0) + point[key]))),
        };
      }, {});
      for (const key in newCentroid) {
        newCentroid[key] /= assignedPoints.length;
      }
      newCentroids.push(newCentroid);
    }

    // Check for convergence (stopping criteria)
    hasConverged = true;
    for (let i = 0; i < k; i++) {
      const oldCentroid = centroids[i];
      const newCentroid = newCentroids[i];
      for (const key in oldCentroid) {
        if (Math.abs(oldCentroid[key] - newCentroid[key]) > 0.001) { // Adjust tolerance as needed
          hasConverged = false;
          break;
        }
      }
      if (!hasConverged) {
        break;
      }
    }

    // Update centroids for the next iteration
    centroids.length = 0;
    centroids.push(...newCentroids);
  }

  // Return the cluster assignments for each data point
  return assignments;
}

// Helper function to calculate Euclidean distance
function euclideanDistance(point1, point2) {
  let sum = 0;
  for (const key in point1) {
    const diff = point1[key] - point2[key];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}



  // Perform k-means clustering
  const k = 3;
  const maxIterations = 100;
  const clusterAssignments = kmeans(binaryVectors, k, maxIterations);

  // Set up D3.js visualization
  const width = 800;
  const height = 600;

  const svg = d3.select('body').append('svg')
    .attr('width', width)
    .attr('height', height);

  const xScale = d3.scaleLinear()
    .domain([d3.min(binaryVectors, d => d[0]), d3.max(binaryVectors, d => d[0])])
    .range([50, width - 50]);

  const yScale = d3.scaleLinear()
    .domain([d3.min(binaryVectors, d => d[1]), d3.max(binaryVectors, d => d[1])])
    .range([50, height - 50]);

  // Plot points with different colors for clusters
  svg.selectAll('circle')
    .data(binaryVectors)
    .enter().append('circle')
    .attr('cx', d => xScale(d[0]))
    .attr('cy', d => yScale(d[1]))
    .attr('r', 5)
    .attr('fill', (d, i) => d3.schemeCategory10[clusterAssignments[i]])
    .on('mouseover', (event, d, i) => {
      // Show tooltip with information
      const tooltip = svg.append('text')
        .attr('x', xScale(d[0]) + 10)
        .attr('y', yScale(d[1]) - 10)
        .attr('class', 'tooltip')
        .text(`Cluster: ${clusterAssignments[i] + 1}`);

      // Adjust position based on mouse position
      if (xScale(d[0]) > width / 2) {
        tooltip.attr('text-anchor', 'end');
      }

      // Remove tooltip on mouseout
      d3.select(this)
        .on('mouseout', () => {
          tooltip.remove();
        });
    });

  // Zoom functionality
  const zoom = d3.zoom()
    .scaleExtent([0.5, 10])
    .on('zoom', (event) => {
      svg.selectAll('circle')
        .attr('cx', d => xScale(event.transform.applyX(d[0])))
        .attr('cy', d => yScale(event.transform.applyY(d[1])));
    });

  svg.call(zoom);
}

function getIngredientsForDishAndRecipe(data, dish, recipe) {
    const dishIngredients = data
        .filter(row => row.dishName === dish && row.recipeName === recipe)
        .reduce((acc, row) => {
            const ingredients = row.ingredients.split(',');
            return acc.concat(ingredients);
        }, []);

    return countOccurrences(dishIngredients);
}

function countOccurrences(arr) {
    const ingredientCount = {};
    arr.forEach(ingredient => {
        ingredientCount[ingredient] = (ingredientCount[ingredient] || 0) + 1;
    });

    return Object.entries(ingredientCount).map(([ingredient, count]) => ({ ingredient, count }));
}

function createGraph(data, title) {
    const container = document.getElementById('graphContainer');
    container.innerHTML = '';

    data.sort(function(b, a) {
        return a.count - b.count;
    });

    const tooltip = d3.select(container)
        .append('div')
        .attr('id', 'tooltip')
        .attr('class', 'tooltip')
        .style('position', 'absolute')
        .style("pointer-events", "none")
        .style('opacity', 0);

    const margin = { top: 20, right: 0, bottom: 100, left: 50 };
    var w = Math.max(window.innerWidth/3, window.innerWidth * data.length / 100);
    var h = window.innerHeight;
    const svg = d3.select(container).append('svg')
        .attr('width', w + margin.left + margin.right)
        .attr('height', h / 3 * 2)
        .append("g")
        .attr("transform",
          "translate(0," + margin.top + ")");

    const width = w;
    const height = h/3*2 - margin.top - margin.bottom;

    const g = svg.append('g')
        .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

    const x = d3.scaleBand()
        .rangeRound([0, width])
        .padding(0.5)
        .domain(data.map(d => d.ingredient));

    const y = d3.scaleLinear()
        .rangeRound([height, 0])
        .domain([0, d3.max(data, d => d.count)]);

    g.append('g')
        .attr('class', 'axis axis-x')
        .attr('transform', 'translate(0,' + height + ')').style("pointer-events", "none")
        .call(d3.axisBottom(x))
    .selectAll("text")
      .attr("transform", "translate(-10,0)rotate(-45)")
        .style("text-anchor", "end");

    g.append('g')
        .attr('class', 'axis axis-y')
        .call(d3.axisLeft(y).ticks(10));

    const bars = g.selectAll('.bar')
        .data(data)
        .enter().append('rect')
        .attr('class', 'bar')
        .attr('x', d => x(d.ingredient))
        .attr('y', d => y(d.count))
        .attr('width', x.bandwidth())
        .attr('height', d => height - y(d.count))
        .attr("fill", "#69b3a2")
        .on('mouseover', function (d) {
            const object = d.target.__data__
            const rect = this.getBoundingClientRect();
            tooltip.html(`<strong>${object.ingredient}</strong><br>${object.count} times`)
                .style('left', `${window.scrollX + rect.left}px`)
                .style('top', `${rect.top - 40}px`)
            tooltip.transition()
                .duration(200)
                .style('opacity', .9);
        })
        .on('mouseout', function () {
            tooltip.transition()
                .duration(500)
                .style('opacity', 0);
        });

}
