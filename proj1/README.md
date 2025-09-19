[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/-VohRijK)

# Explainable AI Assignment 1: Projection Space Exploration
In this assignment, you are challenged to analyze and compare solutions of a problem, game, algorithm, model, or anything else that can be represented by sequential states. For this, you will project the high-dimensional states to the two-dimensional space, connect the states, and add meta-data to the visualization.

Exemplary solutions are provided in the `solution_rubik.ipynb` and `solution_2048.ipynb` notebooks. 

Further examples to analyze are (board) games and approximation algorithms. The 2048 notebook uses [OpenAI Gym](https://gym.openai.com/) to create a game environment and produce state data. There is a variety of first and third party environments for Gym that can be used.

## General Information Submission

For the intermediate submission, please enter the group and dataset information. Coding is not yet necessary.

**Group Members**

| Student ID | First Name | Last Name | E-Mail                    | Workload [%] |
| ----------|------------|-----------|---------------------------|--------------|
| K12140156 | Jaroslav   | Kvasnicka | K12140156@students.jku.at | 25           |
| K01611130 | Severin    | Lechner   | k01611130@students.jku.at | 25           |
| K11929354 | Stepan     | Malysh    | K11929354@students.jku.at | 25           |
| K01119156 | Manuel     | Bichler   | K01119156@students.jku.at | 25           |

### Dataset
Please add your dataset to the repository (or provide a link if it is too large) and answer the following questions about it:

* Which dataset are you using? What is it about?
 
  We are using our own dataset, which is about the game of Tic-Tac-Toe, but taking the first moves from the real world. We found some data on first moves of 200 people on Reddit where a person shared the results collected over a week by playing Tic-Tac-Toe with family, neighbors, colleagues, and friends, capturing a range of choices from participants aged 9 to 67. Each row of our generated dataset represents a state of the game board during a simulated game, including the positions of the pieces and the player's move. 
* Where did you get this dataset from (i.e., source of the dataset)? How was the dataset generated?

  We calculated the probability distribution for each possible first move by Player X using a dataset of observed moves, allowing us to simulate X's first move based on these probabilities. For Player O's response, we conditioned the choice on X's initial move, creating a probability-adjusted response for O’s first turn. We created the dataset ourselves using a Python script that simulates random games of Tic-Tac-Toe. The script records each move, the state of the board at each step, and the outcome of the game (win/loss/draw). You can find the script in `/data/tic_tac_toe/tic_tac_toe.py`. 

PROVIDE THE UPDATED LINKS
  
* What is dataset size in terms of nodes, items, rows, columns, ...?

  The dataset consists of approximately 10,000 simulated games. Each game generates multiple rows of data, one for each move. Each row contains 13 columns: 9 columns for the board positions (pos00 to pos22), 1 column for the current player, 1 column for the step number (move order), 1 column for the game result (win/loss/draw) and one column indicating the state of the game (start/mid/end).
  
* What do you want to analyze?

  We want to analyze patterns and strategies of Tic-Tac-Toe game, specifically focusing on the starting moves made by players and how these moves influence the game's outcome (win, loss, or draw). Through the generated dataset of simulated games, we can observe which starting moves are most commonly selected, identify successful sequences of moves, and evaluate whether certain strategies consistently lead to winnings or draws.
  
* What are you expecting to see?

  Firstly, which starting positions are played more often and which responses are played more often. Secondly, we will pay attention if any optimal or preferred strategies exist which lead more often to the win. Finally, we will study if there are any defensive strategies that consistenly lead to draw. Furthermore we hope to see some Outcome/Game state clusters.

## Final Submission

* Make sure that you pushed your GitHub repository and not just committed it locally.
* Sending us an email with the code is not necessary.
* Update the *environment.yml* file if you need additional libraries, otherwise the code is not executeable.
* Create a single, clearly named notebook with your solution, e.g. solution.ipynb.
* Save your final executed notebook as html (File > Download as > HTML) and add them to your repository.

## Drive and Video:
https://drive.google.com/drive/folders/1iyFMa5rZz5LmxD2QpnXDyt-3MvLnBp42?usp=sharing

## Development Environment

Checkout this repo and change into the folder:
```
git clone https://github.com/jku-icg-classroom/xai_proj_space_2024-<GROUP_NAME>.git
cd xai_proj_space_2024-<GROUP_NAME>
```

Load the conda environment from the shared `environment.yml` file:
```
conda env create -f environment.yml
conda activate xai_proj_space
```

> Hint: For more information on Anaconda and enviroments take a look at the README in our [tutorial repository](https://github.com/JKU-ICG/python-visualization-tutorial).

Then launch Jupyter Lab:
```
jupyter lab
```

Go to http://localhost:8888/ and open the *template* notebook.

Alternatively, you can also work with [binder](https://mybinder.org/), [deepnote](https://deepnote.com/), [colab](https://colab.research.google.com/), or any other service as long as the notebook runs in the standard Jupyter environment.
