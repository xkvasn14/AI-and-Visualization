import random
import pandas as pd
import numpy as np
# from datetime import datetime

"""
The following code creates a simulated dataset of the game Tic-Tac-Toe G

Each position (pos00, pos01, etc.) corresponds to a cell on a 3x3 Tic-Tac-Toe board.

Board with Position Names:

  pos00 | pos01 | pos02 
 -------+-------+-------
  pos10 | pos11 | pos12 
 -------+-------+-------
  pos20 | pos21 | pos22 

Example Board Representation:
-----------------------------
Each position can hold one of the following values:
- 1: Player X
- -1: Player O
- 0: Empty space

Example of a Tic-Tac-Toe board during a game:

  X  |     | O  
-----+-----+-----
     |  X  |     
-----+-----+-----
  O  |     | X  

Board State:
------------
 pos00 = 1    # X (Player 1)
 pos01 = 0    # Empty
 pos02 = -1   # O (Player -1)

 pos10 = 0    # Empty
 pos11 = 1    # X (Player 1)
 pos12 = 0    # Empty

 pos20 = -1   # O (Player -1)
 pos21 = 0    # Empty
 pos22 = 1    # X (Player 1)

"""

def simulate_game():
    board = [0] * 9 
    print(f"Initial empty board: {board}")
    player = random.choice([1, -1])  # Randomly choose who starts (X or O)
    print(f"Player {player} starts the game (1 for X, -1 for O)")
    states = []
    result = None
 
    initial_board_state = {
                'pos00': board[0], 'pos01': board[1], 'pos02': board[2],
                'pos10': board[3], 'pos11': board[4], 'pos12': board[5],
                'pos20': board[6], 'pos21': board[7], 'pos22': board[8]
            }
    # Record the initial empty board as the first state, player is None and result is None
    states.append((initial_board_state, 0, None))
    print(f"Initial board state recorded: {board}.")
    
    while result is None:
            print(f"\nPlayer {player}'s turn")
            empty_positions = get_empty_positions(board)
            
            if not empty_positions:
                print("No more moves possible.")
                result = 0  # Force a draw if no moves left
                break
            
            move = random.choice(empty_positions) # Choose randomly a position of the still empty positions
            print(f"Player {player} chose position {move}")
            board[move] = player
            print(f"Updated board: {board}")
            
            result = check_winner(board)
            
            state_dict = {
                'pos00': board[0], 'pos01': board[1], 'pos02': board[2],
                'pos10': board[3], 'pos11': board[4], 'pos12': board[5],
                'pos20': board[6], 'pos21': board[7], 'pos22': board[8]
            }
            states.append((state_dict, player, result))
            print(f"Recorded state: {state_dict}, player: {player}, result: {result}")
            
            if result is None:  # Only switch players if the game is not finished
                player *= -1
                print(f"Switching to Player {player}")
        
    return states, result

# Get the empty positions that are still available on the board
def get_empty_positions(board):
    empty_positions = []
    for position_index in range(9):
            if board[position_index] == 0:
                empty_positions.append(position_index)
    print(f"Empty positions to choose from: {empty_positions}")
    return empty_positions

# Check for win, lose, or draw
def check_winner(board):
    winning_combinations = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
    for (i, j, k) in winning_combinations:
        if board[i] == board[j] == board[k] != 0:
            print(f"Player {board[i]} wins with combination {i, j, k}")
            return board[i]  # return 1 for X win, -1 for O win
    if 0 not in board:
        print("It's a draw!")  
        return 0  
    return None  # Game ongoing

def main():
    print("Hello from Tic-Tac-Toe data!")
    total_games=10000 # amount of games to simulate
    games = []
    # Just tracking wins/draws to get a feel for the numbers
    player_1_wins = 0
    player_minus_1_wins = 0
    draws = 0
    
    # Simulate games and count results
    for _ in range(total_games):
        states, result = simulate_game()
        print(f"\nGame finished with result: {result}")
        
        # Count the result of each game
        if result == 1:
            player_1_wins += 1
        elif result == -1:
            player_minus_1_wins += 1
        else:
            draws += 1
        
        for step, (state, player, result) in enumerate(states):
            games.append({
                **state,
                'player': player,
                'step': step,
                'result': result  # None during the game, 1 for X win, -1 for O win, 0 for draw
            })
            print(f"Step {step}, state: {state}, player: {player}, result: {result}")
    
    print("\nSummary of Results:")
    print(f"games played: {total_games}")
    print(f"Player 1 (X) wins: {player_1_wins}")
    print(f"Player -1 (O) wins: {player_minus_1_wins}")
    print(f"Draws: {draws}")

    df = pd.DataFrame(games)
    df['result'] = df['result'].astype('Int64')  # Convert result column to nullable integer type

    # Just for prototyping: Get the current timestamp for the file name
    # timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # csv_filename = f'tic_tac_toe_games_{timestamp}.csv'
    csv_filename = "tic_tac_toe_games.csv"
    
    df.to_csv(csv_filename, index=False)
    print(f"\nData saved to {csv_filename}")

if __name__ == "__main__":
    main()