"""
Tic Tac Toe Player
"""

import math
from copy import deepcopy
X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
#Keep track of the number of plays    
    track = 0
    for i,row in enumerate(board):
        for j, col in enumerate(row):
            if board[i][j] != EMPTY:
                track+=1
                
    if board == initial_state():
        return X
    return O if track%2 == 1 else X
    """
    Returns player who has the next turn on a board.
    """

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    moves = set()
    for i,row in enumerate(board):
        for j, col in enumerate(row) :
            if board[i][j] == EMPTY:
                moves.add((i,j))
    return moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i,j = action
    copied_board =  deepcopy(board)
    copied_board[i][j]= player(copied_board)


    if action not in actions(board):
        raise Exception("Move Invalid, Try again.")


    
    return copied_board
    
   

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    if board[0][0] == board[1][1] == board[2][2]:
        if board[0][0] == X:
            return X
        elif board[0][0] == O:
            return O
        else:
            return None
    if board[2][0] == board[1][1] == board[0][2]:
        if board[2][0] == X:
            return X
        elif board[2][0] == O:
            return O
        else:
            return None
    for row in board:
        if row.count(X) ==3:
            return X
        if row.count(O) == 3:
            return O

    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col]:
            if board[0][col] == X:
                return X
            elif board[0][col] == O:
                return O
            else:
                return None
    return None
            
        

def terminal(board):
    if winner(board) == X or (winner(board) == O) or not actions(board):
        return True
    else:
        return None

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    value = {'X':1, 'O':-1, None:0}
    winning = winner(board)
    return value[winning]


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    
    if terminal(board):
        return None
  
    def max_value(board):
        if terminal(board):
            return utility(board)
        maxVal = float("-inf")
        for action in actions(board):
            maxVal = max(maxVal, mini_value(result(board,action)))
        return maxVal
    def mini_value(board):
        if terminal(board):
            return utility(board)
        minVal = float("inf")
        for action in actions(board):
            minVal = min(minVal, max_value(result(board,action)))
        return minVal
    if player(board) == X:
        return max(actions(board),key = lambda action: mini_value(result(board,action)))
    
    else:
        return min(actions(board),key = lambda action: max_value(result(board,action)))


        
