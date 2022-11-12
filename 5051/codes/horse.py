def move_available(position, move):
    x = position[0] + move[0]
    y = position[1] + move[1]
    return 0 <= x < size[0] and 0 <= y < size[1] and board[x][y] == -1

def get_next_positions(position):
    moves = [[1,2], [2,1], [1,-2], [2,-1], [-1,2], [-2,1], [-1,-2], [-2,-1]]
    return [[position[0] + move[0], position[1] + move[1]] for move in moves if move_available(position, move)]

def get_next_positions_sorted_by_least_next_moves(position):
    return sorted(get_next_positions(position), key=lambda next_position: len(get_next_positions(next_position)))

def print_chess_board_state():
    print("\nFound solution: \n")
    for x in range(size[0]):
        print(*board[x],'\n',sep='\t')

def horse_move(position, step=0):
    board[position[0]][position[1]] = step
    if step == size[0] * size[1] - 1:
        print_chess_board_state()
        quit()
    for next_position in get_next_positions_sorted_by_least_next_moves(position):
        horse_move(next_position, step + 1)
    board[position[0]][position[1]] = -1
    


size = [8,8]
board = [[-1 for i in range(size[0])] for j in range(size[1])]
horse_move([0,0])

