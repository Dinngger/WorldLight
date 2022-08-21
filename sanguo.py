import os
import pygame
from queue import Queue
from copy import deepcopy

N = 0
C = 1
X = 2
S = 3
H = 4
Sl = 5
Su = 6
Sul = 7

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

board = [
    [X, C, H, X],
    [X, 0, X, X],
    [S, S, H, 0],
    [H, 0, H, 0],
    [0, H, 0, 0]]

def toTuple(b):
    return tuple(tuple(x) for x in b)
def update(b):
    for i in range(5):
        for j in range(4):
            if b[i][j] not in [C, X, S, H]:
                b[i][j] = 0
    for i in range(5):
        for j in range(4):
            z = b[i][j]
            if z == C:
                b[i][j+1] = Sl
                b[i+1][j] = Su
                b[i+1][j+1] = Sul
            elif z == S:
                b[i+1][j] = Su
            elif z == H:
                b[i][j+1] = Sl
def final(b):
    return b[3][1] == C
def inBounds(x, y):
    return x >= 0 and x < 5 and y >= 0 and y < 4
def do(b, x, y, d):
    z = b[x][y]
    if d == UP:
        b[x-1][y] = z
    elif d == DOWN:
        b[x+1][y] = z
    elif d == LEFT:
        b[x][y-1] = z
    elif d == RIGHT:
        b[x][y+1] = z
    b[x][y] = 0
    update(b)
    return b

def find(init_board):
    queue = Queue()
    if not final(init_board):
        queue.put(init_board)
    history = {toTuple(init_board): 0}
    while not queue.empty():
        b = queue.get()
        for i in range(5):
            for j in range(4):
                z = b[i][j]
                if z not in [C, X, S, H]:
                    continue
                b_nexts = []
                if inBounds(i-1, j) and b[i-1][j] == N and (z in [X, S] or b[i-1][j+1] == N):
                    b_nexts.append(do(deepcopy(b), i, j, UP))
                if inBounds(i, j-1) and b[i][j-1] == N and (z in [X, H] or b[i+1][j-1] == N):
                    b_nexts.append(do(deepcopy(b), i, j, LEFT))
                if ((z in [X, H] and inBounds(i+1, j) and b[i+1][j] == N and (z == X or b[i+1][j+1] == N)) or
                    (z in [C, S] and inBounds(i+2, j) and b[i+2][j] == N and (z == S or b[i+2][j+1] == N))):
                    b_nexts.append(do(deepcopy(b), i, j, DOWN))
                if ((z in [X, S] and inBounds(i, j+1) and b[i][j+1] == N and (z == X or b[i+1][j+1] == N)) or
                    (z in [C, H] and inBounds(i, j+2) and b[i][j+2] == N and (z == H or b[i+1][j+2] == N))):
                    b_nexts.append(do(deepcopy(b), i, j, RIGHT))
                for b_next in b_nexts:
                    b_next_t = toTuple(b_next)
                    if b_next_t in history:
                        continue
                    history[b_next_t] = b
                    if final(b_next):
                        print('find!')
                        path = [b_next]
                        next_path = b
                        while next_path is not 0:
                            path.append(next_path)
                            next_path = history[toTuple(next_path)]
                        return path
                    queue.put(b_next)
    return None

update(board)
path = find(board)
if path is None:
    os._exit(0)

this_i = len(path) - 1
board = path[this_i]
pygame.init()
colors = [pygame.Color(0, 0, 0), pygame.Color(255, 0, 0),
    pygame.Color(255, 255, 0), pygame.Color(0, 0, 255), pygame.Color(0, 255, 0)]
block_size = 50
playSurface = pygame.display.set_mode(
    (4 * block_size, 5 * block_size),
    pygame.DOUBLEBUF)
fpsClock = pygame.time.Clock()
pygame.display.set_caption('三国华容道')
gameover = False
while not gameover:
    fpsClock.tick(10)
    playSurface.fill(colors[0])
    for i in range(5):
        for j in range(4):
            z = board[i][j]
            if z == Sl:
                z = board[i][j-1]
            elif z == Su:
                z = board[i-1][j]
            elif z == Sul:
                z = board[i-1][j-1]
            pygame.draw.rect(playSurface, colors[z],
                pygame.Rect(j * block_size + 1, i * block_size + 1, block_size - 1, block_size - 1))
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            gameover = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT or event.key == ord('d'):
                this_i = this_i - 1 if this_i > 0 else 0
                board = path[this_i]
            if event.key == pygame.K_LEFT or event.key == ord('a'):
                this_i = this_i + 1 if this_i < len(path) - 1 else len(path) - 1
                board = path[this_i]
            if event.key == pygame.K_ESCAPE or event.key == ord('q'):
                pygame.event.post(pygame.event.Event(pygame.QUIT))
pygame.quit()
