import pygame as pg
import numpy as np
from tetris2d.shapes2d import *

class ScorePos:
    def __init__(self, pos=None, score=0):
        self.pos = pos
        self.score = score

    def add_score(self, amount):
        self.score += amount


def game_loop():
    BOARD_SHAPE = (10, 22 + 4) # "staging" area is 4
    GRIDSIZE = 35
    SCREENSIZE = (GRIDSIZE * BOARD_SHAPE[0] + 200, GRIDSIZE * (BOARD_SHAPE[1] - 4))
    DROPTIME = 75
    MULTIPLIER = 50

    board = np.zeros(BOARD_SHAPE)

    BLUE = (0, 255, 0)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    loop = True
    screen = pg.display.set_mode(SCREENSIZE)

    clock = pg.time.Clock()
    DROPTIMER = pg.USEREVENT
    pg.time.set_timer(DROPTIMER, DROPTIME)

    # None -> Void: Draws the board
    def draw_board():
        screen.fill(WHITE)
        for x in range(BOARD_SHAPE[0]):
            for y in range(BOARD_SHAPE[1] - 2):
                if board[x, y]:
                    pg.draw.rect(screen, BLACK, (SCREENSIZE[0] - GRIDSIZE * (x + 1),
                                                 SCREENSIZE[1] - GRIDSIZE * (y + 1),
                                                 GRIDSIZE, GRIDSIZE))

    # ScorePos -> ScorePos: Checks if we filled any rows and updates score/board correctly. Returns updated score pos
    # Information
    def check_board(current_position):
        conseq_clear = 0
        coords = current_position.pos
        for y in [loc[1] for loc in coords]:
            if all(board[:, y]):
                conseq_clear = conseq_clear + 1
                board[:, y] = False
                for cur in range(y, BOARD_SHAPE[1] - 2):
                    board[:, cur] = board[:, cur+1]
        output = ScorePos()
        output.add_score(current_position.score + MULTIPLIER * conseq_clear ** 2)
        return output

    # ScorePos -> ScorePos: Lowers the brick one space when possible, and cements it to the board when it hits black
    # Invokes check_board to handle what happens when we complete line(s)
    def drop_piece(current_position):
        coords = current_position.pos
        for loc in coords:
            if loc[1] == 0 or board[loc[0], loc[1] - 1]:
                for loc2 in coords:
                    board[loc2[0], loc2[1]] = True
                if loc[1] >= 22: # Game Over condition
                    return ScorePos(score=np.NaN)
                return check_board(current_position)
        else:
            return ScorePos([[loc[0], loc[1] - 1] for loc in coords], current_position.score)

    # ScorePos-> Void: draws the piece onto the board
    def draw_piece(current_position):
        for loc in current_position.pos:
            pg.draw.rect(screen, BLUE, (SCREENSIZE[0] - GRIDSIZE * (loc[0] + 1),
                                        SCREENSIZE[1] - GRIDSIZE * (loc[1] + 1),
                                        GRIDSIZE, GRIDSIZE))

    def draw_score(current_position):
        pg.init()
        score = str(current_position.score)
        text = pg.font.SysFont("Fixedsys Excelsior 3.01", 40).render(score, True, BLACK)
        rect = text.get_rect()
        rect.center = (100, 100)
        screen.blit(text, rect)

    # movement and rotation commands: All have signature ScorePos -> ScorePos and are intuitive enough
    def move_left(current_position):
        coords = current_position.pos
        for loc in coords:
            if loc[0] == 0 or board[loc[0] - 1, loc[1]]:
                return current_position
        else:
            return ScorePos([[loc[0] - 1, loc[1]] for loc in coords], current_position.score)

    def move_right(current_position):
        coords = current_position.pos
        for loc in coords:
            if loc[0] == BOARD_SHAPE[0] - 1 or board[loc[0] + 1, loc[1]]:
                return current_position
        else:
            return ScorePos([[loc[0] + 1, loc[1]] for loc in coords], current_position.score)

    def rot_left(current_location):
        pass

    def rot_right(cucrrent_location):
        pass

    # Scorepos -> Scorepos: Chooses a piece and puts it into the "staging area"
    def choose_piece(current_position):
        candidate = shapes[np.random.randint(0, 6)]
        score = current_position.score
        pos = []
        for x in range(4):
            for y in range(2):
                if candidate[y][x]:
                    pos.append([3 + x, BOARD_SHAPE[1] - y - 1])
        return ScorePos(pos, score)

    current_location = ScorePos()

    # MAIN GAME LOOP #
    while loop:

        if pg.event.get(DROPTIMER):
            if current_location.pos:
                current_location.add_score(MULTIPLIER * 1/2)
                current_location = drop_piece(current_location)
            else:
                current_location = choose_piece(current_location)
        keys = pg.key.get_pressed()
        if keys[pg.K_d]:
            current_location = move_left(current_location)
        if keys[pg.K_a]:
            current_location = move_right(current_location)
        try:
            assert (type(current_location) == ScorePos)
        except AssertionError:
            print(type(current_location))
        if np.isnan(current_location.score):
            current_location = ScorePos(score=0)
            board = np.zeros(BOARD_SHAPE)
        check_board(current_location)
        draw_board()
        draw_piece(current_location)
        draw_score(current_location)
        pg.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    game_loop()