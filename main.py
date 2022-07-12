import random

import pygame as pg
import sys
from settings import *

from os import path
from mutilayer_perceptron import *
class Game:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode((WIDTH, HEIGHT))
        pg.display.set_caption(TITLE)
        self.clock = pg.time.Clock()
        pg.key.set_repeat(500, 100)
        self.load_data()

    def load_data(self):
        self.font=path.join("PixelatedRegular-aLKm.ttf")
    def draw_text(self, text, font_name, size, color, x, y, align="nw"):
        font = pg.font.Font(font_name, size)
        text_surface = font.render(text, True, color)
        text_rect = text_surface.get_rect()
        if align == "nw":
            text_rect.topleft = (x, y)
        if align == "ne":
            text_rect.topright = (x, y)
        if align == "sw":
            text_rect.bottomleft = (x, y)
        if align == "se":
            text_rect.bottomright = (x, y)
        if align == "n":
            text_rect.midtop = (x, y)
        if align == "s":
            text_rect.midbottom = (x, y)
        if align == "e":
            text_rect.midright = (x, y)
        if align == "w":
            text_rect.midleft = (x, y)
        if align == "center":
            text_rect.center = (x, y)
        self.screen.blit(text_surface, text_rect)
        return text_rect

    def new(self):
        # initialize all variables and do all the setup for a new game
        self.all_sprites = pg.sprite.Group()
        self.nn = Neural_Network((WIDTH//2, 0), (2, 4, 4, 1))
        self.cells = [[(x, y) for x in range(WIDTH//2//CELL_SIZE)] for y in range(HEIGHT//CELL_SIZE)]
        self.cells_1d = [j for i in self.cells for j in i]
        self.training_points = []
        self.start_training = False



    def run(self):
        # game loop - set self.playing = False to end the game
        self.playing = True
        while self.playing:
            self.dt = self.clock.tick(FPS) / 1000
            self.events()
            self.update()
            self.draw()

    def quit(self):
        pg.quit()
        sys.exit()

    def update(self):
        # update portion of the game loop
        if self.start_training:
            for i in range(3000):


                choice = random.choice(self.training_points)
                self.nn.train(choice[0], choice[1])


    def draw_grid(self):
        for x in range(0, WIDTH, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, TILESIZE):
            pg.draw.line(self.screen, LIGHTGREY, (0, y), (WIDTH, y))

    def draw(self):


        self.screen.fill(BGCOLOR)
        self.nn.draw(self.screen)
        for n, c in enumerate(self.cells_1d):
            rect = pg.Rect(c[0] * CELL_SIZE, c[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            prediction = self.nn.predict([c[0]/(WIDTH/2/CELL_SIZE), c[1]/(HEIGHT/CELL_SIZE)])[0]
            pg.draw.rect(self.screen, (int(255 * prediction), int(255  * prediction), int(255  * prediction)), rect)

        for p in self.training_points:
            pg.draw.circle(self.screen, BLACK if p[1] == [0] else WHITE, (p[0][0] * CELL_SIZE * (WIDTH/2/CELL_SIZE),p[0][1] * CELL_SIZE * (HEIGHT/CELL_SIZE)), 10)
        # fps
        self.draw_text(str(int(self.clock.get_fps())), self.font, 40, WHITE, 50, 50, align="center")
        pg.display.flip()

    def events(self):
        # catch all events here
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            if event.type == pg.MOUSEBUTTONDOWN :
                if event.button == 1:
                    mx,  my = pg.mouse.get_pos()
                    self.training_points.append((((mx/CELL_SIZE)/(WIDTH/2/CELL_SIZE), (my/CELL_SIZE)/(HEIGHT/CELL_SIZE)), [1]))
                elif event.button == 3:
                    mx,  my = pg.mouse.get_pos()
                    self.training_points.append((((mx/CELL_SIZE)/(WIDTH/2/CELL_SIZE), (my/CELL_SIZE)/(HEIGHT/CELL_SIZE)), [0]))
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.quit()
                if event.key == pg.K_SPACE:
                    self.start_training = not self.start_training




# create the game object
g = Game()
g.new()
g.run()
