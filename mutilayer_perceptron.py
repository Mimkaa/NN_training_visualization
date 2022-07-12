import pygame as pg
import math
from os import path
import random
from settings import *
import math
from mpmath import *

vec = pg.Vector2

my_font = path.join("PixelatedRegular-aLKm.ttf")


def draw_text(surf, text, font_name, size, color, x, y, align="nw"):
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
    surf.blit(text_surface, text_rect)
    return text_rect


class Neuron:
    def __init__(self, pos):
        self.value = 0
        self.pos = vec(pos)
        self.error = 0
        self.bias = random.uniform(-1, 1)


class Connection:
    def __init__(self, neuron0, neuron1):
        self.neuron0 = neuron0
        self.neuron1 = neuron1
        self.weight = random.uniform(-1, 1)


class Dense_layer:
    def __init__(self, pos, neuron_num):
        self.pos = vec(pos)
        self.neuron_num = neuron_num
        self.neurons = []
        self.connections = []
        self.vert_dist = HEIGHT / (self.neuron_num + 1)
        for i in range(neuron_num):
            i += 1
            self.neurons.append(Neuron(self.pos + vec(0, self.vert_dist * i)))
        self.radius = self.vert_dist // 4
        # self.bias = Neuron(self.pos + vec(0, self.vert_dist / 2))
        # self.bias.value = 1

    def draw(self, surf):
        # pg.draw.circle(surf, YELLOW, self.bias.pos, self.radius)
        # draw_text(surf, f'{self.bias.value:.2f}', my_font, int(self.radius * 2), RED, self.bias.pos.x, self.bias.pos.y, align="center")
        # for c in self.connections:
        #     draw_text(surf, f'{c.weight:.2f}', my_font, int(self.radius), RED, c.neuron0.pos.x + 50, c.neuron0.pos.y + 50, align="center")
        for i, n in enumerate(self.neurons):
            i += 1
            pg.draw.circle(surf, WHITE, n.pos, self.radius)
            draw_text(surf, f'{n.value:.2f}', my_font, int(self.radius * 2), RED, n.pos.x, n.pos.y, align="center")

    def apply_softmax_activation(self, func, additional):
        for n in self.neurons:
            n.value = func(n.value, additional)

    def apply_activation(self, func):
        for n in self.neurons:
            n.value = func(n.value)

    def set_values(self, arr):
        for i, n in enumerate(self.neurons):
            n.value = arr[i]

    def add_bias(self):
        for n in self.neurons:
            n.value += n.bias

    def nullify(self):
        for n in self.neurons:
            #n.value = 0
            n.error = 0


class Neural_Network:
    def __init__(self, pos, map_layers):
        self.pos = vec(pos)
        self.layers = []
        # create layers
        horisontal_dist = (WIDTH - self.pos.x) / (len(map_layers) + 2)
        for i, n in enumerate(map_layers):
            i += 1
            self.layers.append(Dense_layer(self.pos + vec(horisontal_dist * i, 0), n))
        # make connections
        for i in range(len(self.layers) - 1):
            for n in self.layers[i].neurons:
                for n1 in self.layers[i + 1].neurons:
                    self.layers[i].connections.append(Connection(n, n1))

        # reset the radius
        radiuses = [l.radius for l in self.layers]
        for l in self.layers:
            l.radius = min(radiuses)

        # for line drawing
        weights_values = [c.weight for l in self.layers for c in l.connections]
        self.max_weight = max(weights_values)
        self.min_weight = min(weights_values)
        positions_for_lines_draw_surf = [n.pos for l in self.layers for n in l.neurons]
        min_x, max_x, min_y, max_y = self.get_min_max_coords(positions_for_lines_draw_surf)
        self.lines_surf_pos = vec(min_x, min_y)
        self.lines_surf = pg.Surface((self.get_surf_size(positions_for_lines_draw_surf)), pg.SRCALPHA)

        # to toggle
        self.forward = True
        self.error = True
        self.back_prop = True

        # answers
        # self.ans_layer = Answer_Layer(answers, self.layers[-1].pos + vec(horisontal_dist, 0))

    def get_min_max_coords(self, points):
        y_points = [p.y for p in points]
        x_points = [p.x for p in points]
        min_x = min(x_points)
        max_x = max(x_points)
        min_y = min(y_points)
        max_y = max(y_points)
        return min_x, max_x, min_y, max_y

    def get_surf_size(self, points):
        min_x, max_x, min_y, max_y = self.get_min_max_coords(points)
        width = int(max_x - min_x + 2)
        height = int(max_y - min_y + 2)
        return width, height

    def sigmoid(self, x):

        return 1 / (1 + math.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


    def train(self, input_arr, answers):

        self.nullify_neurons()

        if self.forward:
            self.forward_prop(input_arr)

        if self.error:
            self.distribute_error(answers)

        # backpropagation
        if self.back_prop:
            self.backpropagation()

        weights_values = [c.weight for l in self.layers for c in l.connections]
        self.max_weight = max(weights_values)
        self.min_weight = min(weights_values)

    def nullify_neurons(self):
        for l in self.layers:
            l.nullify()

    def forward_prop(self, input_arr):

        self.layers[0].set_values(input_arr)

        for n, l in enumerate(self.layers):
            if n > 0:
                l.add_bias()
                l.apply_activation(self.sigmoid)
            for c in l.connections:
                # print(c.neuron0.value,c.weight ,  c.neuron0.value * c.weight)
                c.neuron1.value += c.neuron0.value * c.weight

    def distribute_error(self, answers):
        errors = [answers[i] - self.layers[-1].neurons[i].value for i in range(len(answers))]
        # setting the errors for the last layer
        for i, n in enumerate(self.layers[-1].neurons):
            n.error = errors[i]

        for l in self.layers[::-1]:
            for c in l.connections:
                c.neuron0.error += c.neuron1.error * c.weight

    def backpropagation(self):
        for l in self.layers[::-1]:
            for c in l.connections:
                c.weight += c.neuron1.error * LEARNING_RATE * self.sigmoid_prime(c.neuron1.value) * c.neuron0.value
                c.neuron1.bias += c.neuron1.error * LEARNING_RATE * self.sigmoid_prime(c.neuron1.value)

    def predict(self, input_arr):
        self.forward_prop(input_arr)
        return [n.value for n in self.layers[-1].neurons]

    def draw(self, surf):
        # draw connections
        self.lines_surf.fill(BGCOLOR)
        self.lines_surf.set_colorkey(BGCOLOR)
        for l in self.layers:
            for c in l.connections:
                if c.weight < 0:
                    pg.draw.line(self.lines_surf, (255, 0, 0, int((c.weight / self.min_weight) * 255)),
                                 c.neuron0.pos - self.lines_surf_pos, c.neuron1.pos - self.lines_surf_pos)
                else:
                    pg.draw.line(self.lines_surf, (0, 255, 0, int((c.weight / self.max_weight) * 255)),
                                 c.neuron0.pos - self.lines_surf_pos, c.neuron1.pos - self.lines_surf_pos)

        surf.blit(self.lines_surf, (self.lines_surf_pos))

        # draw layers
        for l in self.layers:
            l.draw(surf)
