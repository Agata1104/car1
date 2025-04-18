import sys
import pygame
import random
import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

max_offset = 15
width, height = 600, 400
r = 10
points = []


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dist(self, A):
        return np.sqrt((self.x - A.x) ** 2 + (self.y - A.y) ** 2)

    # генерация случайных точек вокруг текущей
    def spray_points(self):
        around_points = []
        for i in range(np.random.randint(2, 7)):
            alpha = np.random.random() * 2 * np.pi
            radius = np.random.randint(10, 50)
            x = self.x + radius * np.cos(alpha)
            y = self.y + radius * np.sin(alpha)
            around_points.append(Point(x, y))
        return around_points


def colors(n):
    clrs = []
    for i in range(n):
        r = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        b = np.random.randint(0, 256)
        clrs.append((r, g, b))
    clrs.append((255, 0, 0))
    return clrs


def custom_dbscan(X, eps, min_samples):
    # находим соседей в радиусе eps для каждой точки
    nn = NearestNeighbors(radius=eps)
    nn.fit(X)
    _, neighbors = nn.radius_neighbors(X)
    # определяем core-точки (у которых достаточно соседей)
    core_mask = np.array([len(n) >= min_samples for n in neighbors])
    core_indices = np.where(core_mask)[0]
    # если нет core-точек, все точки - шум
    if len(core_indices) == 0:
        return np.full(X.shape[0], -1)
    # строим граф связей между core-точками
    core_points = X[core_mask]
    core_nn = NearestNeighbors(radius=eps)
    core_nn.fit(core_points)
    adj_matrix = core_nn.radius_neighbors_graph(core_points, mode='connectivity')
    # находим связанные компоненты (кластеры)
    _, core_labels = connected_components(adj_matrix)
    # инициализируем метки всех точек как шум (-1)
    labels = np.full(X.shape[0], -1, dtype=int)
    labels[core_mask] = core_labels # присваиваем метки core-точкам
    # присваиваем border-точкам метки их ближайших core-соседей
    for i in np.where(~core_mask)[0]:
        core_neighbors = [n for n in neighbors[i] if core_mask[n]]
        if core_neighbors:
            labels[i] = labels[core_neighbors[0]]

    return labels


def main():
    pygame.init()
    screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
    screen.fill("white")
    play = True
    pygame.display.update()
    while (play):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
                pygame.quit()

            if event.type == pygame.VIDEORESIZE:
                screen.fill("white")
                for p in points:
                    pygame.draw.circle(screen, 'black', (p.x, p.y), r)

            if event.type == pygame.MOUSEMOTION:
                if 1 in event.buttons:
                    point = Point(*event.pos)

                    if len(points) == 0 or point.dist(points[-1]) > 50:
                        points.append(point)
                        pygame.draw.circle(screen, "black", (point.x, point.y), r)
                        around_points = point.spray_points()
                        for p in around_points:
                            pygame.draw.circle(screen, 'black', (p.x, p.y), r)
                            points.append(p)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pnts = np.array([[p.x, p.y] for p in points])
                    labels = custom_dbscan(pnts, eps=50, min_samples=5)
                    clrs = colors(np.max(labels) + 1)
                    for i, p in enumerate(points):
                        pygame.draw.circle(screen, clrs[labels[i]], (p.x, p.y), r)
        pygame.display.update()


if __name__ == "__main__":
    main()
