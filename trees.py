import numpy as np
import pygame
import matplotlib.pyplot as plt
from numba import jit, prange
import time
import os

# Константы
CELL_SIZE = 50
TEXT_AREA_HEIGHT = 150
TREE = 1
FIRE = 2
EMPTY = 0

# Инициализация pygame
pygame.init()

# Загрузка изображений с фоллбэком
try:
    TREE_IMAGE = pygame.transform.scale(pygame.image.load('tree.png'), (CELL_SIZE, CELL_SIZE))
except FileNotFoundError:
    TREE_IMAGE = pygame.Surface((CELL_SIZE, CELL_SIZE))
    TREE_IMAGE.fill((34, 139, 34))  # Зеленый цвет

try:
    FIRE_IMAGE = pygame.transform.scale(pygame.image.load('fire.png'), (CELL_SIZE, CELL_SIZE))
except FileNotFoundError:
    FIRE_IMAGE = pygame.Surface((CELL_SIZE, CELL_SIZE))
    FIRE_IMAGE.fill((255, 0, 0))  # Красный цвет

EMPTY_COLOR = (255, 255, 255)

@jit(nopython=True)
def create_forest(n):
    return np.zeros((n, n), dtype=np.int8)

@jit(nopython=True)
def density_trees(forest):
    return np.mean(forest == TREE)

@jit(nopython=True, parallel=True)
def update_forest_parallel(forest, epsilon, theta):
    n = forest.shape[0]
    new_forest = forest.copy()
    
    # Генерация новых деревьев
    for i in prange(n):
        for j in prange(n):
            if forest[i, j] == EMPTY and np.random.rand() < epsilon:
                new_forest[i, j] = TREE
                
    # Удары молний
    for i in prange(n):
        for j in prange(n):
            if new_forest[i, j] == TREE and np.random.rand() < theta:
                new_forest[i, j] = FIRE
                
    # Распространение огня
    fire_spread = np.zeros((n, n), dtype=np.int8)
    for i in prange(n):
        for j in prange(n):
            if new_forest[i, j] == FIRE:
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        fire_spread[ni, nj] = 1
                        
    for i in prange(n):
        for j in prange(n):
            if fire_spread[i, j] and new_forest[i, j] == TREE:
                new_forest[i, j] = FIRE
                
    # Гашение огня
    for i in prange(n):
        for j in prange(n):
            if forest[i, j] == FIRE:
                new_forest[i, j] = EMPTY
                
    return new_forest

@jit(nopython=True, parallel=False)
def update_forest_serial(forest, epsilon, theta):
    n = forest.shape[0]
    new_forest = forest.copy()
    
    # Генерация новых деревьев (последовательные циклы)
    for i in range(n):
        for j in range(n):
            if forest[i, j] == EMPTY and np.random.rand() < epsilon:
                new_forest[i, j] = TREE
                
    # Удары молний
    for i in range(n):
        for j in range(n):
            if new_forest[i, j] == TREE and np.random.rand() < theta:
                new_forest[i, j] = FIRE
                
    # Распространение огня
    fire_spread = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(n):
            if new_forest[i, j] == FIRE:
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n and 0 <= nj < n:
                        fire_spread[ni, nj] = 1
                        
    for i in range(n):
        for j in range(n):
            if fire_spread[i, j] and new_forest[i, j] == TREE:
                new_forest[i, j] = FIRE
                
    # Гашение огня
    for i in range(n):
        for j in range(n):
            if forest[i, j] == FIRE:
                new_forest[i, j] = EMPTY
                
    return new_forest

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
    
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        fx = self.find(x)
        fy = self.find(y)
        if fx != fy:
            self.parent[fy] = fx

def cluster_count(forest):
    n = forest.shape[0]
    uf = UnionFind(n * n)
    
    for i in range(n):
        for j in range(n):
            if forest[i, j] != TREE:
                continue
            idx = i * n + j
            if i > 0 and forest[i-1, j] == TREE:
                uf.union(idx, (i-1)*n + j)
            if j > 0 and forest[i, j-1] == TREE:
                uf.union(idx, i*n + (j-1))
    
    roots = set()
    for i in range(n):
        for j in range(n):
            if forest[i, j] == TREE:
                roots.add(uf.find(i*n + j))
    
    return len(roots)

def draw_forest(screen, forest, n):
    for i in range(n):
        for j in range(n):
            pos = (j*CELL_SIZE, i*CELL_SIZE)
            if forest[i, j] == TREE:
                screen.blit(TREE_IMAGE, pos)
            elif forest[i, j] == FIRE:
                screen.blit(FIRE_IMAGE, pos)
    pygame.draw.line(screen, (0,0,0), (0, n*CELL_SIZE), (n*CELL_SIZE, n*CELL_SIZE), 2)

def main():
    # Ввод параметров
    n = int(input("Введите размер поля n: "))
    epsilon = float(input("Введите вероятность появления дерева ε (0-1): "))
    theta = float(input("Введите вероятность удара молнии θ (0-1): "))
    
    # Инициализация
    pygame.init()
    width, height = n*CELL_SIZE, n*CELL_SIZE + TEXT_AREA_HEIGHT
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Модель лесных пожаров")
    font = pygame.font.SysFont("arial", 24)
    
    # Создание и копирование исходного состояния
    forest = create_forest(n)
    original_forest = forest.copy()
    
    # Тестирование производительности
    print("\nТестирование производительности...")
    
    # Прогрев JIT
    update_forest_parallel(original_forest.copy(), epsilon, theta)
    update_forest_serial(original_forest.copy(), epsilon, theta)
    
    # Тест параллельной версии
    start = time.perf_counter()
    test_forest = original_forest.copy()
    for _ in range(300):
        test_forest = update_forest_parallel(test_forest, epsilon, theta)
    parallel_time = time.perf_counter() - start
    
    # Тест последовательной версии
    start = time.perf_counter()
    test_forest = original_forest.copy()
    for _ in range(300):
        test_forest = update_forest_serial(test_forest, epsilon, theta)
    serial_time = time.perf_counter() - start
    
    print(f"\nРезультаты на сетке {n}x{n}:")
    print(f"- Параллельная версия: {parallel_time:.2f} сек")
    print(f"- Последовательная версия: {serial_time:.2f} сек")
    print(f"- Ускорение: x{serial_time/parallel_time:.2f}\n")

    # Основная симуляция с визуализацией
    forest = original_forest.copy()
    stats_steps = 300
    density_history, clusters_history = [], []
    total_d, total_c = 0.0, 0.0
    clock = pygame.time.Clock()
    
    # Главный цикл симуляции
    for step in range(1, stats_steps+1):
        forest = update_forest_parallel(forest, epsilon, theta)
        d = density_trees(forest)
        c = cluster_count(forest)
        
        # Обновление статистики
        total_d += d
        total_c += c
        avg_d = total_d / step
        avg_c = total_c / step
        density_history.append(avg_d)
        clusters_history.append(avg_c)
        
        # Отрисовка
        screen.fill(EMPTY_COLOR)
        draw_forest(screen, forest, n)
        
        # Статистика
        stats_text = [
            f"Шаг {step}/{stats_steps}",
            f"Текущая плотность: {d:.2f}  Средняя: {avg_d:.2f}",
            f"Текущие кластеры: {c}  Среднее: {avg_c:.2f}",
            f"Параметры: n={n}, ε={epsilon:.2f}, θ={theta:.2f}"
        ]
        
        for i, text in enumerate(stats_text):
            screen.blit(font.render(text, True, (0,0,0)), 
                       (10, n*CELL_SIZE + 10 + i*25))
        
        pygame.display.flip()
        
        # Обработка выхода
        if any(event.type == pygame.QUIT for event in pygame.event.get()):
            break
        
        clock.tick(20)
    
    pygame.quit()
    
    # Построение финальных графиков
    plt.figure(figsize=(14, 6))
    
    plt.subplot(121)
    plt.plot(density_history, color='green', linewidth=2)
    plt.title(f"Средняя плотность деревьев\nn={n}, ε={epsilon:.2f}, θ={theta:.2f}")
    plt.xlabel("Время (шаги)")
    plt.ylabel("Плотность")
    plt.grid(True)
    
    plt.subplot(122)
    plt.plot(clusters_history, color='red', linewidth=2)
    plt.title(f"Среднее количество кластеров\nn={n}, ε={epsilon:.2f}, θ={theta:.2f}")
    plt.xlabel("Время (шаги)")
    plt.ylabel("Кластеры")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()