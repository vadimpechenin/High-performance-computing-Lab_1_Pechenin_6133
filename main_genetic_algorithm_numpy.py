import numpy as np
import ga_numpy
import copy
import time
# Взято с https://github.com/ahmedfgad/GeneticAlgorithmPython
# https://www.machinelearningmastery.ru/genetic-algorithm-implementation-in-python-5ab67bb124a6/
"""
    Нужно подобрать зависимость:
    y = c1+c2*x+c3*x**2+c4*x**3+c5*x**4
    К входным данным [x и y]
    Используется генетический алгоритм.
"""

"""
Программа с numpy. Подсчет времени на выполнение
"""
# 1. Формирование входных данных.
equation_inputs=np.arange(-10,10,20/500)
noise = np.random.uniform(equation_inputs.min()/5, equation_inputs.max()/5, (equation_inputs.shape[0], 1)).T
equation_outputs=-equation_inputs**2+1+noise
#ga_numpy.plot_res(equation_inputs,equation_outputs,equation_outputs)
# Количество коэффициентов, которые мы хотим оптимизировать.
num_weights = 3

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 1000
num_parents_mating = 500

# Определение численности поколения.
pop_size = (sol_per_pop,num_weights) # У населения будет хромосома sol_per_pop, где каждая хромосома имеет num_weights генов.
#Создание начальной популяции. #Изменить
const = 1
new_population = np.random.uniform(low=-const, high=const, size=pop_size)
print(new_population)

num_generations = 1
#Основной цикл генетического алгоритма
tic = time.time()
for generation in range(num_generations):
    print("Generation : ", generation)
    # Измерение приспособленности каждой хромосомы в популяции. #Изменить
    fitness = ga_numpy.cal_pop_fitness(equation_inputs, equation_outputs, new_population)

    # Выбор лучших родителей в популяции для вязки.
    parents = ga_numpy.select_mating_pool(new_population, fitness,
                                      num_parents_mating)

    # Создание нового поколения с использованием кроссовера.
    offspring_crossover = ga_numpy.crossover(parents,
                                       offspring_size=(pop_size[0]-parents.shape[0], num_weights))

    # Добавление некоторых вариаций потомству с помощью мутации.
    offspring_mutation = ga_numpy.mutation(offspring_crossover,const)

    # Создание новой популяции на основе родителей и потомства.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

    # Лучший результат в текущей итерации.#Изменить
    # Матрица для умножения
    matrix_for_product = np.ones((new_population.shape[1], equation_inputs.shape[0]))
    matrix_for_product[1, :] = copy.deepcopy(equation_inputs)
    matrix_for_product[2, :] = equation_inputs ** 2
    if (matrix_for_product.shape[0]>3):
        matrix_for_product[3, :] = equation_inputs ** 3
        if (matrix_for_product.shape[0] > 4):
            matrix_for_product[4, :] = equation_inputs ** 4
    massiv_of_y = np.dot(new_population, matrix_for_product)
    # Разность
    dist = equation_outputs - massiv_of_y
    # Функция пригодности вычисляет сумму продуктов между каждым входом и соответствующим ему весом.
    fitness_print = np.abs(dist).max(axis=1)
    print("Лучший результат : ", np.min(fitness_print))

# Получение лучшего решения после итераций, завершающих все поколения.
#Сначала рассчитывается пригодность для каждого решения в последнем поколении.
fitness = ga_numpy.cal_pop_fitness(equation_inputs, equation_outputs, new_population)
# Затем верните индекс этого решения, соответствующего наилучшей пригодности.
best_match_idx = np.where(fitness == np.min(fitness))
toc = time.time()
print('Затрачиваемое время numpy: %f с.' % (toc - tic))
# Матрица для умножения
matrix_for_product = np.ones((new_population.shape[1], equation_inputs.shape[0]))
matrix_for_product[1, :] = copy.deepcopy(equation_inputs)
matrix_for_product[2, :] = equation_inputs ** 2
if (matrix_for_product.shape[0] > 3):
    matrix_for_product[3, :] = equation_inputs ** 3
    if (matrix_for_product.shape[0] > 4):
        matrix_for_product[4, :] = equation_inputs ** 4
coefs_final = new_population[best_match_idx, :].reshape(1,new_population.shape[1])
massiv_of_y1 = np.dot(coefs_final, matrix_for_product).reshape(equation_inputs.shape[0],)
ga_numpy.plot_res(equation_inputs,massiv_of_y1,equation_outputs)

print("Лучший результат : ", new_population[best_match_idx, :])
print("Лучшее значения приспособленности : ", fitness[best_match_idx])
