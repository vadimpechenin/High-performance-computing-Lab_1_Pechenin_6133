import cupy as cp
import ga_cupy
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
Программа с cupy. Подсчет времени на выполнение
"""
# 1. Формирование входных данных.
equation_inputs=cp.arange(-10,10,20/1000)
noise = cp.random.uniform(equation_inputs.min()/5, equation_inputs.max()/5, (equation_inputs.shape[0], 1)).T
equation_outputs=-equation_inputs**2+1+noise
#ga_numpy.plot_res(equation_inputs,equation_outputs,equation_outputs)
# Количество коэффициентов, которые мы хотим оптимизировать.
num_weights = 5

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 2000
num_parents_mating = 500

# Определение численности поколения.
pop_size = (sol_per_pop,num_weights) # У населения будет хромосома sol_per_pop, где каждая хромосома имеет num_weights генов.
#Создание начальной популяции. #Изменить
const = 1
new_population = cp.random.uniform(low=-const, high=const, size=pop_size)
print(new_population)

num_generations = 20
#Основной цикл генетического алгоритма
tic = time.time()
for generation in range(num_generations):
    print("Generation : ", generation)
    # Измерение приспособленности каждой хромосомы в популяции.
    fitness = ga_cupy.cal_pop_fitness(equation_inputs, equation_outputs, new_population)

    # Выбор лучших родителей в популяции для вязки.
    parents = ga_cupy.select_mating_pool(new_population, fitness,
                                      num_parents_mating)

    # Создание нового поколения с использованием кроссовера.
    offspring_crossover = ga_cupy.crossover(parents,
                                       offspring_size=(pop_size[0]-parents.shape[0], num_weights))

    # Добавление некоторых вариаций потомству с помощью мутации.
    offspring_mutation = ga_cupy.mutation(offspring_crossover,const)

    # Создание новой популяции на основе родителей и потомства.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

    # Лучший результат в текущей итерации.#Изменить
    # Матрица для умножения
    matrix_for_product = cp.ones((new_population.shape[1], equation_inputs.shape[0]))
    matrix_for_product[1, :] = copy.deepcopy(equation_inputs)
    matrix_for_product[2, :] = equation_inputs ** 2
    if (matrix_for_product.shape[0]>3):
        matrix_for_product[3, :] = equation_inputs ** 3
        if (matrix_for_product.shape[0] > 4):
            matrix_for_product[4, :] = equation_inputs ** 4
    massiv_of_y = cp.dot(new_population, matrix_for_product)
    # Разность
    dist = equation_outputs - massiv_of_y
    # Функция пригодности вычисляет сумму продуктов между каждым входом и соответствующим ему весом.
    fitness_print = cp.abs(dist).max(axis=1)
    print("Лучший результат : ", cp.min(fitness_print))

# Получение лучшего решения после итераций, завершающих все поколения.
#Сначала рассчитывается пригодность для каждого решения в последнем поколении.
fitness = ga_cupy.cal_pop_fitness(equation_inputs, equation_outputs, new_population)
# Затем верните индекс этого решения, соответствующего наилучшей пригодности.
best_match_idx = cp.where(fitness == cp.min(fitness))
toc = time.time()
print('Затрачиваемое время cupy: %f с.' % (toc - tic))
# Матрица для умножения
matrix_for_product = cp.ones((new_population.shape[1], equation_inputs.shape[0]))
matrix_for_product[1, :] = copy.deepcopy(equation_inputs)
matrix_for_product[2, :] = equation_inputs ** 2
if (matrix_for_product.shape[0] > 3):
    matrix_for_product[3, :] = equation_inputs ** 3
    if (matrix_for_product.shape[0] > 4):
        matrix_for_product[4, :] = equation_inputs ** 4
coefs_final = new_population[best_match_idx[0], :].reshape(1,new_population.shape[1])
massiv_of_y1 = cp.dot(coefs_final, matrix_for_product).reshape(equation_inputs.shape[0],)
equation_inputs_n=cp.asnumpy(equation_inputs)
massiv_of_y1_n=cp.asnumpy(massiv_of_y1)
equation_outputs_n=cp.asnumpy(massiv_of_y1)
ga_cupy.plot_res(equation_inputs_n,massiv_of_y1_n,equation_outputs_n)

print("Лучший результат : ", new_population[best_match_idx[0], :])
print("Лучшее значения приспособленности : ", fitness[best_match_idx[0]])
