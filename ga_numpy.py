import numpy as np
import copy
import matplotlib.pyplot as plt

def cal_pop_fitness(equation_inputs, equation_outputs,pop):
    # Вычисление значения пригодности каждого решения в текущей популяции.
    # Матрица для умножения
    matrix_for_product = np.ones((pop.shape[1], equation_inputs.shape[0]))
    matrix_for_product[1, :] = copy.deepcopy(equation_inputs)
    matrix_for_product[2, :] = equation_inputs ** 2
    if (matrix_for_product.shape[0]>3):
        matrix_for_product[3, :] = equation_inputs ** 3
        if (matrix_for_product.shape[0] > 4):
            matrix_for_product[4, :] = equation_inputs ** 4
    massiv_of_y = np.dot(pop, matrix_for_product)
    # Разность
    dist = equation_outputs - massiv_of_y
    # Функция пригодности вычисляет сумму продуктов между каждым входом и соответствующим ему весом.
    fitness = np.abs(dist).max(axis=1)
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Выбор лучших особей текущего поколения в качестве родителей для производства потомства следующего поколения.
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        min_fitness_idx = np.where(fitness == np.min(fitness))
        min_fitness_idx = min_fitness_idx[0][0]
        parents[parent_num, :] = pop[min_fitness_idx, :]
        fitness[min_fitness_idx] = 99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # Точка, в которой происходит кроссовер между двумя родителями. Обычно это в центре.
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Индекс первого скрещенного родителя.
        parent1_idx = k%parents.shape[0]
        # Индекс второго родителя для скрещивания.
        parent2_idx = (k+1)%parents.shape[0]
        # У нового потомства первая половина генов будет заимствована у первого родителя.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # У нового потомства вторая половина генов будет взята от второго родителя.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover,const):
    # Мутация случайным образом меняет один ген в каждом потомстве.
    #for idx in range(offspring_crossover.shape[0]):
    #Случайное значение, добавляемое к гену.
    random_value = np.random.uniform(-const/4, const/4, offspring_crossover.shape[0]).T
    offspring_crossover[:, offspring_crossover.shape[1]-1] = offspring_crossover[:, offspring_crossover.shape[1]-1] + random_value
    return offspring_crossover

def plot_res(massivs_of_x,massivs_of_y,massivs_of_y_points):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(massivs_of_x.min(), massivs_of_x.max())
    ax.set_ylim(massivs_of_y.min(), massivs_of_y.max())
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')
    ax.set_title(u'Данные из файла')
    #рисование графика зависимости первого столбца от
    #нулевого
    ax.plot(massivs_of_x, massivs_of_y, color='grey', label=r'$d_1$')
    #рисование графика зависимости второго столбца от
    #нулевого
    ax.scatter(massivs_of_x, massivs_of_y_points, label=r'$d_2$')
    ax.legend(loc='best')
    fig.show()
    fig.savefig('Результат генетического алгоритма с CPU.jpeg', dpi=300)