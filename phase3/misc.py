from more_itertools import sort_together
import numpy as np
from phase3.check_constraints import feasibility
import random
import copy


def vector_unitario(vector):
    '''
    Recibe un vector y lo retorna normalizado.
    '''

    return vector / np.linalg.norm(vector)


def angulo_eje(vector):
    '''
    Calcula el ángulo que existe entre un vector y el eje i tongo
    '''

    eje_x = (1, 0)
    v_u = vector_unitario(vector)

    if v_u[1] >= eje_x[1]:
        return (np.arccos(np.clip(np.dot(eje_x, v_u), -1.0, 1.0))*180)/np.pi
    else:
        return ((2*np.pi - np.arccos(np.clip(np.dot(eje_x, v_u), -1.0, 1.0)))*180)/np.pi


def angulo_puntos(v1, v2):
    '''
    Calcula el ángulo que existe entre dos vectores v1 y v2
    '''

    return angulo_eje((v2[0]-v1[0], v2[1]-v1[1]))


def cycle():
    nodos = list()
    for it in range(6):
        i = random.randint(0, 10)
        j = random.randint(0, 10)
        nodos.append((i, j))

    pares = list()
    for nodo in nodos:
        pares.append((nodo, angulo_puntos((5, 5), nodo)))
    pares.sort(key=lambda x: x[1])
    pares


def create_population(data, initial_population_number):
    ''' 
    This function encapsulates the initial population creation, which basically
    iterates initial_population_number times, creating a new individual
    in each iteration and appending it to the list of the initial population
    until its finished.

    Parameters
    ----------
    initial_population_number: int
        Number of individuals to be genereted for the initial population.

    Returns
    -------
    population: list
        Its a list of the individuals included in the first population of
        solutions to be used initially by the algorithm.
    '''

    population = []

    for i in range(initial_population_number):
        ind = create_individual(data)
        population.append(ind)
        ind = create_individual_rd(data)
        population.append(ind)

    return population


def create_individual(data):
    '''
    There will enter the "glouton" function the teacher was talking about
    I guess, we have to search for a good way to create initial solutions.

    Parameters
    ----------
    tasks: list
        A list that contains all the tasks of the problem to be solved.
    workers: list
        A list that contains all the workers of the problem to be solved.

    Returns
    -------
    individual: dict
        Returns a dictionary representing an individual in the point of
        view of the algorithm (workers are keys, and its values are the
        lists containing the tasks that they make in a given solution)
    '''

    individual = {w: [] for w in data.Workers}
    tasks = [t for t in data.Tasks]
    work = copy.deepcopy(data.Workers)
    random.shuffle(work)

    for w in work:
        individual[w] = []

        while feasibility(individual, data):
            p = True

            tri = sorted(data.t[data.Houses[w]].keys(),
                         key=lambda t: data.t[data.Houses[w]][t])
            i = 0
            while tri[i] not in tasks or tri[i] not in data.TasksW[w]:
                i += 1
                if i == len(tri):
                    break
            else:
                individual[w].append(tri[i])
                tasks.remove(tri[i])
                p = False

            if p:
                break

        else:
            individual[w].remove(tri[i])
            tasks.append(tri[i])

    return individual


def create_individual_rd(data):
    '''
    There will enter the "glouton" function the teacher was talking about
    I guess, we have to search for a good way to create initial solutions.

    Parameters
    ----------
    tasks: list
        A list that contains all the tasks of the problem to be solved.
    workers: list
        A list that contains all the workers of the problem to be solved.

    Returns
    -------
    individual: dict
        Returns a dictionary representing an individual in the point of
        view of the algorithm (workers are keys, and its values are the
        lists containing the tasks that they make in a given solution)
    '''

    individual = {w: [] for w in data.Workers}
    tasks = [t for t in data.Tasks]
    work = copy.deepcopy(data.Workers)
    random.shuffle(work)

    for w in work:
        individual[w] = []

        while feasibility(individual, data):
            p = True

            tri = sorted(data.t[data.Houses[w]].keys(),
                         key=lambda t: data.t[data.Houses[w]][t])

            i = random.choices(range(len(tri)), weights=[
                               1/(1 + data.t[data.Houses[w]][t]) for t in data.t[data.Houses[w]].keys()])[0]

            while tri[i] not in tasks or tri[i] not in data.TasksW[w]:
                i += 1
                if i == len(tri):
                    break
            else:
                individual[w].append(tri[i])
                tasks.remove(tri[i])
                p = False

            if p:
                break

        else:
            individual[w].remove(tri[i])
            tasks.append(tri[i])

    return individual


def individuals_copy(individuals, number_of_copies):
    '''
    This function receives a list of individuals and creates a list
    of the same individuals, repeated sequentially number_of_copies
    times.

    Parameters
    ----------
    individuals: list
        List of individuals in which every element is a dictionary
        containing an individual, which is a feasible solution to 
        the optimization problem.
    number_of_copies: int
        Number of copies to be made for each individual.
    '''
    copy_of_individuals = [copy.deepcopy(individuals[i]) for i in range(
        number_of_copies) for m in range(number_of_copies)]

    return copy_of_individuals


def select_best_group(generation, n, score_task, score_travel):
    '''
    This function picks the n best solutions from a generation.
    In other words, it calculates the cost value of every individual
    in a group of individuals (we can call it a generation or a group
    of solutions) and picks the ones that have the n best cost value.

    Parameters
    ----------
    generation: list
        List of individuals (dictionaries).
    n: int
        Integer that tells how many solutions you want to pick from
        the generation

    Returns
    -------
    best_group: list
        List containing the n individuals that have the best cost
        values.
    '''

    # generation_cost = [calculate_cost(generation[i])
    #                    for i in range(len(generation))]
    sorted_temp = sort_together(
        [generation, score_task, score_travel], key_list=[1, 2], reverse=True)

    best_group = list(sorted_temp[0])[:n]
    best_scores_task = list(sorted_temp[1])[:n]
    best_scores_travel = list(sorted_temp[2])[:n]

    if len(sorted_temp) > 2*n:
        idx = random.choices(range(n, len(sorted_temp)), n)
        for i in idx:
            best_group.append(sorted_temp[0][i])
            best_scores_task.append(sorted_temp[1][i])
            best_scores_travel.append(sorted_temp[2][i])

    return best_group, best_scores_task, best_scores_travel


def find_best_solution(generation):
    '''
    This is a special case of selecting best group for the case where
    n = 1. It means that this function picks the solution that have the
    best cost value from a generation (a group of individuals).

    Parameters
    ----------
    generation: list
        List of individuals (dictionaries).

    Returns
    -------
    dict
        It returns the individual with the best cost value.
    '''

    return select_best_group(generation, 1)[0]


def calculate_cost(data, individual):
    '''
    This function calculates the cost value for a given individual.
    This means that it should calculate the total task duration of each
    worker and subtract the total travel time from it.
    As a first approach, it does not take into account the unavailabilities.

    Parameters
    ----------
    individual: dict
        Dictionary representing a feasible solution.

    Returns
    -------
    cost: int
        Cost value of the individual
    '''

    total_task_duration = 0
    total_travel_time = 0

    # iterate over the list of tasks of the workers
    for task_list in individual.values():
        for i in range(len(task_list)):
            task_duration = data.d(task_list[i])
            total_task_duration += task_duration
            if i != 0:
                travel_time = data.t({task_list[i], task_list[i-1]})
                total_travel_time += travel_time

        if len(task_list) > 0:

            # retrieve the worker name
            worker = task_list.index(task_list)

            # taking into account the home to first task and last task to
            # home travel time
            home_to_task_travel_time = data.t(
                {data.Houses(worker), task_list[0]})
            total_travel_time += home_to_task_travel_time
            task_to_home_travel_time = data.t(
                {task_list[-1], data.Houses(worker)})
            total_travel_time += task_to_home_travel_time

    cost = total_task_duration - total_travel_time

    return cost
