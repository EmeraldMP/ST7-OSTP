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

    workers = data.Workers
    tasks = data.Tasks

    for i in range(initial_population_number):
        population.append(create_individual(workers, tasks))

    return population


def create_individual(tasks, workers):
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
    individual = {}

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
    copy_of_individuals = [individuals[i] *
                           number_of_copies for i in range(number_of_copies)]

    return copy_of_individuals


def select_best_group(generation, n):
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

    generation_cost = [calculate_cost(generation[i])
                       for i in range(len(generation))]
    sorted_indexes = sorted(range(len(generation_cost)),
                            key=lambda i: generation_cost[i], reverse=True)[:n]
    best_group = [generation[i] for i in sorted_indexes]

    return best_group


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
            task_duration = data.d(task[i])
            total_task_duration += task_duration
            if i != 0:
                travel_time = data.t({task[i], task[i-1]})
                total_travel_time += travel_time

        if len(task_list) > 0:

            # retrieve the worker name
            worker = task_list.index(task_list)

            # taking into account the home to first task and last task to
            # home travel time
            home_to_task_travel_time = data.t({data.Houses(worker), task[0]})
            total_travel_time += home_to_task_travel_time
            task_to_home_travel_time = data.t({task[-1], data.Houses(worker)})
            total_travel_time += task_to_home_travel_time

    cost = total_task_duration - total_travel_time

    return cost
