from ..data import lecture


def create_population(initial_population_number, endroit, instance):
    '''
    There will enter the "glouton" function the teacher was talking about
    I guess, we have to search for a good way to create initial solutions.

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

    df_Workers, _, df_Task, _ = lecture(endroit, instance)

    # Sets
    workers = list(df_Workers.index)
    tasks = list(df_Task.index)

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


def select_best():
    pass


def best_cost():
    pass


def find_best_solution():
    pass
