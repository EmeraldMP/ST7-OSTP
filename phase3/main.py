from misc import create_population, individuals_copy, select_best_group, find_best_solution
from mutation import mutation
from ..data import Data

initial_population_number = 10
best_future_generation_number = 10
number_of_copies = 10
stop_cost = 1
endroit = ""
instance = ""


def main():
    '''
    This function implements the main algorithm to be used in phase 3. It
    orchestrates all the different functions, most importantly dealing with
    the generations, mutations, selecting the best mutations and seeing if
    we reached a good enough solution to the algorithm to be able to stop.
    Feasibility checking is done in the mutation function.
    For the moment, it it just a sketch for us to have an idea of where and how
    the main functions are going to be used, and what we will need to implement.

    It will be needed most importantly to find a good stopping criteria, which is
    in the heart of the algorithm, and probably set a good initialization population,
    as well as try different hyperparameters (number of copies, setting the best ways
    to mutate individuals, etc)

    Parameters
    ----------

    Returns
    -------
    best_solution: dict
        Dictionary containing the best solution found by the algorithm,
        in the standard format.

    '''

    data = Data(endroit, instance)

    individuals = create_population(data, initial_population_number)
    optimum = False
    while not optimum:
        future_generation = individuals_copy(individuals, number_of_copies)
        for individual in future_generation:
            new_individual = mutation(individual)
            future_generation.remove(individual)
            future_generation.append(new_individual)
        # calculate the cost values, select the best ones
        best_new_individuals = select_best_group(
            future_generation, best_future_generation_number)
        # check if there is a best individual that is good enough
        # we're probably gonna change this to keep iterating until
        # we cant find a neighbourhood or something like that, it will
        # depend on the metaheuristic used I guess (VNS, simulated annealing, etc)
        best_solution = find_best_solution(best_new_individuals)
        if calculate_cost(best_solution) <= stop_cost:
            optimum = True

    return best_solution
