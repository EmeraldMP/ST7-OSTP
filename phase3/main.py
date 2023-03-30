from data import Data
from phase3.misc import create_population, individuals_copy, select_best_group, find_best_solution, calculate_cost
from phase3.mutation import mutate
from phase3.check_constraints import feasibility_sc


def process(endroit, instance, data):
    """
    This function implements the main algorithm to be used in phase 3. It
    orchestrates all the different functions, most importantly dealing with
    the generations, mutations, selecting the best mutations and seeing if
    we reached a good enough solution to the algorithm to be able to stop.
    Feasibility checking is done in the mutation function.
    For the moment, it just a sketch for us to have an idea of where and how
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

    """
    try:
        initial_population_number = 10
        best_future_generation_number = 10
        number_of_copies = 10
        stop_cost = 5000
        best_solution = None

        individuals = create_population(data, initial_population_number)
        best_scores = [feasibility_sc(ind, data) for ind in individuals]
        print(individuals)
        optimum = False
        while not optimum:
            future_generation_temp = individuals_copy(
                individuals, number_of_copies)
            future_generation = []
            future_score = []

            i = 1
            for individual in future_generation_temp:
                new_individual, score = mutate(individual, data)
                # do not remove old generation! maybe some old individuals are better than the new ones
                # future_generation.remove(individual)
                future_generation.append(new_individual)
                future_score.append(score)
                i += 1

            # We add the parents to the new generation
            future_generation.extend(individuals)
            future_score.extend(best_scores)

            # calculate the cost values, select the best ones
            individuals, best_scores = select_best_group(
                future_generation, best_future_generation_number, future_score)

            # check if there is a best individual that is good enough
            # we will probably change this to keep iterating until
            # we cant find a neighbourhood or something like that, it will
            # depend on the metaheuristic used I guess (VNS, simulated annealing, etc)
            # find_best_solution(best_new_individuals)
            best_solution = individuals[0]
            print(best_scores[0])
            if best_scores[0] >= stop_cost:
                optimum = True
    except KeyboardInterrupt:
        return best_solution

    return best_solution
