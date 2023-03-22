
def create_population():
    pass


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
