"""
individuals obey the following representation:
a dictionary consisting of several lists, one for each worker and containing their assigned tasks
{"W1":[T1, T5, ...], "W2": [T0, T16, T10], ..., "W10": [T9]}
"""
import random


# mutation strategies:
# exchanging tasks amongst workers
# reassign task of one worker to another worker
# reorder tasks of a single worker
# add undone task to a worker
# remove a task from a worker

def mutate_exchange(individual):
    # Melanie implements this
    pass


def mutate_reassign(individual):
    # ToDo: add probabilities depending on the length of a worker's task list
    num_workers = len(individual)
    worker_IDs = random.sample(range(num_workers), 2)
    worker1 = worker_IDs[0]
    worker2 = worker_IDs[1]
    task = individual[worker1].pop(random.randrange(len(individual[worker1])))
    individual[worker2].append(task)
    return individual


def mutate_reorder(individual):
    num_workers = len(individual)
    worker = random.randrange(num_workers)
    tasks = individual[worker]
    task1 = random.randrange(len(tasks))
    task2 = random.randrange(len(tasks))
    tasks = swapPositions(tasks, task1, task2)
    individual[worker] = tasks
    return individual


def swapPositions(list, pos1, pos2):
    first_ele = list.pop(pos1)
    second_ele = list.pop(pos2-1)

    list.insert(pos1, second_ele)
    list.insert(pos2, first_ele)

    return list


def mutate_remove(individual):
    # ToDo: add probabilities depending on the length of a worker's task list
    num_workers = len(individual)
    worker = random.randrange(num_workers)
    tasks = individual[worker]
    task = random.randrange(len(tasks))
    tasks.pop(task)
    individual[worker] = tasks
    return individual


def mutate_insert(individual, task):
    num_workers = len(individual)
    worker = random.randrange(num_workers)
    tasks = individual[worker]
    pos = random.randrange(len(tasks))
    tasks.insert(pos, task)
    individual[worker] = tasks
    return individual
