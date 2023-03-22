"""
individuals obey the following representation:
a dictionary consisting of several lists, one for each worker and containing their assigned tasks
{'W1': [5, 1], 'W2': [0, 16, 10], 'W3': [9]}
"""
import random
import numpy as np


# mutation strategies:
# exchanging tasks amongst workers
# reassign task of one worker to another worker
# reorder tasks of a single worker
# add undone task to a worker
# remove a task from a worker

def mutate_flip(individual_ini, data):
    # define time matrix as global variable
    t = data.t

    individual = individual_ini.copy()

    # choose a random worker
    w1, w2 = random.sample(list(individual.keys()), 2)

    good_task_w1 = data.TasksW[w1]
    good_task_w2 = data.TasksW[w2]

    # choose random task to flip
    idx1 = random.randint(0, len(individual[w1]) - 1)
    idx2 = random.randint(0, len(individual[w2]) - 1)

    # dist_mat1 = [t[data.Houses[w1]][task] if task in good_task_w2 else 0 for task in individual[w1]]
    # dist_mat2 = [t[data.Houses[w2]][task] if task in good_task_w1 else 0 for task in individual[w2]]
    # print(dist_mat2)

    # if sum(dist_mat1) == 0 or sum(dist_mat2) == 0:
    #     # we make no changes if there is no posibility of change
    #     print('AHHHHHHHHHH')
    #     return individual

    # l_a = list(range(len(individual[w1])))

    # idx1 = random.choices(list(range(len(individual[w1]))), weights=dist_mat1)[0]
    # idx2 = random.choices(list(range(len(individual[w2]))), weights=dist_mat2)[0]

    # print(idx1, idx2)

    # Create distance Matrix
    dist_mat1 = np.cumsum([t[data.Houses[w1]][task] if task in good_task_w2 else 0 for task in individual[w1]])
    dist_mat2 = np.cumsum([t[data.Houses[w2]][task] if task in good_task_w1 else 0 for task in individual[w2]])

    if dist_mat1[-1] == 0 or dist_mat2[-1] == 0:
        # we make no changes
        return individual

    dist_mat1 = dist_mat1 / dist_mat1[-1]
    dist_mat2 = dist_mat2 / dist_mat2[-1]

    alpha1 = random.random();
    alpha2 = random.random()

    comparation1 = list(dist_mat1 < alpha1) + [False]
    comparation2 = list(dist_mat2 < alpha2) + [False]

    idx1 = 0
    idx2 = 0
    while comparation1[idx1]:
        idx1 += 1
    while comparation2[idx2]:
        idx2 += 1

    # flip task
    individual[w1][idx1], individual[w2][idx2] = individual[w2][idx2], individual[w1][idx1]

    return individual


def mutate_reassign(individual, data):
    workers = list(individual)
    # use the number of tasks of a worker as their probability to be affected by reassignment
    # a worker is more likely to be chosen if more tasks are allocated to them
    probs1 = [len(individual[worker]) for worker in workers]

    # pick a worker from which a task is taken
    worker1 = random.choices(workers, weights=probs1, k=1)

    # adjust probs
    # a worker is more likely to be chosen if less tasks are allocated to them
    workers.remove(worker1)
    probs2 = [1 / (1 + len(individual[worker])) for worker in workers]

    # pick another worker who gets an additional task
    worker2 = random.choices(workers, weights=probs2, k=1)

    # check whether the task can be done by worker 2...
    check = True
    cnt = 0
    tasks = individual[worker1]
    # ... as long as a suitable task has been found or there is no task left in tasks
    while check and cnt < len(individual[worker1]):
        # take a random task from worker 1
        task = tasks.pop(random.randrange(len(tasks)))

        # check whether task can be done by worker 2
        if task in data.TasksW[worker2]:
            #  reassign the task to worker 2 at a random place
            individual[worker1].remove(task)
            individual[worker2].insert(random.randrange(len(individual[worker2])+1), task)

            check = False

        cnt += 1

    return individual


def mutate_reorder(individual, data=None):
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


def mutate_remove(individual, data):
    workers = list(individual)
    # use the number of tasks of a worker as their probability to be affected by reassignment
    # a worker is more likely to be chosen if more tasks are allocated to them
    probs_worker = [len(individual[worker]) for worker in workers]

    # pick a worker from which a task is taken
    worker = random.choices(workers, weights=probs_worker, k=1)

    tasks = individual[worker]

    time_matrix = data.t    # access time matrix by e.g. data.t[][] with task or data.Houses[worker]
    probs_task = []
    for task_ID in range(len(tasks)):
        if task_ID == 0:
            probs_task.append()
        elif task_ID == len(tasks)-1:
            probs_task.append()
        else:
            probs_task.append()

    task_ID = random.randrange(len(tasks))
    tasks.pop(task_ID)
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

