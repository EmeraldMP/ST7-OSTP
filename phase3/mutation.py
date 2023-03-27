"""
individuals obey the following representation:
a dictionary consisting of several lists, one for each worker and containing their assigned tasks
{'W1': [T5, T1], 'W2': [T0, T16, T10], 'W3': [T9]}


"""
import random
import numpy as np


def mutate(individual, data):
    """
    meta function for mutation operation. It picks one mutation strategy randomly and executes it.

    mutation strategies:
    - exchanging tasks amongst workers
    - reassign task of one worker to another worker
    - reorder tasks of a single worker
    - add undone task to a worker
    - remove a task from a worker

    :param individual: original individual
    :param data:
    :return: mutated individual
    """
    strategies = ["flip", "reassign", "reorder", "remove", "add"]

    # pick a strategy
    strategy = random.choice(strategies)
    if strategy == "flip":
        individual = mutate_flip(individual, data)
    elif strategy == "reassign":
        individual = mutate_reassign(individual, data)
    elif strategy == "reorder":
        individual = mutate_reorder(individual, data)
    elif strategy == "remove":
        individual = mutate_remove(individual, data)
    else:
        individual = mutate_add(individual, data)

    return individual


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
    #     # we make no changes if there is no possibility of change
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

    alpha1 = random.random()
    alpha2 = random.random()

    comparison1 = list(dist_mat1 < alpha1) + [False]
    comparison2 = list(dist_mat2 < alpha2) + [False]

    idx1 = 0
    idx2 = 0
    while comparison1[idx1]:
        idx1 += 1
    while comparison2[idx2]:
        idx2 += 1

    # flip task
    individual[w1][idx1], individual[w2][idx2] = individual[w2][idx2], individual[w1][idx1]

    return individual


def mutate_reassign(individual, data):
    workers = list(individual)

    # pick a worker from which a task is taken
    worker1 = pickWorker(individual)

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
            individual[worker2].insert(random.randrange(len(individual[worker2]) + 1), task)

            check = False

        cnt += 1

    return individual


def mutate_reorder(individual, data):
    # pick a worker
    worker = pickWorker(individual)

    tasks = individual[worker]

    # cannot reorder if there are less than 2 tasks
    if len(tasks) < 2:
        return individual

    # pick two tasks
    task1 = pickTask(tasks, worker, data)
    task2 = task1
    while task2 == task1:
        task2 = random.randrange(len(tasks))

    # reorder tasks of worker by swapping task1 and task2
    tasks = swapPositions(tasks, task1, task2)
    individual[worker] = tasks

    return individual


def mutate_remove(individual, data):
    num_tasks = 0

    # make sure that the chosen worker has at least one task
    while num_tasks < 1:
        # pick a worker from which a task is removed
        worker = pickWorker(individual)

        tasks = individual[worker]
        num_tasks = len(tasks)

    # remove a task based on probabilities
    task = pickTask(tasks, worker, data)
    tasks.remove(task)

    individual[worker] = tasks

    return individual


def mutate_add(individual, data):
    # pick a worker such that it is more likely to pick one with less tasks
    worker = pickWorker(individual, crit="less")

    task = None
    undoneTasks = []
    already_checked_workers = []
    cnt = 0
    num_workers = len(list(individual))
    while len(undoneTasks) == 0:
        if cnt == num_workers:
            # if there are no undone tasks
            return individual

        # determine which tasks (that can be done by the picked worker) have not been assigned yet
        for t in data.TasksW:
            if t not in individual[worker]:
                undoneTasks.append(t)

        # if there are no undone tasks, pick another worker
        if len(undoneTasks) == 0:
            already_checked_workers.append(worker)
            while worker in already_checked_workers:
                worker = pickWorker(individual, crit="less")
            cnt += 1
        else:
            # pick a task from undone tasks
            task = pickTask(undoneTasks, worker, data)

    assert task is not None, "no task was picked!"

    # insert the task in worker's task list at a random position
    tasks = individual[worker]
    pos = random.randrange(len(tasks))
    tasks.insert(pos, task)
    individual[worker] = tasks

    return individual


def pickWorker(individual, crit="more", num_worker=1):
    """
    picks workers such that a worker is more likely to be picked based on a criterion.
    :param individual: worker-task assignment
    :param crit: criterion based on which probabilities are calculated
                    "more" means higher probability if more tasks
                    "less" means higher probability if less tasks
    :param num_worker: number of workers that should be picked
    :return: list of picked workers
    """
    workers = list(individual)

    assert crit in ["more", "less"], "Entered criterion is unknown"
    # use the number of tasks of a worker to calculate their probability to be picked
    if crit == "more":
        # a worker is more likely to be chosen if more tasks are allocated to them
        probs = [len(individual[worker]) for worker in workers]
    else:
        # a worker is more likely to be chosen if less tasks are allocated to them
        probs = [1 / (1 + len(individual[worker])) for worker in workers]

    # make sure that not more workers than available are picked
    num_worker = min(num_worker, len(workers))

    # pick workers
    return random.choices(workers, weights=probs, k=num_worker)


def pickTask(tasks, worker, data, num_tasks=1):
    """
    picks a task of a worker such that a task more likely to be picked if the travel time to it and from it are larger
    :param tasks: list of tasks
    :param worker:
    :param data:
    :param num_tasks: number of tasks that should be picked
    :return: a task of the worker
    """
    # tasks = individual[worker]

    # make sure that not more tasks than available are picked
    num_tasks = min(num_tasks, len(tasks))

    # compute for each task the travel time to it and from it to following task and use those values as probabilities
    probs_task = []
    for task_ID in range(len(tasks)):
        # first task of worker
        if task_ID == 0:
            task = tasks[task_ID]
            post_task = tasks[task_ID + 1]
            # access time matrix data.t by data.t[][] with task or data.Houses[worker] as arguments
            probs_task.append(data.t[data.Houses[worker]][task] + data.t[task][post_task])
        # last task of worker
        elif task_ID == len(tasks) - 1:
            task = tasks[task_ID]
            pre_task = tasks[task_ID - 1]
            # access time matrix data.t by data.t[][] with task or data.Houses[worker] as arguments
            probs_task.append(data.t[pre_task][task] + data.t[task][data.Houses[worker]])
        else:
            task = tasks[task_ID]
            pre_task = tasks[task_ID - 1]
            post_task = tasks[task_ID + 1]
            # access time matrix data.t by data.t[][] with task or data.Houses[worker] as arguments
            probs_task.append(data.t[pre_task][task] + data.t[task][post_task])

    # pick tasks based on probabilities
    return random.choices(tasks, weights=probs_task, k=num_tasks)


def swapPositions(list, el1, el2):
    """
    swaps the position of two elements in a list
    :param list: original list
    :param el1: first element
    :param el2: second element
    :return: reordered list
    """
    index1 = list.index(el1)
    index2 = list.index(el2)

    list[index1], list[index2] = list[index2], list[index1]

    return list
