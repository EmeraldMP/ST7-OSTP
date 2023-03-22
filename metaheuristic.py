import math
import numpy as np
import random
import csv
import time
import matplotlib.pyplot as plt
from pprint import pprint
import copy
from data import Data, minutes_to_time


def initial_time(fin, task, w, data):
    # print(fin, task)
    begin = max(fin, data.a[task])
    for u in data.Unva[task]:
        if data.C[u][0] <= begin <= data.C[u][1]:
            begin = data.C[1]

    if begin <= data.b[task]:
        return begin
    return None

# On a déjà la génération


def feasibility(gene, data):

    for w in data.Workers:
        Tasks = gene[w]
        Lunch = True
        Indisp = {p: True for p in data.Pauses[w]}
        fin = data.alpha[w]
        lastTask = data.Houses[w]
        for task in Tasks:

            fin += data.t[lastTask][task]
            begin = initial_time(fin, task, w, data)

            if not begin:
                # print(task, "begin")
                return False

            if not begin + data.d[task] <= 13*60 and Lunch:
                Lunch = False
                fin += 60
                begin = initial_time(fin, task, w, data)
                if not begin:
                    # print(w)
                    return False

            fin = begin + data.d[task]

            if fin > data.b[task]:
                # print(fin, data.b[task])
                # print(task, "fin")
                return False

            for p in Indisp.keys():
                #print(task, minutes_to_time(fin + data.t[task][p]), minutes_to_time((data.a[p])), Indisp[p])
                if Indisp[p] and fin + data.t[task][p] > data.a[p]:
                    # print("on est bien dedans")
                    Indisp[p] = False
                    fin = data.b[p]
                    lastTask = p

                    fin += data.t[lastTask][task]
                    begin = initial_time(fin, task, w, data)

                    if not begin:
                        # print(task, "begin")
                        return False

                    if not begin + data.d[task] <= 13*60 and Lunch:
                        Lunch = False
                        fin += 60
                        begin = initial_time(fin, task, w, data)
                        if not begin:
                            # print(w)
                            return False

                    fin = begin + data.d[task]

                    if fin > data.b[task]:
                        print(task, "fin")
                        return False

            # print(task, minutes_to_time(begin), minutes_to_time(fin))

            lastTask = task

        # Spécifique ppour le retour à la maison
        task = data.Houses[w]

        fin += data.t[lastTask][task]
        begin = max(fin, data.beta[w])

        if not begin <= 13*60 and Lunch:
            Lunch = False
            fin += 60
            begin = max(fin, data.beta[w])

        fin = begin

        if fin > data.beta[w]:
            #print(task, "retour 1")
            return False

        for p in Indisp.keys():
            #print(task, minutes_to_time(fin + data.t[task][p]), minutes_to_time((data.a[p])), Indisp[p])
            if Indisp[p]:
                # print("on est bien dedans")
                Indisp[p] = False
                fin = data.b[p]
                lastTask = p

                fin += data.t[lastTask][task]
                begin = max(fin, data.beta[w])

                if not begin <= 13*60 and Lunch:
                    Lunch = False
                    fin += 60
                    begin = max(fin, data.beta[w])

                fin = begin

                if fin > data.beta[w]:
                    #print(task, "retour")
                    return False

    return True


"""def generate_instance(NW=2, NJ=9, ):

    I = dict()

    for w in range(NW):
        I["h" + str(w)] = dict()
        I["h" + str(w)]["pos"] = (random.randint(0, 100),
                                  random.randint(0, 100))
        I["h" + str(w)]["alpha"] = random.randint(60*6, 60*8)
        I["h" + str(w)]["beta"] = random.randint(60*16, 60*20)
        I["h" + str(w)]["done"] = None

    for j in range(NJ):
        I["j" + str(j)] = dict()
        I["j" + str(j)]["pos"] = (random.randint(0, 100),
                                  random.randint(0, 100))
        begin = random.randint(60*7, 60*19)
        duration = random.randint(30, 60*1.5)
        end = begin + duration + random.randint(30, duration)
        I["j" + str(j)]["begin"] = begin
        I["j" + str(j)]["end"] = end
        I["j" + str(j)]["duration"] = duration
        I["j" + str(j)]["done"] = None

    matrix_dist = dict()
    for key1, value1 in I.items():
        for key2, value2 in I.items():
            matrix_dist[(key1, key2)] = abs(
                value1["pos"][0] - value2["pos"][0]) + abs(value1["pos"][1] - value2["pos"][1])
    return I, matrix_dist"""

# dict_I, matrix_dist = generate_instance(NW=2, NJ=9, )
# pprint(dict_I)
# a = dict_I.get('h0')
# print(a)
# pprint(matrix_dist)
# random_list = random.sample(range(10), 10)
# random_list


# # heuristica:

# """estaba pensado en hacer como algunos razonamientos lógicos para podar el problema, por ejemplo, decir que nunca jamas dos nodos podrán estar dentro del mimo trabajador, también darle una cierta preferencia a elegir nodos cercanos para ir de un lugar al otro
# """


# def w_do_job_j_after_i(dict_I, w, j, i, travel, not_available_jobs):
#     if j in not_available_jobs:
#         return False, None, None
#     # Get the worker's data
#     dictI = copy.deepcopy(dict_I)
#     worker_key = f"h{w}"
#     worker_data = dictI.get(worker_key)

#     # Get the job's data
#     job_key = f"j{j}"
#     # print('job_key',job_key)
#     job_data = dictI.get(job_key)
#     # print('job_data',job_data)

#     # Get the node_i's data
#     # print('i',i)
#     #print('travel[i]', travel[i])
#     i_data = dictI.get(travel[i])
#     # print('i_data',i_data)

#     job_data_aux = copy.deepcopy(job_data)
#     job_data_aux = copy.deepcopy(job_data)

#     # Calculate the time it takes to get from the worker's position to the job's position

#     #travel_time = abs(i_data["pos"][0] - job_data["pos"][0]) + abs(i_data["pos"][1] - job_data["pos"][1])
#     travel_time = matrix_dist[(travel[i], job_key)]

#     primer_a_contar = 0

#     nodos_a_donear = dict()
#     nodos_a_donear[job_key] = copy.deepcopy(job_data)

#     if 'h' in travel[i]:
#         # Check if the worker can do the job
#         T = max(worker_data["alpha"] + travel_time,
#                 job_data["begin"]) + job_data["duration"]
#         if job_data["end"] < T:
#             return False, None, None
#         else:
#             primer_a_contar = 1
#             #last_T = T
#             # print(job_data)
#     else:
#         # Check if the worker can do the job
#         T = max(i_data["end"] + travel_time,
#                 job_data["begin"]) + job_data["duration"]
#         if job_data["end"] < T:
#             return False, None, None
#         else:
#             primer_a_contar = 1
#             #last_T = T
#           #  #print(job_data)

#     nodo_precedente = job_key
#     inicializacion = True
#     for node in travel[i + primer_a_contar:]:
#         # print('node',node)
#         # print('nodo_precedente',nodo_precedente)
#         node_data = I.get(node)
#         nodos_a_donear[node] = copy.deepcopy(node_data)
#         nodo_precedente_data = I.get(nodo_precedente)
#         nodos_a_donear[nodo_precedente] = copy.deepcopy(nodo_precedente_data)
#         nodos_a_donear[nodo_precedente]['done'] = T

#         #print("ya pasó una iteracion")
#         # print('node_data',node_data)
#         # print('nodo_precedente_data',nodo_precedente_data)
#         #travel_time = abs(node_data["pos"][0] - nodo_precedente_data["pos"][0]) + abs(node_data["pos"][1] - nodo_precedente_data["pos"][1])
#         travel_time = matrix_dist[(node, nodo_precedente)]
#         if 'h' in node:
#             #print("aaaaaaaa", nodo_precedente, nodos_a_donear[nodo_precedente]["done"])
#             #print('nodos_a_donear[nodo_precedente]', nodos_a_donear[nodo_precedente])
#             T = nodos_a_donear[nodo_precedente]["done"] + travel_time
#             if node_data["beta"] < T:
#                 #print('pas possible 1:', node_data["beta"], T)
#                 return False, None, None
#         else:
#             T = max(nodos_a_donear[nodo_precedente]["done"] +
#                     travel_time, node_data["begin"]) + node_data["duration"]
#             if node_data["end"] < T:
#                 #print('pas possible 1:', node_data["end"], T)
#                 return False, None, None

#         nodos_a_donear[node]["done"] = T
#         #print('nodos_a_donear[node][done]', nodos_a_donear[node]["done"])
#         nodo_precedente = node
#         #print("bbbbbbbbbb", nodo_precedente, nodos_a_donear[nodo_precedente]["done"])
#         nodo_precedente_data = node_data
#     for key, value in nodos_a_donear.items():
#         #print("key: ", key, " | value: ", value)
#         dictI[key]['done'] = value['done']
#     travel.insert(i+1, f"j{j}")

#     return True, travel, dictI


# def createRandomIndividual(dict_I, NW, NJ):
#     dictI = copy.deepcopy(dict_I)
#     sol = dict()
#     not_available_jobs = set()
#     random_list_w = random.sample(range(NW), NW)
#     random_list_j = random.sample(range(NJ), NJ)
#     for w in random_list_w:
#         travel = ['h'+str(w), 'h'+str(w)]
#         i = 0
#         for j in random_list_j:
#             if j not in not_available_jobs:
#                 w_can_do_job_j, sol_w, dictI0 = w_do_job_j_after_i(
#                     dictI, w, j, i, travel, not_available_jobs)
#                 if w_can_do_job_j:
#                     not_available_jobs.add(j)
#                     travel = sol_w
#                     dictI = dictI0
#                     # print('------')
#         sol['w'+str(w)] = travel
#     return sol, not_available_jobs, dictI


# I, matrix_dist = generate_instance(2, 9)
# sol, not_available_jobs, dictI = createRandomIndividual(dict_I, 2, 9)
# sol
# not_available_jobs
# sol['w0']
# # I
# pprint(dictI)


# def createInitialPopulation(dict_I, NW, NJ, populationSize):
#     population = []
#     # To be completed - Begin
#     for i in range(populationSize):
#         sol, not_available_jobs, dictI = createRandomIndividual(dict_I, 2, 9)
#         instance = {
#             'sol': sol,
#             'not_available_jobs': not_available_jobs,
#             'dictI': dictI
#         }
#         population.append(instance)
#     # To be completed - End
#     return population


# i = 0
# for individual in createInitialPopulation(dict_I, 2, 9, 10):
#     i += 1
#     print('i: ', 1)
#     pprint(individual)
#     print()

# # 3.2. Step 2 - Compute the fitness over the population in order to rank individuals


# # 3.2.1. Compute the fitness score of an individual
# def computeFitnessScore(dict_I, individual):
#     fitnessScore = 0
#     for job_done in individual['not_available_jobs']:
#         fitnessScore += dict_I['j'+str(job_done)]['duration']
#     return fitnessScore


# for individual in createInitialPopulation(dict_I, 2, 9, 1):
#     print(computeFitnessScore(dict_I, individual))
# # 3.2.2. Compute ranking data about the population


# def computeRankingData(population):
#     rankedPopulationIndices, rankedPopulationFitnessScores = [], []
#     # To be completed - Begin
#     list_aux = list()
#     for i in range(len(population)):
#         list_aux.append((i, computeFitnessScore(dict_I, population[i])))
#     list_aux = sorted(list_aux, key=lambda x: x[1], reverse=True)
#     for i in list_aux:
#         rankedPopulationIndices.append(i[0])
#         rankedPopulationFitnessScores.append(i[1])
#     # To be completed - End
#     rankingData = rankedPopulationIndices, rankedPopulationFitnessScores
#     return rankingData


# population = createInitialPopulation(dict_I, 2, 9, 10)
# testRankingData = computeRankingData(population)
# testRankedPopulationIndices = testRankingData[0]
# testRankedPopulationFitnessScores = testRankingData[1]
# print(testRankedPopulationIndices)
# print(testRankedPopulationFitnessScores)


# # 3.3. Step 3 - Select the best individuals to form the group of elites
# def selectBestIndividuals(population, rankedPopulationIndices, nbBestIndividuals):
#     bestIndividuals = []
#     # To be completed - Begin
#     for i in range(nbBestIndividuals):
#         bestIndividuals.append(population[rankedPopulationIndices[i]])
#     # To be completed - End
#     return bestIndividuals


# for elite in selectBestIndividuals(population, testRankedPopulationIndices, 2):
#     print(elite)


# def selectMatingPool(population, rankedPopulationIndices, rankedPopulationFitnessScores, nbParents):

#     matingPool = []

#     # Prepare the roulette wheel
#     # (i.e. compute the cumulative ratios of fitness scores over the population)
#     # To be completed - Start
#     ratio_Scores = list()
#     sum_Scores = sum(rankedPopulationFitnessScores)
#     for i in rankedPopulationFitnessScores:
#         ratio_Scores.append(i / sum_Scores)
#     # To be completed - End

#     # Play the roulette wheel several times to form the mating pool
#     # (i.e repeat picking an individual according to the random distribution of the fitness score ratios)
#     # To be completed - Start
#     #print("HERE THE PROBLEM 2: ", len(population), len(ratio_Scores))
#     try:
#         matingPool = random.choices(
#             population, weights=ratio_Scores, k=nbParents)
#     except:
#         print('population: ', population)
#         print('rankedPopulationFitnessScores: ', rankedPopulationFitnessScores)
#         print('rankedPopulationIndices: ', rankedPopulationIndices)
#         print('population[rankedPopulationIndices[0]]: ',
#               population[rankedPopulationIndices[0]])
#         print('ratio_Scores: ', ratio_Scores)
#     # To be completed - End

#     return matingPool


# for parent in selectMatingPool(population, testRankedPopulationIndices,
#                                testRankedPopulationFitnessScores, 2):
#     print(parent)


# random.randint(0, 8)
# """aca se puede ser así: """


# parent1 = {
#     'w0': ['h0', 'j1', 'j2', 'j3', 'j4', 'h0'],
#     'w1': ['h1', 'j5', 'j6', 'j7', 'j8', 'h1'],
# }
# parent2 = {
#     'w0': ['h0', 'j1', 'j3', 'j5', 'j7', 'h0'],
#     'w1': ['h1', 'j2', 'j4', 'j6', 'j8', 'h1'],
# }

# """ -> chequeando factibildad obviamente: robarle dos seguidos a parent 1 w0, mezclarlo con parent 2 w0 y lo q sobra dejarlo igual.
# """


# child = {
#     'w0': ['h0', 'j1', 'j2', 'j5', 'j7', 'h0'],
#     'w1': ['h1', 'j3', 'j4', 'j6', 'j8', 'h1'],
# }


# def crossover(parent1, parent2):
#     # Inicializar el cromosoma hijo
#     child = {}

#     # Recorrer cada trabajo en el cromosoma padre
#     for work in parent1:
#         # Generar un punto de cruce aleatorio
#         crossover_point = random.randint(1, len(parent1[work]) - 1)

#         # Combinar las tareas de los padres en el hijo, evitando repeticiones
#         child[work] = parent1[work][:crossover_point] + \
#             [task for task in parent2[work][crossover_point:]
#                 if task not in parent1[work]]

#     return child


# def crossovers(parent1, parent2):
#     # Inicializar el cromosoma hijo
#     child = {}

#     # Recorrer cada trabajo en el cromosoma padre
#     for work in parent1:
#         # Generar un punto de cruce aleatorio
#         crossover_point = random.randint(1, len(parent1[work]) - 1)

#         # Combinar las tareas de los padres en el hijo, evitando repetir elementos
#         child[work] = []
#         assigned_items = set()
#         for item in parent1[work][:crossover_point] + parent2[work][crossover_point:]:
#             if item not in assigned_items:
#                 child[work].append(item)
#                 assigned_items.add(item)

#     return child


# parent1 = {
#     'w0': ['h0', 'j1', 'j2', 'j3', 'j4', 'h0'],
#     'w1': ['h1', 'j5', 'j6', 'j7', 'j8', 'h1'],
# }
# parent2 = {
#     'w0': ['h0', 'j1', 'j3', 'j5', 'j7', 'h0'],
#     'w1': ['h1', 'j2', 'j4', 'j6', 'j8', 'h1'],
# }
# child = crossovers(parent1, parent2)
# child

# # Define the list of points with their coordinates and time windows
# points = [
#     {'id': 1, 'x': 0, 'y': 0, 'start_time': 0, 'end_time': 1},
#     {'id': 2, 'x': 3, 'y': 1, 'start_time': 0, 'end_time': 2},
#     {'id': 3, 'x': 2, 'y': 2.5, 'start_time': 3, 'end_time': 4},
#     {'id': 4, 'x': 1, 'y': 3, 'start_time': 6, 'end_time': 8},
#     {'id': 5, 'x': 0, 'y': 0, 'start_time': 8, 'end_time': 9},
# ]

# # Define the order in which the points are visited by the vehicle
# route = [1, 2, 3, 4, 5]

# # Create a scatter plot of the points
# fig, ax = plt.subplots()
# ax.scatter([p['x'] for p in points], [p['y'] for p in points])

# # Add a bar for each point that shows the time window
# for p in points:
#     ax.barh(p['y'], 10, left=1+p['x'], height=0.05, color='red', alpha=0.1)
#     ax.barh(p['y'], p['end_time'] - p['start_time'], left=1 +
#             p['x'] + p['start_time'], height=0.05, color='green')


# # Connect the points in the order they are visited by the vehicle
# route_x = [points[route[i]-1]['x'] for i in range(len(route))]
# route_y = [points[route[i]-1]['y'] for i in range(len(route))]
# ax.plot(route_x, route_y, '-o')

# # Add a label for the route
# route_time = sum([points[route[i]-1]['end_time'] for i in range(len(route))])
# ax.set_title(f"Route Time: {route_time}")

# plt.show()


# def breedParents(parent1, parent2):
#     child = []
#     # To be completed - Begin
#     random1 = random.randint(0, len(parent1)-2)
#     random2 = random.randint(random1+1, len(parent1)-1)
#     parent1_gen = parent1[random1: random2]
#     #print("le set: ", parent1_gen)
#     i = 0
#     j = 0
#     while i <= len(parent2)-1:
#         if parent2[i] not in parent1_gen:
#             child.append(parent2[i])
#             j += 1
#             if j == random1:
#                 i += 1
#                 break
#         i += 1
#     #print('child1: ', child)
#     child += parent1_gen
#     #print('child2: ', child)
#     # if random2 < len(parent1):
#     j = random2
#     while i <= len(parent2)-1:
#         if parent2[i] not in parent1_gen:
#             child.append(parent2[i])
#             j += 1
#             if j == len(parent1)+1:
#                 break
#         i += 1
#     #print('child3: ', child)
#     # To be completed - End
#     return child


# exampleParent1 = exampleMatingPool[0]
# exampleParent2 = exampleMatingPool[-1]
# exampleChild = [(73, 93), (139, 181), (163, 196), (50, 123), (68, 187),
#                 (121, 149), (116, 25), (196, 130), (0, 35), (103, 137)]
# print(exampleParent1)
# print(exampleParent2)
# print(exampleChild)


# print(exampleParent1)
# print(exampleParent2)
# print(breedParents(exampleParent1, exampleParent2))

# # Breed mating pool


# def breedMatingPool(matingPool, nbChildren):
#     children = []
#     # To be completed - Start
#     for i in range(nbChildren):
#         parent1 = random.choice(matingPool)
#         while True:
#             parent2 = random.choice(matingPool)
#             if parent2 != parent1:
#                 break
#         children.append(breedParents(parent1, parent2))
#     # To be completed - End
#     return children


# exampleChildren = [
#     [(0, 35), (103, 137), (116, 25), (68, 187), (196, 130),
#      (50, 123), (163, 196), (121, 149), (139, 181), (73, 93)],
#     [(73, 93), (103, 137), (0, 35), (50, 123), (139, 181),
#      (121, 149), (163, 196), (116, 25), (196, 130), (68, 187)],
#     [(103, 137), (139, 181), (116, 25), (68, 187), (196, 130),
#      (50, 123), (163, 196), (121, 149), (0, 35), (73, 93)],
#     [(73, 93), (103, 137), (50, 123), (139, 181), (121, 149),
#      (163, 196), (0, 35), (116, 25), (68, 187), (196, 130)],
#     [(73, 93), (50, 123), (139, 181), (163, 196), (116, 25),
#      (196, 130), (103, 137), (0, 35), (121, 149), (68, 187)],
#     [(73, 93), (50, 123), (139, 181), (196, 130), (0, 35),
#      (121, 149), (163, 196), (116, 25), (68, 187), (103, 137)],
#     [(103, 137), (139, 181), (163, 196), (50, 123), (68, 187),
#      (121, 149), (0, 35), (116, 25), (73, 93), (196, 130)],
#     [(73, 93), (139, 181), (163, 196), (50, 123), (68, 187),
#      (121, 149), (0, 35), (103, 137), (116, 25), (196, 130)],
#     [(103, 137), (116, 25), (196, 130), (50, 123), (73, 93),
#      (0, 35), (139, 181), (121, 149), (163, 196), (68, 187)],
#     [(163, 196), (103, 137), (0, 35), (50, 123), (139, 181), (68, 187), (121, 149), (116, 25), (73, 93), (196, 130)]]


# for child in breedMatingPool(exampleMatingPool, 10):
#     print(child)
# # 3.4.3. Step 4.C - Apply random mutations to the children


# def mutateIndividual(individual, mutationRate):
#     # To be completed - Start
#     length = len(individual)
#     for i in range(length-2):
#         if random.random() <= mutationRate:
#             swaped_i = random.randint(i, length-1)
#             original = individual[i]
#             individual[i] = individual[swaped_i]
#             individual[swaped_i] = original
#     # To be completed - End
#     return individual


# exampleMutatedChild = [(73, 93), (0, 35), (116, 25), (196, 130), (68, 187),
#                        (163, 196), (139, 181), (50, 123), (121, 149), (103, 137)]
# print(exampleChild)
# print(exampleMutatedChild)

# print(exampleChild)
# print(mutateIndividual(exampleChild.copy(), 0.1))


# # Mutate a population
# def mutatePopulation(population, mutationRate):
#     # To be completed - Start
#     population_aux = list()
#     for individual in population:
#         population_aux.append(mutateIndividual(
#             individual.copy(), mutationRate))
#     population = population_aux
#     # To be completed - End
#     return population


# exampleMutatedChildren = [
#     [(163, 196), (103, 137), (73, 93), (68, 187), (196, 130),
#      (50, 123), (139, 181), (0, 35), (121, 149), (116, 25)],
#     [(103, 137), (73, 93), (163, 196), (139, 181), (68, 187),
#      (196, 130), (0, 35), (50, 123), (121, 149), (116, 25)],
#     [(0, 35), (196, 130), (163, 196), (68, 187), (116, 25),
#      (50, 123), (139, 181), (121, 149), (73, 93), (103, 137)],
#     [(116, 25), (163, 196), (50, 123), (139, 181), (121, 149),
#      (103, 137), (0, 35), (73, 93), (68, 187), (196, 130)],
#     [(73, 93), (50, 123), (0, 35), (139, 181), (68, 187),
#      (163, 196), (103, 137), (121, 149), (196, 130), (116, 25)],
#     [(116, 25), (50, 123), (0, 35), (196, 130), (68, 187),
#      (121, 149), (139, 181), (163, 196), (73, 93), (103, 137)],
#     [(50, 123), (196, 130), (73, 93), (0, 35), (68, 187),
#      (121, 149), (103, 137), (116, 25), (163, 196), (139, 181)],
#     [(73, 93), (121, 149), (163, 196), (0, 35), (50, 123),
#      (139, 181), (103, 137), (116, 25), (68, 187), (196, 130)],
#     [(196, 130), (121, 149), (139, 181), (103, 137), (73, 93),
#      (0, 35), (116, 25), (50, 123), (163, 196), (68, 187)],
#     [(163, 196), (116, 25), (0, 35), (50, 123), (139, 181), (68, 187), (121, 149), (73, 93), (103, 137), (196, 130)]]


# for child in exampleChildren:
#     print(child)
# print()
# for mutatedChild in mutatePopulation([individual.copy() for individual in exampleChildren], 0.1):
#     print(mutatedChild)


# def mergePopulation(population1, population2):
#     newPopulation = []
#     # To be completed - Start
#     newPopulation += population1 + population2
#     # To be completed - End
#     return newPopulation

# # 3.6. Step 6 - Repeat steps 2-5 with the new generation until reaching a stopping criterion


# def produceNewGeneration(generation, nbElites, nbParents, nbChildren, mutationRate):
#     newGeneration = []
#     nbIndividualsPerGeneration = len(generation)
#     # To be completed - Start
#     #print('size :', len(generation))
#     population = generation
#     nbBestIndividuals = nbElites

#     # elite
#     rankedPopulationIndices, rankedPopulationFitnessScores = computeRankingData(
#         population)

#     elite = selectBestIndividuals(
#         population, rankedPopulationIndices, nbBestIndividuals)
#     #print("HERE THE PROBLEM:")
#     matingPool = selectMatingPool(
#         population, rankedPopulationIndices, rankedPopulationFitnessScores, nbParents)

#     # childrens
#     childrens = breedMatingPool(matingPool, nbChildren)

#     #mutated_population = mutatePopulation(childrens, mutationRate)
#     mutated_population = childrens

#     rankedmutated_PopulationIndices, rankedmutated_PopulationFitnessScores = computeRankingData(
#         mutated_population)

#     nbChildrens = len(population) - nbElites
#     final_childrens = selectBestIndividuals(
#         mutated_population, rankedmutated_PopulationIndices, nbChildrens)

#     # merge
#     newGeneration = mergePopulation(elite, mutated_population)
#     # to be completed - End
#     return newGeneration
# # 3.7. Step 7 - Select the best individual of the last generation


# def runGeneticAlgorithm(instance,
#                         generationSize, nbElites, nbParents, nbChildren, mutationRate,
#                         nbGenerations=100, maximumSolvingTimeInSeconds=60, timeCriterion=True):

#     # Initialize stopping criterion data
#     criterionSatisfaction = True
#     nbGenerationsDone = 0
#     startTime = 0

#     # Check criterion satisfaction
#     def checkCriterionSatisfaction():
#         if timeCriterion:
#             return (time.time() - startTime) < maximumSolvingTimeInSeconds
#         else:
#             return nbGenerationsDone < nbGenerations

#     # Create initial population
#     generation = createInitialPopulation(generationSize, instance)

#     # Repeat producing generation until reaching the stopping criterion
#     startTime = time.time()
#     bestFitnessScoresEvolution = []
#     # while checkCriterionSatisfaction():
#     i = 0
#     while i <= nbGenerations:
#         i += 1
#         print(i, end=" ")
#         generation = produceNewGeneration(
#             generation, nbElites, nbParents, nbChildren, mutationRate)
#         bestIndividual = generation[0]
#         bestFitnessScoresEvolution.append(computeFitnessScore(bestIndividual))
#         nbGenerationsDone += 1
#     rankedGenerationIndices, rankedGenerationFitnessScores = computeRankingData(
#         generation)
#     bestIndividual = generation[rankedGenerationIndices[0]]
#     bestFitnessScore = computeFitnessScore(bestIndividual)
#     bestFitnessScoresEvolution.append(bestFitnessScore)

#     return bestIndividual, bestFitnessScore, bestFitnessScoresEvolution


# # 4. Functions for displaying results

# def plotEvolution(bestDistancesEvolution):
#     plt.plot(bestDistancesEvolution)
#     plt.ylabel('total distance')
#     plt.xlabel('generation index')
#     plt.show()


# def plotRoute(route):
#     xs = [city[0] for city in route] + [route[0][0]]
#     ys = [city[1] for city in route] + [route[0][1]]
#     plt.plot(xs, ys, 'o-')
#     plt.ylabel('x')
#     plt.xlabel('y')
#     plt.show()
# # 5. Numerical experiments


# fileName = "TSPInstance50.csv"
# instance = loadFixedInstance(fileName)
# # To be completed - Start
# bestIndividual, bestFitnessScore, bestFitnessScoresEvolution = runGeneticAlgorithm(
#     instance, generationSize=100, nbElites=30, nbParents=50, nbChildren=70, mutationRate=0.3,
#     nbGenerations=10000, maximumSolvingTimeInSeconds=60, timeCriterion=True)
# # To be completed - End

# bestDistancesEvolution = 1/np.array(bestFitnessScoresEvolution)
# plotEvolution(bestDistancesEvolution)
# plotRoute(bestIndividual)
# print("Total distance:" + str(1/bestFitnessScore))


# fileName = "TSPInstance100.csv"
# instance = loadFixedInstance(fileName)
# # To be completed - Start
# bestIndividual, bestFitnessScore, bestFitnessScoresEvolution = runGeneticAlgorithm(
#     instance, generationSize=300, nbElites=50, nbParents=100, nbChildren=250, mutationRate=0.25,
#     nbGenerations=3000, maximumSolvingTimeInSeconds=20, timeCriterion=True)
# # To be completed - End

# bestDistancesEvolution = 1/np.array(bestFitnessScoresEvolution)
# plotEvolution(bestDistancesEvolution)
# plotRoute(bestIndividual)
# print("Total distance:" + str(1/bestFitnessScore))


# # Define the problem
# num_workers = 3
# num_jobs = 5
# worker_time_windows = [[0, 10], [3, 8], [5, 12]]
# job_time_windows = [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]]
# job_durations = [2, 3, 2, 4, 3]

# # Define the cost function


# def cost_function(solution):
#     cost = 0
#     for i in range(num_jobs):
#         start_time = solution[i]
#         end_time = start_time + job_durations[i]
#         for j in range(num_workers):
#             if start_time >= worker_time_windows[j][0] and end_time <= worker_time_windows[j][1]:
#                 cost += 1
#     return cost

# # Define the initial solution


# def initial_solution():
#     solution = []
#     for i in range(num_jobs):
#         start_time = random.randint(
#             job_time_windows[i][0], job_time_windows[i][1] - job_durations[i])
#         solution.append(start_time)
#     return solution

# # Define the neighbor function


# def neighbor(solution):
#     new_solution = solution.copy()
#     i = random.randint(0, num_jobs - 1)
#     new_start_time = random.randint(
#         job_time_windows[i][0], job_time_windows[i][1] - job_durations[i])
#     new_solution[i] = new_start_time
#     return new_solution

# # Define the acceptance probability function


# def acceptance_probability(old_cost, new_cost, temperature):
#     if new_cost < old_cost:
#         return 1.0
#     else:
#         return math.exp((old_cost - new_cost) / temperature)

# # Define the simulated annealing algorithm


# def simulated_annealing():
#     # Set initial temperature and cooling rate
#     temperature = 1000.0
#     cooling_rate = 0.03

#     # Initialize current solution and cost
#     current_solution = initial_solution()
#     current_cost = cost_function(current_solution)

#     # Initialize best solution and cost
#     best_solution = current_solution
#     best_cost = current_cost

#     # Loop until temperature is too low
#     while temperature > 1.0:
#         # Generate a neighbor solution
#         neighbor_solution = neighbor(current_solution)
#         neighbor_cost = cost_function(neighbor_solution)

#         # Determine whether to accept the neighbor solution
#         ap = acceptance_probability(current_cost, neighbor_cost, temperature)
#         if ap > random.random():
#             current_solution = neighbor_solution
#             current_cost = neighbor_cost

#         # Update the best solution if necessary
#         if current_cost < best_cost:
#             best_solution = current_solution
#             best_cost = current_cost

#         # Decrease the temperature
#         temperature *= 1 - cooling_rate

#     return best_solution, best_cost


# # Run the simulated annealing algorithm
# best_solution, best_cost = simulated_annealing()

# # Print the best solution and cost
# print("Best solution:", best_solution)
# print("Best cost:", best_cost)


# # Define the problem
# num_workers = 3
# num_jobs = 5
# worker_time_windows = [[0, 10], [3, 8], [5, 12]]
# job_time_windows = [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]]
# job_durations = [2, 3, 2, 4, 3]

# # Define the fitness function


# def fitness_function(solution):
#     cost = 0
#     for i in range(num_jobs):
#         start_time = solution[i]
#         end_time = start_time + job_durations[i]
#         for j in range(num_workers):
#             if start_time >= worker_time_windows[j][0] and end_time <= worker_time_windows[j][1]:
#                 cost += 1
#     return cost

# # Define the initial population


# def initial_population(population_size):
#     population = []
#     for i in range(population_size):
#         individual = []
#         for j in range(num_jobs):
#             start_time = random.randint(
#                 job_time_windows[j][0], job_time_windows[j][1] - job_durations[j])
#             individual.append(start_time)
#         population.append(individual)
#     return population

# # Define the selection function


# def selection(population, tournament_size):
#     tournament = random.sample(population, tournament_size)
#     winner = max(tournament, key=fitness_function)
#     return winner

# # Define the crossover function


# def crossover(parent1, parent2):
#     child = []
#     for i in range(num_jobs):
#         if random.random() < 0.5:
#             child.append(parent1[i])
#         else:
#             child.append(parent2[i])
#     return child

# # Define the mutation function


# def mutation(individual, mutation_rate):
#     for i in range(num_jobs):
#         if random.random() < mutation_rate:
#             individual[i] = random.randint(
#                 job_time_windows[i][0], job_time_windows[i][1] - job_durations[i])
#     return individual

# # Define the genetic algorithm


# def genetic_algorithm(population_size, tournament_size, crossover_rate, mutation_rate, num_generations):
#     # Initialize the population
#     population = initial_population(population_size)

#     # Loop through generations
#     for generation in range(num_generations):
#         # Create a new population
#         new_population = []

#         # Add the best individual from the previous generation
#         best_individual = max(population, key=fitness_function)
#         new_population.append(best_individual)

#         # Generate the rest of the population through selection, crossover, and mutation
#         while len(new_population) < population_size:
#             parent1 = selection(population, tournament_size)
#             parent2 = selection(population, tournament_size)
#             child = crossover(parent1, parent2)
#             child = mutation(child, mutation_rate)
#             new_population.append(child)

#         # Update the population
#         population = new_population

#     # Return the best individual
#     best_individual = max(population, key=fitness_function)
#     best_fitness = fitness_function(best_individual)
#     return best_individual, best_fitness


# # Run the genetic algorithm
# best_individual, best_fitness = genetic_algorithm(
#     population_size=100,
#     tournament_size=5,
#     crossover_rate=0.8,
#     mutation_rate=0.1,
#     num_generations=100
# )

# # Print the best individual and fitness
# print("Best individual:", best_individual)
# print("Best fitness:", best_fitness)


# # Define the problem
# num_workers = 3
# num_jobs = 5
# worker_time_windows = [[0, 10], [3, 8], [5, 12]]
# job_time_windows = [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]]
# job_durations = [2, 3, 2, 4, 3]
# job_earnings = [10, 15, 12, 18, 13]

# # Define the fitness function


# def fitness_function(solution):
#     total_cost = 0
#     total_earnings = 0
#     for i in range(num_jobs):
#         start_time = solution[i]
#         end_time = start_time + job_durations[i]
#         worker = -1
#         for j in range(num_workers):
#             if start_time >= worker_time_windows[j][0] and end_time <= worker_time_windows[j][1]:
#                 worker = j
#                 break
#         if worker == -1:
#             total_cost += float("inf")
#         else:
#             # Calculate the cost based on the distance traveled by the worker
#             if i == 0:
#                 distance_cost = 0
#             else:
#                 prev_job_end_time = solution[i - 1] + job_durations[i - 1]
#                 distance_cost = abs(job_time_windows[i][0] - job_time_windows[i - 1][1]) + abs(
#                     worker_time_windows[worker][0] - job_time_windows[i][0])
#             total_cost += distance_cost
#             # Calculate the earnings based on the money earned by the worker for doing the job
#             total_earnings += job_earnings[i]
#     return total_earnings - total_cost

# # Define the initial population


# def initial_population(population_size):
#     population = []
#     for i in range(population_size):
#         individual = []
#         for j in range(num_jobs):
#             start_time = random.randint(
#                 job_time_windows[j][0], job_time_windows[j][1] - job_durations[j])
#             individual.append(start_time)
#         population.append(individual)
#     return population

# # Define the selection function


# def selection(population, tournament_size):
#     tournament = random.sample(population, tournament_size)
#     winner = max(tournament, key=fitness_function)
#     return winner

# # Define the crossover function


# def crossover(parent1, parent2):
#     child = []
#     for i in range(num_jobs):
#         if random.random() < 0.5:
#             child.append(parent1[i])
#         else:
#             child.append(parent2[i])
#     return child

# # Define the mutation function


# def mutation(individual, mutation_rate):
#     for i in range(num_jobs):
#         if random.random() < mutation_rate:
#             individual[i] = random.randint(
#                 job_time_windows[i][0], job_time_windows[i][1] - job_durations[i])
#     return individual

# # Define the genetic algorithm


# def genetic_algorithm(population_size, tournament_size, crossover_rate, mutation_rate, num_generations):
#     # Initialize the population
#     population = initial_population(population_size)

#     # Loop through generations
#     for generation in range(num_generations):
#         # Create a new population
#         new_population = []

#         # Add the best individual from the previous generation
#         best_individual = max(population, key=fitness_function)
#         new_population.append(best_individual)

#         # Generate the rest of the population through selection, crossover,


# population_size = 50
# tournament_size = 5
# crossover_rate = 0.8
# mutation_rate = 0.1
# num_generations = 100

# best_solution = genetic_algorithm(
#     population_size, tournament_size, crossover_rate, mutation_rate, num_generations)

# print("Best solution found:")
# if best_solution is not None:
#     print("Best solution found:")
#     print(best_solution)
#     print("Fitness value:", fitness_function(best_solution))
# else:
#     print("No feasible solution found within the given number of generations.")


# # Define problem parameters
# NUM_WORKERS = 3
# NUM_JOBS = 13
# MAX_DURATION = 8  # in hours
# MAX_SPEED = 50  # in km/h

# # Define chromosome
# chromosome_length = NUM_JOBS * NUM_WORKERS


# def generate_chromosome():
#     return [random.randint(0, NUM_JOBS-1) for _ in range(chromosome_length)]

# # Define fitness function


# def calculate_fitness(chromosome):
#     # Convert chromosome to schedule for each worker
#     schedules = [[] for _ in range(NUM_WORKERS)]
#     for i in range(chromosome_length):
#         worker_index = i % NUM_WORKERS
#         job_index = chromosome[i]
#         schedules[worker_index].append(job_index)

#     # Calculate total earnings for each worker
#     total_earnings = [0.0 for _ in range(NUM_WORKERS)]
#     for worker_index in range(NUM_WORKERS):
#         current_time = 0.0  # start at home
#         current_x, current_y = jobs[0][:2]  # start at home
#         for job_index in schedules[worker_index]:
#             # Calculate travel time to job
#             job_x, job_y, job_duration, job_window_start, job_window_end, job_pay = jobs[
#                 job_index]
#             travel_time = math.sqrt(
#                 (job_x - current_x)**2 + (job_y - current_y)**2) / MAX_SPEED
#             arrival_time = current_time + travel_time

#             # Check if job can be done within time window and before end of day
#             if arrival_time >= job_window_start and arrival_time + job_duration <= job_window_end and current_time + travel_time + job_duration <= MAX_DURATION:
#                 # Add earnings for job
#                 total_earnings[worker_index] += job_pay

#                 # Update current time and location
#                 current_time = arrival_time + job_duration
#                 current_x = job_x
#                 current_y = job_y

#         # Calculate travel time back home
#         home_x, home_y, _, _, _, _ = jobs[0]
#         travel_time = math.sqrt((home_x - current_x) **
#                                 2 + (home_y - current_y)**2) / MAX_SPEED
#         current_time += travel_time

#     # Return total earnings for all workers
#     return sum(total_earnings)


# jobs = [
#     (0, 0, 0, 0, 24, 0),  # home
#     (10, 20, 2, 9, 11, 10),
#     (20, 30, 3, 10, 14, 15),
#     (30, 10, 1, 7, 8, 5),
#     (40, 20, 2, 11, 13, 10),
#     (50, 30, 3, 9, 14, 15),
#     (60, 10, 1, 8, 9, 5),
#     (70, 20, 2, 10, 12, 10),
#     (80, 30, 3, 12, 15, 15),
#     (90, 10, 1, 7, 8, 5),
#     (100, 20, 2, 9, 11, 10),
#     (110, 30, 3, 8, 12, 15),
#     (120, 10, 1, 10, 11, 5),
# ]

# cromos = generate_chromosome()
# calculate_fitness(cromos)
