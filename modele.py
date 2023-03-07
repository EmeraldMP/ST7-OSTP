from gurobipy import *
import time


class Variable():
    def __init__(self, Data, modele):
        self.m, self.Xm, self.Tm, self.Ym = modele(Data)
        self.integerate()

    def opti(self):  # Demand to optimize the model
        t1 = time.time()
        self.m.optimize()
        delta = time.time() - t1
        self.integerate()
        return delta

    def integerate(self):  # Define to dictionnaries corresponding to the variables, were the variables are replaced by their values
        try:
            self.X = {(i, j, w): [self.Xm[(i, j, w)].x, self.Xm[(
                i, j, w)].VarName] for (i, j, w) in self.Xm.keys()}
            self.T = {i: [self.Tm[i].x, self.Tm[i].VarName] for i in self.Tm.keys() if type(self.Tm[i]) != int} | {
                i: self.Tm[i] for i in self.Tm.keys() if type(self.Tm[i]) == int}
        except:
            self.X = {(i, j, w): [0, self.Xm[(
                i, j, w)].VarName] for (i, j, w) in self.Xm.keys()}
            self.T = {i: [0, self.Tm[i].VarName] for i in self.Tm.keys() if type(self.Tm[i]) != int} | {
                i: self.Tm[i] for i in self.Tm.keys() if type(self.Tm[i]) == int}

        try:
            """self.Y = {}
            for i in self.Tm.keys():
                sum = 0
                for w in self.Data.Workers:
                    if (i, w) in self.Ym:
                        print(i, w)
                        if self.Ym[(i, w)].getValue():
                            self.Y[i] = 1
                            break
                else:
                    self.Y[i] = 0
"""
            self.Y = {i: [sum([self.Ym[(i, w)].getValue() for w in self.Data.Workers if (
                i, w) in self.Ym])] for i in self.Tm.keys() if type(self.Tm[i]) != int} | {i: 1 for i in self.Tm.keys() if type(self.Tm[i]) == int}
            print(self.Y)

        except:

            self.Y = {i: 0 for i in self.Tm.keys()}

    def debug(self):  # If you want to see the constraints defined in the model
        self.m.params.outputflag = 1
        self.m.display()


def modele_v1_1(Data):
    # 1) it  necesary to put the option that workers do a trip from their house to their house, because we are making that them has to get out once and have to get bac once. If they do nothing in the optimal, they will pick this fictional arc that basicly means: do nothing.

    ####################################
    ##    Initialisation du modèle    ##
    ####################################
    m = Model("Phase 1")

    ####################################
    ##  Initialisation des variables  ##
    ####################################
    # - X_{ijw} = 1 if worker w makes trip from i to j
    #             0 otherwise
    # - There are four cases: 1) i node and j node. 2) i node and j house. 3) i house and j node. 4) i house and j house.
    #
    # - Note that: it does not exist the trip from one node to the same node if node is not a house. And that workers
    #              can not return to another worker's house
    X = {(i, j, w): m.addVar(vtype=GRB.BINARY, name=f"{w}_fait_le_trajet_{i}_à_{j}") for w in Data.Workers for j in Data.Tasks + Data.Pauses[w] for i in Data.Tasks + Data.Pauses[w] if j != i} |\
        {(i, Data.Houses[w], w): m.addVar(vtype=GRB.BINARY, name=f"{w}_fait_le_trajet_{i}_à_{Data.Houses[w]}") for w in Data.Workers for i in Data.Tasks + Data.Pauses[w]} |\
        {(Data.Houses[w], j, w): m.addVar(vtype=GRB.BINARY, name=f"{w}_fait_le_trajet_{Data.Houses[w]}_à_{j}") for w in Data.Workers for j in Data.Tasks + Data.Pauses[w]} |\
        {(Data.Houses[w], Data.Houses[w], w): m.addVar(vtype=GRB.BINARY,
                                                       name=f"{w}_fait_le_trajet_{Data.Houses[w]}_à_{Data.Houses[w]}") for w in Data.Workers}

    # - T_i = time of begining of task i
    T = {i: m.addVar(vtype=GRB.INTEGER, name=f"temps_début_tâche_{i}")
         for i in Data.Tasks} | {p: Data.a[p] for w in Data.Workers for p in Data.Pauses[w]}

    # Variables additionnelles
    # - These variable represent the fact that the worker w do the task or the pause i, they are binary because of the first constraint

    Y = {(i, w): LinExpr(quicksum([X[(i, j, w)] for j in Data.Tasks + Data.Pauses[w] + [
        Data.Houses[w]] if j != i])) for w in Data.Workers for i in Data.Tasks + Data.Pauses[w]}

    Y_bis = {(i, w): LinExpr(quicksum([X[(j, i, w)] for j in Data.Tasks + Data.Pauses[w] + [
        Data.Houses[w]] if j != i])) for w in Data.Workers for i in Data.Tasks + Data.Pauses[w]}

    ####################################
    ## Initialisation des contraintes ##
    ####################################

    # 1- All tasks have to be done once ✅
    ContrDone = {i: m.addConstr(
        quicksum([Y[(i, w)] for w in Data.Workers]) == 1) for i in Data.Tasks}

    # 2 -Workers have to be capable of doing the Tasks ✅
    MS = 10
    ContrSkill = {(i, w, s): m.addConstr(
        Data.r[i][s] <= Data.l[w][s] + MS*(1 - Y[(i, w)])) for i in Data.Tasks for w in Data.Workers for s in Data.Skills}

    # 3- flow restriction ✅
    ContrFlow = {(i, w): m.addConstr(Y[(i, w)] == Y_bis[(i, w)])
                 for w in Data.Workers for i in Data.Tasks + Data.Pauses[w]}

    # 4- Border flow conditions ✅
    ContrBorderL = {w: m.addConstr(quicksum(
        [X[(i, Data.Houses[w], w)] for i in Data.Tasks + Data.Pauses[w] + [Data.Houses[w]]]) == 1) for w in Data.Workers}
    ContrBorderR = {w: m.addConstr(quicksum(
        [X[(Data.Houses[w], j, w)] for j in Data.Tasks + Data.Pauses[w] + [Data.Houses[w]]]) == 1) for w in Data.Workers}

    # 5- Task disponibility ✅
    ContrTaskDisp = {i: m.addConstr(Data.a[i] <= T[i]) for i in Data.Tasks}
    ContrTaskDisp = {i: m.addConstr(
        T[i] + Data.d[i] <= Data.b[i]) for i in Data.Tasks}

    # 6- task sequence is possible ✅
    MT = 24*60
    ContrSeq = {(i, j, w): m.addConstr(T[i] + Data.d[i] + Data.t[i][j] <= T[j] + MT*(1 - X[(i, j, w)]))
                for w in Data.Workers for i in Data.Tasks + Data.Pauses[w] for j in Data.Tasks + Data.Pauses[w] if i != j}

    # 7- Task sequence borders conditions ✅
    ContrBorderSeqDeb = {(Data.Houses[w], j, w): m.addConstr(Data.alpha[w] + Data.t[Data.Houses[w]][j] <=
                                                             T[j] + MT*(1 - X[(Data.Houses[w], j, w)])) for w in Data.Workers for j in Data.Tasks + Data.Pauses[w]}
    ContrBorderSeqFin = {(i, Data.Houses[w], w): m.addConstr(T[i] + Data.d[i] + Data.t[i][Data.Houses[w]] <=
                                                             Data.beta[w] + MT*(1 - X[(i, Data.Houses[w], w)])) for w in Data.Workers for i in Data.Tasks + Data.Pauses[w]}

    # 8- Employees have unavailabilities
    ContrPausDone = {i: m.addConstr(Y[(i, w)] == 1)
                     for w in Data.Workers for i in Data.Pauses[w]}

    ####################################
    ##  Initialisation de l'objectif  ##
    ####################################
    m.setObjective(quicksum([Data.t[i][j]*X[(i, j, w)]
                             for (i, j, w) in X.keys()]), GRB.MINIMIZE)

    m.update()

    return m, X, T, Y


def modele_v1_2(Data):
    # This second model is equivalent to the first the only difference is that the 2 constraint on the requirement of skill level
    #  disappear and we only define the variable Xijw when it is possible for the worker to do the tasks i and j. In other words the
    #  binary variales taking zero as final value obviously because of the skill are not define, which rduce the number of variables in the model
    #  Note: It appears to be a little bit faster in the execution (according to test done on Poland that take approximatively two minutes of computation)

    ####################################
    ##    Initialisation du modèle    ##
    ####################################
    m = Model("Phase 1 rapide")

    ####################################
    ##  Initialisation des variables  ##
    ####################################
    # - X_{ijw} = 1 if worker w makes trip from i to j
    #             0 otherwise
    # - There are four cases: 1) i node and j node. 2) i node and j house. 3) i house and j node. 4) i house and j house.
    #
    # - Note that: it does not exist the trip from one node to the same node if node is not a house. And that workers
    #              can not return to another worker's house
    X = {(i, j, w): m.addVar(vtype=GRB.BINARY, name=f"{w}_fait_le_trajet_{i}_à_{j}") for w in Data.Workers for j in Data.TasksW[w] + Data.Pauses[w] for i in Data.TasksW[w] + Data.Pauses[w] if j != i} |\
        {(i, Data.Houses[w], w): m.addVar(vtype=GRB.BINARY, name=f"{w}_fait_le_trajet_{i}_à_{Data.Houses[w]}") for w in Data.Workers for i in Data.TasksW[w] + Data.Pauses[w]} |\
        {(Data.Houses[w], j, w): m.addVar(vtype=GRB.BINARY, name=f"{w}_fait_le_trajet_{Data.Houses[w]}_à_{j}") for w in Data.Workers for j in Data.TasksW[w] + Data.Pauses[w]} |\
        {(Data.Houses[w], Data.Houses[w], w): m.addVar(vtype=GRB.BINARY,
                                                       name=f"{w}_fait_le_trajet_{Data.Houses[w]}_à_{Data.Houses[w]}") for w in Data.Workers}

    # - T_i = time of begining of task i
    T = {i: m.addVar(vtype=GRB.INTEGER, name=f"temps_début_tâche_{i}")
         for i in Data.Tasks} | {p: Data.a[p] for w in Data.Workers for p in Data.Pauses[w]}

    # Variables additionnelles

    Y = {(i, w): LinExpr(quicksum([X[(i, j, w)] for j in Data.TasksW[w] + Data.Pauses[w] + [Data.Houses[w]] if j != i])) for w in Data.Workers for i in Data.TasksW[w] + Data.Pauses[w]} |\
        {(i, w)         : 0 for w in Data.Workers for i in Data.Tasks if i not in Data.TasksW[w]}
    Y_bis = {(i, w): LinExpr(quicksum([X[(j, i, w)] for j in Data.TasksW[w] + Data.Pauses[w] + [Data.Houses[w]] if j != i])) for w in Data.Workers for i in Data.TasksW[w] + Data.Pauses[w]} |\
            {(i, w)             : 0 for w in Data.Workers for i in Data.Tasks if i not in Data.TasksW[w]}

    ####################################
    ## Initialisation des contraintes ##
    ####################################

    # 1- All tasks have to be done once ✅
    ContrDone = {i: m.addConstr(
        quicksum([Y[(i, w)] for w in Data.Workers]) == 1) for i in Data.Tasks}

    # 2 - have to be capable of doing the Tasks ✅

    # 3- flow restriction ✅
    ContrFlow = {(i, w): m.addConstr(Y[(i, w)] == Y_bis[(i, w)])
                 for w in Data.Workers for i in Data.TasksW[w] + Data.Pauses[w]}

    # 4- Border flow conditions ✅
    ContrBorderL = {w: m.addConstr(quicksum(
        [X[(i, Data.Houses[w], w)] for i in Data.TasksW[w] + Data.Pauses[w] + [Data.Houses[w]]]) == 1) for w in Data.Workers}
    ContrBorderR = {w: m.addConstr(quicksum(
        [X[(Data.Houses[w], j, w)] for j in Data.TasksW[w] + Data.Pauses[w] + [Data.Houses[w]]]) == 1) for w in Data.Workers}

    # 5- Task disponibility ✅
    ContrTaskDisp = {i: m.addConstr(Data.a[i] <= T[i]) for i in Data.Tasks}
    ContrTaskDisp = {i: m.addConstr(
        T[i] + Data.d[i] <= Data.b[i]) for i in Data.Tasks}

    # 6- task sequence is possible ✅
    MT = 24*60
    ContrSeq = {(i, j, w): m.addConstr(T[i] + Data.d[i] + Data.t[i][j] <= T[j] + MT*(1 - X[(i, j, w)]))
                for w in Data.Workers for i in Data.TasksW[w] + Data.Pauses[w] for j in Data.TasksW[w] + Data.Pauses[w] if i != j}

    # 7- Task sequence borders conditions ✅
    ContrBorderSeqDeb = {(Data.Houses[w], j, w): m.addConstr(
        Data.alpha[w] + Data.t[Data.Houses[w]][j] <= T[j] + MT*(1 - X[(Data.Houses[w], j, w)])) for w in Data.Workers for j in Data.TasksW[w]}
    ContrBorderSeqFin = {(i, Data.Houses[w], w): m.addConstr(T[i] + Data.d[i] + Data.t[i][Data.Houses[w]]
                                                             <= Data.beta[w] + MT*(1 - X[(i, Data.Houses[w], w)])) for w in Data.Workers for i in Data.TasksW[w]}

    # 8- Employees have unavailabilities
    ContrPausDone = {i: m.addConstr(Y[(i, w)] == 1)
                     for w in Data.Workers for i in Data.Pauses[w]}

    ####################################
    ##  Initialisation de l'objectif  ##
    ####################################
    m.setObjective(quicksum([Data.t[i][j]*X[(i, j, w)]
                             for (i, j, w) in X.keys()]), GRB.MINIMIZE)
    m.update()

    return m, X, T, Y


def modele_v2_1(Data):
    # This second model is equivalent to the first the only difference is that the 2 constraint on the requirement of skill level
    #  disappear and we only define the variable Xijw when it is possible for the worker to do the tasks i and j. In other words the
    #  binary variales taking zero as final value obviously because of the skill are not define, which rduce the number of variables in the model
    #  Note: It appears to be a little bit faster in the execution (according to test done on Poland that take approximatively two minutes of computation)

    ####################################
    ##    Initialisation du modèle    ##
    ####################################
    m = Model("Phase 2")

    ####################################
    ##  Initialisation des variables  ##
    ####################################
    # - X_{ijw} = 1 if worker w makes trip from i to j
    #             0 otherwise
    # - There are four cases: 1) i node and j node. 2) i node and j house. 3) i house and j node. 4) i house and j house.
    #
    # - Note that: it does not exist the trip from one node to the same node if node is not a house. And that workers
    #              can not return to another worker's house
    X = {(i, j, w): m.addVar(vtype=GRB.BINARY, name=f"{w}_fait_le_trajet_{i}_à_{j}") for w in Data.Workers for j in Data.TasksW[w] + Data.Pauses[w] for i in Data.TasksW[w] + Data.Pauses[w] if j != i} |\
        {(i, Data.Houses[w], w): m.addVar(vtype=GRB.BINARY, name=f"{w}_fait_le_trajet_{i}_à_{Data.Houses[w]}") for w in Data.Workers for i in Data.TasksW[w] + Data.Pauses[w]} |\
        {(Data.Houses[w], j, w): m.addVar(vtype=GRB.BINARY, name=f"{w}_fait_le_trajet_{Data.Houses[w]}_à_{j}") for w in Data.Workers for j in Data.TasksW[w] + Data.Pauses[w]} |\
        {(Data.Houses[w], Data.Houses[w], w): m.addVar(vtype=GRB.BINARY,
                                                       name=f"{w}_fait_le_trajet_{Data.Houses[w]}_à_{Data.Houses[w]}") for w in Data.Workers}

    # - Choice of the moment of the lunch break
    D = {(i, j, w): m.addVar(vtype=GRB.BINARY,
                             name=f"{w}_fait_une_pause_sur_le_trajet_{i}_à_{j}") for (i, j, w) in X.keys()}

    # - Implements the unavailabilities of the tasks
    M1 = {u: m.addVar(vtype=GRB.BINARY, name=f"{i}_commmence_après_le_début_de_l'indisponibilité_{u}")
          for i in Data.Task for u in Data.Unva[i]}
    M2 = {u: m.addVar(vtype=GRB.BINARY, name=f"{i}_commmence_après_la_fin_de_l'indisponibilité_{u}")
          for i in Data.Task for u in Data.Unva[i]}

    # - T_i = time of begining of task i
    T = {i: m.addVar(vtype=GRB.INTEGER, name=f"temps_début_tâche_{i}")
         for i in Data.Tasks} | {p: Data.a[p] for w in Data.Workers for p in Data.Pauses[w]}

    # Variables additionnelles

    Y = {(i, w): LinExpr(quicksum([X[(i, j, w)] for j in Data.TasksW[w] + Data.Pauses[w] + [Data.Houses[w]] if j != i])) for w in Data.Workers for i in Data.TasksW[w] + Data.Pauses[w]} |\
        {(i, w)         : 0 for w in Data.Workers for i in Data.Tasks if i not in Data.TasksW[w]}
    Y_bis = {(i, w): LinExpr(quicksum([X[(j, i, w)] for j in Data.TasksW[w] + Data.Pauses[w] + [Data.Houses[w]] if j != i])) for w in Data.Workers for i in Data.TasksW[w] + Data.Pauses[w]} |\
            {(i, w)
              : 0 for w in Data.Workers for i in Data.Tasks if i not in Data.TasksW[w]}

    ####################################
    ## Initialisation des contraintes ##
    ####################################

    # 1- All tasks have to be done at most once
    ContrDone = {i: m.addConstr(
        quicksum([Y[(i, w)] for w in Data.Workers]) <= 1) for i in Data.Tasks}

    # 3- flow restriction
    ContrFlow = {(i, w): m.addConstr(Y[(i, w)] == Y_bis[(i, w)])
                 for w in Data.Workers for i in Data.TasksW[w] + Data.Pauses[w]}

    # 4- Border flow conditions
    ContrBorderL = {w: m.addConstr(quicksum(
        [X[(i, Data.Houses[w], w)] for i in Data.TasksW[w] + Data.Pauses[w] + [Data.Houses[w]]]) == 1) for w in Data.Workers}
    ContrBorderR = {w: m.addConstr(quicksum(
        [X[(Data.Houses[w], j, w)] for j in Data.TasksW[w] + Data.Pauses[w] + [Data.Houses[w]]]) == 1) for w in Data.Workers}

    # 5- Task disponibility
    ContrTaskDisp = {i: m.addConstr(Data.a[i] <= T[i]) for i in Data.Tasks}
    ContrTaskDisp = {i: m.addConstr(
        T[i] + Data.d[i] <= Data.b[i]) for i in Data.Tasks}

    # 6- task sequence is possible
    MT = 24*60
    ContrSeq = {(i, j, w): m.addConstr(T[i] + Data.d[i] + Data.t[i][j] + 60*D[(i, j, w)] <= T[j] + MT*(1 - X[(i, j, w)]))
                for w in Data.Workers for i in Data.TasksW[w] + Data.Pauses[w] for j in Data.TasksW[w] + Data.Pauses[w] if i != j}

    # 7- Task sequence borders conditions
    ContrBorderSeqDeb = {(Data.Houses[w], j, w): m.addConstr(
        Data.alpha[w] + Data.t[Data.Houses[w]][j] + 60*D[(Data.Houses[w], j, w)] <= T[j] + MT*(1 - X[(Data.Houses[w], j, w)])) for w in Data.Workers for j in Data.TasksW[w]}
    ContrBorderSeqFin = {(i, Data.Houses[w], w): m.addConstr(T[i] + Data.d[i] + Data.t[i][Data.Houses[w]] + 60*D[(i, Data.Houses[w], w)]
                                                             <= Data.beta[w] + MT*(1 - X[(i, Data.Houses[w], w)])) for w in Data.Workers for i in Data.TasksW[w]}

    # 8- Employees have unavailabilities
    ContrPausDone = {i: m.addConstr(Y[(i, w)] == 1)
                     for w in Data.Workers for i in Data.Pauses[w]}

    # Contraintes sur les pauses de midi

    ContrMid1 = {(i, j, w): m.addConstr(
        D[(i, j, w)] <= X[(i, j, w)]) for (i, j, w) in X.keys()}
    ContrMid2 = {(i, j, w): m.addConstr(
        D[(i, j, w)]*13*60 <= T[j]) for (i, j, w) in X.keys() if j in T}
    ContrMid3 = {(i, j, w): m.addConstr(
        MT*(1 - D[(i, j, w)]) + 13*60 >= T[i]) for (i, j, w) in X.keys() if i in T}
    ContrMid4 = {w: m.addConstr(quicksum([D[(i, j, w)] for i in Data.TasksW[w] + Data.Pauses[w] + [
        Data.Houses[w]] for j in Data.TasksW[w] + Data.Pauses[w] + [Data.Houses[w]] if (i, j, w) in X.keys()]) == 1) for w in Data.Workers}

    # Indisponibilité des tâches

    ContrM1_1 = {u: m.addConstr(-MT*(1 - M1[u]) <= T[i] - (
        Data.m[u][0] + Data.d[i])) for i in Data.Task for u in Data.Unva[i]}
    ContrM1_2 = {u: m.addConstr(T[i] - (Data.m[u][0] + Data.d[i]) <= -1 + MT*M1[u])
                 for i in Data.Task for u in Data.Unva[i]}

    ContrM2_1 = {u: m.addConstr(-MT*(1 - M2[u]) <= T[i] - Data.m[u][1])
                 for i in Data.Task for u in Data.Unva[i]}
    ContrM2_2 = {u: m.addConstr(T[i] - Data.m[u][1] <= -1 + MT*M2[u])
                 for i in Data.Task for u in Data.Unva[i]}

    ContrEq = {u: m.addConstr(M1[u] == M2[u])
               for i in Data.Task for u in Data.Unva[i]}

    ####################################
    ##  Initialisation de l'objectif  ##
    ####################################
    DurTask = LinExpr(quicksum([Data.d[i]*Y[(i, w)] for (i, w) in Y.keys()]))
    DurTrav = LinExpr(quicksum([Data.t[i][j]*X[(i, j, w)]
                      for (i, j, w) in X.keys()]))

    m.setObjective(DurTask - 0.0001 * DurTrav, GRB.MAXIMIZE)
    m.update()

    return m, X, T, Y, DurTask, DurTrav
