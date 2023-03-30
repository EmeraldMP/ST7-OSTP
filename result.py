from data import minutes_to_time
from phase3.check_constraints import initial_time, feasibility_sc
import pickle
import folium
from gurobipy import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
import pandas as pd


def minutes_to_time_pd(minutes_since_midnight):
    minutes_since_midnight = int(minutes_since_midnight)

    # create timedelta object from minutes since midnight
    timedelta_obj = pd.Timedelta(minutes=minutes_since_midnight)

    # create Timestamp object for today's date at the given time
    timestamp_obj = pd.Timestamp.today().normalize() + timedelta_obj

    return timestamp_obj


class Result:

    def __init__(self, Data, Var, endroit, instance, méthode, version):
        self.Data = Data
        self.Var = Var
        self.endroit = endroit
        self.instance = instance
        self.méthode = méthode
        self.version = version

    def resultat_timeline(self, ajout="", show=True):

        data = []

        n = 0
        for w in self.Data.Workers:
            for (w_, t_) in self.Var.Indicateur["moments repas"]:
                if w_ == w:
                    data.append({'Task': 'Lunch', 'Start': minutes_to_time_pd(t_),
                                 'Finish': minutes_to_time_pd(t_ + 60), 'Worker': w, 'shape': 2})
                    break
            prec = self.Data.Houses[w]
            for i in self.all_rutes[w]:

                if i[3] - self.Data.t[prec][i[0]] < t_ + 60 and i[3] > t_ + 60:
                    data.append({'Task': 'Trajectory', 'Start': minutes_to_time_pd(i[3] - self.Data.t[prec][i[0]] - 60),
                                 'Finish': minutes_to_time_pd(t_), 'Worker': w, 'shape': 1})
                    data.append({'Task': 'Trajectory', 'Start': minutes_to_time_pd(t_ + 60),
                                 'Finish': minutes_to_time_pd(i[3]), 'Worker': w, 'shape': 1})
                elif t_ <= i[3] - self.Data.t[prec][i[0]] < t_ + 60:
                    data.append({'Task': 'Trajectory', 'Start': minutes_to_time_pd(i[3] - self.Data.t[prec][i[0]] - 60),
                                 'Finish': minutes_to_time_pd(t_), 'Worker': w, 'shape': 1})
                else:
                    data.append({'Task': 'Trajectory', 'Start': minutes_to_time_pd(i[3] - self.Data.t[prec][i[0]]),
                                 'Finish': minutes_to_time_pd(i[3]), 'Worker': w, 'shape': 1})

                data.append({'Task': i[0], 'Start': minutes_to_time_pd(i[3]),
                             'Finish': minutes_to_time_pd(i[3] + self.Data.d[i[0]]), 'Worker': w, 'shape': 1})
                prec = i[0]

            data.append({'Task': 'Trajectory', 'Start': minutes_to_time_pd(i[3] + self.Data.d[i[0]]),
                         'Finish': minutes_to_time_pd(i[3] + self.Data.d[i[0]] + self.Data.t[i[0]][self.Data.Houses[w]]),
                         'Worker': w, 'shape': 0})

        len_color = len(data)//10 + 1

        fig = px.timeline(data, x_start="Start", x_end="Finish", y="Worker", color="Task", pattern_shape="Task", pattern_shape_sequence=['\\'] + ['' for i in range(len(data))],
                          color_discrete_sequence=['#3B8BDA'] + ['#DBDBDB'] + px.colors.qualitative.Plotly*len_color)

        # Update the figure layout and show the plot
        fig.update_layout(title='Daily Task Timeline',
                          xaxis=dict(tickformat='%H:%M:%S'),
                          height=400)
        fig.write_html(
            f"solutions/Timeline{self.endroit}V{self.instance}ByM{self.méthode}{self.version}{ajout}.html")

        if show:
            fig.show()

    def process_result(self):
        # Translate the informations held in the variables so that we can visualise the results

        self.txt = "taskId;performed,employeeName,startTime;"
        # all_rutes = {worker:['HouseOf'+str(worker)] for worker in Workers}
        self.all_rutes = {worker: [] for worker in self.Data.Workers}
        for tasks in self.Var.T.keys():
            self.result = [tasks]
            try:
                if self.Var.Y[tasks][0] == 0:
                    self.result.append(0)
                    self.result += ['', '']
                else:
                    self.result.append(1)
                    for (i, j, w) in self.Var.X.keys():
                        if i == tasks:
                            if self.Var.X[(i, j, w)][0]:
                                self.result.append(w)
                                self.all_rutes[w].append(self.result)
                    self.result.append(int(self.Var.T[tasks][0]))
                self.txt += "\n"
                for el in self.result:
                    self.txt += str(el)+";"
            except:
                if self.Var.Y[tasks] == 0:
                    self.result.append(0)
                    self.result += ['', '']
                else:
                    self.result.append(1)
                    for (i, j, w) in self.Var.X.keys():
                        if i == tasks:
                            if self.Var.X[(i, j, w)][0]:
                                self.result.append(w)
                                self.all_rutes[w].append(self.result)
                    self.result.append(self.Var.T[tasks])

        if self.Var.Indicateur["moments repas"]:
            self.txt += "\n\n"
            self.txt += "employeeName,lunchBreakStartTime;"
            for (w, t) in self.Var.Indicateur["moments repas"]:
                self.txt += "\n"
                self.txt += str(w) + ";" + str(t) + ";"

        self.all_rutes
        # sort the routes in order of time for each worker
        for w in self.all_rutes.keys():
            self.all_rutes[w] = sorted(self.all_rutes[w], key=lambda x: x[3])

        self.routes_lat_log = {worker: [] for worker in self.Data.Workers}
        for w in self.Data.Workers:
            self.routes_lat_log[w].append(self.Data.nodes[self.Data.Houses[w]])
            self.routes_lat_log[w] += [self.Data.nodes[task[0]]
                                       for task in self.all_rutes[w]]
            self.routes_lat_log[w].append(self.Data.nodes[self.Data.Houses[w]])

    def save_map(self, ajout=""):
        # Represent the solution in a map, to easily see the spacial repartition of the workers tasks

        m = folium.Map(
            location=self.Data.nodes[self.Data.Houses[self.Data.Workers[0]]], zoom_start=10)

        colors = ['blue', 'red', 'green', 'orange', 'pink', 'cadetblue', 'black', 'darkblue', 'darkgreen',
                  'darkred', 'gray', 'lightblue', 'lightgreen', 'purple']*((len(self.Data.Workers)//14) + 1)

        for j in range(len(self.Data.Workers)):
            color = colors[j]
            route = self.routes_lat_log[self.Data.Workers[j]]
            for i in range(1, len(route)-1):
                folium.Marker(location=route[i], icon=folium.Icon(
                    icon='map-marker', color=color)).add_to(m)
                if i != 0 and i != len(route)-1:
                    if self.all_rutes[self.Data.Workers[j]][i-1][0][0:5] == 'Pause':
                        folium.Marker(location=route[i], icon=folium.Icon(
                            icon='star', color=color)).add_to(m)
            folium.Marker(location=route[0], icon=folium.Icon(
                icon='home', color=color)).add_to(m)
            folium.PolyLine(locations=route, color=color, weight=4).add_to(m)

        # show the map
        m.save(
            f"solutions/map{self.endroit}V{self.instance}ByM{self.méthode}{self.version}{ajout}.html")

    def save_txt(self, ajout=""):
        # Generate the result
        with open(f"solutions\Solution{self.endroit}V{self.instance}ByM{self.méthode}{self.version}{ajout}.txt", "w") as file:
            file.write(self.txt)

    def resultat_simple(self):
        for (i, j, w) in self.Var.X.keys():
            if self.Var.X[(i, j, w)][0]:
                if j in self.Data.Tasks:
                    print(self.Var.X[(i, j, w)][1], "à",
                          minutes_to_time(self.Var.T[j][0]))
                else:
                    print(self.Var.X[(i, j, w)][1])

    def resultat_graph(self, ajout="", show=False):
        # Represent the solution in a time graph, to easily see the period of work, travel and the respect of availabilities

        col = list(mcolors.TABLEAU_COLORS)

        n = 0
        for w in self.Data.Workers:
            n += 1
            ax = plt.subplot(len(self.Data.Workers), 1, n)

            ax.plot([self.Data.alpha[w], self.Data.alpha[w]],
                    [-1, 2], color='black', zorder=11)
            ax.plot([self.Data.beta[w], self.Data.beta[w]],
                    [-1, 2], color='black', zorder=11)

            ax.set_title(w)

            prec = self.Data.Houses[w]
            p = 0
            for i in self.all_rutes[w]:
                p = (p+1) % 10
                ax.plot([i[3] - self.Data.t[prec][i[0]], i[3]],
                        [1, 1], color='black')
                ax.plot([i[3], i[3] + self.Data.d[i[0]]],
                        [0, 0], color=col[p], label=i[0])
                ax.plot([self.Data.a[i[0]], self.Data.a[i[0]]],
                        [-1, 3 + p], color=col[p], zorder=11 - p)
                ax.plot([self.Data.b[i[0]], self.Data.b[i[0]]],
                        [-1, 3 + p], color=col[p], zorder=11 - p)
                prec = i[0]

            ax.plot([i[3] + self.Data.d[i[0]],  i[3] + self.Data.d[i[0]] + self.Data.t[i[0]]
                    [self.Data.Houses[w]]], [1, 1], color='black')
            ax.set_ylim(-15, 15)
            ax.set_xlim(0, 1440)
            ax.legend()

        plt.savefig(
            f"solutions\Graph{self.endroit}V{self.instance}ByM{self.méthode}{self.version}{ajout}.png")
        if show:
            plt.show()

    def save_res(self, ajout=""):
        # Save the value of the variable for the solution find
        with open(f"solutions\X{self.endroit}V{self.instance}ByM{self.méthode}{self.version}{ajout}.pkl", "wb") as tf:
            pickle.dump(self.Var.X, tf)
        with open(f"solutions\T{self.endroit}V{self.instance}ByM{self.méthode}{self.version}{ajout}.pkl", "wb") as tf:
            pickle.dump(self.Var.T, tf)
        with open(f"solutions\Y{self.endroit}V{self.instance}ByM{self.méthode}{self.version}{ajout}.pkl", "wb") as tf:
            pickle.dump(self.Var.Y, tf)
        with open(f"solutions\Indic{self.endroit}V{self.instance}ByM{self.méthode}{self.version}{ajout}.pkl", "wb") as tf:
            pickle.dump(self.Var.Indicateur, tf)

    def load_res(self, ajout=""):
        # Load the value of the variable for the solution find before
        with open(f"solutions\X{self.endroit}V{self.instance}ByM{self.méthode}{self.version}{ajout}.pkl", "rb") as tf:
            self.Var.X = pickle.load(tf)
        with open(f"solutions\T{self.endroit}V{self.instance}ByM{self.méthode}{self.version}{ajout}.pkl", "rb") as tf:
            self.Var.T = pickle.load(tf)
        try:
            with open(f"solutions\Y{self.endroit}V{self.instance}ByM{self.méthode}{self.version}{ajout}.pkl", "rb") as tf:
                self.Var.Y = pickle.load(tf)
        except:
            self.Var.Y = {i: [sum([self.Var.X[(i, j, w)][0] for w in self.Data.Workers for j in self.Data.Tasks +
                                   self.Data.Pauses[w] + [self.Data.Houses[w]] if (i, j, w) in self.Var.X])] for i in self.Var.T.keys() if type(self.Var.T[i]) != int} |\
                {i: sum([self.Var.X[(i, j, w)][0] for w in self.Data.Workers for j in self.Data.Tasks +
                         self.Data.Pauses[w] + [self.Data.Houses[w]] if (i, j, w) in self.Var.X]) for i in self.Var.T.keys() if type(self.Var.T[i]) == int}
            print(self.Var.Y)
            with open(f"solutions\Y{self.endroit}V{self.instance}ByM{self.méthode}{self.version}{ajout}.pkl", "wb") as tf:
                pickle.dump(self.Var.Y, tf)

        try:
            with open(f"solutions\Indic{self.endroit}V{self.instance}ByM{self.méthode}{self.version}{ajout}.pkl", "rb") as tf:
                self.Var.Indicateur = pickle.load(tf)

        except:
            pass

    def convert_to_individual(self):
        gene = {w: [i[0] for i in self.all_rutes[w] if i[0]
                    not in self.Data.Pauses[w]] for w in self.Data.Workers}
        return gene

    def convert_from_individual(self, gene, tproc, nb_it):

        X = {(i, j, w): [0, f"{w}_fait_le_trajet_{i}_à_{j}"] for w in self.Data.Workers for j in self.Data.TasksW[w] + self.Data.Pauses[w] for i in self.Data.TasksW[w] + self.Data.Pauses[w] if j != i} |\
            {(i, self.Data.Houses[w], w): [0, f"{w}_fait_le_trajet_{i}_à_{self.Data.Houses[w]}"] for w in self.Data.Workers for i in self.Data.TasksW[w] + self.Data.Pauses[w]} |\
            {(self.Data.Houses[w], j, w): [0, f"{w}_fait_le_trajet_{self.Data.Houses[w]}_à_{j}"] for w in self.Data.Workers for j in self.Data.TasksW[w] + self.Data.Pauses[w]} |\
            {(self.Data.Houses[w], self.Data.Houses[w], w): [
                0, f"{w}_fait_le_trajet_{self.Data.Houses[w]}_à_{self.Data.Houses[w]}"] for w in self.Data.Workers}

        rep = []

        # - T_i = time of begining of task i
        T = {i: [0, f"temps_début_tâche_{i}"] for i in self.Data.Tasks} | {
            p: self.Data.a[p] for w in self.Data.Workers for p in self.Data.Pauses[w]}

        for w in self.Data.Workers:
            Tasks = gene[w]
            Lunch = True
            Indisp = {p: True for p in self.Data.Pauses[w]}
            fin = self.Data.alpha[w]
            lastTask = self.Data.Houses[w]
            for task in Tasks:

                fin += self.Data.t[lastTask][task]
                begin = initial_time(fin, task, w, self.Data)

                if begin + self.Data.d[task] > 13*60 and Lunch:
                    Lunch = False
                    fin += 60
                    if lastTask == self.Data.Houses[w]:
                        rep.append(
                            [w, int(max(720, self.Data.alpha[w]))])
                    else:
                        rep.append(
                            [w, int(max(720, T[lastTask][0] + self.Data.d[lastTask]))])
                    begin = initial_time(fin, task, w, self.Data)

                fin = begin + self.Data.d[task]

                for p in Indisp.keys():

                    if Indisp[p] and fin + self.Data.t[task][p] > self.Data.a[p]:

                        X[(lastTask, p, w)][0] = 1
                        Indisp[p] = False
                        fin = self.Data.b[p]
                        lastTask = p

                        fin += self.Data.t[lastTask][task]
                        begin = initial_time(fin, task, w, self.Data)

                        if not begin + self.Data.d[task] <= 13*60 and Lunch:
                            Lunch = False
                            fin += 60
                            if lastTask == self.Data.Houses[w]:
                                rep.append(
                                    [w, int(max(720, self.Data.alpha[w]))])
                            else:
                                rep.append(
                                    [w, int(max(720, T[lastTask] + self.Data.d[lastTask]))])

                            begin = initial_time(fin, task, w, self.Data)

                        fin = begin + self.Data.d[task]

                # print(task, minutes_to_time(begin), minutes_to_time(fin))
                X[(lastTask, task, w)][0] = 1
                T[task][0] = begin

                lastTask = task

            task = self.Data.Houses[w]

            fin += self.Data.t[lastTask][task]
            begin = max(fin, self.Data.beta[w])

            if not begin <= 13*60 and Lunch:
                Lunch = False
                fin += 60
                if lastTask == self.Data.Houses[w]:
                    rep.append([w, int(max(720, self.Data.alpha[w]))])
                else:
                    rep.append(
                        [w, int(max(720, T[lastTask][0] + self.Data.d[lastTask]))])
                begin = max(fin, self.Data.beta[w])

                if begin < 13*60:
                    begin = 13*60
                    for u in self.Data.Unva[task]:
                        if self.Data.C[u][0] <= begin <= self.Data.C[u][1]:
                            begin = self.Data.C[u][1]

                    if begin > self.Data.b[task]:
                        return False

            fin = begin

            for p in Indisp.keys():
                # print(task, minutes_to_time(fin + self.Data.t[task][p]), minutes_to_time((self.Data.a[p])), Indisp[p])
                if Indisp[p]:

                    X[(lastTask, p, w)][0] = 1
                    Indisp[p] = False
                    fin = self.Data.b[p]
                    lastTask = p

                    fin += self.Data.t[lastTask][task]
                    begin = max(fin, self.Data.beta[w])

                    if not begin <= 13*60 and Lunch:
                        Lunch = False
                        fin += 60
                        if lastTask == self.Data.Houses[w]:
                            rep.append([w, int(max(720, self.Data.alpha[w]))])
                        else:
                            rep.append(
                                [w, int(max(720, T[lastTask] + self.Data.d[lastTask]))])
                        begin = max(fin, self.Data.beta[w])

                        if begin < 13*60:
                            begin = 13*60
                            for u in self.Data.Unva[task]:
                                if self.Data.C[u][0] <= begin <= self.Data.C[u][1]:
                                    begin = self.Data.C[u][1]

                            if begin > self.Data.b[task]:
                                return False

                    fin = begin

            X[(lastTask, task, w)][0] = 1

        self.Var.X = X
        self.Var.T = T

        self.Var.Y = {i: [sum([self.Var.X[(i, j, w)][0] for w in self.Data.Workers for j in self.Data.Tasks +
                               self.Data.Pauses[w] + [self.Data.Houses[w]] if (i, j, w) in self.Var.X])] for i in self.Var.T.keys() if type(self.Var.T[i]) != int} |\
            {i: sum([self.Var.X[(i, j, w)][0] for w in self.Data.Workers for j in self.Data.Tasks +
                     self.Data.Pauses[w] + [self.Data.Houses[w]] if (i, j, w) in self.Var.X]) for i in self.Var.T.keys() if type(self.Var.T[i]) == int}

        dw, dt = feasibility_sc(gene, self.Data)

        self.Var.Indicateur = {"temps execution": tproc, "tdurée tâches": dw,
                               "tdurée trajet": -dt, "iterations": nb_it,  "moments repas": rep}

    def save_descent(self, Best, Av, ajout="", show=False):
        # Represent the solution in a time graph, to easily see the period of work, travel and the respect of availabilities
        ax = plt.subplot()
        ax.set_title("Amelioration of the solution on the iterations")

        ax.scatter(range(len(Best)), Best, color='red',
                   label="Score of the best solution")
        ax.scatter(range(len(Av)), Av, color='blue',
                   label="Average score over the generation")

        ax.legend()

        plt.savefig(
            f"solutions\Courbe{self.endroit}V{self.instance}ByM{self.méthode}{self.version}{ajout}.png")
        if show:
            plt.show()
