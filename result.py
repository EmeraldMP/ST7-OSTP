from data import minutes_to_time
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
            prec = self.Data.Houses[w]
            for i in self.all_rutes[w]:
                data.append({'Task': 'Trajectory', 'Start': minutes_to_time_pd(i[3] - self.Data.t[prec][i[0]]),
                             'Finish': minutes_to_time_pd(i[3]), 'Worker': w})
                data.append({'Task': i[0], 'Start': minutes_to_time_pd(i[3]),
                             'Finish': minutes_to_time_pd(i[3] + self.Data.d[i[0]]), 'Worker': w})
                prec = i[0]
            data.append({'Task': 'Trajectory', 'Start': minutes_to_time_pd(i[3] + self.Data.d[i[0]]),
                         'Finish': minutes_to_time_pd(i[3] + self.Data.d[i[0]] + self.Data.t[i[0]][self.Data.Houses[w]]),
                         'Worker': w})

        len_color = len(data)//10 + 1

        fig = px.timeline(data, x_start="Start", x_end="Finish", y="Worker", color="Task",
                          color_discrete_sequence=['#DBDBDB'] + px.colors.qualitative.Plotly*len_color)

        # Update the figure layout and show the plot
        fig.update_layout(title='Daily Task Timeline',
                          xaxis=dict(tickformat='%H:%M:%S'),
                          height=400)
        fig.write_html(f"solutions/Timeline{self.endroit}V{self.instance}ByM{self.méthode}{self.version}{ajout}.html")

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

        colors = ['blue', 'red', 'green', 'orange', 'pink', 'cadetblue', 'black', 'darkblue', 'darkgreen', 'darkpurple',
                  'darkred', 'gray', 'lightblue', 'lightgray', 'lightgreen', 'lightred', 'purple', 'white']

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
            cygdjwhjgsk
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
