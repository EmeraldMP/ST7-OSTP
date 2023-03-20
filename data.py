import numpy as np
import pandas as pd

# Définition of the funstions used to read data


class Data:
    def __init__(self, endroit, instance):
        self.Workers, self.Skills, self.Tasks, self.TasksW, self.Houses, \
            self.Pauses, self.Unva, self.l, self.r, self.s, self.t, self.d, self.a, \
            self.b, self.alpha, self.beta, self.nodes, self.m = créer_ensemble(
                endroit, instance)

# function to transform time given in the excel in minute of the day


def time_to_minutes(time_str):
    str_hour = time_str[:-2]
    am_pm = time_str[-2:]
    hour_str, minute = str_hour.split(':')
    hour = int(hour_str)
    if am_pm == 'pm':
        if hour == 12:
            return hour * 60 + int(minute)
        hour += 12
    return hour * 60 + int(minute)

# inverse of the precedent function


def minutes_to_time(total_min):
    hour = int(total_min//60)
    min_ = int(total_min % 60)
    am_pm = 'am'
    if hour > 12:
        am_pm = 'pm'
        hour -= 12
    if hour == 12:
        am_pm = 'pm'
    return f"{hour:02d}:{min_:02d}{am_pm}"


def read_lat_log(df, alias=None):
    nodes = {}
    for name, dic_inf in df.items():
        if alias:
            name = alias[name]
        nodes[name] = (dic_inf["Latitude"], dic_inf["Longitude"])
    return nodes

# Calculate the distance between to point on the hearth given by their coordinates


def haversine(pt1, pt2):
    R = 6371  # radius of the Earth in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [pt1[0], pt1[1], pt2[0], pt2[1]])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * \
        np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def distance_matrix(nodes):
    VELOCITY = 50
    # Create distance matrix with haversine distance
    dist_matrix = {}
    for node_i in nodes.keys():
        dist_matrix[node_i] = {}
        for node_j in nodes.keys():
            dist = haversine(nodes[node_i], nodes[node_j])
            dist_matrix[node_i][node_j] = int(np.ceil((dist / VELOCITY) * 60))
    return dist_matrix

# read the data in the excel to transfom them in panda dataframes


def lecture(endroit, instance):

    path = f"instances\Instance{endroit}V{instance}.xlsx"
    # Employees
    df_Workers = pd.read_excel(path, sheet_name=0, index_col='EmployeeName')
    # Employees unvabilites
    df_Workers_un = pd.read_excel(path, sheet_name=1)

    # Task
    df_Task = pd.read_excel(path, sheet_name=2, index_col='TaskId')

    # Task unvabilites
    df_Task_un = pd.read_excel(path, sheet_name=3)

    return df_Workers, df_Workers_un, df_Task, df_Task_un


def créer_ensemble(endroit, instance):

    df_Workers, df_Workers_un, df_Task, df_Task_un = lecture(endroit, instance)

    # Employees
    dict_Workers = df_Workers.to_dict('index')
    # Employees unvabilites
    dict_Workers_un = df_Workers_un.to_dict('index')
    # Task
    dict_Task = df_Task.to_dict('index')
    # Task unvabilites
    dict_Task_un = df_Task_un.to_dict('index')

    # Sets
    Workers = list(df_Workers.index)
    Skills = list(df_Workers["Skill"].unique())
    Tasks = list(df_Task.index)
    Houses = {w: "HouseOf" + w for w in Workers}

    df_aux = pd.DataFrame()
    for skill in Skills:
        df_aux[skill] = df_Workers.apply(
            lambda x: x['Level'] if x['Skill'] == skill else 0, axis=1)

    l = df_aux.to_dict('index')

    # create de dictionary with the information of pauses
    Pauses = {}
    PauseNode = {}
    a_pause = {}
    b_pause = {}
    d_pause = {}

    for w in Workers:
        w_pauses_df = df_Workers_un[df_Workers_un['EmployeeName'] == w]
        if w_pauses_df.shape[0] == 0:
            Pauses[w] = []
        else:
            pause_list = []
            for i in range(w_pauses_df.shape[0]):
                pause_name = f'Pause{w}{i+1}'
                pause_list.append(pause_name)
                PauseNode[pause_name] = (
                    w_pauses_df.iloc[i, 1], w_pauses_df.iloc[i, 2])
                a_pause[pause_name] = time_to_minutes(w_pauses_df.iloc[i, 3])
                b_pause[pause_name] = time_to_minutes(w_pauses_df.iloc[i, 4])
                d_pause[pause_name] = b_pause[pause_name] - a_pause[pause_name]
            Pauses[w] = pause_list

    # create de dictionary with the information of tasks unavailability
    Unva = {}
    m = {}

    for i in Tasks:
        i_unva_df = df_Task_un[df_Task_un['TaskId'] == i]
        if i_unva_df.shape[0] == 0:
            Unva[i] = []
        else:
            unva_list = []
            for n in range(i_unva_df.shape[0]):
                unva_name = f'Unvalaibility{i}{n+1}'
                unva_list.append(unva_name)

                m[unva_name] = [time_to_minutes(
                    i_unva_df.iloc[n, 1]), time_to_minutes(i_unva_df.iloc[n, 2])]

            Unva[i] = unva_list

    # Opening time for taks i
    a = df_Task.apply(lambda x: int(time_to_minutes(
        x['OpeningTime'])), axis=1).to_dict() | a_pause

    # Closing time for taks i
    b = df_Task.apply(lambda x: int(time_to_minutes(
        x['ClosingTime'])), axis=1).to_dict() | b_pause

    # time worker w start working
    alpha = df_Workers.apply(lambda x: int(
        time_to_minutes(x['WorkingStartTime'])), axis=1).to_dict()

    # time worker w end working
    beta = df_Workers.apply(lambda x: int(
        time_to_minutes(x['WorkingEndTime'])), axis=1).to_dict()

    # Duration of the task i
    d = df_Task['TaskDuration'].to_dict() | d_pause

    # Skill requierd by task i
    s = df_Task['Skill'].to_dict()

    # Level requierd by task i on the skill s
    df_aux = pd.DataFrame()
    for skill in Skills:
        df_aux[(skill)] = df_Task.apply(lambda x: x['Level']
                                        if x['Skill'] == skill else 100, axis=1)

    r = df_aux.to_dict('index')

    # Denfine the nodes
    nodes = read_lat_log(dict_Workers, Houses) | read_lat_log(
        dict_Task) | PauseNode
    nodes
    # Define the time matrix in minutes round ceil between a node (task or worker) with another (task or worker)
    t = distance_matrix(nodes)

    # Define a dictionnary to access the tasks availables for a given worker according to their skills
    Cap = {}
    for w in Workers:
        for i in Tasks:
            for s in Skills:
                if r[i][s] > l[w][s]:
                    Cap |= {(i, w): False}
                    break
            else:
                Cap |= {(i, w): True}
    TasksW = {w: [i for i in Tasks if Cap[(i, w)]] for w in Workers}

    return Workers, Skills, Tasks, TasksW, Houses, Pauses, Unva, l, r, s, t, d, a, b, alpha, beta, nodes, m
