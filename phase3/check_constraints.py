
def initial_time(fin, task, w, data):
    # print(fin, task)
    begin = max(fin, data.a[task])
    for u in data.Unva[task]:
        if data.C[u][0] <= begin <= data.C[u][1]:
            begin = data.C[u][1]

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

            if begin + data.d[task] > 13*60 and Lunch:
                Lunch = False
                fin += 60
                begin = initial_time(fin, task, w, data)
                if not begin:
                    # print(w)
                    return False

                if begin < 13*60:
                    begin = 13*60
                    for u in data.Unva[task]:
                        if data.C[u][0] <= begin <= data.C[u][1]:
                            begin = data.C[u][1]

                    if begin > data.b[task]:
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

                        if begin < 13*60:
                            begin = 13*60
                            for u in data.Unva[task]:
                                if data.C[u][0] <= begin <= data.C[u][1]:
                                    begin = data.C[u][1]

                            if begin > data.b[task]:
                                return False

                    fin = begin + data.d[task]

                    if fin > data.b[task]:
                        # print(task, "fin")
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
            # print(task, minutes_to_time(fin + data.t[task][p]), minutes_to_time((data.a[p])), Indisp[p])
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
                    # print(task, "retour")
                    return False

    return True


def feasibility_sc(gene, data):
    sc_task = 0
    sc_trav = 0
    eps = 0.001

    for w in data.Workers:
        Tasks = gene[w]
        Lunch = True
        Indisp = {p: True for p in data.Pauses[w]}
        fin = data.alpha[w]
        lastTask = data.Houses[w]

        for task in Tasks:
            sc_task += data.d[task]

            fin += data.t[lastTask][task]
            begin = initial_time(fin, task, w, data)

            if not begin:
                # print(task, "begin")
                return 0, 0

            if begin + data.d[task] > 13*60 and Lunch:
                Lunch = False
                fin += 60
                begin = initial_time(fin, task, w, data)
                if not begin:
                    # print(w)
                    return 0, 0

                if begin < 13*60:
                    begin = 13*60
                    for u in data.Unva[task]:
                        if data.C[u][0] <= begin <= data.C[u][1]:
                            begin = data.C[u][1]

                    if begin > data.b[task]:
                        return 0, 0

            fin = begin + data.d[task]

            if fin > data.b[task]:
                # print(fin, data.b[task])
                # print(task, "fin")
                return 0, 0

            for p in Indisp.keys():
                # print(task, minutes_to_time(fin + data.t[task][p]), minutes_to_time((data.a[p])), Indisp[p])
                if Indisp[p] and fin + data.t[task][p] > data.a[p]:
                    # print("on est bien dedans")
                    sc_trav += data.t[lastTask][p]
                    Indisp[p] = False
                    fin = data.b[p]
                    lastTask = p

                    fin += data.t[lastTask][task]
                    begin = initial_time(fin, task, w, data)

                    if not begin:
                        # print(task, "begin")
                        return 0, 0

                    if not begin + data.d[task] <= 13*60 and Lunch:
                        Lunch = False
                        fin += 60
                        begin = initial_time(fin, task, w, data)
                        if not begin:
                            # print(w)
                            return 0, 0

                        if begin < 13*60:
                            begin = 13*60
                            for u in data.Unva[task]:
                                if data.C[u][0] <= begin <= data.C[u][1]:
                                    begin = data.C[u][1]

                            if begin > data.b[task]:
                                return 0, 0

                    fin = begin + data.d[task]

                    if fin > data.b[task]:
                        # print(task, "fin")
                        return 0, 0

            # print(task, minutes_to_time(begin), minutes_to_time(fin))
            sc_trav += data.t[lastTask][task]
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
            return 0, 0

        for p in Indisp.keys():
            # print(task, minutes_to_time(fin + data.t[task][p]), minutes_to_time((data.a[p])), Indisp[p])
            if Indisp[p]:
                # print("on est bien dedans")
                sc_trav += data.t[lastTask][p]
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
                    # print(task, "retour")
                    return 0, 0

        sc_trav += data.t[lastTask][task]

    return sc_task, -sc_trav
