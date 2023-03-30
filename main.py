import data
import result
import modele
import time
from gurobipy import *
from phase3.main import process

# Paramètres de la simulation
endroit = "Ukraine"
instance = 3
méthode = 3
version = 1
opt = False
ajout = ""

# Processus
Data = data.Data(endroit, instance)
Var = modele.Variable(Data, méthode, version)

if opt:
    if méthode == 3:
        tdeb = time.time()
        individu, Best, Av, nb_it = process(Data)
        tproc = time.time() - tdeb
        Result = result.Result(Data, Var, endroit, instance, méthode, version)
        Result.convert_from_individual(individu, tproc, nb_it)
        Result.save_descent(Best, Av)
        Result.save_res()

    else:
        Var.opti()
        Result = result.Result(Data, Var, endroit, instance, méthode, version)
        Result.save_res(ajout=ajout)
else:
    Result = result.Result(Data, Var, endroit, instance, méthode, version)
    Result.load_res(ajout=ajout)

Result.process_result()
Result.save_txt(ajout=ajout)
Result.save_map(ajout=ajout)
Result.resultat_simple()
Result.resultat_timeline(ajout=ajout, show=False)


print(Var.Indicateur)
