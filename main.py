import data
import result
import modele
import time
from gurobipy import *


endroit = "Poland"
instance = 2
méthode = 2
version = 2
opt = True
ajout = "_test"


Data = data.Data(endroit, instance)
Var = modele.Variable(Data, méthode, version)

if opt:
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
