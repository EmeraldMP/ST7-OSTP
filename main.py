import data
import result
import modele
import time
from metaheuristic import feasibility
from gurobipy import *


endroit = "Bordeaux"
instance = 2
méthode = 2
version = 2
opt = False
ajout = ""


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

gene = Result.convert_to_gene()
print(gene)
print(feasibility(gene, Data))

print(Var.Indicateur)
