import data
import result
import modele
import time
from metaheuristic import feasibility
from gurobipy import *


endroit = "Poland"
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
gene = {'Irena': ['T5', 'T14', 'T8', 'T15', 'T1', 'T2', 'T7'], 'Karol': [
    'T3', 'T9', 'T12', 'T11', 'T4', 'T17', 'T18', 'T19', 'T6', 'T13']}
print(feasibility(gene, Data))

print(Var.Indicateur)
