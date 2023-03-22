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
Var2 = modele.Variable(Data, 3, 1)
Var = modele.Variable(Data, méthode, version)

if opt:
    Var.opti()
    Result = result.Result(Data, Var, endroit, instance, méthode, version)
    Result.save_res(ajout=ajout)
else:
    Result = result.Result(Data, Var, endroit, instance, méthode, version)
    Result.load_res(ajout=ajout)

Result.process_result()
# Result.save_txt(ajout=ajout)
# Result.save_map(ajout=ajout)
Result.resultat_simple()
# Result.resultat_timeline(ajout=ajout, show=False)
print(Var.Indicateur)

gene = Result.convert_to_gene()
print(gene)

Result2 = result.Result(Data, Var2, endroit, instance, 3, 1)
Result2.convert_from_gene(gene)
Result2.process_result()
Result2.save_txt(ajout=ajout)
Result2.save_map(ajout=ajout)
Result2.resultat_simple()
Result2.resultat_timeline(ajout=ajout, show=False)


print(Var2.Indicateur)

# gene = {'Tom': ['T1', 'T2', 'T5', 'T6', 'T7', 'T9', 'T4', 'T3', 'T8']}
# gene2 = {'Tom': ['T1', 'T2', 'T6', 'T8', 'T7', 'T5', 'T9', 'T4', 'T3']}
# print(feasibility(gene, data.Data("GuineaGolf", 1)))
