import data
import result
import modele
import time
from gurobipy import *


endroit = "Bordeaux"
instance = 2
méthode = 2
version = 1

Data = data.Data(endroit, instance)
Var = modele.Variable(Data, méthode, version)

print(Var.opti())

Result = result.Result(Data, Var, endroit, instance, méthode)
Result.save_res()
Result.process_result()
Result.resultat_simple()
Result.resultat_graph(show=True)
