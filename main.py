import data
import result
import modele
import time
from gurobipy import *


endroit = "Poland"
instance = 1
méthode = 1

Data = data.Data(endroit, instance)
Var = modele.Variable(Data, modele.modele_v1_2)

print(Var.opti())

Result = result.Result(Data, Var, endroit, instance, méthode)
Result.load_res("_2")
Result.save_txt()
