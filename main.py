import data
import result
import modele
import time
from gurobipy import *


endroit = "Finland"
instance = 1
méthode = 1

Data = data.Data(endroit, instance)
Var = modele.Variable(Data, modele.modele_v1_2)

t1 = time.time()
Var.opti()
print(time.time() - t1)


Result = result.Result(Data, Var, endroit, instance, méthode)
Result.save_res()
Result.process_result()

Result.save_map()
Result.save_txt()
Result.resultat_graph()
