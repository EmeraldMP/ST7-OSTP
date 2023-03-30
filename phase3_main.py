from phase3.main import process
from data import Data

endroit = "Ukraine"
instance = 3

data = Data(endroit, instance)

process(endroit, instance, data)
