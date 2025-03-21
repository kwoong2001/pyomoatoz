# Pyomo Tutorial V1 (2025.03.19) 

""" Call modules """
import pyomo.environ as pyo 
import math

""" INPUTS """
solver = 'ipopt' # Select solver

""" Create model for testing """
# Create the model
model = pyo.ConcreteModel()

""" Create Variables """
# Variables
model.x = pyo.Var(within=pyo.NonNegativeReals,initialize=10.0)
model.y = pyo.Var(within=pyo.NonNegativeReals,initialize=math.pi/2) # 

""" Create Constraints """
#constraints
def xregion1(model):
    return model.x>=10 # return model.x==10
model.Boundx1 = pyo.Constraint(rule=xregion1)

def xregion2(model):
    return model.x<=20 # return model.x==20
model.Boundx2 = pyo.Constraint(rule=xregion2)

def yregion1(model):
    return model.y<=math.pi/2*5
model.Boundy1 = pyo.Constraint(rule=yregion1)

def yregion2(model):
    return model.y>=math.pi/2*2
model.Boundy2 = pyo.Constraint(rule=yregion2)

""" Create Objective function """
# Objective function
def obj_rule(model):                                        
    return  model.x + pyo.sin(model.y) # x + sin(y)
model.obj = pyo.Objective(rule=obj_rule,sense=pyo.minimize) # if you want to maximize objective, use 'sense=pyo.maximize'

""" Solve """
optimizer = pyo.SolverFactory(solver)
Problem = optimizer.solve(model,tee=True)                             
                                                     
""" Print Solution """
print('x= ' + str(model.x.value))
print('y= ' + str(model.y.value))
print('Objective function= ' + str(model.obj.expr()))
