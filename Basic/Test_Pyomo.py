# Pyomo Tutorial V1 (2025.03.19) 

""" Call modules """
import pyomo.environ as pyo 

""" INPUTS """
solver = 'ipopt' # Select solver

""" Create model for testing """
# Create the model
model = pyo.ConcreteModel()

""" Create Variables """
# Variables
model.x = pyo.Var(within=pyo.NonNegativeReals,initialize=9.01) # 10<x<inf
model.y = pyo.Var(within=pyo.NonNegativeReals,initialize=1.57*2) # 3/2pi

""" Create Constraints """
#constraints
def xregion(model):
    return model.x>=10.01
model.Boundx = pyo.Constraint(rule=xregion)

def yregion1(model):
    return model.y<=1.57*5
model.Boundy1 = pyo.Constraint(rule=yregion1)

def yregion2(model):
    return model.y>=1.57*2
model.Boundy2 = pyo.Constraint(rule=yregion2)

""" Create Objective function """
# Objective function
def obj_rule(model):                                        
    return  model.x + pyo.sin(model.y) # x + sin(y)
model.obj = pyo.Objective(rule=obj_rule)

""" Solve """
optimizer = pyo.SolverFactory(solver)
Problem = optimizer.solve(model,tee=True)                             
                                                     
""" Print Solution """
print('x= ' + str(model.x.value))
print('y= ' + str(model.y.value))
print('Objective function= ' + str(model.obj.expr()))
