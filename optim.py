import pyomo.environ as pyo

x = [[1, 1], [2, 1], [3, 1], [4, 1], [5, 1], [6, 1]]
# x = [1, 2, 3, 4, 5, 6]
y = [-1, -1, -1, 1, 1, 1]
data_num = len(x)
feature_num = len(x[0])

# Define the model
model = pyo.ConcreteModel()

# Define Sets
model.data_index    = pyo.Set(initialize=[i for i in range(data_num)])
model.feature_index = pyo.Set(initialize=[i for i in range(feature_num)])

# Define Parameters
model.data_num      = pyo.Param(domain=pyo.NonNegativeIntegers, initialize=data_num)
model.feature_num   = pyo.Param(domain=pyo.NonNegativeIntegers, initialize=feature_num)
model.C             = pyo.Param(domain=pyo.Reals, default=1e9)

# Define decision variables
model.u     = pyo.Var(model.feature_index, domain=pyo.NonNegativeReals)
model.w     = pyo.Var(model.feature_index, domain=pyo.Reals)
model.b     = pyo.Var(domain=pyo.Reals)
model.relax = pyo.Var(model.data_index, domain=pyo.NonNegativeReals)

# Define the objective
def objective(model: pyo.ConcreteModel):
    return pyo.summation(model.u) + model.C * pyo.summation(model.relax)

model.obj   = pyo.Objective(rule=objective, sense=pyo.minimize)

# Define Constraints
for i in range(feature_num):
    setattr(model, f'abs_constraint_1_{i}', pyo.Constraint(expr=model.u[i] >= model.w[i]))
    setattr(model, f'abs_constraint_2_{i}', pyo.Constraint(expr=model.u[i] >= -model.w[i]))

for i in range(data_num):
    setattr(model, f'constraint_{i+1}', pyo.Constraint(expr=y[i] * (pyo.summation(x[i], model.w) + model.b) + model.relax[i] >= 1))

# Determine LP solver
solver_name = 'glpk'
opt = pyo.SolverFactory(solver_name)
result=opt.solve(model)

# Show output
model.display()
model.pprint()