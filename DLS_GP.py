import operator
import random

import numpy as np
from deap import base
from deap import creator
from deap import gp
from deap import tools
from deap.algorithms import eaSimple
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from selection import doubleLexicase, selAutomaticEpsilonLexicaseFast


# Define new functions
def protectedDiv(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


# Load the diabetes dataset
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=0)

# Standardize the input features and target values based on the training data
scaler_X = StandardScaler().fit(X_train)
X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_y = StandardScaler().fit(y_train.reshape(-1, 1))
y_train = scaler_y.transform(y_train.reshape(-1, 1)).ravel()
y_test = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

pset = gp.PrimitiveSet("MAIN", X_train.shape[1])
pset.addPrimitive(np.add, 2, name="vadd")
pset.addPrimitive(np.subtract, 2, name="vsub")
pset.addPrimitive(np.multiply, 2, name="vmul")
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(np.negative, 1, name="vneg")
pset.addPrimitive(np.cos, 1, name="vcos")
pset.addPrimitive(np.sin, 1, name="vsin")
pset.addEphemeralConstant("rand101", lambda: random.uniform(-1, 1))
for i in range(X_train.shape[1]):
    pset.renameArguments(**{f"ARG{i}": f"x{i}"})

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the sum of squared difference between the expression
    predicted_values = func(*X_train.T)
    individual.case_values = (predicted_values - y_train) ** 2
    diff = np.mean((predicted_values - y_train) ** 2)
    return diff,


toolbox.register("evaluate", evalSymbReg)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))


def main():
    random.seed(0)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    eaSimple(pop, toolbox, 0.9, 0.1, 50, stats, halloffame=hof)

    # Perform ensemble prediction on test data using individuals in hof
    func_hof = toolbox.compile(expr=hof[0])
    y_pred_list_hof = func_hof(*X_test.T)
    mse_test_hof = np.mean((y_pred_list_hof - y_test) ** 2)
    print("Mean squared error on test data for the best model:", mse_test_hof)
    print('Tree size', len(hof[0]))
    return pop, stats, hof


def no_selection(x, k):
    return x[-k:]


if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)

    # Mean squared error on test data for the best model: 0.648794571670637
    # Tree size 28
    toolbox.register("select", selAutomaticEpsilonLexicaseFast)
    main()

    # Mean squared error on test data for the best model: 0.5864317222171147
    # Tree size 11
    toolbox.register("select", doubleLexicase)
    main()
