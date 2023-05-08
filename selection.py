import numpy as np
from numba import njit


@njit(cache=True)
def selAutomaticEpsilonLexicaseNumba(case_values, fit_weights, k):
    selected_individuals = []
    avg_cases = 0

    for i in range(k):
        candidates = list(range(len(case_values)))
        cases = np.arange(len(case_values[0]))
        np.random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            errors_for_this_case = np.array([case_values[x][cases[0]] for x in candidates])
            median_val = np.median(errors_for_this_case)
            median_absolute_deviation = np.median(np.array([abs(x - median_val) for x in errors_for_this_case]))
            if fit_weights > 0:
                best_val_for_case = np.max(errors_for_this_case)
                min_val_to_survive = best_val_for_case - median_absolute_deviation
                candidates = list([x for x in candidates if case_values[x][cases[0]] >= min_val_to_survive])
            else:
                best_val_for_case = np.min(errors_for_this_case)
                max_val_to_survive = best_val_for_case + median_absolute_deviation
                candidates = list([x for x in candidates if case_values[x][cases[0]] <= max_val_to_survive])
            cases = np.delete(cases, 0)
        avg_cases = (avg_cases * i + (len(case_values[0]) - len(cases))) / (i + 1)
        selected_individuals.append(np.random.choice(np.array(candidates)))
    return selected_individuals, avg_cases


def selAutomaticEpsilonLexicaseFast(individuals, k, return_avg_cases=False):
    fit_weights = individuals[0].fitness.weights[0]
    case_values = np.array([ind.case_values for ind in individuals])
    index, avg_cases = selAutomaticEpsilonLexicaseNumba(case_values, fit_weights, k)
    selected_individuals = [individuals[i] for i in index]
    if return_avg_cases:
        return selected_individuals, avg_cases
    else:
        return selected_individuals


def doubleLexicase(pop, k):
    lexicase_round = 10
    size_selection = 'Roulette'

    chosen = []
    for _ in range(k):
        candidates = selAutomaticEpsilonLexicaseFast(pop, lexicase_round)
        size_arr = np.array([len(x) for x in candidates])
        if size_selection == 'Roulette':
            size_arr = np.max(size_arr) + np.min(size_arr) - size_arr
            index = np.random.choice([i for i in range(0, len(size_arr))], p=size_arr / size_arr.sum())
        elif size_selection == 'Min':
            index = np.argmin(size_arr)
        else:
            raise Exception('Unknown Size Selection Operator!')
        chosen.append(candidates[index])
    return chosen
