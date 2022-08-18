def numeric_derivative(function=None,
                       variable_name=None,
                       param_dict=None):
    EPSILON = 1e-4
    val = param_dict[variable_name]
    incremented_dict = {key: value for key, value in param_dict.items()}
    incremented_dict[variable_name] = val + EPSILON
    decremented_dict = {key: value for key, value in param_dict.items()}
    decremented_dict[variable_name] = val - EPSILON
    difference = function(**incremented_dict) - function(**decremented_dict)
    return difference / (2 * EPSILON)

def numeric_gradient(neural_net=None):
    pass


