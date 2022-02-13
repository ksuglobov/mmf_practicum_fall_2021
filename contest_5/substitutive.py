from functools import wraps


def substitutive(func, fixed_args=tuple()):
    n = func.__code__.co_argcount

    @wraps(func)
    def wrapper(*args):
        res_args = fixed_args + args
        if n == len(res_args):
            return func(*res_args)
        elif n > len(res_args):
            return substitutive(func, res_args)
        else:
            raise TypeError
    return wrapper
