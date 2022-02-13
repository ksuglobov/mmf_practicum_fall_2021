from functools import wraps


def check_arguments(*arg_types):
    def check_decorator(func):
        @wraps(func)
        def wrapper(*args):
            if len(arg_types) > len(args):
                raise TypeError
            for arg, arg_type in zip(args, arg_types):
                if not isinstance(arg, arg_type):
                    raise TypeError
            return func(*args)
        return wrapper
    return check_decorator
