def linearize(obj):
    try:
        for i in obj:
            if i is obj:
                yield i
            else:
                yield from linearize(i)
    except TypeError:
        yield obj
