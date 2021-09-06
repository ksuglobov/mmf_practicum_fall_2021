import math


def find_word_in_circle(circle, word):
    a = len(circle)
    b = len(word)
    if not (a > 0 and b > 0):
        return -1
    circle *= math.ceil((a + b - 1) / a)
    i = circle.find(word)
    if i != -1:
        return (i, 1)
    i = circle[::-1].rfind(word, 0, a + b - 1)
    if i != -1:
        return (a - i - 1, -1)
    return -1
