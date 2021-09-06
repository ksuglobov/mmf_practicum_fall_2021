def check_first_sentence_is_second(str1, str2):
    d1 = {}
    list1 = str1.split(' ')
    for i in list1:
        if i in d1:
            d1[i] += 1
        else:
            d1[i] = 1
    d2 = {}
    list2 = str2.split(' ')
    for i in list2:
        if i in d2:
            d2[i] += 1
        else:
            d2[i] = 1
    d1[''] = 1
    d2[''] = 1

    for i in d2.keys():
        if not (i in d1) or d2[i] > d1[i]:
            return False
    return True
