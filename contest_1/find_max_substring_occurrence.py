def find_max_substring_occurrence(input_string):
    str_len = len(input_string)
    for i in range(1, str_len + 1):
        if not str_len % i:
            k = str_len // i
            s = input_string[0:i]
            chk = 1
            for j in range(1, k):
                if s != input_string[j * i:(j + 1) * i]:
                    chk = 0
                    break
            if chk:
                return k
