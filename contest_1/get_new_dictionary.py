def get_new_dictionary(input_dict_name, output_dict_name):
    dict = {}

    file = open(input_dict_name, 'r')
    q1 = int(file.readline().strip())
    for i in range(q1):
        line = file.readline()
        info = line.split(' - ')
        for key in info[1].strip().split(', '):
            if key in dict:
                dict[key].append(info[0])
            else:
                dict[key] = [info[0]]
    file.close()

    file = open(output_dict_name, 'w')
    q2 = len(dict.keys())
    file.write(str(q2) + '\n')
    for word in sorted(dict.keys()):
        file.write(word + ' - ' + ', '.join(sorted(dict[word])) + '\n')
    file.close()
