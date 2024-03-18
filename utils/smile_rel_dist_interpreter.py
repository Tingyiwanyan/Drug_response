import numpy as np

def smile_cl_converter(smile):
    new_smile = ''
    for i in range(len(smile)):
        if smile[i] == 'C':
            if not i == len(smile) - 1:
                if smile[i+1] == 'l':
                    new_smile += 'L'
                else:
                    new_smile += smile[i]
        elif smile[i] == 'l' and smile[i-1] == 'C':
            continue

        elif smile[i] == 'B':
            if not i == len(smile) - 1:
                if smile[i+1] == 'r':
                    new_smile += 'B'
                else:
                    new_smile += smile[i]
        elif smile[i] == 'r' and smile[i-1] == 'B':
            continue
            
        else:
            #new_smile.append(smile[i])
            new_smile+=smile[i]
    return new_smile

def symbol_converter(symbol):
    if symbol == '(':
        return (None, 2)
    elif symbol == ')':
        return (None, 2)
    elif symbol == '=':
        return (None, 1)
    elif symbol == '#':
        return (None, 1)
    elif symbol == '1':
        return (None, 2)
    elif symbol == '2':
        return (None, 2)
    elif symbol == '3':
        return (None, 2)
    elif symbol == '4':
        return (None, 2)
    elif symbol == '5':
        return (None, 2)
    elif symbol == '6':
        return (None, 2)
    elif symbol == '7':
        return (None, 2)
    elif symbol == '8':
        return (None, 2)
    else:
        return (symbol, 2)

def prop_dist(smile, pos, direction):
    length = len(smile)
    rel_distance = np.zeros(130)
    projection = np.zeros(130)
    interpret_smile = []
    accumulate_left = 0
    accumulate_right = 0
    flag_double = 0
    if direction == "left":
        flag = 0
        for i in range(pos):
            rel_pos = int(pos-i-1)
            symbol, dist = symbol_converter(smile[rel_pos])
            if symbol == None:
                if dist == 2:
                    #rel_distance[pos - i - 2] = 2 + accumulate_left
                    #accumulate_left += 2
                    flag = 0
                else:
                    #rel_distance[pos - i - 2] = 1 + accumulate_left
                    #accumulate_left += 1
                    flag = 1
            else:
                projection[pos - i - 1] = rel_pos
                if rel_pos == 0:
                    projection[pos - i - 1] = -2 
                interpret_smile.append(symbol)
                if flag == 1:
                    rel_distance[pos - i - 1] = 1 + accumulate_left
                    accumulate_left += 1
                else:
                    rel_distance[pos - i - 1] = 2 + accumulate_left
                    accumulate_left += 2
                flag = 0

        interpret_smile.reverse()
        return rel_distance, interpret_smile, projection
                
    if direction == "right":
        flag = 0
        for i in range(length-pos-1):
            #print(i)
            rel_pos = int(pos+i+1)
            symbol, dist = symbol_converter(smile[rel_pos])
            if symbol == None:
                if rel_pos == length-1:
                    continue
                if dist == 2:
                    #rel_distance[pos + i + 2] = 2 + accumulate_right
                    #accumulate_right += 2
                    flag = 0
                else:
                    #rel_distance[pos + i + 2] = 1 + accumulate_right
                    #accumulate_right += 1
                    flag = 1
            else:
                projection[pos + i + 1] = rel_pos
                interpret_smile.append(symbol)
                if flag == 1:
                    rel_distance[pos + i + 1] = 1 + accumulate_right
                    accumulate_right += 1
                else:
                    rel_distance[pos + i + 1] = 2 + accumulate_right
                    accumulate_right += 2
                flag = 0

        return rel_distance, interpret_smile, projection
            
def smile_rel_dis_interpreter(smile, pos):
    rel_distance = np.zeros(130)
    pos_left = prop_dist(smile, pos,'left')  
    pos_right = prop_dist(smile, pos, 'right')
    rel_distance = pos_left[0] + pos_right[0]
    rel_distance[pos] = -1 #make specific mark for current position
    symbol, dist = symbol_converter(smile[pos])
    interpret_smile = pos_left[1] + list(symbol) + pos_right[1]
    projection = pos_left[2] + pos_right[2]
    projection[pos] = pos
    if pos == 0:
        projection[pos] = -2 #mark the initial pos position
    interpret_smile_ = ""
    for k in interpret_smile:
        interpret_smile_ += k
    projection_ = []
    [projection_.append(i) for i in projection if i!=0]
    if projection_[0] == -2:
        projection_[0] = 0
    rel_distance_ = []
    [rel_distance_.append(i) for i in rel_distance if i!=0]
    for i in range(len(rel_distance_)):
        if rel_distance_[i] == -1:
            rel_distance_[i] = 0

    return rel_distance_, interpret_smile_, projection_

def generate_rel_dist_matrix(smile):
    new_smile = smile_cl_converter(smile)
    length = len(new_smile)
    rel_distance_whole = []
    interpret_smile_whole = []
    projection_whole = []
    for i in range(length):
        symbol, dist = symbol_converter(new_smile[i])
        if symbol == None:
            continue
        else:
            rel_distance, interpret_smile, projection = smile_rel_dis_interpreter(new_smile, i)
            #length_seq = len(rel_distance)
            rel_distance_whole.append(rel_distance)
            interpret_smile_whole.append(interpret_smile)
            projection_whole.append(projection)

    rel_distance_whole = np.stack(rel_distance_whole)
    projection_whole = np.stack(projection_whole)
    interpret_smile_whole = np.stack(interpret_smile_whole)

    return rel_distance_whole, interpret_smile_whole, projection_whole