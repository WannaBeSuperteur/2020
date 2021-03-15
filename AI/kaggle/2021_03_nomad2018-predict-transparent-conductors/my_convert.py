import math
import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np

# train.csv -> train_converted.csv
# test.csv  -> test_converted.csv

# distance
def dist(a0, a1):
    for i in range(3):
        a0[i] = float(a0[i])
        a1[i] = float(a1[i])
        
    return math.sqrt(pow(a0[0] - a1[0], 2) + pow(a0[1] - a1[1], 2) + pow(a0[2] - a1[2], 2))

# get nearest atom info
def getNearestAtomInfo(detail):

    # find 3 nearest atoms
    # with weight -> 1st near: 4, 2nd near: 2, 3rd near: 1

    # total 16 columns
    # column  1~ 4: Al -> % of Al, Ga, In and O
    # column  5~ 8: Ga -> % of Al, Ga, In and O
    # column  9~12: In -> % of Al, Ga, In and O
    # column 13~16: O  -> % of Al, Ga, In and O

    result = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    kinds = {'Al':0, 'Ga':1, 'In':2, 'O':3}

    data = detail[6:]
    atoms = len(data)

    for i in range(atoms):

        # xyz of atom index of i
        i_xyz = [float(data[i][1]), float(data[i][2]), float(data[i][3])]
        i_kind = data[i][4]

        # list of [distance_ij, atom_j_kind]
        distances = []
        
        for j in range(atoms):
            if i != j:
                j_xyz = [float(data[j][1]), float(data[j][2]), float(data[j][3])]
                j_kind = data[j][4]

                dist_ij = dist(i_xyz, j_xyz)

                #print(np.array(i_xyz), i_kind, np.array(j_xyz), j_kind, dist_ij)
                
                distances.append([dist_ij, j_kind])

        # sort distance
        distances.sort(key=lambda x:x[0])

        #print(i_xyz)
        #for j in range(len(distances)):
        #    print(distances[j])

        # add weight for each nearest atom j for atom i
        nearest_1 = distances[0]
        result[kinds[i_kind]][kinds[nearest_1[1]]] += 4

        nearest_2 = distances[1]
        result[kinds[i_kind]][kinds[nearest_2[1]]] += 2

        nearest_3 = distances[2]
        result[kinds[i_kind]][kinds[nearest_3[1]]] += 1

        #print(np.array(result))

    # return final result
    final_result = []
    
    for i in range(4):
        for j in range(4):
            try:
                final_result.append(result[i][j] / sum(result[i]))
            except:
                final_result.append(0)

    return final_result

def convert_input(_input, trainTest):

    # add percent_atom_o
    
    # add detailed degree
    # ->  0.00 ~ +7.49 :  log(1 + log_mul *  detail_deg)
    # -> -7.49 ~  0.00 : -log(1 + log_mul * -detail_deg)

    converted_input = []
    log_mul = 200

    for i in range(len(_input)):
        if i % 10 == 0: print(i)
        
        temp = []

        # read detailed info from *.xyz files
        detail = RD.loadArray(trainTest + '/' + str(i+1) + '/geometry.xyz', ' ')

        Al_count = 0
        Ga_count = 0
        In_count = 0
        O_count = 0
        
        for j in range(6, len(detail)):
            if detail[j][4] == 'Al':
                Al_count += 1
            elif detail[j][4] == 'Ga':
                Ga_count += 1
            elif detail[j][4] == 'In':
                In_count += 1
            elif detail[j][4] == 'O':
                O_count += 1

        # spacegroup and number_of_total_atoms
        for j in range(2):
            temp.append(_input[i][j])

        # compute percent_atom_* (percent of O is always 0.6)
        total_atoms = float(_input[i][1])
        Al = Al_count / total_atoms
        Ga = Ga_count / total_atoms
        In = In_count / total_atoms

        temp.append(Al)
        temp.append(Ga)
        temp.append(In)

        # find 3 nearest atoms and the kinds of them
        nearest_atom_info = getNearestAtomInfo(detail)

        for j in range(len(nearest_atom_info)):
            temp.append(nearest_atom_info[j])

        # lattice_vector_*_ang
        for j in range(5, 8):
            temp.append(_input[i][j])

        # lattice_angle
        for j in range(8, 11):
            val = float(_input[i][j])
            major = int((val + 7.5) / 15)
            detail = val - 15 * major

            if detail >= 0:
                detail = math.log(1 + log_mul * detail)
            else:
                detail = -math.log(1 + log_mul * -detail)

            temp.append(major)
            temp.append(detail)
    
        converted_input.append(temp)

    return converted_input

if __name__ == '__main__':

    train_rows = 2400
    test_rows = 600
    input_cols = 30

    # TRAIN
    train_data = RD.loadArray('train.csv', ',')
    
    train_input = np.array(train_data)[1:, 1:12]
    train_output0 = np.array(train_data)[1:, 12:13] # formation_energy_ev_natom
    train_output1 = np.array(train_data)[1:, 13:14] # bandgap_energy_ev

    # TEST
    test_data = RD.loadArray('test.csv', ',')
    
    test_input = np.array(test_data)[1:, 1:]

    # convert input
    train_input_converted = convert_input(train_input, 'train')
    test_input_converted = convert_input(test_input, 'test')

    print(np.shape(train_input_converted))
    print(np.shape(test_input_converted))

    # merge training data
    train_final = [['id', 'spacegroup', 'number_of_total_atoms',
                    'percent_atom_al', 'percent_atom_ga', 'percent_atom_in',
                    'nearest_al_al', 'nearest_al_ga', 'nearest_al_in', 'nearest_al_o',
                    'nearest_ga_al', 'nearest_ga_ga', 'nearest_ga_in', 'nearest_ga_o',
                    'nearest_in_al', 'nearest_in_ga', 'nearest_in_in', 'nearest_in_o',
                    'nearest_o_al', 'nearest_o_ga', 'nearest_o_in', 'nearest_o_o',
                    'lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang',
                    'la_alpha_degree', 'la_alpha_detailed_degree',
                    'la_beta_degree', 'la_beta_detailed_degree',
                    'la_gamma_degree', 'la_gamma_detailed_degree',
                    'formation_energy_ev_natom', 'bandgap_energy_ev']]

    for i in range(train_rows):
        
        temp = [i+1]
        for j in range(input_cols):
            temp.append(train_input_converted[i][j])
        temp.append(train_output0[i][0])
        temp.append(train_output1[i][0])

        train_final.append(temp)

    # merge test data
    test_final = [['id', 'spacegroup', 'number_of_total_atoms',
                   'percent_atom_al', 'percent_atom_ga', 'percent_atom_in',
                   'nearest_al_al', 'nearest_al_ga', 'nearest_al_in', 'nearest_al_o',
                   'nearest_ga_al', 'nearest_ga_ga', 'nearest_ga_in', 'nearest_ga_o',
                   'nearest_in_al', 'nearest_in_ga', 'nearest_in_in', 'nearest_in_o',
                   'nearest_o_al', 'nearest_o_ga', 'nearest_o_in', 'nearest_o_o',
                   'lattice_vector_1_ang', 'lattice_vector_2_ang', 'lattice_vector_3_ang',
                   'la_alpha_degree', 'la_alpha_detailed_degree',
                   'la_beta_degree', 'la_beta_detailed_degree',
                   'la_gamma_degree', 'la_gamma_detailed_degree']]

    for i in range(test_rows):
        
        temp = [i+1]
        for j in range(input_cols):
            temp.append(test_input_converted[i][j])

        test_final.append(temp)

    # save array
    RD.saveArray('train_converted.csv', train_final, ',', 500)
    RD.saveArray('test_converted.csv', test_final, ',', 500)
