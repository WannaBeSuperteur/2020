import math
import sys
sys.path.insert(0, '../../AI_BASE')
import readData as RD
import numpy as np

def convert_input(_input, trainTest):

    # add percent_atom_o
    
    # add detailed degree
    # ->  0.00 ~ +7.49 :  log(1 + log_mul *  detail_deg)
    # -> -7.49 ~  0.00 : -log(1 + log_mul * -detail_deg)

    converted_input = []
    log_mul = 200

    for i in range(len(_input)):
        if i % 25 == 0: print(i)
        
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

        # compute percent_atom_*
        total_atoms = float(_input[i][1])
        Al = Al_count / total_atoms
        Ga = Ga_count / total_atoms
        In = In_count / total_atoms
        # O = O_count / total_atoms (always 0.6)

        temp.append(Al)
        temp.append(Ga)
        temp.append(In)
        # temp.append(O) (always 0.6)

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
    input_cols = 14

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

    # merge training data
    train_final = [['id', 'spacegroup', 'number_of_total_atoms',
                    'percent_atom_al', 'percent_atom_ga', 'percent_atom_in',
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
