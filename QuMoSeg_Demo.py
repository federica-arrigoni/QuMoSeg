
###             QuMoSeg-v1/-v2  Demo           ### 

# import libraries 
import math 
import numpy as np 
import json 
from scipy.io import loadmat 
from scipy.io.matlab import savemat
import dimod 
import dwavebinarycsp 
import dwave.inspector 
from dwave.system import DWaveSampler, EmbeddingComposite 

# parameters that change rarely  
anneal_params = dict(anneal_schedule=[[0.0, 0.0], [20.0, 1.0]]) 

# input/output 
folder = "./QuMoSeg_Data/" 
output_folder = "./QuMoSeg_Results/" 

all_base_files = [] 
all_base_files.append("data_3-8,9-102_000") 
all_base_files.append("data_5-9,9-180_000") 

# 1: v1, 2: v2 
strategy_ = 1 

# parameters of the chain strength function 
if strategy_ == 1: 
    a_coeff = 0.13508 
    b_coeff = -1.94207 
if strategy_ == 2: 
    a_coeff = 0.1238 
    b_coeff = -1.3180 

# algorithm and demo launch parameters 

lambda_1_v1 = 10.0         # v1 only 
lambda_1_v2 = 10.0         # v2 only 
lambda_2_v2 = 4.0          # v2 only 


epsilon = 0.001         # threshold for an effective zero value 
CHAIN_STR = 10.0        # default value (overridden later) 
Nsamples = 1000         # number of samples/annealings 
N_CASES = 20            # each case with the same number of qubits is represented by N_CASES different problems 


# different problem types (two in this demo) 
for base_file in range(0, len(all_base_files)): 

    # statistics variables 
    N_ERRORS = {}                               # errors over lowest-energy samples 
    N_ERRORS_ALL_SAMPLES = {}                   # errors of all samples 
    N_PHYS_VARIABLES = {}                       # number of physical qubits in the minor embedding 
    MAX_CHAIN_LENGTHS = {}                      # maximum chain lengths 
    E_DIFF = {}                                 # differences in the energies (lowest energy solution vs energy of the lowest-error sample) 
    N_GT_SOLUTIONS = 0                          # number of measured ground-truth solutions 
    P_OPTIMAL = {}                              # probability to measure optimal solution as lowest-energy solution (N_GT_SOLUTIONS / Nsamples) 

    # saved samples 
    LOWEST_ENERGY_SAMPLES_ = [[]] 
    LOWEST_ERROR_SAMPLES_ = [[]] 
    LOWEST_ENERGY_SAMPLES_ENERGIES = [] 
    LOWEST_ERROR_SAMPLES_ENERGIES = [] 


    ### DIM is required for the initialisation of CHAIN_STR (taken from the first file) 
    filename_ = folder + all_base_files[base_file] + "00" +  ".mat" 
    matfile = loadmat(filename_) 
    Q_ = matfile['Q'] 
    DIM_ = len(Q_) 
    CHAIN_STR = a_coeff * DIM_ + b_coeff  
    ### 


    # initialisation 
    LOWEST_ENERGY_SAMPLES_ = [[0 for i in range(0, DIM_)] for i in range(0, N_CASES)] 
    LOWEST_ERROR_SAMPLES_ = [[0 for i in range(0, DIM_)] for i in range(0, N_CASES)] 
    LOWEST_ENERGY_SAMPLES_ENERGIES = [0 for i in range(0, N_CASES)] 
    LOWEST_ERROR_SAMPLES_ENERGIES = [0 for i in range(0, N_CASES)] 


    ### the loop over the problem instances 
    for f in range(0, N_CASES, 1): 

        str_f = str(f) 
        str_f_padded = str_f .rjust(2, '0') 
        filename = folder + all_base_files[base_file] + str_f_padded + ".mat" 
        print(filename) 
        matfile = loadmat(filename) 
        
        Q = matfile['Q']        # for strategirs 1,2,3,4 
        Qd = matfile['Qd']      # for strategy 5 
        DIM = len(Q) 
        A1 = matfile['A1'] 
        A2 = matfile['A2'] 
        b1 = matfile['b1'] 
        b2 = matfile['b2'] 
        x_gt_perm = matfile['x_gt_perm'] 
        x_gt_1 = x_gt_perm.T[0] 
        x_gt_2 = x_gt_perm.T[1] 
        A1T = A1.transpose() 
        A2T = A2.transpose() 
        b1T = b1.T 
        b2T = b2.T  
        A1TA1 = np.matmul(A1T, A1) 
        A2TA2 = np.matmul(A2T, A2) 
        b1TA1 = np.matmul(b1T, A1) 
        b2TA2 = np.matmul(b2T, A2) 


        # calculate QUBO weights (see the paper) 
        Q_RECTIFIED = [] 
        C_RECTIFIED = [] 
        
        if strategy_ == 1: 
            Q_RECTIFIED = Qd + lambda_1_v1 * A1TA1 
            C_RECTIFIED = np.diag(Q_RECTIFIED) - 2 * (lambda_1_v1 * b1TA1) 

        if strategy_ == 2: 
            Q_RECTIFIED = Q + lambda_1_v2 * A1TA1 + lambda_2_v2 * A2TA2 
            C_RECTIFIED = np.diag(Q_RECTIFIED) - 2 * (lambda_1_v2 * b1TA1) - 2 * (lambda_2_v2 * b2TA2) 
        # 


        # the QUBO weights (linear and quadratic) 
        linear = {} 
        quadratic = {} 

        # qubit bises
        for i in range(0, DIM): 
            linear[i+1] = C_RECTIFIED[0][i] 

        # qubit couplings (only non-zero values) 
        for i in range(0, DIM): 
            for j in range(0, DIM): 
                if  ( (i < j) and (abs(Q_RECTIFIED[i, j]) > epsilon) ): 
                    quadratic[(i+1, j+1)] =  2 * Q_RECTIFIED[i, j] 


        # initialise the problem and pass it to Adv4.1 
        offset = 0.0 
        vartype = dimod.BINARY 
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype) 
        sampler = EmbeddingComposite(DWaveSampler(solver='Advantage_system4.1')) 
        #  

        # sample Nsamples times 
        sampleset = sampler.sample(bqm, chain_strength = CHAIN_STR, num_reads = Nsamples, **anneal_params) 

        # select the lowest-energy sample 
        best_sample = sampleset.first.sample 

        # visualise graph problem and print the sampleset (comment out if required) 
        # dwave.inspector.show(bqm, sampleset) 
        # print(sampleset) 


        # compare x_gt (two variants) and the result and return Hamming distance 
        n_correct_1 = 0
        for k in range(0, DIM): 
            if (best_sample[k+1] == x_gt_1[k]):  
                n_correct_1 = n_correct_1 + 1
        n_correct_2 = 0
        for k in range(0, DIM): 
            if (best_sample[k+1] == x_gt_2[k]):  
                n_correct_2 = n_correct_2 + 1 
        n_correct = 0 
        if n_correct_1 > n_correct_2: 
            n_correct = n_correct_1 
        else: 
            n_correct = n_correct_2 

        # collecting individual results for statistics 
        N_ERRORS[f] = n_correct / DIM 
        print("n_correct_1 = ", n_correct_1) 
        print("n_correct_2 = ", n_correct_2) 
        print("n_correct = ", n_correct) 
        print("DIM = ", DIM) 
        print("N_CORRECT = ", n_correct / DIM) 

        if n_correct == DIM: 
            N_GT_SOLUTIONS += 1 
            P_OPTIMAL[f] = sampleset.record.num_occurrences[0] / Nsamples 
        else: 
            P_OPTIMAL[f] = 0 


        ### check the accuracy of all samples (perhaps there is a sample with higher energy but higher accuracy) 
        id_sample_lowest_error = 0 
        lowest_error = 0.0 
        lowest_energy = 10^10 
        energy_lowest_error = 0.0 
        lowest_energy_sample = {} 

        Nsamples_actual = sampleset.record.sample.shape[0] 
        for s_ in range(0, Nsamples_actual): 

            current_sample = sampleset.record.sample[s_] 
            current_energy = sampleset.record.energy[s_] 
            ### lowest energy over all samples 
            if lowest_energy > current_energy: 
                lowest_energy = current_energy 
                lowest_energy_sample = current_sample 
            ### 

            n_correct_s1 = 0 
            for k in range(0, DIM): 
                if (current_sample[k] == x_gt_1[k]): 
                    n_correct_s1 = n_correct_s1 + 1 

            n_correct_s2 = 0 
            for k in range(0, DIM): 
                if (current_sample[k] == x_gt_2[k]): 
                    n_correct_s2 = n_correct_s2 + 1 

            n_correct_current = 0 
            if n_correct_s1 > n_correct_s2: 
                n_correct_current = n_correct_s1 
            else: 
                n_correct_current = n_correct_s2 

            error_current = n_correct_current / DIM 
            if lowest_error < error_current: 
                id_sample_lowest_error = s_ 
                lowest_error = error_current 
                energy_lowest_error = current_energy 

        for k in range(0, DIM): 
            LOWEST_ENERGY_SAMPLES_[f][k]    = lowest_energy_sample[k] 
            LOWEST_ERROR_SAMPLES_[f][k]     = sampleset.record.sample[id_sample_lowest_error][k] 

        LOWEST_ENERGY_SAMPLES_ENERGIES[f]   = lowest_energy 
        LOWEST_ERROR_SAMPLES_ENERGIES[f]    = sampleset.record.energy[id_sample_lowest_error] 


        
        N_ERRORS_ALL_SAMPLES[f] = lowest_error                      # accuracy over all samples 
        E_DIFF[f] = abs(energy_lowest_error - lowest_energy)        # difference in the lowest energy and the energy resulting in the lowest error 
                                                                    # (Hamming Distance close to or equal to 1.0) 
        # save chain lengths 
        LENS_ = {} 
        lenv = 0 
        for key in sampleset.info['embedding_context']['embedding']: 
            LENS_[lenv] = len(sampleset.info['embedding_context']['embedding'][key]) 
            lenv += 1 

        N_PHYS_VARIABLES[f] = sum(LENS_.values()) 
        MAX_CHAIN_LENGTHS[f] = max(LENS_.values()) 

    # END: for f in range(0, N_CASES, 1): 


    # evaluating the collected values 
    listr1 = [] 
    for value in N_ERRORS.values(): 
        listr1.append(value) 
    STD_DEV_ERROR = np.std(listr1, ddof=1) 
    MEAN_ERROR = np.mean(listr1) 

    listr2 = [] 
    for value in MAX_CHAIN_LENGTHS.values(): 
        listr2.append(value) 
    STD_DEV_MAX_CHAIN_LENGTHS = np.std(listr2, ddof=1) 
    MEAN_MAX_CHAIN_LENGTHS = np.mean(listr2) 

    listr3 = [] 
    for value in N_ERRORS_ALL_SAMPLES.values(): 
        listr3.append(value) 
    STD_DEV_ERROR_ALL_SAMPLES = np.std(listr3, ddof=1) 
    MEAN_ERROR_ALL_SAMPLES = np.mean(listr3) 


    listr4 = [] 
    for value in E_DIFF.values(): 
        listr4.append(value) 
    STD_DEV_E_DIFF = np.std(listr4, ddof=1) 
    MEAN_E_DIFF = np.mean(listr4) 


    listr5 = [] 
    for value in N_PHYS_VARIABLES.values(): 
        listr5.append(value) 
    STD_DEV_PHYS_VAR = np.std(listr5, ddof=1) 
    MEAN_PHYS_VAR = np.mean(listr5) 


    listr = [] 
    for value in P_OPTIMAL.values(): 
        listr.append(value) 
    MEAN_P_OPTIMAL = np.mean(listr) 


    # STATISTICS 
    STATISTICS_ = [] 
    STATISTICS_ = [0 for i in range(0, 12)] 
    STATISTICS_[0] = MEAN_ERROR 
    STATISTICS_[1] = STD_DEV_ERROR 
    STATISTICS_[2] = MEAN_ERROR_ALL_SAMPLES 
    STATISTICS_[3] = STD_DEV_ERROR_ALL_SAMPLES 
    STATISTICS_[4] = MEAN_E_DIFF 
    STATISTICS_[5] = STD_DEV_E_DIFF 
    STATISTICS_[6] = MEAN_MAX_CHAIN_LENGTHS 
    STATISTICS_[7] = STD_DEV_MAX_CHAIN_LENGTHS 
    STATISTICS_[8] = MEAN_PHYS_VAR 
    STATISTICS_[9] = STD_DEV_PHYS_VAR 
    STATISTICS_[10] = N_GT_SOLUTIONS 
    STATISTICS_[11] = MEAN_P_OPTIMAL 

    print("") 
    print("MEAN_ERROR                = ", MEAN_ERROR) 
    print("STD_DEV_ERROR             = ", STD_DEV_ERROR) 
    print("MEAN_ERROR_ALL_SAMPLES    = ", MEAN_ERROR_ALL_SAMPLES) 
    print("STD_DEV_ERROR_ALL_SAMPLES = ", STD_DEV_ERROR_ALL_SAMPLES) 
    print("MEAN_E_DIFF               = ", MEAN_E_DIFF) 
    print("STD_DEV_E_DIFF            = ", STD_DEV_E_DIFF) 
    print("MEAN_MAX_CHAIN_LENGTHS    = ", MEAN_MAX_CHAIN_LENGTHS) 
    print("STD_DEV_MAX_CHAIN_LENGTHS = ", STD_DEV_MAX_CHAIN_LENGTHS) 
    print("MEAN_PHYS_VAR             = ", MEAN_PHYS_VAR) 
    print("STD_DEV_PHYS_VAR          = ", STD_DEV_PHYS_VAR) 
    print("N_GT_SOLUTIONS            = ", N_GT_SOLUTIONS) 
    print("MEAN_P_OPTIMAL            = ", MEAN_P_OPTIMAL) 


    ### SAVE .csv and .txt files 
    output_file_name_lens           = output_folder + all_base_files[base_file] + "_lenergy_s_" + str(strategy_) + ".csv"                   # lowest energy samples 
    output_file_name_lers           = output_folder + all_base_files[base_file] + "_lerror_s_" + str(strategy_) + ".csv"                    # lowest error samples 
    output_file_name_lens_energies  = output_folder + all_base_files[base_file] + "_lenergy_s_energies_" + str(strategy_) + ".csv"          # lowest energy sample energies 
    output_file_name_lers_energies  = output_folder + all_base_files[base_file] + "_lerror_s_energies_" + str(strategy_) + ".csv"           # lowest error sample energies 
    output_file_statistics          = output_folder + all_base_files[base_file] + "_statistics_" + str(strategy_) + ".txt"                  # entire statistics 

    np.savetxt(output_file_name_lens, LOWEST_ENERGY_SAMPLES_, fmt='%d') 
    np.savetxt(output_file_name_lers, LOWEST_ERROR_SAMPLES_, fmt='%d') 
    np.savetxt(output_file_name_lens_energies, LOWEST_ENERGY_SAMPLES_ENERGIES, fmt='%d') 
    np.savetxt(output_file_name_lers_energies, LOWEST_ERROR_SAMPLES_ENERGIES, fmt='%d') 
    np.savetxt(output_file_statistics, STATISTICS_,  fmt='%1.4f') 
