import genetic as ga
import artificialNeuralNet as ann
import pandas
import numpy as np

pool_size = 200
chrom_size = 50
# The network contains 2 hidden layers
I1_s = 1 # 1 Neuron in input layer
H1_s = 3 # 3 Neurons in hidden layer
H2_s = 3 ##
O1_s = 1 ##

pm = 0.2 # Probability of mutation
pc = 0.5 # Probability of crossover

wt_max = float(input('Enter max wt value: '))
wt_min = float(input('Enter min wt value: '))

string_length = chrom_size*(I1_s*H1_s + H1_s*H2_s + H2_s*O1_s) # String for GA

Input = ann.collectData('train_2_x.xlsx')
data_size = int(np.shape(Input)[0])

pool = np.zeros((pool_size, 1, string_length))
error_mat = np.zeros((pool_size, 1))

iteration = 20

# Initial pool
for j in range(pool_size):

    w1 = ga.genRandomWt(I1_s, H1_s, chrom_size)
    string12 = ga.convMat2Str(w1)
    w1 = ga.getValWt(wt_max, wt_min, w1)
    w2 = ga.genRandomWt(H1_s, H2_s, chrom_size)
    string23 = ga.convMat2Str(w2)
    w2 = ga.getValWt(wt_max, wt_min, w2)
    w3 = ga.genRandomWt(H2_s, O1_s, chrom_size)
    string34 = ga.convMat2Str(w3)
    w3 = ga.getValWt(wt_max, wt_min, w3)
    string13 = ga.joinString(string12, string23)
    string14 = ga.joinString(string13, string34)

    error_sum = 0

    for i in range (data_size):
        
        I1 = ann.inputNeurons(Input, i)
        H1 = ann.getNextLayer(I1, w1)
        H2 = ann.getNextLayer(H1, w2)
        O1 = ann.getNextLayer(H2, w3)

        e = ann.getError(Input, i, O1)
        error_sum += e

    pool[j] = string14
    error_mat[j] = error_sum/data_size

e_min = error_mat[0].copy()
p_min = pool[0].copy()

while (iteration > 0):
    [pool, error_mat] = ga.arrangePool(pool, error_mat, string_length)
    print('Error:', error_mat[0])

    mating_pool = ga.matingPool(pool)
    children = ga.uniformCrossOver(mating_pool, pc)
    children = ga.mutation(children, pm)

    for j in range(pool_size):

        [string12, string24] = ga.breakString(children[j], (I1_s*H1_s*chrom_size))
        w1 = ga.convStr2ValMat(1, 3, chrom_size, string12, wt_max, wt_min)

        [string23, string34] = ga.breakString(string24, (H1_s*H2_s*chrom_size))
        w2 = ga.convStr2ValMat(3, 3, chrom_size, string23, wt_max, wt_min)

        w3 = ga.convStr2ValMat(3, 1, chrom_size, string34, wt_max, wt_min)

        string13 = ga.joinString(string12, string23)
        string14 = ga.joinString(string13, string34)

        error_sum = 0

        for i in range (data_size):
            
            I1 = ann.inputNeurons(Input, i)
            H1 = ann.getNextLayer(I1, w1)
            H2 = ann.getNextLayer(H1, w2)
            O1 = ann.getNextLayer(H2, w3)
            
            e = ann.getError(Input, i, O1)
            error_sum += e

        pool[j] = string14
        error_mat[j] = error_sum/data_size

    [pool, error_mat] = ga.arrangePool(pool, error_mat, string_length)
    e_after = error_mat[0].copy()

    if (e_after < e_min):
        e_min = e_after
        p_min = pool[0].copy()
        print('Best e:', e_min)

    iteration -= 1



print('')
print('Updated E: ', e_min)

[string12, string24] = ga.breakString(p_min, (I1_s*H1_s*chrom_size))
w1 = ga.convStr2ValMat(1, 3, chrom_size, string12, wt_max, wt_min)
[string23, string34] = ga.breakString(string24, (H1_s*H2_s*chrom_size))
w2 = ga.convStr2ValMat(3, 3, chrom_size, string23, wt_max, wt_min)
w3 = ga.convStr2ValMat(3, 1, chrom_size, string34, wt_max, wt_min)
string13 = ga.joinString(string12, string23)
string14 = ga.joinString(string13, string34)

print('')
I1 = float(input('Please provide the input: '))
H1 = ann.getNextLayer(I1, w1)
H2 = ann.getNextLayer(H1, w2)
O1 = ann.getNextLayer(H2, w3)
print('From the network: ')
print('Output is:', O1)