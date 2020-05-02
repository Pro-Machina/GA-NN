# Functions for genetic algorithms implementation in Nural networks
# These functions can be used for following conditions:
# (1) Weights are represented in matrices
# (2) The GA strings are continous arrays of all the weights in a defined sequence (suggested order: I -> H -> O)

import random
import numpy as np

def genRandomWt (prev, nxt, l):
    """ Returns an array of chromosomes with random genes i.e wt matrix (input: neurons in previous and next layers, length of the string) """

    row = prev # Rows and cols are defined to avoid confusion
    col = nxt

    wt = np.zeros((row, col, l)) # Forms a wt matrix of size row X col X [string length]
    # Using this format is necessary to iterate in the matrix

    for r in range(row):
        for c in range(col):
            for g in range(l):
                if (random.uniform(0,1) < 0.5): # Random genes allocated to chromosomes
                    wt[r][c][g] = 0
                else:
                    wt[r][c][g] = 1

    wt = wt.tolist()
    return wt   


def convMat2Str (wt):
    """ Returns a continous string for the given weight matrix (input: weight matrix) """

    gene = len(wt[0][0])
    row = len(wt) # Rows and cols are defined to avoid confusion
    col = len(wt[0])
    s_len = gene*row*col # The final string will contain all the genes in the wt matrix arranged linearly
    string = [0]*s_len
    count = 0

    for r in range(row):
        for c in range(col):
            for g in range(gene):
                if (count < s_len):
                    string[count] = wt[r][c][g]
                count = count + 1

    return string


def convStr2Mat (prev, nxt, l, string):
    """ Returns the weight matrix from the string, (input: neurons in previous and next layer, length of a chromosome, string) """

    row = prev # Rows and cols are defined to avoid confusion
    col = nxt
    s_len = len(string)

    wt = [[ [0 for g in range(l)] for c in range(col)] for r in range(row)] # Forms a wt matrix of size row X col X [string length]
    count = 0

    for r in range(row):
        for c in range(col):
            for g in range(l):
                if (count < s_len):
                    wt[r][c][g] = string[count]
                    count = count + 1

    return wt


def convStr2ValMat (prev, nxt, l, string, wt_max, wt_min):
    """ Returns the weight value matrix from the string, (input: neurons in previous and next layer, length of a chromosome, string) """

    row = prev # Rows and cols are defined to avoid confusion
    col = nxt
    s_len = len(string)

    wt = [[ [0 for g in range(l)] for c in range(col)] for r in range(row)] # Forms a wt matrix of size row X col X [string length]
    count = 0

    for r in range(row):
        for c in range(col):
            for g in range(l):
                if (count < s_len):
                    wt[r][c][g] = string[count]
                    count = count + 1

    wt_val = getValWt(wt_max, wt_min, wt)
    return wt_val


def getValWt (wt_max, wt_min, wt):
    """ Returns a weight matrix with decoded values of the chromosomes in the range wt_max to wt_min """

    gene = len(wt[0][0])
    row = len(wt) # Rows and cols are defined to avoid confusion
    col = len(wt[0])

    wt_val = [[ 0 for c in range(col)] for r in range(row)] # Forms a wt matrix of size row X col

    for r in range(row):
        for c in range(col):
            d = 0
            x = 0
            count = gene - 1

            for g in wt[r][c]:
                d = d + ( g*(2**count) )
                count = count - 1

            x = wt_min + ( ( (wt_max - wt_min)/((2**gene) - 1) )*d )

            wt_val[r][c] = x

    if (row == 1):
        return wt_val[0]
    else:
        return wt_val


def joinString (string1, string2):
    """ Returns a continous string joining string1 and string2 in that order """

    s_len = len(string1) + len(string2) # Length of the combined string
    string = [0]*s_len
    count = 0

    for i in range(len(string1)):
        if (count < s_len):
            string[count] = string1[i] # Reading string 1
            count = count + 1
    
    for j in range(len(string2)):
        if (count < s_len):
            string[count] = string2[j] # Reading string 2
            count = count + 1

    return string


def breakString (string, size1):
    """ Breakes the given string into two stings with string1 of size1 and string2 of remaining size """

    if (string.ndim == 1):
        string1 = string[:size1].copy()
        string2 = string[size1:].copy()
    else:
        string1 = string[0][:size1].copy()
        string2 = string[0][size1:].copy()

    return [string1, string2]


def arrangePool(pool, e_mat, chrom_size):
    """ Arranges the pool from fittest to most unfit chromosome and returns both rearranged pool and error matrix """

    row_pool = int(np.shape(pool)[0])
    col_pool = int(np.shape(pool)[1])
    
    temp_pool = np.zeros((1, chrom_size))
    temp_e = 0
    count = 20

    while(count > 0):
        count = count - 1
        for r in range(row_pool - 1):
            if (e_mat[r+1] < e_mat[r]):
                temp_e = e_mat[r].copy()
                e_mat[r] = e_mat[r+1].copy()
                e_mat[r+1] = temp_e.copy()

                temp_pool[0] = pool[r].copy()
                pool[r] = pool[r+1].copy()
                pool[r+1] = temp_pool[0].copy()

                count = count + 1

    return [pool, e_mat]


def singlePCrossOver (pool, cross):
    """ Returns the pool with children chromosomes after single point crossover from the cross point """
    # Not recomended for neural net weight updation 

    parent = pool.copy()
    childern = pool.copy()

    row = int(np.shape(pool)[0])
    col = int(np.shape(pool)[1]) - 1
    chrom = int(np.shape(pool)[2])

    temp_spc1 = np.zeros((1, int(chrom/2)))
    temp_spc2 = np.zeros((1, int(chrom/2)))
    
    for i in range(row-1):
        temp_spc1 = parent[i][col][:cross].copy()
        temp_spc2 = parent[i+1][col][:cross].copy()

        childern[i][col][:cross] = temp_spc2.copy()
        childern[i+1][col][:cross] = temp_spc1.copy()

    return childern


def sumNumber(num):
    """ Gives the sum of all the positive numbers before input number """
    # Used for rank based selection

    sum_num = 0
    for i in range(num+1):
        sum_num += i

    return sum_num

def matingPool (pool_arranged):
    """ Generates a mating pool from the input pool based on rank selection method """

    row = int(np.shape(pool_arranged)[0])
    col = int(np.shape(pool_arranged)[1]) 
    chrom = int(np.shape(pool_arranged)[2])

    mating_pool = np.zeros((row, col, chrom))
    col -= 1
    j = 0
    denom = sumNumber(row)

    while (j < row):
        for r in range(row):
            if (j < row):
                if (not any(mating_pool[j][col])):
                    if ( random.uniform(0, 1) < ((row - (r + 1))/denom) ):
                        mating_pool[j][col] = pool_arranged[r][col].copy() # Fills the mating pool based on rank based selection
                        j += 1

    return mating_pool


def mutation (pool, pm):
    """ Uniformly mutates the genes of the pool as per the mutation probability pm """

    row = int(np.shape(pool)[0])
    col = int(np.shape(pool)[1]) - 1
    chrom = int(np.shape(pool)[2])

    mutation_pool = pool.copy()

    for r in range(row): 
        for g in range(chrom):
            if (random.uniform(0, 1) < pm):
                if (mutation_pool[r][col][g] == 1):
                    mutation_pool[r][col][g] = 0 # Flips the mutating gene
                else:
                    mutation_pool[r][col][g] = 1 # Flips the mutating gene

    return mutation_pool


def uniformCrossOver (pool, pc):
    """ Returns the pool with children chromosomes after single point crossover from the cross point """

    parent = pool.copy()
    childern = pool.copy()

    row = int(np.shape(pool)[0])
    col = int(np.shape(pool)[1]) - 1
    chrom = int(np.shape(pool)[2])

    temp_gene1 = 0
    temp_gene2 = 0
    i = 0
    
    while (i < (row - 1)):
        for g in range(chrom):
            if (random.uniform(1, 0) < pc):
                temp_gene1 = parent[i][col][g]
                childern[i+1][col][g] = temp_gene1

                temp_gene2 = parent[i+1][col][g]
                childern[i][col][g] = temp_gene2
        i += 2

    return childern

