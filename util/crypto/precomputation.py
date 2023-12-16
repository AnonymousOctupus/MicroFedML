from RandomOracle import RandomOracle
from FiniteField import FiniteField

import pickle

class PrecomputedTable:
    def __init__(self, prime = -1):
        if prime == -1:
            self.ff = FiniteField()
        else:
            self.ff = FiniteField(p = prime)
        self.ro = RandomOracle(self.ff)

    # def listing_generators_01(iterations, vec_len)
    #     l = [None] * iterations
    #     for k in range(iterations):
    #         l[k] = ro.query(k, vec_len)

    def precompute(self, file_name, iterations, vec_len, max_expo):
        # file_name = file_name
        table = open(file_name, 'wb')
        # self.vec_len = vec_len

        num = iterations * vec_len
        print(f"num = {num}")
        l = self.ro.query(0, num)
        for i in range(num): 
            # with open(file_name, 'ab') as result:
            # self.table.write(l[i])
            # exponents_p = {}
            # exponents_n = {}
            exponents = {}
            # exponents_p = [(0, 0)] * (max_expo * 2)
            # exponents_n = [(0, 0)] * (max_expo * 2)
            gen_inv = self.ff.divide(1, l[i])
            temp_p = l[i]
            temp_n = gen_inv
            for j in range (1, max_expo + 1):
                # exponents[j] = (temp, j)
                # exponents[n] = (temp, j)
                exponents[temp_p] = j
                exponents[temp_n] = -j
                temp_p = self.ff.multiply(temp_p, l[i])
                temp_n = self.ff.multiply(temp_n, gen_inv)
            # exponents_p.sort(key = get_expo)
            # exponents_n.sort(key = get_expo)
            exponents[self.ff.getPrime()] = 0
            line = (i, l[i], exponents)
            # line = (i, l[i], exponents_p, exponents_n)
            pickle.dump(line, table, 4)
            # table.write("\n")
        
        table.close()

    def read_table(self, filename, vec_len):
        self.table = open(filename, 'rb')
        self.vec_len = vec_len
        self.current_line = 0
        return

    def discrete_log(self, a, iteration, vec_index):
        line_num = iteration * self.vec_len + vec_index
        # l = self.table.readline()
        line = pickle.load(self.table)
        # print(type(line[2]))
        # print((line[2]))
        print(line[1])
        if (line_num == line[0]):
            # if line[2][a] != None:
            if a in line[2]:
                return line[2][a]
            else:
                return self.ff.log(a, line[1])
        else:
            print(f"currently at line {line[0]}; requiring line {line_num}")

    def get_expo(e):
        return e[0]

    # def search(a, list):


iteration = 2
vec_len = 2
max_expo = 10
file_name = "pre-computed-table.txt"

pctbl = PrecomputedTable()
pctbl.precompute(file_name, iteration, vec_len, max_expo)
pctbl.read_table(file_name, vec_len)

ff = FiniteField()
ro = RandomOracle(ff)
test_gen = ro.query(0, vec_len)
print(iteration, vec_len, test_gen[0])
test_expo = 8
test_a = ff.power(test_gen[0], test_expo)
print(pctbl.discrete_log(test_a, 0, 0))



test_expo = -5
test_a = ff.power(test_gen[1], test_expo)
print(pctbl.discrete_log(test_a, 0, 1))



test_gen = ro.query(1, vec_len)
test_expo = 3
test_a = ff.power(test_gen[0], test_expo)
print(pctbl.discrete_log(test_a, 1, 0))

test_expo = 20
test_a = ff.power(test_gen[1], test_expo)
print(pctbl.discrete_log(test_a, 1, 1))

