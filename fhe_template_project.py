from eva import EvaProgram, Input, Output, evaluate
from eva.ckks import CKKSCompiler
from eva.seal import generate_keys
from eva.metric import valuation_mse
import timeit
import networkx as nx
from random import random
import numpy as np

# Using networkx, generate a random graph
# You can change the way you generate the graph
def generateGraph(n, k, p):
    #ws = nx.cycle_graph(n)
    ws = nx.watts_strogatz_graph(n,k,p)
    return ws

# If there is an edge between two vertices its weight is 1 otherwise it is zero
# You can change the weight assignment as required
# Two dimensional adjacency matrix is represented as a vector
# Assume there are n vertices
# (i,j)th element of the adjacency matrix corresponds to (i*n + j)th element in the vector representations
def serializeGraphZeroOne(GG,vec_size):
    n = GG.order() # changed from size() which ouputs number of edges 
    graphdict = {}
    g = []
    for row in range(n):
        for column in range(n):
            if GG.has_edge(row, column): # or row==column: I did not assume the vertices are connected to themselves
                weight = 1
            else:
                weight = 0 
            g.append(weight)  
            key = str(row)+'-'+str(column)
            graphdict[key] = [weight] # EVA requires str:listoffloat
    # EVA vector size has to be large, if the vector representation of the graph is smaller, fill the eva vector with zeros
    # g = [0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0]
    for _ in range(vec_size - n*n): 
       g.append(0.0)
    return g, graphdict

# To display the generated graph
def printGraph(graph,n):
    for row in range(n):
        for column in range(n):
            print("{:.5f}".format(graph[row*n+column]), end = '\t')
        print() 

# Eva requires special input, this function prepares the eva input
# Eva will then encrypt them
def prepareInput(n, m):
    input = {}
    GG = generateGraph(n,3,0.5)
    graph, graphdict = serializeGraphZeroOne(GG,m)
    input['Graph'] = graph
    return input, GG

def rowSum(graph, n):
    # summing rows of the matrix code is taken from eva.std.numeric.horizontal_sum
    vec_size = graph.program.vec_size 
    x = graph
    i = 1 
    while i < n: # different from summing all the elements in the vector, it sums up to node count, computing sum of each row 
        y = x << i
        x = y + x
        i <<= 1
    
    column = n * ([1.0] + (n-1) * [0.0]) + (vec_size - n*n) * [0.0] # each row-sum is in the first column of the row
    return column * x

def computeGLaplacian(graph, n):
    vec_size = graph.program.vec_size 
    x = rowSum(graph, n)
    # computing the diagonal degree matrix
    for i in range(n):
        diag = vec_size*[0.0]
        diag[i*n+i] = 1
        if i==0:
            degree = diag * x
        else:
            degree += diag * x
        x >>= 1
        
    # computing graph laplacian
    neg_adj = (vec_size *[-1]) * graph
    laplacian = degree + neg_adj
    return laplacian

def copyFirstN2Times(vector, n): 
    # the first element of the vector copied n^2 times to fill first n^2 elements
    vec_size = vector.program.vec_size
    temp = ([1.0] + (vec_size-1)*[0.0]) * vector
    i = 1
    while i < n*n:
        temp2 = temp >> i
        temp = temp + temp2
        i <<= 1
    return temp

# def getColumnMask(n, c, value, vec_size):
#     mask = (vec_size)*[0.0]
#     for i in range(n):
#         mask[i*n+c] = 1.0
#     return mask
def getColumnMask(n, c, vec_size, value=1.0):
    mask = np.zeros(vec_size)
    mask[np.arange(0,n,1)*n+c] = value
    return mask.tolist()

def getColumnMask2(n, c, vec_size, value=1.0):
    mask = np.ones(vec_size)
    mask[np.arange(0,n,1)*n+c] = value
    return mask.tolist()

def getRowMask(n, r, vec_size):
    mask = r*n*[0.0] + n*[1.0] +  (vec_size-(r+1)*n)* [0.0]
    return mask

def maskMatrix(graph, mask):
    return graph*mask

def horizontal_concat(vector, n):
    vec_size = vector.program.vec_size
    num = int(vec_size/(n*n))
    for i in range(n):
        mask = getRowMask(n, i, vec_size)
        if i == 0:
            concat = mask * vector
        else:
            concat += (mask * vector)  >> ((num-i)*n)
    while i < num:
        concat += concat >> (i*n*n)
        i <<= 2
    return concat 
def vertical_concat(vector, n, remove=False):
    # vec_size = 4 * n*n
    # for matrix multiplication vec_size == n for rotations
    # with this operation rotation will on vector will be like vec_size == n
    vec_size = vector.program.vec_size
    if remove:
        temp = vector* ((n*n) * [1.0] + (vec_size - n*n) * [0.0])
    else:
        temp = vector
    
    i = 1
    while i < int(vec_size/(n*n)):
        temp += temp >> (i*n*n)
        i <<= 2
    return temp 

def prep_sigma_perm_dict(n):
    sigma_perm_dict = {}
    for k in range(-n+1, n):
        sigma_perm_dict[k] = get_sigma_perm_vector(n, k)
    return sigma_perm_dict

def prep_tau_perm_dict(n):
    tau_perm_dict = {}
    for k in range(1, n):
        tau_perm_dict[k] =  get_tau_perm_vector(n,k)
    return tau_perm_dict

def prep_col_shift_perm_dict(n):
    col_shift_perm_dict = {}
    for k in range(n):
        col_shift_perm_dict[k] = get_col_shift_perm(n, k)
    return col_shift_perm_dict

def get_sigma_perm_vector(n, k):
    perm = np.arange(0, n*n, 1)
    if k >= 0:
        perm -= n*k
        perm = np.logical_and(0 <= perm, perm < (n-k))
    else:
        perm -= (n+k)*n
        perm = np.logical_and(-k <= perm, perm < n)
    return (perm.astype(float)+ 1e-10).tolist()

def get_tau_perm_vector(n,k):
    perm = np.zeros(n*n)
    index = k+n*np.arange(0,n, 1).astype(int)
    perm[index] = 1
    return (perm.astype(float)+ 1e-10).tolist() 

def get_col_shift_perm(n, k):
    perm = np.arange(0,n*n,1) % n
    return (np.logical_and(0 <= perm, perm < (n-k)).astype(float)+ 1e-10).tolist(), (np.logical_and((n-k) <= perm, perm < n).astype(float)+ 1e-10).tolist()

def lin_transformation_sigma(vector, n):
    for k in range(-n+1, n):
        if k == 1-n:
            ret = sigma_perm_dict[k] * (vector << k)
        else:
            ret += sigma_perm_dict[k] * (vector << k)
    return ret 

def lin_transformation_tau(vector, n):
    ret = get_tau_perm_vector(n, 0) * vector
    for k in range(1, n):
        ret += tau_perm_dict[k] * (vector << n*k)
    return ret

def lin_transformation_cs(vector, n, k):
    perm1, perm2 = col_shift_perm_dict[k]
    return  ((vector << k) * perm1) + ((vector << (k-n)) * perm2)

def lin_transformation_rs(vector, n, k):
    return vector << (n*k)

def matrix_mult(A, B, n):
    vec_size = A.program.vec_size
    A = vertical_concat(A, n)
    B = vertical_concat(B, n)
    A0 = lin_transformation_sigma(A, n)
    B0 = lin_transformation_tau(B, n)
    A0 = vertical_concat(A0, n, remove=True)
    B0 = vertical_concat(B0, n, remove=True)
    AB = A0*B0
    # A0 = horizontal_concat(A0, n)
    for k in range(1, n):
        A = lin_transformation_cs(A0, n, k)
        B = lin_transformation_rs(B0, n, k)
        AB += A*B
    return AB

def computeDooLittleDecomp(graph, n, iteration, inverse):
    vec_size = graph.program.vec_size 
    print("IN SERVER:", iteration)
    i = iteration
    if iteration == 0:
        laplacian = computeGLaplacian(graph, n)
        remover = n * [0.0] + (n-1) * ([0.0] + (n-1)* [1.0]) + (vec_size - n*n) * [0.0]
        laplacian = remover * laplacian + ([1.0] + (vec_size-1) * [0.0])
        upper = (n*n)*[0.0]+[1.0] + (vec_size-n*n-1 )*[0.0] 
        lower = (2*n*n)*[0.0]+[1.0] + (vec_size-2*n*n-1)*[0.0]  
        ret = laplacian + upper + lower  
    else:
        laplacian = graph * ((n*n) * [1.0] + (vec_size-n*n) * [0.0])
        upper = graph * ((n*n) * [0.0] + (n*n) * [1.0] + (vec_size-2*n*n) * [0.0]) << (n*n)
        lower = graph * ((2*n*n) * [0.0] + (n*n) * [1.0] + (vec_size-3*n*n) * [0.0]) << (2*n*n)
        # mask = getColumnMask2(n, i-1, vec_size, inverse)
        # lower *= mask        
        multiplication = matrix_mult(lower, upper, n)
        row_mask = getRowMask(n,i,vec_size)
        diff = laplacian - multiplication
        upper += maskMatrix(diff, row_mask)
        col_mask = getColumnMask(n, i, vec_size)            
        lower += maskMatrix(diff, col_mask)
        ret = laplacian + (upper >> n*n) + (lower >> 2*n*n) 
    return ret

# This is the dummy analytic service
# You will implement this service based on your selected algorithm
# you can use other parameters as global variables !!! do not change the signature of this function

iteration = 0
inverse = 1
sigma_perm_dict, tau_perm_dict, col_shift_perm_dict = None, None, None

def graphanalticprogram(graph):
    # vec_size = graph.program.vec_size
    # vector1 = graph * ((n*n) * [1.0] + (vec_size-n*n) * [0.0])
    # vector2 = graph * ((n*n) * [0.0] + (n*n) * [1.0] + (vec_size-2*n*n) * [0.0]) << (n*n)
    # # vector1 = copy_vector_in(vector1, n)
    # # vector2 = copy_vector_in(vector2, n)

    # return matrix_mult(vector1, vector2, n)
    return computeDooLittleDecomp(graph, n, iteration, inverse)

    
# Do not change this 
# the parameter n can be passed in the call from simulate function
class EvaProgramDriver(EvaProgram):
    def __init__(self, name, vec_size=4096, n=4):
        self.n = n
        super().__init__(name, vec_size)

    def __enter__(self):
        super().__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)

def checkLaplacian(vector, GG, n):
    computed_laplacian = np.array(vector[:n*n]).reshape((n,n))
    correct_laplacian = nx.laplacian_matrix(GG).toarray()
    correct_laplacian[0,0] = 1
    correct_laplacian[0,1:n] = 0
    correct_laplacian[1:n,0] = 0
    diff = ~(np.abs(computed_laplacian-correct_laplacian) < 1e-2)
    index = np.where(diff)
    print("Laplacian Difference:", np.sum(diff)) 
    if np.sum(diff) > 0:
        print(index, computed_laplacian[index], correct_laplacian[index])

def computeDiagonalMult(vector, n):
    mult = 1
    for i in range(n):
        mult *= vector[i*n+i]
    return mult

def get_new_inputs(outputs, iteration):
    global inverse
    arr = outputs["ReturnedValue"]
    upper = arr[n*n:2*n*n]
    lower = np.array(arr[2*n*n:3*n*n])
    inverse = 1.0/ upper[iteration*n+iteration]
    mask = np.array(getColumnMask2(n, iteration, n*n, inverse))
    lower *= mask 
    arr[2*n*n:3*n*n] = lower.tolist() 
    new_inputs = {"Graph": arr}
    return new_inputs

def run_one_sim_loop(inputs, config, iteration, vec_size, node_size):
    graphanaltic = EvaProgramDriver("graphanaltic", vec_size=vec_size,n=node_size)
    with graphanaltic:
        graph = Input('Graph')
        reval = graphanalticprogram(graph)
        Output('ReturnedValue', reval)
   
    prog = graphanaltic
    prog.set_output_ranges(45)
    prog.set_input_scales(45)

    start = timeit.default_timer()
    compiler = CKKSCompiler(config=config)
    compiled_multfunc, params, signature = compiler.compile(prog)
    compiletime = (timeit.default_timer() - start) * 1000.0 #ms

    start = timeit.default_timer()
    public_ctx, secret_ctx = generate_keys(params)
    keygenerationtime = (timeit.default_timer() - start) * 1000.0 #ms
    
    start = timeit.default_timer()
    encInputs = public_ctx.encrypt(inputs, signature)
    encryptiontime = (timeit.default_timer() - start) * 1000.0 #ms

    start = timeit.default_timer()
    encOutputs = public_ctx.execute(compiled_multfunc, encInputs)        
    executiontime = (timeit.default_timer() - start) * 1000.0 #ms

    start = timeit.default_timer()
    outputs = secret_ctx.decrypt(encOutputs, signature)
    decryptiontime = (timeit.default_timer() - start) * 1000.0 #ms

    start = timeit.default_timer()
    reference = evaluate(compiled_multfunc, inputs)
    referenceexecutiontime = (timeit.default_timer() - start) * 1000.0 #ms

    mse = valuation_mse(outputs, reference) # since CKKS does approximate computations, this is an important measure that depicts the amount of error

    return outputs, (compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, mse)

def prep_matricies(n):
    matrix1 = np.random.rand(n,n)
    matrix2 = np.random.rand(n,n)
    multiplication = np.matmul(matrix1, matrix2)
    inp_vector = matrix1.reshape(-1).tolist() + matrix2.reshape(-1).tolist() + (2*n*n) * [0.0]
    inputs = {"Graph": inp_vector}
    return inputs, multiplication

# Repeat the experiments and show averages with confidence intervals
# You can modify the input parameters
# n is the number of nodes in your graph
# If you require additional parameters, add them
def simulate(n):
    global iteration
    global sigma_perm_dict, tau_perm_dict, col_shift_perm_dict
    sigma_perm_dict = prep_sigma_perm_dict(n)
    tau_perm_dict = prep_tau_perm_dict(n)
    col_shift_perm_dict = prep_col_shift_perm_dict(n)

    m = n*n*4
    print("Will start simulation for ", n)
    config = {}
    config['warn_vec_size'] = 'false'
    config['lazy_relinearize'] = 'true'
    config['rescaler'] = 'always'
    config['balance_reductions'] = 'true'
    inputs, GG = prepareInput(n, m)
    times_list = []
    # inputs, multiplication = prep_matricies(n)
    # outputs, others = run_one_sim_loop(inputs, config, iteration, m, n) 
    # computed_multiplication = np.array(outputs["ReturnedValue"][:n*n]).reshape((n,n))  
    # diff = ~(np.abs(computed_multiplication-multiplication) < 1e-2)
    # index = np.where(diff)
    # print("Multiplication Difference:", np.sum(diff)) 
    while iteration < n:
        outputs, others = run_one_sim_loop(inputs, config, iteration, m, n)
        inputs = get_new_inputs(outputs, iteration)
        #checkLaplacian(outputs["ReturnedValue"], GG, n)
        times_list.append(np.array(others))
        iteration += 1

    determinant = computeDiagonalMult(outputs["ReturnedValue"][n*n:2*n*n], n)
    print("FHE Sol:", determinant, "Correct Sol", len(list(nx.algorithms.tree.mst.SpanningTreeIterator(GG))))
    iteration = 0
    return np.array(times_list).sum(0).tolist() # others 

if __name__ == "__main__":
    simcnt = 100 #The number of simulation runs, set it to 3 during development otherwise you will wait for a long time
    # For benchmarking you must set it to a large number, e.g., 100
    #Note that file is opened in append mode, previous results will be kept in the file
    resultfile = open("results.csv", "a")  # Measurement results are collated in this file for you to plot later on
    resultfile.write("NodeCount,PathLength,SimCnt,CompileTime,KeyGenerationTime,EncryptionTime,ExecutionTime,DecryptionTime,ReferenceExecutionTime,Mse\n")
    resultfile.close()
    
    print("Simulation campaing started:")
    for nc in range(4,64,4): # Node counts for experimenting various graph sizes
        n = nc
        resultfile = open("results.csv", "a") 
        for i in range(simcnt):
            #Call the simulator
            try:
                compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, mse = simulate(n)
                res = str(n) + "," + str(i) + "," + str(compiletime) + "," + str(keygenerationtime) + "," +  str(encryptiontime) + "," +  str(executiontime) + "," +  str(decryptiontime) + "," +  str(referenceexecutiontime) + "," +  str(mse) + "\n"
                print(res)
                resultfile.write(res)
            except Exception as e:
                print(e, "at node count:", n, "sim count:", i)
      
        resultfile.close()