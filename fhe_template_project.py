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

def getColumnMask(n, c, vec_size):
    mask = (vec_size)*[0.0]
    for i in range(n):
        mask[i*n+c] = 1.0
    return mask

def getRowMask(n, r, vec_size):
    mask = r*n*[0.0] + n*[1.0] +  (vec_size-(r+1)*n)* [0.0]
    return mask

def maskMatrix(graph, mask):
    return graph*mask

def computeDooLittleDecomp(graph, n, inverse_times_client):
    vec_size = graph.program.vec_size 
    print("IN SERVER:", inverse_times_client)
    i = inverse_times_client
    if inverse_times_client == 0:
        laplacian = computeGLaplacian(graph, n)
        remover = n * [0.0] + (n-1) * ([0.0] + (n-1)* [1.0]) + (vec_size - n*n) * [0.0]
        laplacian = remover * laplacian + ([1.0] + (vec_size-1) * [0.0])
        upper = (n*n)*[0.0]+[1.0] + (vec_size-n*n-1 )*[0.0] #(n*[1.0] + (vec_size-n)*[0.0]) * laplacian
        lower = (2*n*n)*[0.0]+[1.0] + (vec_size-2*n*n-1)*[0.0] #(n*([1.0] + (n-1)*[0.0]) + (vec_size-n*n)* [0.0]) * laplacian
        ret = laplacian + upper + lower #(upper >> n*n) + (lower >> 2*n*n)
    else:
        matrix = graph * ((n*n) * [1.0] + (vec_size-n*n) * [0.0])
        upper = graph * ((n*n) * [0.0] + (n*n) * [1.0] + (vec_size-2*n*n) * [0.0]) << (n*n)
        lower = graph * ((2*n*n) * [0.0] + (n*n) * [1.0] + (vec_size-3*n*n) * [0.0]) << (2*n*n)
        inv_element = graph << (3*n*n)
        mask = getColumnMask(n, i-1, vec_size)
        inverse = copyFirstN2Times(inv_element, n)
        (n*([1.0] + (n-1)*[0.0]) + (vec_size-n*n)* [0.0])
        mask2 = n*((i-1) * [1.0] + [0.0] + (n-i)*[1.0]) + (vec_size-n*n)*[0.0]
        lower = maskMatrix(lower, inverse*mask) + mask2*lower

        lij = (i*n*[0.0] + i*[1.0] + (vec_size - (n+1)*i)* [0.0]) * lower
        temp1 = vec_size * [0.0]
        for j in range(i):
            temp1 += (lij << ((i-j)*n+j)) * (j*n * [0.0] + [1.0] + (vec_size -j*n-1) * [0.0])
        for k in range(n):
            if k == 0:
                temp2 = upper * temp1
            else:
                temp2 += upper * temp1 
            temp1 >>= 1  
        temp3 = vec_size * [0.0]
        for t in range(i):
            temp3 += temp2 * (n*[1.0] + (vec_size-n)*[0.0]) #+ (temp2 << t*n) * (n*[1.0] + (vec_size-n)*[0.0])
            temp2 <<= n
        if i == 1:
            temp3 = temp2
        temp3 >>= i*n

        mask = getRowMask(n, i, vec_size)
        upper += maskMatrix(matrix - temp3, mask) 
        
        uji = (i*(i*[0.0] + [1.0] + (n-1-i) * [0.0])+ (vec_size-i*n)*[0.0]) * upper
        temp1 = vec_size * [0.0]
        for j in range(i):
            temp1 += (uji << (i + j*(n-1))) * (j*[0.0] + [1.0] + (vec_size-j-1) * [0.0])
        for k in range(n):
            if k == 0:
                temp2 = lower * temp1
            else:
                temp2 += lower * temp1
            temp1 >>= n
        temp3 = vec_size * [0.0]
        for t in range(i):
            mask = getColumnMask(n,t, vec_size)
            temp3 += maskMatrix(temp2, mask) << t
        if i == 1:
            temp3 = temp2        
        temp3 >>= i
        mask = getColumnMask(n, i, vec_size)            
        lower += maskMatrix(matrix - temp3, mask)
        ret = matrix + (upper >> n*n) + (lower >> 2*n*n)
    return ret


def computeDiagonalMult(vector, n):
    mult = 1
    for i in range(n):
        mult *= vector[i*n+i]
    return mult
# This is the dummy analytic service
# You will implement this service based on your selected algorithm
# you can use other parameters as global variables !!! do not change the signature of this function

inverse_times_client = 0 # same variable for client  
def graphanalticprogram(graph):
    ret = computeDooLittleDecomp(graph, n, inverse_times_client)
    return ret
    
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

# Repeat the experiments and show averages with confidence intervals
# You can modify the input parameters
# n is the number of nodes in your graph
# If you require additional parameters, add them
def simulate(n):
    global inverse_times_client
    m = 32*32*4
    print("Will start simulation for ", n)
    config = {}
    config['warn_vec_size'] = 'false'
    config['lazy_relinearize'] = 'true'
    config['rescaler'] = 'always'
    config['balance_reductions'] = 'true'
    inputs, GG = prepareInput(n, m)

    while inverse_times_client < n:
        graphanaltic = EvaProgramDriver("graphanaltic", vec_size=m,n=n)
        with graphanaltic:
            graph = Input('Graph')
            reval = graphanalticprogram(graph)
            Output('ReturnedValue', reval)
        
        prog = graphanaltic
        prog.set_output_ranges(45)
        prog.set_input_scales(45)

        print("Compiling the Program")
        start = timeit.default_timer()
        compiler = CKKSCompiler(config=config)
        compiled_multfunc, params, signature = compiler.compile(prog)
        compiletime = (timeit.default_timer() - start) * 1000.0 #ms

        print("Generating the Keys")
        start = timeit.default_timer()
        public_ctx, secret_ctx = generate_keys(params)
        keygenerationtime = (timeit.default_timer() - start) * 1000.0 #ms
        
        print("Encrypt Data")
        start = timeit.default_timer()
        encInputs = public_ctx.encrypt(inputs, signature)
        encryptiontime = (timeit.default_timer() - start) * 1000.0 #ms
        print("Execute")
        start = timeit.default_timer()
        encOutputs = public_ctx.execute(compiled_multfunc, encInputs)
        outputs = secret_ctx.decrypt(encOutputs, signature)
        # print(type(outputs))
        arr = outputs["ReturnedValue"]
        upper = arr[n*n:2*n*n]

        laplacian = arr[:n*n]
        inverseee = 1/ upper[inverse_times_client*n+inverse_times_client]
        print("inv ujj", inverseee)
        arr[3*n*n] = inverseee
        inputs = {"Graph": arr}
        inverse_times_client += 1
        executiontime = (timeit.default_timer() - start) * 1000.0 #ms

        start = timeit.default_timer()
        outputs = secret_ctx.decrypt(encOutputs, signature)
        decryptiontime = (timeit.default_timer() - start) * 1000.0 #ms

        start = timeit.default_timer()
        reference = evaluate(compiled_multfunc, inputs)
        referenceexecutiontime = (timeit.default_timer() - start) * 1000.0 #ms
    determinant = computeDiagonalMult(upper,n)

    # Change this if you want to output something or comment out the two lines below
    # for key in outputs:
    # print(upper)
    print(np.array(laplacian).reshape(n,n), nx.laplacian_matrix(GG).toarray())
    print(np.array(upper).reshape(n,n))
    print()
    print(np.array(arr[2*n*n:3*n*n]).reshape(n,n))
    print(determinant, len(list(nx.algorithms.tree.mst.SpanningTreeIterator(GG))))


    mse = valuation_mse(outputs, reference) # since CKKS does approximate computations, this is an important measure that depicts the amount of error

    return compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, mse


if __name__ == "__main__":
    simcnt = 3 #The number of simulation runs, set it to 3 during development otherwise you will wait for a long time
    # For benchmarking you must set it to a large number, e.g., 100
    #Note that file is opened in append mode, previous results will be kept in the file
    resultfile = open("results.csv", "a")  # Measurement results are collated in this file for you to plot later on
    resultfile.write("NodeCount,PathLength,SimCnt,CompileTime,KeyGenerationTime,EncryptionTime,ExecutionTime,DecryptionTime,ReferenceExecutionTime,Mse\n")
    resultfile.close()
    
    print("Simulation campaing started:")
    for nc in range(32,64,4): # Node counts for experimenting various graph sizes
        n = nc
        resultfile = open("results.csv", "a") 
        for i in range(simcnt):
            #Call the simulator
            compiletime, keygenerationtime, encryptiontime, executiontime, decryptiontime, referenceexecutiontime, mse = simulate(n)
            res = str(n) + "," + str(i) + "," + str(compiletime) + "," + str(keygenerationtime) + "," +  str(encryptiontime) + "," +  str(executiontime) + "," +  str(decryptiontime) + "," +  str(referenceexecutiontime) + "," +  str(mse) + "\n"
            print(res)
            resultfile.write(res)
            break
        break
        resultfile.close()