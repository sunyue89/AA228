import sys
import pandas as pd
import time
from scipy.special import loggamma
import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import graphviz
from ch02 import Variable
from ch05 import bayesian_score


def write_gph(G, filename):
    edges = []
    for key, values in nx.to_dict_of_lists(G).items():
        for value in values:
            edges.append("{}, {}\n".format(key, value))

    with open(filename, 'w') as f:
        f.writelines(edges)

def init_graph(data):
    G = nx.DiGraph()
    G.add_nodes_from(data.columns)
    return G

def init_variables(data):
    variables = []
    for i in data.columns:
        variables.append(Variable(i, max(data[i])))
    return variables

def init_data(data):
    return data.to_numpy().T-1

def draw_graph(G, filename):
    with open(filename,'w') as f:
        write_dot(G,f.name)
        img_file = graphviz.render('dot','png',f.name)

def compute(infile, outfile1, outfile2):
    rawdata = pd.read_csv(infile)
    G = init_graph(rawdata)
    variables = init_variables(rawdata)
    data = init_data(rawdata)
    score = bayesian_score(variables,G,data)
    print(score)

    start = time.time()

    while True:
        k2(variables,G,data)
        localsearch(variables,G,data,1)
        new_score = bayesian_score(variables, G, data)
        print(new_score)
        if(new_score <= score):
            break
        score = new_score

    write_gph(G, outfile1)
    draw_graph(G, outfile2)
    end = time.time()
    print(end-start)


def k2(variables, G, data):
    for k, i in enumerate(G):
        y = bayesian_score(variables, G, data)
        while True:
            y_best, j_best = -np.inf, list(G)[0]
            for j in range (k):
                nj = list(G)[j]
                if not G.has_edge(nj, i):
                    G.add_edge(nj, i)
                    y_prime = bayesian_score(variables, G, data)
                    if nx.is_directed_acyclic_graph(G) and y_prime > y_best:
                        y_best, j_best = y_prime, nj
                    G.remove_edge(nj, i)
            if y_best > y:
                y = y_best
                G.add_edge(j_best, i)
            else:
                break

def localsearch(variables, G,  data, k_max):
    y = bayesian_score(variables, G, data)
    for k in range(k_max):
        graph_prime = rand_graph_neighbor(G)
        if len(list(nx.simple_cycles(graph_prime))) == 0:
            y_prime = bayesian_score(variables, graph_prime, data)
        else:
            y_prime = -np.inf
        if y_prime > y:
            y, G = y_prime, graph_prime

def rand_graph_neighbor(graph) -> nx.DiGraph:
    n = graph.number_of_nodes()
    i = np.random.randint(low=0, high=n)
    j = (i + np.random.randint(low=1, high=n) - 1) % n
    ni = list(graph)[i]
    nj = list(graph)[j]
    graph_prime = graph.copy()
    if graph.has_edge(ni, nj):
        graph_prime.remove_edge(ni, nj)
    else:
        graph_prime.add_edge(ni, nj)
    return graph_prime

def main():
    if len(sys.argv) != 4:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename1 = sys.argv[2]
    outputfilename2 = sys.argv[3]
    compute(inputfilename, outputfilename1, outputfilename2)

if __name__ == '__main__':
    main()
