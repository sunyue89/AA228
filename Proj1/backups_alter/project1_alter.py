import sys
import pandas as pd
import time
# from scipy.special import gammaln
from scipy.special import loggamma
import numpy as np
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
# import itertools
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
    G.add_nodes_from(range(len(data.columns)))
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
        score = bayesian_score(variables, G, data)
        k2(variables,G,data)
        new_score = bayesian_score(variables, G, data)
        print(new_score)
        if(new_score <= score):
            break
    #edge_direction_optimization(G, data)

    write_gph(G, outfile1)
    draw_graph(G, outfile2)
    end = time.time()
    print(end-start)


def k2(variables, G, data):

    for k, i in enumerate(G):
        y = bayesian_score(variables, G, data)
        while True:
            y_best, j_best = -np.inf, 0
            for j in range (k):
                if not G.has_edge(j, i):
                    G.add_edge(j, i)
                    y_prime = bayesian_score(variables, G, data)
                    if nx.is_directed_acyclic_graph(G) and y_prime > y_best:
                        y_best, j_best = y_prime, j
                    G.remove_edge(j, i)
            if y_best > y:
                y = y_best
                G.add_edge(j_best, i)
            else:
                break

# def edge_direction_optimization(G, data):
#     scores = {node: bayesian_score_one_node(G, node, data) for node in G}

#     for edge in list(G.edges):
#         curr_score = sum(scores.values())
#         Gtemp = G.copy()

#         parent, child = edge
#         parent_score = scores[parent]
#         child_score = scores[child]
#         Gtemp.remove_edge(*edge)
#         Gtemp.add_edge(*edge[::-1])

#         if not nx.is_directed_acyclic_graph(Gtemp):
#             continue

#         new_parent_score = bayesian_score_one_node(Gtemp, parent, data)
#         new_child_score = bayesian_score_one_node(Gtemp, child, data)
#         if new_parent_score + new_child_score > parent_score + child_score:
#             G.remove_edge(*edge)
#             G.add_edge(*edge[::-1])
#             scores[parent] = new_parent_score
#             scores[child] = new_child_score


def main():
    if len(sys.argv) != 4:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename1 = sys.argv[2]
    outputfilename2 = sys.argv[3]
    compute(inputfilename, outputfilename1, outputfilename2)


if __name__ == '__main__':
    main()
