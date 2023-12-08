import sys
import pandas as pd
import time
from scipy.special import gammaln
import networkx as nx
from networkx.drawing.nx_pydot import write_dot
import itertools
import graphviz


def write_gph(G, filename):
    edges = []
    for key, values in nx.to_dict_of_lists(G).items():
        for value in values:
            edges.append("{}, {}\n".format(key, value))

    with open(filename, 'w') as f:
        f.writelines(edges)

def init_graph(data):
    G = nx.DiGraph()
    G.add_nodes_from(list(data.columns))
    return G

def draw_graph(G, filename):
    with open(filename,'w') as f:
        write_dot(G,f.name)
        img_file = graphviz.render('dot','png',f.name)

def compute(infile, outfile1, outfile2):
    data = pd.read_csv(infile)
    G = init_graph(data)

    start = time.time()
    while True:
        score = bayesian_score(G, data)
        k2(G,data)
        edge_direction_optimization(G, data)
        if score == bayesian_score(G, data):
            break
    write_gph(G, outfile1)
    draw_graph(G, outfile2)
    end = time.time()
    print(end-start)

def bayesian_score_one_node(G, node, data):
    score = 0
    ri = max(data[node])
    parents = sorted(list(G.pred[node]))
    parent_inst = itertools.product(*[data[p].unique() for p in parents])

    for inst in parent_inst:
        tmp = data.copy()
        for l in range(len(parents)):
            tmp = tmp[tmp[parents[l]] == inst[l]]

        score += gammaln(ri) - gammaln(ri + len(tmp))

        for k in range(ri):
            m = len(tmp[tmp[node] == k + 1])
            score += gammaln(1+m)

    return score

def k2(G, data):
    scores = [bayesian_score_one_node(G, node, data) for node in G]

    for idx, node in enumerate(G):
        parents = []

        while True:
            curr_score = sum(scores)
            curr_best_local_score = scores[idx]
            curr_best_parent = None

            for new_parent in G:
                Gtemp = G.copy()

                if new_parent == node or new_parent in Gtemp.pred[node] or node in Gtemp.pred[new_parent]:
                    continue

                Gtemp.add_edge(new_parent, node)
                new_local_score = bayesian_score_one_node(Gtemp, node, data)
                if nx.is_directed_acyclic_graph(Gtemp) and new_local_score > curr_best_local_score:
                    curr_best_local_score = new_local_score
                    curr_best_parent = new_parent

            if curr_best_parent:
                G.add_edge(curr_best_parent, node)
                parents.append(curr_best_parent)
                scores[idx] = curr_best_local_score

                if len(parents) >= 1:
                    break

            else:
                break

def edge_direction_optimization(G, data):
    scores = {node: bayesian_score_one_node(G, node, data) for node in G}

    for edge in list(G.edges):
        curr_score = sum(scores.values())
        Gtemp = G.copy()

        parent, child = edge
        parent_score = scores[parent]
        child_score = scores[child]
        Gtemp.remove_edge(*edge)
        Gtemp.add_edge(*edge[::-1])

        if not nx.is_directed_acyclic_graph(Gtemp):
            continue

        new_parent_score = bayesian_score_one_node(Gtemp, parent, data)
        new_child_score = bayesian_score_one_node(Gtemp, child, data)
        if new_parent_score + new_child_score > parent_score + child_score:
            G.remove_edge(*edge)
            G.add_edge(*edge[::-1])
            scores[parent] = new_parent_score
            scores[child] = new_child_score

def bayesian_score(G,data):
    return sum(bayesian_score_one_node(G,node,data) for node in G)

def main():
    if len(sys.argv) != 4:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename1 = sys.argv[2]
    outputfilename2 = sys.argv[3]
    compute(inputfilename, outputfilename1, outputfilename2)


if __name__ == '__main__':
    main()
