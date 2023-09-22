from heapq import heappop, heappush
from collections import UserDict, defaultdict
from heapq import heappop
from turtle import color
from pyvis.network import Network
import networkx as nx
import tqdm
import pickle
import os

def draw_nodes(net, nodes, group_num=0, size=8):
    for node in nodes.values():
        # form = "{:.1f}".format(node['score'])
        net.add_node(
            node['uid'], label=f"{node['text']}", shape='dot', group=group_num, size=size)


def draw_edges(net, edges):
    for edge in edges.values():
        form = "{:.1f}".format(edge['score'])
        net.add_edge(edge['src'], edge['tgt'], title=form,
                     arrowStrikethrough=False)


def viz_result(completed, tokenizer):
    print(f"Number of completed hypothesis: {len(completed)}")

    net = Network(height='1500px', width='100%', directed=True)

    seen_nodes = {}
    seen_edges = {}
    # net.toggle_stabilization(False)
    # while frontier:
    #     _, hypo = heappop(frontier)
    #     nodes, edges = hypo.visualization(tokenizer)
    #     draw_nodes(net,nodes, 9999, size=6)
    #     draw_edges(net,edges,9999)
    # first set all nodes and edges
    for idx, hypo in enumerate(completed):
        nodes, edges = hypo.visualization(tokenizer)
        seen_nodes = seen_nodes | nodes
        seen_edges = seen_edges | edges
    draw_nodes(net, seen_nodes, idx, size=9)
    draw_edges(net, seen_edges)
    return net


def construct(frontier, completed, tokenizer):
    if frontier and isinstance(frontier[0], tuple):
        frontier = [x[1] for x in frontier]
    if isinstance(completed[0], tuple):
        completed = [x[1] for x in completed]
    edges = {}
    nodes = {}

    # record all of the children nodes, top_down['root'] = [root_node_uid]
    top_down = defaultdict(list)
    seen = {}

    def dfs(node, state):
        if not node:
            return

        uid = node.uid
        if uid in seen:
            return
        seen[uid] = True

        step = node.glob_step
        token_idx = node.token_idx
        uid = node.uid
        prob = node.prob
        acc_score = node.acc_score
        length = node.length

        nodes[uid] = {
            'uid': uid,
            'step': step,
            'token_idx': token_idx,
            'tok':tokenizer.decode(token_idx),
            'prob': prob,
            'acc_score': acc_score,
            'length': length,
            'compl': state
        }
        if not node.prev:
            top_down['root'] = [uid]
            return
        prev_node = node.prev[0]
        prev_uid, prev_prob = prev_node.uid, prev_node.prob
        edges[f"{prev_uid}_{uid}"] = {
            'src': prev_uid,
            'tgt': uid,
            'prob': prob
        }
        top_down[prev_uid].append(uid)
        dfs(prev_node, state)

    for comp in completed:
        dfs(comp, True)
    if frontier:
        for uncomp in frontier:
            dfs(uncomp, False)
    return edges, nodes, top_down

from collections import Counter

def count_children_number(top_down):
    counter = Counter()
    start = top_down['root'][0]

    def dfs(node_id):
        if node_id in counter:
            return counter[node_id]
        kids = top_down[node_id]
        counter[node_id] = 1
        for k in kids:
            counter[node_id] += dfs(k)
        return counter[node_id]
    total_cnt = dfs(start)
    return counter, total_cnt

def assign_pos(node_uid, children_cnt, start_pos, edges, nodes, top_down,location_map):

    location_map[node_uid] = [nodes[node_uid]['length'], start_pos]
    kids = top_down[node_uid]
    offset = 0
    for k in kids:

        assign_pos(k, children_cnt, start_pos + offset , edges, nodes, top_down,location_map)
        offset += children_cnt[k]
    

def visualize_fixed(output_dict, tokenizer):
    
    net = Network(height='100%', width='100%')
    # first draw the incomplete trees, and then completed so we can overide the completed states
    completed = output_dict['completed']
    frontier = output_dict['frontier']
    edges, nodes, top_down = construct(frontier, completed, tokenizer)
    children_cnt, total_cnt = count_children_number(top_down)
    print('...')
    # start span, end span
    # assign x and y to all nodes, x is time step, y is the position
    location_map = {}
    assign_pos(top_down['root'][0], children_cnt,0,edges, nodes, top_down,location_map)
    
    # draw nodes
    for node in nodes.values():
        if node['compl']:
            c = 'red'
        else:
            c = 'blue'
        net.add_node(node['uid'], label=f"{node['tok']}", shape='dot', x=location_map[node['uid']][0]*150, y=location_map[node['uid']][1] * 10, color=c, size = 2.5)


    # draw edges
    for edge in edges.values():
        form = "{:.1f}".format(edge['prob'])
        net.add_edge(edge['src'], edge['tgt'], title=form, width=2 * edge['prob'], arrowStrikethrough=False)  #arrowStrikethrough=False

    for n in net.nodes:
        n.update({'physics': False})
    return net

if __name__ == "__main__":
    # # execute only if run as a script
    # prefix = 'sum'
    # suffix = '.pkl'
    # suffix = 'astar_15_35_False_0.4_True_False_4_5_imp_False_0.0_0.9.pkl'
    # files = os.listdir('vizs')
    # files = [f for f in files if f.endswith(suffix) and f.startswith(prefix)]
    print(os.getcwd())
    with open('vizs/bs_1.05_10_5_40375560.pkl','rb') as fd:
        output_dict = pickle.load(fd)
    print(output_dict)
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-xsum')

    files = os.listdir('vizs')
    files = [x for x in files if x.endswith('pkl')]
    for f in files:
        name = ".".join(f.split('.')[:-1])
        with open(f"vizs/{f}", 'rb') as fd:
            output_dict = pickle.load(fd)
        net = visualize_fixed(output_dict)
        net.show(f"vizs/new-{name}.html")
        print(name)
        break

