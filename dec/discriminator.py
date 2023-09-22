from collections import defaultdict


def get_discrim_score():
    return 0

class Book:
    # book keeping all the nodes and their edges
    def __init__(self) -> None:
        self.children = defaultdict(list)
        self.all_nodes = {}
    def add_child(self, par, kid):
        self.children[par].append(kid)
