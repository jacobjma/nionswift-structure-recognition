import os
from .graph import GeometricGraph
import json

def walk_dir(path, ending):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if file[-len(ending):] == ending:
                files.append(os.path.join(r, file))

    return files


def load_library(name):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.join('../libraries/', name))
    paths = walk_dir(path, '.json')

    library = {}
    alias = {}
    for path in paths:
        if os.path.split(path)[-1] == 'alias.json':
            with open(path) as f:
                alias = json.load(f)
        else:
            library[os.path.basename(path)] = GeometricGraph.read(path)

    return library, alias
