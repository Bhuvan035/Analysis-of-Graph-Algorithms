# explanations for these functions are provided in requirements.py

from graph import Graph
from collections import deque

def get_diameter(graph: Graph) -> int:
    def bfs(start: int) -> dict:
        distances={start: 0}
        queue=deque([start])
        
        while queue:
            node=queue.popleft()
            for i in graph.get_neighbors(node):
                if i not in distances:
                    distances[i]=distances[node] + 1
                    queue.append(i)
        
        return distances

    Dmax=0
	#Finding the first farthest node
    r=0
    distances =bfs(r)
    #print(distances)
    farthest_node = max(distances, key=distances.get)
    farthest_distance = distances[farthest_node]
    #print(farthest_distance)
    if farthest_distance > Dmax:
        Dmax=farthest_distance

	#Repeating for w
    r=farthest_node
    distances=bfs(r)
    farthest_node=max(distances, key=distances.get)
    farthest_distance = distances[farthest_node]
    if farthest_distance > Dmax:
        Dmax= farthest_distance

    return Dmax

'''def count_triangles(graph: Graph) -> int:
    triangles_count = 0
    
    for u in graph.neighbors_dict:
        for v in graph.neighbors_dict[u]:
            if v > u:  
                for w in graph.neighbors_dict[v]:
                    if w > v and w in graph.neighbors_dict[u]:
                        triangles_count += 1 
    
    return triangles_count'''

def generate_degeneracy_list(graph: Graph) -> list[int]:
    L =[]  #Output list
    degree_list ={}  # List for all degrees
    for i in range(graph.get_num_nodes()):
        degree_list[i] =len(graph.get_neighbors(i))  
    
    D =[]  #Vertices with same degree
    for i in range(graph.get_num_nodes()+1):
        D.append([])  # Initially empty
    
    for vertex, degree in degree_list.items():
        D[degree].append(vertex)

    #Nv ={}  #For list of neighbors 
    #for i in range(graph.get_num_nodes()):
    #    Nv[i] =[]

    k = 0  
    HL = set() 

    while len(L) < graph.get_num_nodes():
        for i in range(len(D)):
            if D[i]:
                k = max(k, i)  
                v = D[i].pop()  
                L.append(v) 
                HL.add(v)  

                for neighbor in graph.get_neighbors(v):
                    if neighbor not in HL:  
                        old_d = degree_list[neighbor]  #Subtracting 1 from dw
                        new_d = old_d - 1
                        degree_list[neighbor] = new_d

                        D[old_d].remove(neighbor)  #moving to the new cell
                        D[new_d].append(neighbor)
                        
                        #Nv[neighbor].append(v)
                break 

    return L


def count_triangles(graph: Graph) -> int:
    L = generate_degeneracy_list(graph)

    pos_in_L = {}
    for index, v in enumerate(L):
        pos_in_L[v] = index
    
    triangles_count = 0
    
    for v in L:
        for u in graph.get_neighbors(v):
            if pos_in_L[u] < pos_in_L[v]:  
                for w in graph.get_neighbors(v):
                    if pos_in_L[w] < pos_in_L[v] and w != u: 
                        if w in graph.get_neighbors(u) and u < w:  
                            triangles_count += 1

    return triangles_count


def get_clustering_coefficient(graph: Graph) -> float:
    def get_two_edge_paths(graph: Graph) -> int:
        two_edge_pathscount = 0
        
        for i in graph.neighbors_dict:
            d = len(graph.neighbors_dict[i])  
            two_edge_pathscount += d*(d-1)//2
        
        return two_edge_pathscount

    #print(count_triangles(graph),get_two_edge_paths(graph))

    return (3*count_triangles(graph))/get_two_edge_paths(graph)
	
def get_degree_distribution(graph: Graph) -> dict[int, int]:
    degree_distribution_dict = {}
 
    for i in graph.neighbors_dict.values():
        d = len(i)  
        if d in degree_distribution_dict:
            degree_distribution_dict[d] += 1  
        else:
            degree_distribution_dict[d] = 1  
    
    return degree_distribution_dict
