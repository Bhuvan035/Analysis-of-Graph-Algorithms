# explanations for member functions are provided in requirements.py
# each file that uses a graph should import it from this file.

from collections.abc import Iterable

class Graph:
	def __init__(self, num_nodes: int, edges: Iterable[tuple[int, int]]):
		self.num_nodes=num_nodes
		self.neighbors_dict = {}
		for u,v in edges:
			if u in self.neighbors_dict:
				self.neighbors_dict[u].append(v)
			else:
				self.neighbors_dict[u]=[v]
			
			if v in self.neighbors_dict:
				self.neighbors_dict[v].append(u)
			else:
				self.neighbors_dict[v]=[u]
			
		
	def get_num_nodes(self) -> int:
		return self.num_nodes
		# raise NotImplementedError

	def get_num_edges(self) -> int:
		count=0
		for i in self.neighbors_dict.values():
			count+=len(i)
		return count//2   #Double edge count
		# raise NotImplementedError

	def get_neighbors(self, node: int) -> Iterable[int]:
		if node in self.neighbors_dict:
			return self.neighbors_dict[node]
		else:
			return []
		# raise NotImplementedError

	# feel free to define new methods in addition to the above
	# fill in the definitions of each required member function (above),
	# and for any additional member functions you define
