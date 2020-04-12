#!/usr/bin/env python
# coding: utf-8

# In[81]:


import networkx as nx


# In[82]:


def jaccard_wt(graph, node):
    """
    The weighted jaccard score, defined above.
    Args:
      graph....a networkx graph
      node.....a node to score potential new edges for.
    Returns:
      A list of ((node, ni), score) tuples, representing the 
                score assigned to edge (node, ni)
                (note the edge order)
    """
    answer = set(graph.neighbors(node))
    credits = []
    
    for n in graph.nodes():
        if n != node and not graph.has_edge(node,n):
            answer_b = set(graph.neighbors(n))
            answer_c = answer & answer_b
            degrees = 0.0
            for h in answer_c:
                degrees += 1.0/float(graph.degree(h))

            degree_1 = 0.0
            degree_2 = 0.0
      
            for h in answer:
                degree_1 += graph.degree(h)
            
            degree_1 = 1/degree_1  
        
            for h in answer_b:
                degree_2 += graph.degree(h)

            degree_2 = 1/degree_2
            
            score_temp = degrees/(degree_1+degree_2)
            
            credits.append(((node,n),score_temp))
    
    return sorted(credits, key=lambda x: (-x[1],x[0][1]))


# In[ ]:




