
# coding: utf-8

# In[1]:


from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time 
from TwitterAPI import TwitterAPI


# In[2]:


consumer_key = 'tNkGzG2gFIQCfLwBXb8YeclEh'
consumer_secret = 'kX88kGqkWD7yvpttAP6DtJUrEKnvnn4X2l0kQn7qszK4sYN38N'
access_token = '1086860929608880133-jsHyGz5fJqKpM5OO6ASTdKGE7FwUU9'
access_token_secret = 'S9KNFCuvi8EzaCO7fDDE6CL0nPYpBG5MYadUpaQg6p2CF'


# In[3]:


def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


# In[4]:


def read_screen_names(filename):
    path = 'E:/MS in CS/SEM 2/OSNA/Assignments/juhideshpande/a0/candidates.txt'
    candidates_file = open(path,'r')
    return(candidates_file.read().split())


# In[5]:


# read_screen_names('candidates.txt')


# In[6]:


def robust_request(twitter, resource, params, max_tries=5):
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


# In[7]:


def get_users(twitter, screen_names):
    r1=list(robust_request(twitter,'users/lookup',{'screen_name': screen_names},max_tries=5))
    rq=[r for r in r1]
    return rq


# In[8]:


# >>> twitter = get_twitter()
# >>> users = get_users(twitter, ['twitterapi', 'twitter'])
# >>> [u['id'] for u in users]
# [6253282, 783214]


# In[9]:


def get_friends(twitter, screen_name):
    r1=list(robust_request(twitter,'friends/ids',{'screen_name': screen_name},max_tries=5))
    return sorted(r1)


# In[10]:


# twitter = get_twitter()
# get_friends(twitter, 'aronwc')[:5]
#[695023, 1697081, 8381682, 10204352, 11669522]


# In[11]:


def add_all_friends(twitter, users):
    for m in users:  
        m['friends']=get_friends(twitter,m['screen_name'])       


# In[12]:


# twitter = get_twitter()
# users = [{'screen_name': 'aronwc'}]
# add_all_friends(twitter, users)
# users[0]['friends'][:5]
#[695023, 1697081, 8381682, 10204352, 11669522]


# In[13]:


def print_num_friends(users):
# #     for m in users:  
# # #         m['friends']=get_friends(twitter,m['screen_name'])
# #         a = add_all_friends(twitter, m['friends'])
# # #         print(len(users[m]))
# #         print(str(len(a)))
#      for i in users:
#         print(i['screen_name'] + " " + str(len(i['friends'])))

     for i in users:
         print(i['screen_name'] + " " +str(len(i['friends'])))

#     data = sorted(users, key=lambda x: x['screen_name'])
#     for i in range(0,len(data)):
#         print(i['screen_name'],str(len(i['friends'])),end='\n')


# In[14]:


# twitter = get_twitter()
# users = [{'screen_name': 'aronwc'}]
# print_num_friends(users)
# users[0]['friends'][:5]


# In[15]:


def count_friends(users):
    a = Counter()
    for m in users:
        a.update(m['friends'])
    return a


# In[16]:


# c = count_friends([{'friends': [1,2]}, {'friends': [2,3]}, {'friends': [2,3]}])
# c.most_common()


# In[17]:


# print(users)


# In[18]:


def friend_overlap(users):
    check_list=[]
    for m in users:
        for n in users:
            if(m['screen_name']!=n['screen_name']):
                o=(m['screen_name'],n['screen_name'],len(set(m['friends']).intersection(n['friends'])))
                check_list.append(o)
    sorted_list=[]
    for k in check_list:                          
        if not (k in sorted_list or tuple([k[1], k[0], k[2]]) in sorted_list): 
             sorted_list.append(k)
    sorted_list.sort(key=lambda tup: (-tup[2],tup[0],tup[1]))
    return sorted_list


# In[19]:


#   >>> friend_overlap([{'screen_name': 'a', 'friends': ['1', '2', '3']}, {'screen_name': 'b', 'friends': ['2', '3', '4']}, {'screen_name': 'c', 'friends': ['1', '2', '3']}])
#     [('a', 'c', 3), ('a', 'b', 2), ('b', 'c', 2)]


# In[20]:


def followed_by_hillary_and_donald(users, twitter):
    twitter = get_twitter()
    followed_list = []
    hc=get_friends(twitter,'HillaryClinton')
    dt=get_friends(twitter,'realDonaldTrump')
    both=sorted(set(hc).intersection(dt))   
    info = robust_request(twitter,'users/lookup',{'user_id': both},max_tries=5)
    for i in info:
        followed_list.append(i['screen_name'])
    return followed_list


# In[21]:


def create_graph(users, friend_counts):
    graph = nx.Graph()
    for sn in users:
        graph.add_node(sn['screen_name'])
        for friend in sn['friends']:
            if(friend_counts[friend]>1):
                graph.add_node(friend)
                graph.add_edge(sn['screen_name'], friend)                
    nx.draw(graph, with_labels=True)         
    return graph  


# In[22]:


def draw_network(graph, users, filename):
    layout=nx.spring_layout(graph)
    nodeLabels={}
    nl=[]
    for sn in users:
        nl.append(sn['screen_name'])
    for each in nl:
        nodeLabels[each]=each
    plt.figure(figsize=(10,10))
    nx.draw_networkx(graph,pos=layout, with_labels=False)
    nx.draw_networkx_labels(graph,layout,node_size=50,alpha=.6,labels=nodeLabels,with_labels=True,arrows=False,font_size=16,font_color='r', width=.2,)
    plt.axis("off")
    plt.savefig(filename, format="PNG",dpi=40)
    plt.show()    


# In[23]:


def main():
    """ Main method. You should not modify this. """
    twitter = get_twitter()
    screen_names = read_screen_names('candidates.txt')
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
    add_all_friends(twitter, users)
    print('Friends per candidate:')
    print_num_friends(users)
    friend_counts = count_friends(users)
    print('Most common friends:\n%s' % str(friend_counts.most_common(5)))
    print('Friend Overlap:\n%s' % str(friend_overlap(users)))
    print('User followed by Hillary and Donald: %s' % followed_by_hillary_and_donald(users, twitter))

    graph = create_graph(users, friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    draw_network(graph, users, 'network.png')
    print('network drawn to network.png')


if __name__ == '__main__':
    main()

