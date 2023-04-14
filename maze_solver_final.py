#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import numpy as np

#create a maze
#0: clear path
#1: wall
#2: start point
#3: goal point

#maze = np.array([
#    [0, 1, 0, 1, 0, 1, 0, 0, 1, 3],
#    [0, 1, 0, 1, 0, 0, 0, 1, 1, 0],
#    [0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
#    [0, 0, 0, 1, 0, 1, 0, 1, 1, 0],
#    [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
#    [1, 0, 1, 1, 0, 1, 0, 0, 0, 0],
#    [1, 0, 0, 0, 0, 1, 0, 1, 1, 1],
#    [1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
#    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
#    [2, 1, 1, 1, 1, 1, 0, 1, 1, 1]
#])

#height = 10
#width = 10

#print(maze)


# In[ ]:


class Node():
    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.x = state[0]
        self.y = state[1]
        self.parent = parent
        self.action = action
        self.manhattan_distance = manhattan_distance(state)
        self.path_cost = path_cost
        self.evaluation_func = self.manhattan_distance + path_cost
    
    def get_manhattan_distance(self):
        return self.manhattan_distance

    def get_evaluation_func(self):
        return self.evaluation_func


# In[ ]:


class stack_frontier():
    def __init__(self):
        self.frontier = []
    
    #add node to the frontier
    def add(self, node):
        self.frontier.append(node)
    
    #remove last node of the frontier *LIFO
    def remove(self):
        node = self.frontier[-1]
        self.frontier = self.frontier[:-1]
        return node
    
    #check if the frontier is empty
    def isEmpty(self):
        return len(self.frontier) == 0
    
    # return length of the frontier
    def length(self):
        return len(self.frontier)
    
    # check whether a node is in the frontier or not
    def check_node(self, state):
        return any(state == node.state for node in self.frontier)
    
class queue_frontier(stack_frontier):
    #remove last node of the frontier *FIFO
    def remove(self):
        node = self.frontier[0]
        self.frontier = self.frontier[1:]
        return node
    
class heuristic_frontier(queue_frontier):
    # sort nodes in frontier by h(n)
    def sort_gbfs(self):
        self.frontier.sort(key=Node.get_manhattan_distance)
    
    # sort nodes in frontier by f(n)
    def sort_astar(self):
        self.frontier.sort(key=Node.get_evaluation_func)
    
    # return a node in the frontier given the state
    def get_node(self, state):
        for node in self.frontier:
            if node.state == state:
                return node
    
    #add node to the frontier
    def add(self, node):
        self.frontier.insert(0, node)


# In[ ]:


def goal_test(state):
    return state == finish_state

# calculate manhattan distance
def manhattan_distance(state):
    x, y = state
    return (abs(x - finish_state[0]) + abs(y - finish_state[1]))


# In[ ]:


# create a boolean matrix, True -> wall
def walls_bool(maze):
    """
    convert maze matrix to boolean matrix
    If a state is a wall (1) -> True
    If a sate is a clear path (0) or a start point (2) or a finish point (3) -> False
    """
    result = []
    for i in range(len(maze)):
        row = []
        for j in range(len(maze[0])):
            if maze[i][j] == 1:
                row.append(True)
            else:
                row.append(False)
        result.append(row)
    return result

# transition model
# input: state -> output: all possible (action, next state)
def generate_neighbors(state):
    '''
    Input a state
    Output possible (action, next state)
    '''
    x, y = state
    
    # action
    # all possible moves
    # (action, next state)
    moves = [
        ("up", (x - 1, y)),
        ("down", (x + 1, y)),
        ("left", (x, y - 1)),
        ("right", (x, y + 1))
    ]
    
    result = []
    for action, (x, y) in moves:
        if (0 <= x <= len(maze)-1) and (0 <= y <= len(maze[0])-1):
            try:
                if not walls_bool(maze)[x][y]:
                    result.append((action, (x, y)))
            except:
                pass
    
    return result


# In[ ]:


def breadth_first_search():
    print("BREADTH FIRST SEARCH")
    
    # initialize start point
    start = Node(start_state, None, None, 0)
    
    # keep track number of nodes kept in memory
    num_memory = 0
        
    # initialize frontier
    frontier = queue_frontier()
    
    # use to store explored nodes
    explored = []
    
    # mark the explored status of the node, status = True <-> node have been removed from frontier
    status = [[False for i in range(width)] for j in range(height)]
    
    # add starting point to frontier
    frontier.add(start)
    
    while not frontier.isEmpty():
        # remove node from the frontier
        node = frontier.remove()
        
        # mark explored status as True
        status[node.x][node.y] = True
        
        # add state to explored list
        explored.append(node.state)

        # check goal_state and return solution
        if goal_test(node.state):
            print("Time complexity (number of nodes expanded in order to solve the maze):", len(explored))
            print("Space complexity (max number of nodes kept in memory):", num_memory)
            
            actions = []
            coordinates = []
            
            while node.parent != None:
                actions.append(node.action)
                coordinates.append(node.state)
                node = node.parent
            
            actions.reverse()
            coordinates.reverse()
            solution = (actions, coordinates, explored)
            
            return solution
        
        for action, state in generate_neighbors(node.state):
            i, j = state
            if not frontier.check_node(state) and status[i][j] == False:
                child = Node(state=state, parent=node, action=action, path_cost=node.path_cost + 1)
                frontier.add(child)
                
        if num_memory < len(frontier.frontier):
            num_memory = len(frontier.frontier)   
    return "Unsolvable"


# In[ ]:


def depth_first_search():
    print("DEPTH FIRST SEARCH")
    
    # initialize start point
    start = Node(start_state, None, None, 0)
    
    # keep track number of nodes kept in memory
    num_memory = 0
        
    #initialize frontier
    frontier = stack_frontier()
    
    # use to store explored nodes
    explored = []
    
     # mark the explored status of the node, status = True <-> node have been removed from frontier
    status = [[False for i in range(width)] for j in range(height)]
    
    # add starting point to frontier
    frontier.add(start)
    
    while not frontier.isEmpty():
        node = frontier.remove()
        status[node.x][node.y] = True
        explored.append(node.state)
       
        # check goal_state
        if goal_test(node.state):
            print("Time complexity (number of nodes expanded in order to solve the maze):", len(explored))
            print("Space complexity (max number of nodes kept in memory):", num_memory)
            actions = []
            coordinates = []
            while node.parent != None:
                actions.append(node.action)
                coordinates.append(node.state)
                node = node.parent
            actions.reverse()
            coordinates.reverse()
            solution = (actions, coordinates, explored)
            return solution
        
        for action, state in generate_neighbors(node.state):
            i, j = state
            if not frontier.check_node(state) and status[i][j] == False:
                child = Node(state=state, parent=node, action=action, path_cost=node.path_cost + 1)
                frontier.add(child)
        if num_memory < len(frontier.frontier):
            num_memory = len(frontier.frontier)   
    return "Unsolvable"


# In[ ]:


def greedy_best_first_search():
    print("GREEDY BEST FIRST SEARCH")
    start = Node(start_state, None, None, 0)
    #keep track of number of nodes kept in memory
    num_memory = 0
        
    #initialize frontier and explored set
    frontier = heuristic_frontier()
    explored = []
    
    status = [[False for i in range(width)] for j in range(height)]

    #add starting point to frontier
    frontier.add(start)
    
    while not frontier.isEmpty():
        frontier.sort_gbfs()

        node = frontier.remove()
        status[node.x][node.y] = True
        explored.append(node.state)
       
        # check goal_state
        if goal_test(node.state):
            actions = []
            coordinates = []
            print("Time complexity (number of nodes expanded in order to solve the maze):", len(explored))
            print("Space complexity (max number of nodes kept in memory):", num_memory)
            while node.parent != None:
                actions.append(node.action)
                coordinates.append(node.state)
                node = node.parent
            
            actions.reverse()
            coordinates.reverse()
            solution = (actions, coordinates, explored)
            return solution
        
        for action, state in generate_neighbors(node.state):
            i, j = state
            if not frontier.check_node(state) and status[i][j] == False:
                child = Node(state=state, parent=node, action=action, path_cost=node.path_cost + 1)
                frontier.add(child)
        if num_memory < len(frontier.frontier):
            num_memory = len(frontier.frontier)   
    return "Unsolvable"


# In[ ]:


def a_star_search():
    print("A* SEARCH")
    start = Node(start_state, None, None, 0)
    #keep track of number of nodes kept in memory
    num_memory = 0
        
    #initialize frontier and explored set
    frontier = heuristic_frontier()
    explored = []
    
    status = [[False for i in range(width)] for j in range(height)]

    #add starting point to frontier
    frontier.add(start)
    
    
    while not frontier.isEmpty():
        frontier.sort_astar()
        node = frontier.remove()
        status[node.x][node.y] = True
        
        explored.append(node.state)
       
        # check goal_state
        if goal_test(node.state):
            actions = []
            coordinates = []
            print("Time complexity (number of nodes expanded in order to solve the maze):", len(explored))
            print("Space complexity (max number of nodes kept in memory):", num_memory)
            while node.parent != None:
                actions.append(node.action)
                coordinates.append(node.state)
                node = node.parent
            
            actions.reverse()
            coordinates.reverse()
            solution = (actions, coordinates, explored)
            return solution
        
        for action, state in generate_neighbors(node.state):
            i, j = state
            if status[i][j] == False:
                if frontier.check_node(state):
                    # handle duplicate nodes (reopening closed node)
                    if (node.path_cost + 1) < frontier.get_node(state).path_cost:
                        x = frontier.get_node(state)
                        x.parent = node
                        x.action = action
                        x.path_cost = node.path_cost + 1
                else:
                    child = Node(state=state, parent=node, action=action, path_cost=node.path_cost + 1)
                    frontier.add(child)
        if num_memory < len(frontier.frontier):
            num_memory = len(frontier.frontier)       
    return "Unsolvable"


# In[ ]:


from PIL import Image, ImageDraw
import PIL

def show_image(maze, solution, app):
    white = (255, 255, 255)
    blue = (43, 104, 149)
    red = (255, 105, 97)
    yellow = (255, 255, 128)
    green = (11, 102, 35)
    light_green = (119, 221, 119)
    
    # pixel of width and height
    w = 100 * width
    h = 100 * height

    # draw a board
    im = Image.new('RGB', (w, h), (0, 0, 0))
    idraw= ImageDraw.Draw(im)
    
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            # draw wall box
            if maze[i][j] == 1:
                idraw.rectangle([(j*100, i*100), (j*100+100,i*100+100)], fill=blue)
            # draw start box
            elif maze[i][j] == 2:
                idraw.rectangle([(j*100, i*100), (j*100+100,i*100+100)], fill=red)
            # draw finish box
            elif maze[i][j] == 3:    
                idraw.rectangle([(j*100, i*100), (j*100+100,i*100+100)], fill=yellow)

    # draw explored box
    for i, j in solution[2]:
        if (i, j) != finish_state and (i, j) != start_state:
            idraw.rectangle([(j*100, i*100), (j*100+100,i*100+100)], fill=light_green)

    # draw solution box
    for i, j in solution[1]:
        if (i, j) != finish_state:
            idraw.rectangle([(j*100, i*100), (j*100+100,i*100+100)], fill=green)
            
    # draw vertical line
    for i in range(100, h, 100):
        idraw.line([(0, i), (w, i)], fill=white, width=1)
        
    # draw horizontal line   
    for i in range(100, w, 100):
        idraw.line([(i, 0), (i, h)], fill=white, width=1)
    
    # show and save image
    im.show()
    if app == "BFS":
        im.save("1_BFS.png")
    elif app == "DFS":
        im.save("2_DFS.png")
    elif app == "GBFS":
        im.save("3_GBFS.png")
    elif app == "ASS":
        im.save("4_ASS.png")


# In[ ]:


'''
from maze_generator import Maze

def main():
    global height, width, maze, start_state, finish_state
    height = int(input("Input the height of the maze: "))
    width = int(input("Input the width of the maze: "))
    show_img = str(input("Want to show solution images? (y/n) "))
    if show_img == 'y':
        print("Images will be popped up in order: BFS -> DFS -> GBFS -> A*")
    
    maze = Maze.generator(height, width)
    
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == 2:
                start_state = (i, j)
            elif maze[i][j] == 3:
                finish_state = (i, j)
    
    print("________________________________________________________________________________")
    solution_bfs = breadth_first_search()
    print("Path cost:", len(solution_bfs[0]))
    print("________________________________________________________________________________")
    solution_dfs = depth_first_search()
    print("Path cost:", len(solution_dfs[0]))
    print("________________________________________________________________________________")
    solution_gbfs = greedy_best_first_search()
    print("Path cost:", len(solution_gbfs[0]))
    print("________________________________________________________________________________")
    solution_ass = a_star_search()
    print("Path cost:", len(solution_ass[0]))
    print("________________________________________________________________________________")
    
    print("Optimal path cost:", len(solution_bfs[0]))
    for i in range(len(solution_bfs[0])):
        print(solution_bfs[0][i], solution_bfs[1][i])
    
    if show_img == 'y':
        show_image(maze, solution_bfs, "BFS")
        show_image(maze, solution_dfs, "DFS")
        show_image(maze, solution_gbfs, "GBFS")
        show_image(maze, solution_ass, "ASS")
    
main()
'''


# In[ ]:


from square_maze_generator import Square_Maze

def main():
    global height, width, maze, start_state, finish_state
    N = int(input("Input size of the maze: "))
    show_img = str(input("Want to show solution images? (y/n) "))
    if show_img == 'y':
        print("Images will be popped up in order: BFS -> DFS -> GBFS -> A*")

    height = width = N
    maze = Square_Maze.generator(1, N)
    
    maze[0][0] = 2
    maze[N-1][N-1] = 3
    
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == 2:
                start_state = (i, j)
            elif maze[i][j] == 3:
                finish_state = (i, j)
    
    print("________________________________________________________________________________")
    solution_bfs = breadth_first_search()
    print("Path cost:", len(solution_bfs[0]))
    print("________________________________________________________________________________")
    solution_dfs = depth_first_search()
    print("Path cost:", len(solution_dfs[0]))
    print("________________________________________________________________________________")
    solution_gbfs = greedy_best_first_search()
    print("Path cost:", len(solution_gbfs[0]))
    print("________________________________________________________________________________")
    solution_ass = a_star_search()
    print("Path cost:", len(solution_ass[0]))
    print("________________________________________________________________________________")
    
    print("Optimal path cost:", len(solution_bfs[0]))
    for i in range(len(solution_bfs[0])):
        print(solution_bfs[0][i], solution_bfs[1][i])
    
    if show_img == 'y':
        show_image(maze, solution_bfs, "BFS")
        show_image(maze, solution_dfs, "DFS")
        show_image(maze, solution_gbfs, "GBFS")
        show_image(maze, solution_ass, "ASS")
    
main()

