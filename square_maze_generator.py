import random

class Square_Maze():
    def generator(m, N):
        # m maze paths
        # N: size of the maze
        maze = [[1 for x in range(N)] for y in range(N)]
        dx = [0, 1, 0, -1]; dy = [-1, 0, 1, 0] # 4 directions to move in the maze
        stack = [] # array of stacks

        stack.append((0,0))
        maze[0][0] = 0

        cont = True # continue
        while cont:
            cont = False
            if len(stack) > 0:
                cont = True # continue as long as there is a non-empty stack
                (cx, cy) = stack[-1]
                # find a new cell to add
                nlst = [] # list of available neighbors
                for i in range(4):
                    nx = cx + dx[i]; ny = cy + dy[i]
                    if nx >= 0 and nx < N and ny >= 0 and ny < N:
                        if maze[ny][nx] == 1:
                            # of occupied neighbors must be 1
                            ctr = 0
                            for j in range(4):
                                ex = nx + dx[j]; ey = ny + dy[j]
                                if ex >= 0 and ex < N and ey >= 0 and ey < N:
                                    if maze[ey][ex] ==  0: ctr += 1
                            if ctr == 1: nlst.append(i)
                # if 1 or more neighbors available then randomly select one and add
                if len(nlst) > 0:
                    ir = nlst[random.randint(0, len(nlst) - 1)]
                    cx += dx[ir]; cy += dy[ir]
                    maze[cy][cx] =  0
                    stack.append((cx, cy))
                else: stack.pop()

        #randomly unblocked
        for i in range(int(N*N*0.15)):
            if maze[random.randint(0,N-1)][random.randint(0,N-1)] == 1:
                maze[random.randint(0,N-1)][random.randint(0,N-1)] = 0

        return maze