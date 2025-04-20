import pygame
import numpy as np
import math
import cv2
from numpy.linalg import inv, pinv
import matplotlib.pyplot as plt 
from collections import deque 
import time

class Envir:
    def __init__(self, dimentions):
        #color
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.green = (0, 255, 0)
        self.blue  = (0, 0,  255)
        self.red   = (255, 0, 0)
        self.yel   = (255, 255, 0)
        #map dims
        self.height = dimentions[0]
        self.width = dimentions[1]
        #window settings
        pygame.display.set_caption("Robot 2")
        self.trail_set = []
        self.map = pygame.display.set_mode((self.width, self.height))
        
    def trail(self, pos):
        for i in range(0, len(self.trail_set) - 2,  2):
            pygame.draw.line(self.map, self.green, (self.trail_set[i][0], self.trail_set[i][1]),
                             (self.trail_set[i + 1][0], self.trail_set[i + 1][1]), 15)
        if(self.trail_set.__sizeof__() > 15000):
            self.trail_set.pop(0)
        self.trail_set.append(pos)

class Robot(pygame.sprite.Sprite):
    def __init__(self, startPos, endPos, robotImg, robot_size, th):
        pygame.sprite.Sprite.__init__(self)
        self.robot_size = robot_size
        self.start_pos = startPos
        self.end_pos = endPos
        self.w = robot_size * M2P
        self.x = startPos[0]
        self.y = startPos[1]
        self.theta = 3 * math.pi / 2
        self.vl = 0
        self.vr = 0
        self.maxspeed = 40
        self.minspeed = -40
        self.th = th
        self.image = pygame.image.load(robotImg)
        self.image = pygame.transform.scale(self.image, (self.w, self.w))
        self.image = pygame.Surface.convert_alpha(self.image)
        self.rotated = self.image
        self.mask  = pygame.mask.from_surface(self.rotated)
        self.rect = self.rotated.get_rect(center = (self.x, self.y))

        self.sensors_values = [0] * 5
        self.sensors_pos = [
            [[[self.w * 0.5], [self.w * 0.4]], math.pi/2],
            [[[self.w * 0.65], [self.w * 0.25]], math.pi/4],
            [[[self.w * 0.65], [0]], 0],
            [[[self.w * 0.65], [-self.w * 0.25]], -math.pi/4],
            [[[self.w * 0.5], [-self.w * 0.4]], -math.pi/2],
        ]
        self.sensors_lines = [[(0, 0), (0, 0)] for i in range(5)]
        self.sensors_max = 180 
        
        self.fitness_val = 0
        self.crash = False
        self.totalTime = 0.0
        self.ckTime = 0.0
        self.last_x = self.x;
        self.last_y = self.y;

        self.l = self.robot_size/2
        self.r = 0.015
        self.alpha1 = math.pi/2
        self.alpha2 = -math.pi/2
        self.pinv_j1 = pinv([
             [math.sin(self.alpha1), -math.cos(self.alpha1), (-self.l)],
             [math.sin(self.alpha2), -math.cos(self.alpha2), (-self.l)]
        ])
        self.j2 = np.array([
            [self.r, 0.0], 
            [0.0, self.r]
        ])
        
    def draw(self):
        environment.map.blit(self.rotated, self.rect)
        #text_surface = font.render(str(self.fitness_val), True, (255, 0, 0))
        #environment.map.blit(text_surface, self.rect)
        for s in range(5):
            pygame.draw.aaline(environment.map, (0, 0, 255), self.sensors_lines[s][0],  self.sensors_lines[s][1])
            
    def reset(self):
        self.x = self.start_pos[0]
        self.y = self.start_pos[1]
        self.last_x = self.x;
        self.last_y = self.y;
        self.theta = 3*math.pi/2
        self.vl = 0
        self.vr = 0
        self.fitness_val = 0
        self.totalTime = 0.0
        self.ckTime = 0.0
        self.crash = False
        
    def sensors(self):
        R = np.array([
            [math.cos(self.theta), -math.sin(self.theta)],
            [math.sin(self.theta), math.cos(self.theta)]
        ])
        for s in range(5):
            i = 0
            Gsensor_pos = R @ self.sensors_pos[s][0]
            end_x = 0
            end_y = 0
            R1 = [
                [math.cos(self.theta + self.sensors_pos[s][1])],
                [math.sin(self.theta + self.sensors_pos[s][1])]
            ]
            while(i < self.sensors_max):
                b1 = [[R1[0][0] * i], [R1[1][0] * i]]
                end_x = int(self.x + Gsensor_pos[0][0] + b1[0][0])
                end_y = int(self.y - Gsensor_pos[1][0] - b1[1][0])
                if(end_x >= 900):
                    end_x = 899
                    break
                elif(end_x < 0):
                    end_x = 0
                    break
                if(end_y >= 1000):
                    end_y = 999
                    break
                elif(end_y < 0):
                    end_y = 0
                    break
                if(environment.map.get_at((end_x, end_y))[:3] == (0, 0, 0)):
                    break
                i+=5
            
            self.sensors_lines[s][0] = (self.x + Gsensor_pos[0][0], self.y - Gsensor_pos[1][0])
            self.sensors_lines[s][1] = (end_x, end_y)
            self.sensors_values[s] = i / self.sensors_max #Chuan hoa gia tri sensor [0 - 1]

    def move(self):
        to = self.Forward_kinematic()
        self.totalTime += dt
        self.x += to[0][0] * M2P * dt
        self.y -= to[1][0] * M2P * dt
        self.theta = (self.theta + to[2][0] * dt) % (math.pi * 2) #Chuan hoa goc quay nam trong [0 - 2pi]
        self.rotated = pygame.transform.rotozoom(self.image, np.degrees(self.theta), 1)
        #centered between the two drive
        b = [
            [np.cos(self.theta) * self.w * 0.25],
            [np.sin(self.theta) * self.w * 0.25]
        ]
        
        self.rect = self.rotated.get_rect(center = (self.x + b[0][0], self.y - b[1][0]))
        self.mask = pygame.mask.from_surface(self.rotated)
        
    def set_spinning_speed(self, v_left, v_right):
        self.vl = self.minspeed + (self.maxspeed - self.minspeed) * v_left  #Dat toc do theo pha tram
        self.vr = self.minspeed + (self.maxspeed - self.minspeed) * v_right #Dat toc do theo pha tram
    
    def Forward_kinematic(self):
        phi_dot = np.array([
            [self.vl], 
            [-self.vr]
        ])
        R = np.array([
            [math.cos(self.theta), math.sin(self.theta), 0.0],
            [-math.sin(self.theta), math.cos(self.theta), 0.0],
            [0.0, 0.0, 1.0]
        ])
        to = inv(R) @ self.pinv_j1 @ self.j2 @ phi_dot
        return to

    def is_crash(self):
        if(self.crash or pygame.sprite.collide_mask(maze, self) or self.y < self.start_pos[1] - 20):
            self.crash = True
            return True
        return False

    def sigmoid(self, x):
        return(1/(1 + np.exp(-x)))
    
    def neural_network_forward(self, weights):
        if(self.crash):
            return
        self.sensors()
        current_pos = [self.x / 900, self.y / 950, self.theta / (math.pi * 2)] #Chuan hoa gia tri vi tri [0 - 1]
        inp = np.array(self.sensors_values + current_pos).reshape((num_input, 1))
        w1 = weights[weights1_range[0]:weights1_range[1]].reshape((num_input, num_hidden_neurons))
        bias = weights[bias_hidden_range[0]:bias_hidden_range[1]].reshape(num_bias_hidden, 1)
        w2 = weights[weights2_range[0]:weights2_range[1]].reshape((num_hidden_neurons, num_output))
        bias_output = weights[bias_output_range[0]:bias_output_range[1]].reshape((num_bias_output, 1))
        output = self.sigmoid(w2.T @ self.sigmoid(w1.T @ inp + bias) + bias_output)
        self.set_spinning_speed(output[0][0], output[1][0])
        self.move()
        
    def fitness(self):
        global NS_archive, count_evaluate
        #Tinh toan do moi la tai vi tri ket thuc
        if(self.crash):
            pos = np.array([(self.x, self.y)])
            if(NS_archive.shape[0] < k + 1):
                NS_archive = np.append(NS_archive, pos, axis = 0)
                return
            euclidean_distance = np.linalg.norm(pos - NS_archive, axis = 1)
            k_nearest = np.partition(euclidean_distance, k)[:k]
            self.fitness_val = np.mean(k_nearest)
            count_evaluate+=1
            if(self.fitness_val > novelty_threshold and np.min(k_nearest) > 0.01):
                map_ns.set_at((int(self.x), int(self.y)), (255, 0, 0, 255))
                NS_archive = np.append(NS_archive, pos, axis = 0)
                count_evaluate = 0
        elif(self.y > self.end_pos[1]):
            self.crash = True
            #Luu ca the thoat khoi me cung
            with open('best_%s_seed=%d_w=%f_c1=%f_c2=%f.npy' % (str(time.time()), seed, w, c1, c2), 'wb') as f:
                np.save(f, position[self.th])
        #Kiem tra thoi gian toi da
        elif(self.totalTime > 120):
            self.crash = True
        else:
            #Kiem tra robot bi ket
            if((self.totalTime - self.ckTime) > 1):
                if(math.sqrt((self.last_x - self.x)**2 + (self.last_y - self.y)**2) < 30):
                    self.crash = True
                self.ckTime = self.totalTime
                self.last_x = self.x
                self.last_y = self.y
                
            
class Maze(pygame.sprite.Sprite):
    def __init__(self, mazeImg, width, rect):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(mazeImg).convert_alpha()
        self.rect = self.image.get_rect(center = (rect[0] + width[0]/2, rect[1] + width[1]/2))
        self.maze_start_pos = (411, 8)
        self.maze_end_pos = (497, 900)
        self.mask = pygame.mask.from_surface(self.image)




seed = 10
np.random.seed(seed) #Tai tao random giong nhau cho moi lan chay neu seed giong nhau
M2P = 300 #300px/meter
P2M = 1.0 / M2P
ROBOT_SIZE = 0.04 #robot 4cm
dims = (1000, 900)
start_maze = (0, 50) #Diem dat me cung trong cua so pygame

k = 30 # k neasest
novelty_threshold = 20 #Nguong them diem moi la

N = 1000  # cá thể
max_N = 40 # So ca the toi da co the chay cung luc
num_input = 8 # ngõ vào noron
num_hidden_neurons = 15  # noron lớp ẩn
num_bias_hidden = num_hidden_neurons
num_output = 2 # ngõ ra noron
num_bias_output = num_output
num_total = num_input * num_hidden_neurons + num_hidden_neurons * num_output + num_bias_hidden + num_bias_output
min_val = -1
max_val = 1

weights1_range = [0, num_input * num_hidden_neurons] #[start_index, end_index]
bias_hidden_range = [num_input * num_hidden_neurons, num_input * num_hidden_neurons + num_bias_hidden]
weights2_range = [num_input * num_hidden_neurons + num_bias_hidden, num_total - num_bias_output]
bias_output_range = [num_total - num_bias_output, num_total]
print("Number of weights:", num_total)
print(weights1_range, bias_hidden_range, weights2_range, bias_output_range)

#------------------------------------------------------------------------------------------#
#Train

checkpoint_flag = False 
checkpoint_path = 'checkpoint.npz'

if(checkpoint_flag): #Tai checkpoint
    loaded_data = np.load(checkpoint_path)
    c1 = loaded_data['params'][0]
    c2 = loaded_data['params'][1]
    w = loaded_data['params'][2]
    position = loaded_data['position']
    velocity = loaded_data['velocity']
    iteration = int(loaded_data['params'][3])
    pVal_best = loaded_data['pVal_best'].tolist()
    pPos_best = loaded_data['pPos_best']
    pEnd_best = loaded_data['pEnd_best'].tolist()
    gVal_best = loaded_data['params'][4]
    gPos_best = loaded_data['gPos_best']
    gEnd_best = loaded_data['gEnd_best'].tolist()
    count_evaluate = loaded_data['params'][5]
    print("c1:", c1, "c2:", c2, "w:", w)
else: #Tao moi
    c1 = 1.5
    c2 = 1.5
    w = 0.9

    position = np.random.uniform(min_val, max_val, size = (N, num_total)).astype(np.float64)   
    velocity = np.random.uniform(min_val, max_val, size = (N, num_total)).astype(np.float64)  
    
    pVal_best = [-100000000000] * N
    pPos_best = np.random.uniform(min_val, max_val, size = (N, num_total)).astype(np.float64)
    pEnd_best = [[0, 0] for i in range(N)]
    gVal_best = -100000000000
    gPos_best = np.random.uniform(min_val, max_val, size = num_total).astype(np.float64)
    gEnd_best = [0, 0]



pygame.init()
environment = Envir(dims)
maze = Maze('M_07.png', (900, 900), start_maze)
robot = [Robot((maze.maze_start_pos[0] + start_maze[0], maze.maze_start_pos[1] + start_maze[1] + 20), 
               (maze.maze_end_pos[0] + start_maze[0], maze.maze_end_pos[1] + start_maze[1]),
               "micro.png", ROBOT_SIZE, i) for i in range(N)]

font = pygame.font.Font(pygame.font.get_default_font(), 20)

map_ns = pygame.display.set_mode((dims[1], dims[0]))
map_ns = map_ns.convert_alpha()
map_ns.fill((0, 0, 0, 0))

if checkpoint_flag:
    NS_archive = loaded_data['NS_archive'] #Luu tru diem cuoi moi la cua cac ca the
    for pos in NS_archive:
        map_ns.set_at((int(pos[0]), int(pos[1])), (255, 0, 0, 255))
else:
    NS_archive = np.array([maze.maze_start_pos])
    iteration = 0
    count_evaluate = 0

dt = 0
running = True
lasttime = pygame.time.get_ticks()
max_iteration = 10000000

while iteration < max_iteration:
    if(not running):
        break
    for i in range(N):
        robot[i].reset()
    crash = [False] * N
    lasttime = pygame.time.get_ticks()
    max_n = max_N
    while(1):
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                pygame.display.quit()
                running = False
        if(not running):
            break
            
        dt = (pygame.time.get_ticks() - lasttime)/1000
        lasttime = pygame.time.get_ticks()
        
        environment.map.fill(environment.white)
        environment.map.blit(maze.image, start_maze)
        environment.map.blit(map_ns, (0, 0))
        iter_text = font.render("Iter: " + str(iteration), True, (255, 0, 0))
        environment.map.blit(iter_text, (700, 3))
        pygame.draw.circle(environment.map, (0, 0, 255), (int(gEnd_best[0]), int(gEnd_best[1])), 2)
                
        for i in range(max_n):
            robot[i].neural_network_forward(position[i])
            if(robot[i].is_crash() and not crash[i]):
                max_n = min(max_n + 1, N) #Them mot ca the neu mot ca the khac chet
            crash[i] = robot[i].is_crash()
            if(not crash[i]):
                robot[i].fitness()
        for i in range(max_n):
            pygame.draw.circle(environment.map, (0, 255, 0), (int(pEnd_best[i][0]), int(pEnd_best[i][1])), 2)
            if(not crash[i]):
                robot[i].draw()
        pygame.display.update()
        if(all(crash)):
            break
       

    for i in range(N):
        robot[i].fitness()

    
    #Doi vi tri tot nhat neu khong co phat hien diem moi la
    if(count_evaluate > 2000):
        r = np.random.randint(0, N)
        gVal_best = pVal_best[r]
        gPos_best = np.copy(pPos_best[r])
        gEnd_best[0] = pEnd_best[r][0]
        gEnd_best[1] = pEnd_best[r][1]
        count_evaluate = 0

    iteration+=1    
    if(NS_archive.shape[0] > k + 1):
        if(gEnd_best != [0, 0]):
            euclidean_distance = np.linalg.norm(np.array(gEnd_best) - NS_archive, axis = 1)
            k_nearest = np.partition(euclidean_distance, k)[:k]
            gVal_best = np.mean(k_nearest)
    else:
        continue

    for i in range(N):
        for k in range(num_total): 
            r1 = np.random.uniform(0, 1)  # Tăng khả năng khám phá
            r2 = np.random.uniform(0, 0.5)  # Giảm dao động
            velocity[i][k] = ((w * velocity[i][k]) +
                    (c1 * r1 * (pPos_best[i][k] - position[i][k])) + 
                    (c2 * r2 * (gPos_best[k] - position[i][k])))  
            if velocity[i][k] < min_val:
                velocity[i][k] = min_val
            elif velocity[i][k] > max_val:
                velocity[i][k] = max_val
        for k in range(num_total): 
            position[i][k] += velocity[i][k]
            
        if(NS_archive.shape[0] > k + 1 and pEnd_best[i] != [0, 0]):
            euclidean_distance = np.linalg.norm(np.array(pEnd_best[i]) - NS_archive, axis = 1)
            k_nearest = np.partition(euclidean_distance, k)[:k]
            pVal_best[i] = np.mean(k_nearest)
            
        fitness_val = robot[i].fitness_val
        if fitness_val > pVal_best[i]:
            pVal_best[i] = fitness_val
            pPos_best[i] = np.copy(position[i])
            pEnd_best[i][0] = robot[i].x
            pEnd_best[i][1] = robot[i].y
        if fitness_val > gVal_best:
            print("New point: x=%d y=%d novetly=%f" % (int(robot[i].x), int(robot[i].y), float(fitness_val)))
            gVal_best = fitness_val
            gPos_best = np.copy(position[i])
            gEnd_best[0] = robot[i].x
            gEnd_best[1] = robot[i].y

    #Luu checkpoint dinh ki
    if(iteration % 3 == 0):
        params = np.array([c1, c2, w, iteration, gVal_best, count_evaluate])
        np.savez(checkpoint_path, params = params, 
                            NS_archive = NS_archive,
                            position = position,
                            velocity = velocity,
                            pVal_best = np.array(pVal_best),
                            pPos_best = pPos_best, 
                            pEnd_best = np.array(pEnd_best), 
                            gPos_best = gPos_best,
                            gEnd_best = np.array(gEnd_best)),
pygame.display.quit()



#---------------------------------------------------------------------------------------------------------------------------------#
# #Chay ca the thoat khoi me cung

# pygame.init()
# environment = Envir(dims)
# maze = Maze('M_10.png', (900, 900), start_maze)
# robot = Robot((maze.maze_start_pos[0] + start_maze[0], maze.maze_start_pos[1] + start_maze[1] + 20), 
#                (maze.maze_end_pos[0] + start_maze[0], maze.maze_end_pos[1] + start_maze[1]),
#                "micro.png", ROBOT_SIZE, 0)
# position = np.load('best_1744005804.3067148_seed=10_w=0.700000_c1=2.000000_c2=1.500000.npy')
# font = pygame.font.Font(pygame.font.get_default_font(), 10)
# print("Position:\n", position)
# dt = 0
# running = True
# lasttime = pygame.time.get_ticks()
# while(1):
#     for event in pygame.event.get():
#         if(event.type == pygame.QUIT):
#             pygame.display.quit()
#             running = False
#     if(not running):
#         break
        
#     dt = (pygame.time.get_ticks() - lasttime)/1000
#     lasttime = pygame.time.get_ticks()
#     environment.map.fill(environment.white)
#     environment.map.blit(maze.image, start_maze)   
#     robot.neural_network_forward(position)
#     robot.draw()
#     if(robot.is_crash()):
#         pass
#     pygame.display.update()
