import numpy as np
import os
from copy import deepcopy
from string import ascii_lowercase, ascii_uppercase
import re


def solve_day_n(n, input_path = 'input_data/'):
    x = process_input_day_n(n, input_path)
    loc = {}
    exec(f"""soln = soln_day_{str(n)}({x})""", globals(), loc)
    soln = loc['soln']
    return soln

def process_input_day_n(n, input_path = 'input_data/'):
    input_fn = f"""input_{n:02d}.txt"""
    filepath = os.path.join(input_path, input_fn)
    x = []
    
    if n in [1]:
        running_tot = 0
        with open(filepath) as fp:
            for line in fp:
                if line == '\n':
                    x.append(running_tot)
                    running_tot = 0
                else:
                    running_tot += int(line)
        return x

    elif n in [2]:
        with open(filepath) as fp:
            for line in fp:
                x.append(line.strip('\n').split(' ') )
        return x
    
    elif n in [3]:
        with open(filepath) as fp:
            for line in fp:
                x.append(line.strip('\n'))
        return x
    
   
    elif n in [4]:
        with open(filepath) as fp:
            for line in fp:
                x.append([list(map(int, x.split('-'))) for x in line.strip('\n').split(',')])
        return x
    
    elif n in [5, 6]:
        with open(filepath) as fp:
            for line in fp:
                x.append(line)
        return(x)
        
    else:
        return 'Not implemented yet'

def soln_day_1(x):
   
    ## Part 1
    soln_pt_1 = np.max(x)
    
    ## Part 2
    soln_pt_2 = np.sum(sorted(x)[-3:])

    return (soln_pt_1, soln_pt_1)


def soln_day_2(x):
   
    ## Part 1
    encoding = {'X': 0, 'Y': 1, 'Z': 2,
                'A': 0, 'B': 1, 'C': 2}
    points = 0
    for a in x:
        select_pts = encoding[a[1]] + 1
        battle_pts = 3 * ((1 + encoding[a[1]] - encoding[a[0]]) % 3)
        points += select_pts
        points += battle_pts
    soln_pt_1 = points

    ## Part 2
    points = 0
    for a in x:
        battle_pts = 3 * encoding[a[1]]
        select_pts = 1 + ((encoding[a[0]] + encoding[a[1]] - 1) % 3)
        points += select_pts
        points += battle_pts
    soln_pt_2 = points
    return (soln_pt_1, soln_pt_2)

    
        
def soln_day_3(x):

    ## Part 1
    priorities = []
    for rucksack in x:
        rucksack_size = int(len(rucksack)/2)
        comp_a, comp_b = (rucksack[:rucksack_size], rucksack[rucksack_size:])
        for item in comp_a:
            if item in comp_b:
                priorities.append(1 + (ascii_lowercase + ascii_uppercase).index(item))
                break
    soln_pt_1 = sum(priorities)

    ## Part 2
    priorities = []
    for i in range(int(len(x) / 3)):
        rucksack_1 = set(x[3*i])
        rucksack_2 = set(x[3*i + 1])
        rucksack_3 = set(x[3*i + 2])

        (badge,) = rucksack_1.intersection(rucksack_2).intersection(rucksack_3)
        priorities.append(1 + (ascii_lowercase + ascii_uppercase).index(badge))
    
    soln_pt_2 = sum(priorities)
    
    return (soln_pt_1, soln_pt_2)



def soln_day_4(x):

    ## Part 1
    num_redundant = 0
    for elf_1, elf_2 in x:
        max_individ_range = max(elf_1[1] - elf_1[0], elf_2[1] - elf_2[0])
        overall_range = max(elf_1[1], elf_2[1]) - min(elf_1[0], elf_2[0])
        if max_individ_range == overall_range:
            num_redundant += 1
    soln_pt_1 = num_redundant

    ## Part 2
    num_redundant = 0
    for elf_1, elf_2 in x:
        tot_individ_range = (1 + elf_1[1] - elf_1[0]) + (1 + elf_2[1] - elf_2[0])
        overall_range = 1 + max(elf_1[1], elf_2[1]) - min(elf_1[0], elf_2[0])
        if overall_range < tot_individ_range:
            num_redundant += 1
    soln_pt_2 = num_redundant
    
    return (soln_pt_1, soln_pt_2)



def soln_day_5(x):
    start_config = []
    instructions = []
    i = 0
    while len(x[i]) == 36:
        start_config.append([*x[i][1::4]])
        i += 1
    start_config = list(map(list, zip(*start_config))) # transpose
    start_config = [[a for a in start_stack if a != ' ' and not a.isdigit()] for start_stack in start_config] # non-empty items only
    instructions = [re.findall(r'\d+', line) for line in x[i+1:]] # instructions[0] times, move from instructions[1] to instructions[2]

    ## Part 1
    pt1_config = deepcopy(start_config)
    for instruction in instructions:
        for _ in range(int(instruction[0])):
            move_from = int(instruction[1]) - 1
            move_to = int(instruction[2]) - 1
            char_to_move = pt1_config[move_from].pop(0)
            pt1_config[move_to].insert(0, char_to_move)
    soln_pt_1 = ''.join([stack[0] for stack in pt1_config])

    ## Part 2
    pt2_config = deepcopy(start_config)
    for instruction in instructions:
        num_to_move = int(instruction[0])
        move_from = int(instruction[1]) - 1
        move_to = int(instruction[2]) - 1
        char_to_move = pt2_config[move_from][:num_to_move]
        del pt2_config[move_from][:num_to_move]
        pt2_config[move_to] = char_to_move + pt2_config[move_to]
    soln_pt_2 = ''.join([stack[0] for stack in pt2_config])
    
    return (soln_pt_1, soln_pt_2)




def soln_day_6(x):

    ## Part 1
    i = 4
    while len(set(x[0][i-4:i])) < 4:
        i += 1
    soln_pt_1 = i

    ## Part 2
    i = 14
    while len(set(x[0][i-14:i])) < 14:
        i += 1
    soln_pt_2 = i
    
    return (soln_pt_1, soln_pt_2)

def soln_day_7(x):

    ## Part 1

    soln_pt_1 = 0

    ## Part 2

    soln_pt_2 = 0
    
    return (soln_pt_1, soln_pt_2)