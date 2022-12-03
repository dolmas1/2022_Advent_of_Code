import numpy as np
import os
from string import ascii_lowercase, ascii_uppercase


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




