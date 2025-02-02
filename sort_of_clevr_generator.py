import cv2
import os
import numpy as np
import random
#import cPickle as pickle
import pickle
import warnings
import argparse
import time

import pandas as pd
import tqdm

parser = argparse.ArgumentParser(description='Sort-of-CLEVR dataset generator')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--t-subtype', type=int, default=-1,
                    help='Force ternary questions to be of a given type')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

train_size = 9800
test_size = 200
img_size = 75
size = 5
question_size = 18  ## 2 x (6 for one-hot vector of color), 3 for question type, 3 for question subtype
q_type_idx = 12
sub_q_type_idx = 15
"""Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""

nb_questions = 10
dirs = './data'

colors = [
    (0,0,255),##r
    (0,255,0),##g
    (255,0,0),##b
    (0,156,255),##o
    (128,128,128),##k
    (0,255,255)##y
]

colors_name = ['Red','Green','Blue','Orange','Grey','Yellow']
shape_name = {
    'r': 'Rectangle',
    'c': 'Circle'
}


try:
    os.makedirs(dirs)
except:
    print('directory {} already exists'.format(dirs))

def center_generate(objects):
    while True:
        pas = True
        center = np.random.randint(0+size, img_size - size, 2)   # Returns a list of two numbers between [size,img_size-size)     
        if len(objects) > 0:
            for name,c,shape in objects:
                if ((center - c) ** 2).sum() < ((size * 2) ** 2):   # Checking if the already given object overlap with one present in image
                    pas = False
        if pas:
            return center

def expand_unary(relations):
    ques,ans = relations
    
    ques_expanded = []
    ans_expanded = []
    subtype = []
    n = len(ques)
    colors_name = ['red','green','blue','orange','grey','yellow']
    
    for i in range(n):
        colors = None
        for j in range(len(colors_name)):
            if ques[i][j]==1:
                colors = colors_name[j]
                break
        if ques[i][15]==1:
            ques_expanded.append(f'What is the shape of the {colors} object?')
            subtype.append(1)
            if ans[i]==2:
                ans_expanded.append('Rectangle')
            else:
                ans_expanded.append('Circle')
        elif ques[i][16]==1:
            ques_expanded.append(f'Is there a {colors} colored object in right half of the image?')
            subtype.append(2)
            if ans[i]==0:
                ans_expanded.append('No')
            else:
                ans_expanded.append('Yes')
        elif ques[i][17]==1:
            ques_expanded.append(f'Is there a {colors} colored object in top half of the image?')
            subtype.append(3)
            if ans[i]==0:
                ans_expanded.append('No')
            else:
                ans_expanded.append('Yes')
    
    return [ques_expanded,ans_expanded,subtype]

def expand_binary(relations):
    ques,ans = relations
    
    ques_expanded = []
    ans_expanded = []
    subtype = []
    
    n = len(ques)
    colors_name = ['red','green','blue','orange','grey','yellow']
    
    for i in range(n):
        colors = None
        for j in range(len(colors_name)):
            if ques[i][j]==1:
                colors = colors_name[j]
                break
        if ques[i][15]==1:
            ques_expanded.append(f'What is the shape of the closest object to {colors} colored object?')
            subtype.append(1)
            if ans[i]==2:
                ans_expanded.append('Rectangle')
            else:
                ans_expanded.append('Circle')
        elif ques[i][16]==1:
            ques_expanded.append(f'What is the shape of the furtherest object to {colors} colored object?')
            subtype.append(2)
            if ans[i]==2:
                ans_expanded.append('Rectangle')
            else:
                ans_expanded.append('Circle')
        elif ques[i][17]==1:
            ques_expanded.append(f'How many objects are similiar in shape to {colors} colored object?')
            subtype.append(3)
            ans_expanded.append(ans[i]-4)
    
    return [ques_expanded,ans_expanded,subtype]

def expand_ternary(relations):
    ques,ans = relations
    
    ques_expanded = []
    ans_expanded = []
    
    n = len(ques)
    colors_name = ['red','green','blue','orange','grey','yellow']
    
    for i in range(n):
        colors = []
        for j in range(len(colors_name)):
            if ques[i][j]==1:
                colors.append(colors_name[j])
                break
        for j in range(6,2*len(colors_name)):
            if ques[i][j]==1:
                colors.append(colors_name[j-6])
        # print(colors)
        if ques[i][15]==1:
            ques_expanded.append(f'How many objects are there between {colors[0]} and {colors[1]} colored object?')
            ans_expanded.append(ans[i]-4)
        elif ques[i][16]==1:
            ques_expanded.append(f'Is there any object on the line joining center of {colors[0]} and {colors[1]} object?')
            if ans[i]==0:
                ans_expanded.append('Yes')
            else:
                ans_expanded.append('No')
        elif ques[i][17]==1:
            ques_expanded.append(f'How many obtuse angled triangle can be formed with two vertices being {colors[0]} and {colors[1]} object center?')
            ans_expanded.append(ans[i]-4)
    
    return [ques_expanded,ans_expanded]


def build_dataset(ind,type,dirs):
    objects = []
    state = pd.DataFrame()
    img = np.ones((img_size,img_size,3)) * 255  #(75X75X3) (W,H,C)
    for color_id,color in enumerate(colors):  
        center = center_generate(objects)
        if random.random()<0.5:
            start = (center[0]-size, center[1]-size) # Range is [0,img_size-2*size)
            end = (center[0]+size, center[1]+size)  # Range is  [2*size,img_size)
            cv2.rectangle(img, start, end, color, -1)   # Drawing a square

            # Wierd way of storing a object
            objects.append((color_id,center,'r'))   # Making it color equal to color_id from enumerate colors.
        else:
            center_ = (center[0], center[1])   
            cv2.circle(img, center_, size, color, -1) # Else Drawing a circle of radius size
            objects.append((color_id,center,'c'))

    state['center'] = [i[1] for i in objects]
    state['color'] = [colors_name[i[0]] for i in objects]
    state['shape'] = [shape_name[i[2]] for i in objects]
    state['size'] = [size]*len(objects)         # For Square/Rectangle it is side length, For circle it is radius

    # os.makedirs(f'{dirs}/{type}/state',exist_ok=True)
    # state.to_csv(f'{dirs}/{type}/state/state_{ind}.csv')

    os.makedirs(f'{dirs}/{type}/img',exist_ok=True)
    np.savez(f'{dirs}/{type}/img/img_{ind}.npz',img=img)


    ternary_questions = []
    binary_questions = []
    norel_questions = []
    ternary_answers = []
    binary_answers = []
    norel_answers = []

    '''
    10 Non-Relational Type Questions

    Subtype 0: What is the shape of the object?
    Subtype 1: Whether the object is in right half or left half of the image? 0 for left and 1 for right
    Subtype 2: Whether the object is in top half or bottom half of the image? 0 for bottom and 1 for top
    Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]
    '''
    for _ in range(nb_questions):
        question = np.zeros((question_size))    # 2 x (6 for one-hot vector of color), 3 for question type, 3 for question subtype
        color = random.randint(0,5)
        question[color] = 1                     # Randomly choosing a color out of six possible colors 
        question[q_type_idx] = 1                # 12th index, of the questions array is 1 array
        subtype = random.randint(0,2)           # Randomly choosing the question-subtype
        question[subtype+sub_q_type_idx] = 1    
        norel_questions.append(question)        # Appending it to non relation questions
        
        if subtype == 0:
            """query shape->rectangle/circle"""
            if objects[color][2] == 'r':
                answer = 2
            else:
                answer = 3

        elif subtype == 1:
            """query horizontal position->yes/no"""
            if objects[color][1][0] < img_size / 2:
                answer = 0
            else:
                answer = 1

        elif subtype == 2:
            """query vertical position->yes/no"""
            if objects[color][1][1] < img_size / 2:
                answer = 0
            else:
                answer = 1
        norel_answers.append(answer)
    
    '''
    10 Binary relation questions

    Subtype 0: What is the closest object (a rectangle or circle) to a randomly choosen colored object?
    Subtype 1: What is the furthest object (a rectangle or circle) to a randomly choosen colored object?
    Subtype 2: Number of objects similiar in shape to a randomly choosen colored object. Max can be 6
    '''

    for _ in range(nb_questions):
        question = np.zeros((question_size))
        color = random.randint(0,5)
        question[color] = 1
        question[q_type_idx+1] = 1
        subtype = random.randint(0,2)
        question[subtype+sub_q_type_idx] = 1
        binary_questions.append(question)

        if subtype == 0:
            """closest-to->rectangle/circle"""
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            dist_list[dist_list.index(0)] = 999
            closest = dist_list.index(min(dist_list))
            if objects[closest][2] == 'r':
                answer = 2
            else:
                answer = 3
                
        elif subtype == 1:
            """furthest-from->rectangle/circle"""
            my_obj = objects[color][1]
            dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            furthest = dist_list.index(max(dist_list))
            if objects[furthest][2] == 'r':
                answer = 2
            else:
                answer = 3

        elif subtype == 2:
            """count->1~6"""
            my_obj = objects[color][2]
            count = -1
            for obj in objects:
                if obj[2] == my_obj:
                    count +=1 
            answer = count+4

        binary_answers.append(answer)

    '''
    10 Ternary Relational questions

    Subtype 0: Checks how many objects lie between the two randomly choosen objects. Max can be 4
    Subtype 1: Checks if any object center lies on the line connecting the centers of two randomly choosen objects
    Subtype 2: Checks whether the traingle formed with all objects with the two randomly choosen objects is obtuse or not
    '''
    for _ in range(nb_questions):

        '''
        Choosing Two random colored objects
        '''
        question = np.zeros((question_size))
        rnd_colors = np.random.permutation(np.arange(5))
        # 1st object
        color1 = rnd_colors[0]
        question[color1] = 1
        # 2nd object
        color2 = rnd_colors[1]
        question[6 + color2] = 1

        question[q_type_idx + 2] = 1
        
        if args.t_subtype >= 0 and args.t_subtype < 3:
            subtype = args.t_subtype
        else:
            subtype = random.randint(0, 2)

        question[subtype+sub_q_type_idx] = 1
        ternary_questions.append(question)

        # get coordiantes of object from question
        A = objects[color1][1]
        B = objects[color2][1]

        if subtype == 0:
            """between->1~4"""

            between_count = 0 
            # check is any objects lies inside the box
            for other_obj in objects:
                # skip object A and B
                if (other_obj[0] == color1) or (other_obj[0] == color2):
                    continue

                # Get x and y coordinate of third object
                other_objx = other_obj[1][0]
                other_objy = other_obj[1][1]

                if (A[0] <= other_objx <= B[0] and A[1] <= other_objy <= B[1]) or \
                   (A[0] <= other_objx <= B[0] and B[1] <= other_objy <= A[1]) or \
                   (B[0] <= other_objx <= A[0] and B[1] <= other_objy <= A[1]) or \
                   (B[0] <= other_objx <= A[0] and A[1] <= other_objy <= B[1]):
                    between_count += 1

            answer = between_count + 4
        elif subtype == 1:
            """is-on-band->yes/no"""
            
            grace_threshold = 12  # half of the size of objects
            epsilon = 1e-10  
            m = (B[1]-A[1])/((B[0]-A[0]) + epsilon ) # add epsilon to prevent dividing by zero
            c = A[1] - (m*A[0])

            answer = 1  # default answer is 'no'

            # check if any object lies on/close the line between object A and object B
            for other_obj in objects:
                # skip object A and B
                if (other_obj[0] == color1) or (other_obj[0] == color2):
                    continue

                other_obj_pos = other_obj[1]
                
                # y = mx + c
                y = (m*other_obj_pos[0]) + c
                if (y - grace_threshold)  <= other_obj_pos[1] <= (y + grace_threshold):
                    answer = 0
        elif subtype == 2:
            """count-obtuse-triangles->1~6"""

            obtuse_count = 0

            # disable warnings
            # the angle computation may fail if the points are on a line
            warnings.filterwarnings("ignore")
            for other_obj in objects:
                # skip object A and B
                if (other_obj[0] == color1) or (other_obj[0] == color2):
                    continue

                # get position of 3rd object
                C = other_obj[1]
                # edge length
                a = np.linalg.norm(B - C)
                b = np.linalg.norm(C - A)
                c = np.linalg.norm(A - B)
                # angles by law of cosine
                alpha = np.rad2deg(np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))
                beta = np.rad2deg(np.arccos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c)))
                gamma = np.rad2deg(np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b)))
                max_angle = max(alpha, beta, gamma)
                if max_angle >= 90 and max_angle < 180:
                    obtuse_count += 1

            warnings.filterwarnings("default")
            answer = obtuse_count + 4

        ternary_answers.append(answer)

    ternary_relations = (ternary_questions, ternary_answers)
    binary_relations = (binary_questions, binary_answers)
    norelations = (norel_questions, norel_answers)
    
    ternary_exp = expand_ternary(ternary_relations)
    binary_exp = expand_binary(binary_relations)
    unary_exp = expand_unary(norelations)

    img = img/255.
    dataset = (img, ternary_relations, binary_relations, norelations)

    # Only taking Unary and binary 
    n_un,n_bi = len(unary_exp[0]),len(binary_exp[0])

    new_df = pd.DataFrame({
        'Image': [f'./data/img/img_{ind}.npz']*(n_un+n_bi),
        'State': [state]*(n_un+n_bi),
        'Question': unary_exp[0] + binary_exp[0],
        'Answer': unary_exp[1] + binary_exp[1],
        'Relation': ['unary']*(n_un) + ['binary']*(n_bi),
        'Ques type': unary_exp[2] + binary_exp[2]
    })

    # new_row = pd.Series({'Image':f'./data/img/img_{ind}.npz',
    #             'State':state,
    #             'Ternary QA': ternary_relations,
    #             'Binary QA': binary_relations,
    #             'Unary QA': norelations,
    #             "Ternary Exp": ternary_exp,
    #             "Binary Exp": binary_exp,
    #             "Unary Exp": unary_exp})

    # df = pd.concat([df,new_row.to_frame().T],ignore_index=True)
    
    # df = pd.concat([df,new_df],ignore_index=True)

    return dataset,new_df

# test_df = pd.DataFrame(columns=["Image","State",
#                                 "Question","Answer",
#                                 "Relation","Ques Type"])

# train_df = pd.DataFrame(columns=["Image","State",
#                                 "Question","Answer",
#                                 "Relation","Ques Type"])
test_datasets = []
train_datasets = []

test_df_list = []
train_df_list = []

print('Building test datasets...')
for i in tqdm.tqdm(range(test_size),desc='Test Set'):
    tmp_test,tmp_test_df = build_dataset(i,'test',dirs)
    test_df_list.append(tmp_test_df)
    test_datasets.append(tmp_test)

start = time.time()
print(f'Combining test dataset and saving....')
test_df = pd.concat(test_df_list,ignore_index=True)
test_df.to_pickle(f'{dirs}/test_df.pkl')
print(f'Done. Time taken: {time.time()-start}')

print('Building train datasets...')
for i in tqdm.tqdm(range(train_size),desc='Train Set'):
    tmp_train,tmp_train_df = build_dataset(i,'train',dirs)
    train_df_list.append(tmp_train_df)
    train_datasets.append(tmp_train)

start = time.time()
print(f'Combining train dataset and saving....')
train_df = pd.concat(train_df_list,ignore_index=True)
train_df.to_pickle(f'{dirs}/train_df.pkl')
print(f'Done. Time taken: {time.time()-start}')

'''
Dataset is stored as (Number of samples) -> (img,ternary relations, binary relations, norelations) 
img -> 75X75X3
ternary relations -> (ternary_questions, ternary_answers) 
    10 ternary questions per image with shape 2 x (6 for one-hot vector of color), 3 for question type, 3 for question subtype
    10 ternary answers with shape being [yes, no, rectangle, circle, r, g, b, o, k, y]. Note that r,b,g,o,k,y act as number in some question types
Same for binary and no relations questions
'''

filename = os.path.join(dirs,'sort-of-clevr.pickle')
with  open(filename, 'wb') as f:
    pickle.dump((train_datasets, test_datasets), f)
print('datasets saved at {}'.format(filename))
