import random
import os
import glob

def get_whole_name(path:str,filter=None):

    if isinstance(path,str):
        if filter is None:
            name_list=os.listdir(path)
            with open('whole_name.txt','w') as f:
                for i in range(len(name_list)):
                    f.write(name_list[i]+'\n')
        else:
            name_list=glob.glob(path+filter)

            with open('whole_name.txt','w') as f:
                for i in range(len(name_list)):
                    f.write(os.path.basename(name_list[i])+'\n')
        
        print('done!')

    else:
        raise ValueError('please input a path str')

def ratio_split(input,test_ratio:float):

    if isinstance(input,str):
        with open(input,'r') as f:
            name_list = [t.strip() for t in f.readlines()]
    
    elif isinstance(input,list):
        name_list=input
    
    else:
        raise ValueError('please input a path str or a list of names!')
    
    random.shuffle(name_list)

    num_whole=len(name_list)   
    num_test=int(test_ratio*num_whole)

    test_list=random.sample(name_list,num_test)
    train_list=list(set(name_list).difference(set(test_list)))

    with open('test_index.txt','w') as f:
        for i in range(len(test_list)):
            f.write(test_list[i]+'\n')
    
    with open('train_index.txt', 'w') as f:
        for i in range(len(train_list)):
            f.write(train_list[i]+'\n')
    
    print('done!')

def k_fold_split(input,k:int):

    if isinstance(input,str):
        with open(input,'r') as f:
            name_list = [t.strip() for t in f.readlines()]
    
    elif isinstance(input,list):
        name_list=input
    
    else:
        raise ValueError('please input a path str or a list of names!')
    
    random.shuffle(name_list)
    num_block=len(name_list)//k

    for i in range(k):

        st=i*num_block
        ed=i*num_block+num_block

        if i==k-1:
            ed=len(name_list)
        
        with open('train_f%d.txt'%(i),'w') as f:
            for i in range(st,ed):
                f.write(name_list[i]+'\n')
    
    print('done!')
