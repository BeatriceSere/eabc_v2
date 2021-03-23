#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:43:57 2021

@author: giulialatini & beatriceserenellini
"""

import random 
'''
    k-subalphabets
    --------------
    
    All k sub-alphabets are the same length
'''

def k_subalphabets (alphabet,kappa_subal,classes,Log_alphabet):
    ksubalphabets=[]
    sum_quality=0
    n=int(len(alphabet)/kappa_subal)
    for symb in alphabet:
        sum_quality=symb.quality+sum_quality
    if sum_quality == 0:
        for _ in range(kappa_subal):
            ksub_set=set(random.choices(alphabet, weights =None, k=n))
            k_sub=list(ksub_set)
            k_sub=k_sub[:1000]
            for swarmClass in classes:
                number_symbols_class=len([sym for sym in k_sub if sym.classSymb==swarmClass])
                if number_symbols_class==0:
                    print('** Add symbol **')
                    k_sub.append(Log_alphabet[swarmClass][0])
            ksubalphabets.append(k_sub)
    else:   
        prob=[]
        for symb in alphabet:
            prob.append(symb.quality/sum_quality)
        for counter,value in enumerate(prob):
            if value==0:
               prob[counter]=random.random()
        for _ in range(kappa_subal):
            ksub_set=set(random.choices(alphabet,weights =prob,k=n))
            k_sub=list(ksub_set)
            k_sub=sorted(k_sub, key=lambda x: x.quality, reverse=True)
            #k_sub.sort(reverse=True, key=k_sub[:].quality)
            k_sub=k_sub[:1000]
            for swarmClass in classes:
                number_symbols_class=len([sym for sym in k_sub if sym.classSymb==swarmClass])
                if number_symbols_class==0:
                    print('** Add symbol **')
                    k_sub.append(Log_alphabet[swarmClass][0])
            ksubalphabets.append(k_sub)
            
    return(ksubalphabets)    

    

'''
    k-subalphabets
    --------------

    Generation of k sub-alphabets choosing symbols from general alphabet containing the symbols
    of all classes of the g-th generation. The probability of choosing one symbol over another
    is linked to symbol quality.
    
    Special cases:
        
        - FIRST CYCLE, all symbols have zero quality: in this case all symbols are chosen 
        with the random probability;
        
        - SYMBOLS NEVER CHOSEN, therefore with a null quality: in this case these symbols 
        are chosen with the random probability.
    
'''
'''
def k_subalphabets (alphabet,kappa_subal,classes,Log_alphabet):
    ksubalphabets=[]
    sum_quality=0
    n=int(len(alphabet)/kappa_subal)
    for symb in alphabet:
        sum_quality=symb.quality+sum_quality
    if sum_quality == 0:
        for _ in range(kappa_subal):
            ksub_set=set(random.choices(alphabet, weights =None, k=n))
            k_sub=list(ksub_set)
            for swarmClass in classes:
                number_symbols_class=len([sym for sym in k_sub if sym.classSymb==swarmClass])
                if number_symbols_class==0:
                    print('** Add symbol **')
                    k_sub.append(Log_alphabet[swarmClass][0])
            ksubalphabets.append(k_sub)
    else:   
        prob=[]
        for symb in alphabet:
            prob.append(symb.quality/sum_quality)
        for counter,value in enumerate(prob):
            if value==0:
               prob[counter]=random.random()
        for _ in range(kappa_subal):
            ksub_set=set(random.choices(alphabet,weights =prob,k=n))
            k_sub=list(ksub_set)
            for swarmClass in classes:
                number_symbols_class=len([sym for sym in k_sub if sym.classSymb==swarmClass])
                if number_symbols_class==0:
                    print('** Add symbol **')
                    k_sub.append(Log_alphabet[swarmClass][0])
            ksubalphabets.append(k_sub)
            
    return(ksubalphabets)    
'''
'''
    l-subalphabets
    --------------

   Generation of l sub-alphabets starting from the previously generated k. 
   The creation of the l-alphabets takes place by carrying out the following 
   operations on the k-alphabets:
    
        - UNION: union of two sub-alphabets, choosen randomly 
        
        - INTERSECTION: intersection of two sub-alphabets, choosen randomly 
        
        - CUT: deletion of two symbols of one of k-subalphabets , choosen randomly 
            
    The choice of an operation is random.
    
'''



def l_subalphabets (subalphabets,l):
    alph_offsprings=[]
    for _ in range(l):
        op_choice1 = random.random()
        if op_choice1<0.3:                    # Apply union
            ind1 = random.choice(subalphabets)
            ind2 = random.choice(subalphabets)
            while ind1 == ind2:
                ind1 = random.choice(subalphabets)
                ind2 = random.choice(subalphabets)
            alph_offspring = set(ind1) | set(ind2)
            alph_offspring = list(alph_offspring)
            alph_offsprings.append(alph_offspring[:1000])
        elif 0.3<op_choice1<0.6:                                   # Apply intersection
            ind1 = random.choice(subalphabets)
            ind2 = random.choice(subalphabets)
            alph_offspring = set(ind1) & set(ind2)
            while len(alph_offspring) == 0:
                ind1 = random.choice(subalphabets)
                ind2 = random.choice(subalphabets)
                alph_offspring = set(ind1) & set(ind2)
            alph_offspring= set(ind1) & set(ind2)
            alph_offspring = list(alph_offspring)
            alph_offsprings.append(alph_offspring[:1000])
        else:
            ind1 = random.choice(subalphabets)
            n1= random.randint(0, len(ind1))
            del ind1[n1]
            n2= random.randint(0, len(ind1))
            del ind1[n2]
            alph_offspring =ind1
            alph_offsprings.append(alph_offspring[:1000])
    return(alph_offsprings)       