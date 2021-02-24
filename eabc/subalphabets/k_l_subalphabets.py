#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 09:43:57 2021

@author: giulialatini
"""




import random
def k_subalphabets (alphabet,k):
    ksubalphabets=[]
    sum_quality=0
    n=int(len(alphabet)/k)
    for symb in alphabet:
        sum_quality=symb.quality+sum_quality
    if sum_quality == 0:
        for _ in range(k):
            ksub_set=set(random.choices(alphabet, weights =None, k=n))
            k_sub=list(ksub_set)
            ksubalphabets.append(k_sub)
    else:
        prob=[]
        for symb in alphabet:
            prob.append(symb.quality/sum_quality)
        for i,n in enumerate(prob):
            if n==0:
               prob[i]=random.random()
        for _ in range(k):
            ksub_set=set(random.choices(alphabet,weights =prob,k=n))
            k_sub=list(ksub_set)
            ksubalphabets.append(k_sub)
    return(ksubalphabets)    


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
            alph_offsprings.append(alph_offspring)
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
            alph_offsprings.append(alph_offspring)
        else:
            ind1 = random.choice(subalphabets)
            n1= random.randint(0, len(ind1))
            del ind1[-n1]
            n2= random.randint(0, len(ind1))
            del ind1[n2]
            alph_offspring =ind1
            alph_offsprings.append(alph_offspring)
    return(alph_offsprings)       