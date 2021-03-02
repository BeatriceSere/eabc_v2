
from deap import base, creator,tools
from deap.algorithms import varOr
import numpy as np
import random
import pickle
import networkx as nx
import multiprocessing
from functools import partial
import copy

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from scipy.optimize import differential_evolution

from Datasets.IAM import IamDotLoader
from Datasets.IAM import Letter,GREC,AIDS
from eabc.datasets import graph_nxDataset
from eabc.extractors import Extractor
from eabc.extractors import randomwalk_restart
from eabc.embeddings import SymbolicHistogram
from eabc.extras.featureSelDE import FSsetup_DE,FSfitness_DE 
from eabc.environments.nestedFS import eabc_Nested
from eabc.granulators.granule import Granule
from eabc.subalphabets.k_l_subalphabets import k_subalphabets, l_subalphabets
def IAMreader(parser,path):
    
    delimiters = "_", "."      
    
    Loader = IamDotLoader.DotLoader(parser,delimiters=delimiters)
    
    graphDict = Loader.load(path)
    
    graphs,classes=[],[]
    for g,label in graphDict.values():
        graphs.append(g)
        classes.append(label)
    
    return graphs, classes 


def main(dataTR,dataVS,dataTS,N_subgraphs,mu,lambda_,ngen,maxorder,cxpb,mutpb):
    
    print("Setup...")
    #Graph decomposition
    # extract_func = randomwalk_restart.extr_strategy(max_order=maxorder)
    extract_func = randomwalk_restart.extr_strategy()
    subgraph_extr = Extractor(extract_func)

    expTRSet = subgraph_extr.decomposeGraphDataset(dataTR,maxOrder= maxorder)
    expVSSet = subgraph_extr.decomposeGraphDataset(dataVS,maxOrder= maxorder)
    expTSSet = subgraph_extr.decomposeGraphDataset(dataTS,maxOrder= maxorder)
    
    
    
    ####################################
    # Evaluate the individuals with an invalid fitness
    DEBUG_FIXSUBGRAPH = False
    DEBUG_FITNESS = False
    DEBUG_INDOCC = True
    print("Initializing populations...")
    if DEBUG_FIXSUBGRAPH:
        print("DEBUG SUBGRAPH STOCHASTIC TRUE")
    if DEBUG_FITNESS:
        print("DEBUG FITNESS TRUE")
    if DEBUG_INDOCC:
        print("DEBUG REPEATED IND TRUE")        
    classes= dataTR.unique_labels()
    
    #Initialize a dict of swarms - {key:label - value:deap popolution}
    population = {thisClass:toolbox.population(n=mu) for thisClass in classes}
    IDagentsHistory = {thisClass:[ind.ID for ind in population[thisClass]] for thisClass in classes}
    
    if DEBUG_FIXSUBGRAPH:
        subgraphsByclass = {thisClass:[] for thisClass in classes}
        
    for swarmClass in classes:

        thisClassPatternIDs = np.where(np.asarray(dataTR.labels)==swarmClass)[0]
        classAwareTR = dataTR[thisClassPatternIDs.tolist()]
        ##
        if DEBUG_FIXSUBGRAPH:
            subgraphsByclass[swarmClass] = subgraph_extr.randomExtractDataset(classAwareTR, N_subgraphs)
            subgraphs = [subgraphsByclass[swarmClass] for _ in population[swarmClass]]
        else:
            subgraphs = [subgraph_extr.randomExtractDataset(classAwareTR, N_subgraphs) for _ in population[swarmClass]]       
    ####################################
     
    ################## BEGINNING OF GENERATIONS ##################    
    # Log book
    LogAgents = {gen: {thisClass:[] for thisClass in classes} for gen in range(ngen+1)}
    LogPerf = {thisClass:[] for thisClass in classes}
    Log_alphabet={thisClass:[] for thisClass in classes}
    LogAccuracy = []
    alphabet=[]
    
    # Begin the generational process   
    for gen in range(1, ngen + 1):
            print("######################## Generation: {} ########################".format(gen))
            
            for swarmClass in classes:
                print("############")
                
                # Generate the offspring: mutation OR crossover OR reproduce and individual as it is
                offspring = toolbox.varOr(population=population[swarmClass],toolbox=toolbox,lambda_=lambda_, idHistory=IDagentsHistory[swarmClass])
                
                # Selecting data for this swarm               
                thisClassPatternIDs = np.where(np.asarray(dataTR.labels)==swarmClass)[0]
                classAwareTR = dataTR[thisClassPatternIDs.tolist()]
                
                # Select both old and offspring for evaluation in order to run agents
                population[swarmClass] = population[swarmClass] + offspring
                
                # Select class agent with a property 'classAgent' in environments->nestedFS
                for agent in population[swarmClass]:
                    agent.classAgents=swarmClass
                print('Loading class',agent.classAgents)
                
                # Select pop number of buckets to be assigned to agents
                if DEBUG_FIXSUBGRAPH:
                    subgraphs = [subgraphsByclass[swarmClass] for _ in population[swarmClass]]
                else:
                    subgraphs = [subgraph_extr.randomExtractDataset(classAwareTR, N_subgraphs) for _ in population[swarmClass]]
                
                # Run individual and return the partial fitness comp+card
                fitnesses,alphabets = zip(*toolbox.map(toolbox.evaluate, zip(population[swarmClass],subgraphs)))
                
                ''' Generate IDs for agents that pushed symbols in class bucket
                    E.g. idAgents       [ 0   0    1   1  1     2    -  3    .... ]
                         alphabets      [[s1 s2] [ s3  s4 s5]  [s6] []  [s7] .... ]
                    Identify the agent that push s_i symbol
                     idAgents=[]
                     for i in range(len(pop)):
                         if alphabets[i]:
                             for _ in range(len(alphabets[i])):
                                 idAgents.append(i)'''
                
                # Concatenate symbols if not empty
                alphabet = sum(alphabets,[]) + alphabet
                Log_alphabet[swarmClass] = alphabet 
            alphabet=list(set(alphabet))
            
            # Generate k+l subalphabets
            ### View k_l_subalphabets into eabc directory
            print('########### SUBALPHABETS #############')
            ksubalphabets = k_subalphabets(alphabet,3)
            print('ksubalphabets =', len(ksubalphabets))
            lsubalphabets = l_subalphabets(ksubalphabets,2)
            print('lsubalphabets =', len(lsubalphabets)) #'len_l=', len(lsubalphabets[0]), len(lsubalphabets[1]), len(lsubalphabets[2]))
            klsubalphabets = ksubalphabets + lsubalphabets 
            
            ##################
            # Restart with previous symbols
            #thisGenClassAlphabet = alphabets + ClassAlphabets[swarmClass]
            ##################
            
            sub_position = 0           
            for sub in klsubalphabets:
                embeddingStrategy = SymbolicHistogram(isSymbolDiss=True,isParallel=True)
        
                # Embedding with current symbols
                embeddingStrategy.getSet(expTRSet, sub)
                TRembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
                TRpatternID = embeddingStrategy._embeddedIDs
        
                embeddingStrategy.getSet(expVSSet, sub)
                VSembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
                VSpatternID = embeddingStrategy._embeddedIDs        
        
                # Resorting matrix for consistency with dataset        
                TRorderID = np.asarray([TRpatternID.index(x) for x in dataTR.indices])
                VSorderID = np.asarray([VSpatternID.index(x) for x in dataVS.indices])        
                TRMat = TRembeddingMatrix[TRorderID,:]
                VSMat = VSembeddingMatrix[VSorderID,:]        
                
                # Labels 
                TRlabels = np.array(dataTR.labels)
                #VSlabels = np.array(dataVS.labels)
                
                # Classifier
                classifier = KNN()
                classifier.fit(TRMat,TRlabels)
                predictedVSLabels = classifier.predict(VSMat)
                #print(VSlabels)
                #print(predictedVSLabels)
                #print(dataVS.labels,predictedVSLabels) 
                #print("{},{}".format(len(VSlabels),len(predictedVSLabels)))
                
                # Model accuracy
                accuracy = sum(predictedVSLabels==np.asarray(dataVS.labels))/len(dataVS.labels)
                t = (sub_position,accuracy)
                LogAccuracy.append(t)
                print(sub_position,'-th subalphabet')
                sub_position = sub_position + 1
                print("Accuracy {} - alphabet = {}".format(accuracy,len(sub)))
            #print(LogAccuracy) 
            
            ##################
            # Find the best subalphabets by evaluating the accuracy of the model
            LogAccuracy.sort(key=lambda x: x[1],reverse=True)
            kappa=len(ksubalphabets)
            winning_alphabets = []
            for i in range(kappa):
                winning_alphabets.append(klsubalphabets[LogAccuracy[i][0]])
            print('Length of the winning alphabets:',len(winning_alphabets))
            ##################
            
            ################ REWARD ####################
            print('###### REWARD ######')
            # Assign the final fitness to agents
            fitnessesRewarded = list(fitnesses)
            
            # Winners symbols
            Symbols=list(set(np.concatenate((np.array(winning_alphabets)))))
            
            # Log book
            rewardLog = []
            qualityLog = []
            pop= []
            
            for swarmClass in classes:               
                for agent in range(len(population[swarmClass])):
                    
                    # ID and Class of each agent
                    agentID = population[swarmClass][agent].ID
                    classAgent= population[swarmClass][agent].classAgents
                    
                    NagentSymbolsInModels=len([sym for sym in Symbols if sym.owner==str(agentID)+classAgent])
                    #print('Simbols for agent=', NagentSymbolsInModels)
                    if NagentSymbolsInModels == 0:
                        reward=0
                    else:
                        for position,winner in enumerate(winning_alphabets):
                                for sym in winner:
                                    if sym.owner==str(agentID)+classAgent:
                                        if LogAccuracy[position][1] <= 0.75:
                                            sym.quality = sym.quality-1
                                        elif LogAccuracy[position][1] >= 0.85:
                                             sym.quality = sym.quality+10
                                        else:
                                             sym.quality = sym.quality+1
                        for sym in Symbols:
                            if sym.owner==str(agentID)+classAgent:
                                qualityLog.append(sym.quality)
                        reward = sum(qualityLog)/NagentSymbolsInModels
                    rewardLog.append(reward)
                    if DEBUG_FITNESS:
                        fitnessesRewarded[agent] = reward,
                    else:
                        fitnessesRewarded[agent] = fitnesses[agent][0]+reward
                    qualityLog = []
                    #print('fitness=', fitnessesRewarded[agent]) 
            ####################################        

                
 
            if DEBUG_INDOCC:
                fitmean = []
            for ind, fit in zip(pop, fitnessesRewarded):
                if DEBUG_INDOCC:
                    ids = np.asarray([thisInd.ID for thisInd in pop])
                    fitness = np.asarray(fitnessesRewarded)
                    indices = np.where(ids == ind.ID)
                    fit = np.mean(fitness[indices]),
                ind.fitness.values = fit
                if DEBUG_INDOCC:
                    fitmean.append(fit)
            ##           
            # x = np.asarray([ind.fitness.values[0] for ind in pop])
            # y = np.asarray([fit[0] for fit in fitmean])
            # if not np.all(x == y):
            #     pause = input("Stop Error")
            #     print("in pop")
            #     print(np.asarray([ind.fitness.values[0] for ind in pop]))
            #     pause = input()
            #     print("fitness list ")
            #     print(np.asarray([fit[0] for fit in fitmean]))
            #     #print(np.asarray([fit[0] for fit in fitnesses]))                    
            #     pause = input()
            #     print(np.where(x!=y))
            #     pause = input()
            #     for ind in pop:
            #         print(ind.ID,ind.fitness.valid)
            ##
            
            # Select the next generation population for the current swarm
            for swarmClass in classes:
                population[swarmClass] = toolbox.select(population[swarmClass], mu)
                
            # Save Informedness for class and gen
            ###LogPerf[swarmClass].append([J,sum(np.asarray(best_GA2)==1),len(best_GA2)])
            
            # Save population at g = gen
            #LogAgents[gen][swarmClass].append([pop,fitnesses,rewardLog,fitnessesRewarded])
            
            
            Log_sym_quality = {gen: {sym.owner:sym.quality for sym in Symbols} for gen in range(1,ngen+1)}
            #print(Log_sym_quality)
            print("------------------------------------------------------------------")
    print("Test phase")
    #Collect class alphabets and embeddeding TR,VS,TS with concatenated Alphabets
    ALPHABETS=[alphabets for alphabets in ClassAlphabets.values()]   
    ALPHABETS = sum(ALPHABETS,[])
    
    embeddingStrategy.getSet(expTRSet, ALPHABETS)
    TRembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
    TRpatternID = embeddingStrategy._embeddedIDs

    embeddingStrategy.getSet(expVSSet, ALPHABETS)
    VSembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
    VSpatternID = embeddingStrategy._embeddedIDs

    #Resorting matrix for consistency with dataset        
    TRorderID = np.asarray([TRpatternID.index(x) for x in dataTR.indices])
    VSorderID = np.asarray([VSpatternID.index(x) for x in dataVS.indices])   

    TRMat = TRembeddingMatrix[TRorderID,:]
    VSMat = VSembeddingMatrix[VSorderID,:]        

    #Feature Selection                  
    #bounds_GA2, CXPB_GA2, MUTPB_GA2, DE_Pop = FSsetup_DE(len(ALPHABETS), -1)
    #FS_accDE= partial(FSfitness_DE,perfMetric = 'accuracy')
    #TuningResults_GA2 = differential_evolution(FS_accDE, bounds_GA2, 
    #                                           args=(TRMat,
    #                                                 VSMat, 
    #                                                 dataTR.labels, 
    #                                                 dataVS.labels),
    #                                                 maxiter=100, init=DE_Pop, 
    #                                                 recombination=CXPB_GA2,
    #                                                 mutation=MUTPB_GA2, 
    #                                                 workers=-1, 
    #                                                 polish=False, 
    #                                                 updating='deferred')
    
    #best_GA2 = [round(i) for i in TuningResults_GA2.x]
    #print("Selected {}/{} feature".format(sum(np.asarray(best_GA2)==1), len(best_GA2)))
    
    #Embedding with best alphabet
    #mask = np.array(best_GA2,dtype=bool)
    classifier = KNN()
    classifier.fit(TRMat, dataTR.labels)
    #predictedVSmask=classifier.predict(VSMat[:, mask])
    
    #accuracyVS = sum(predictedVSmask==np.asarray(dataVS.labels))/len(dataVS.labels)
    #print("Accuracy on VS with global alphabet: {}".format(accuracyVS))

    #Embedding TS with best alphabet
    #ALPHABET = np.asarray(ALPHABETS,dtype = object)[mask].tolist()
    embeddingStrategy.getSet(expTSSet, ALPHABETS)
    TSembeddingMatrix = np.asarray(embeddingStrategy._embeddedSet)
    TSpatternID = embeddingStrategy._embeddedIDs   
    TSorderID = np.asarray([TSpatternID.index(x) for x in dataTS.indices]) 
    TSMat = TSembeddingMatrix[TSorderID,:]
    
    predictedTS=classifier.predict(TSMat)
    accuracyTS = sum(predictedTS==np.asarray(dataTS.labels))/len(dataTS.labels)
    print("Accuracy on TS with global alphabet: {}".format(accuracyTS))    
       

    return LogAgents,LogPerf,ClassAlphabets,TRMat,VSMat,predictedVSmask,dataVS.labels,TSMat,predictedTS,dataTS.labels,ALPHABETS,ALPHABET,mask

if __name__ == "__main__":


    seed = 64
    random.seed(seed)
    np.random.seed(seed)
    # Parameter setup
    # They should be setted by cmd line
    path ="/home/LabRizzi/eabc_v2/Datasets/IAM/Letter3/"
    name = "LetterH"
    # path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/GREC/"
    # name = "GREC"  
    # path = "/home/luca/Documenti/Progetti/E-ABC_v2/eabc_v2/Datasets/IAM/AIDS/"
    # name = "AIDS" 
    N_subgraphs = 20
    ngen = 50
    mu = 10
    lambda_= 50
    maxorder = 5
    CXPROB = 0.45
    MUTPROB = 0.45
    INDCXP = 0.3
    INDMUTP = 0.3
    TOURNSIZE = 3
    QMAX = 500


    ###Preprocessing
    print("Loading...")    
    
    if name == ('LetterH' or 'LetterM' or 'LetterL'):
        parser = Letter.parser
    elif name == 'GREC':
        parser = GREC.parser
    elif name == 'AIDS':
        parser = AIDS.parser
    else:
        raise FileNotFoundError
        
    
    IAMreadergraph = partial(IAMreader,parser)
    rawtr = graph_nxDataset(path+"Training/", name, reader = IAMreadergraph)
    rawvs = graph_nxDataset(path+"Validation/", name, reader = IAMreadergraph)
    rawts = graph_nxDataset(path+"Test/", name, reader = IAMreadergraph)

    ####
    if name == ('LetterH' or 'LetterM' or 'LetterL'):  
        weights = Letter.normalize('coords',rawtr.data,rawvs.data,rawts.data)
    elif name == 'GREC':
        weights = GREC.normalize(rawtr.data,rawvs.data,rawts.data)
    elif name == 'AIDS':
        weights = AIDS.normalize(rawtr.data,rawvs.data,rawts.data)
    ###
    
    #Slightly different from dataset used in pygralg
    dataTR = rawtr
    dataVS = rawvs
    dataTS = rawts
    
    #Create type for the problem
    if name == 'GREC':
        Dissimilarity = GREC.GRECdiss
    elif name == ('LetterH' or 'LetterM' or 'LetterL'):
        Dissimilarity = Letter.LETTERdiss
    elif name == 'AIDS':
        Dissimilarity = AIDS.AIDSdiss


    #Maximizing
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Agent", list, fitness=creator.FitnessMax, ID = None)
    
    toolbox = base.Toolbox()
    
    #Multiprocessing map
    # pool = multiprocessing.Pool()
    # toolbox.register("map", pool.map)

    eabc_Nested = eabc_Nested(DissimilarityClass=Dissimilarity,problemName = name,DissNormFactors=weights)
    
    #Q scaling
    scale_factor = len(np.unique(dataTR.labels,dataVS.labels,dataTS.labels)[0])
    scaledQ = round(QMAX/scale_factor)
    
    toolbox.register("attr_genes", eabc_Nested.gene_bound,QMAX = scaledQ) 
    toolbox.register("agent", tools.initIterate,
                    creator.Agent, toolbox.attr_genes)
    
    
    toolbox.register("population", eabc_Nested.initAgents, toolbox.agent,n=100)  
    toolbox.register("evaluate", eabc_Nested.fitness)
    toolbox.register("mate", eabc_Nested.customXover,indpb=INDCXP)
    #Setup mutation
    toolbox.register("mutate", eabc_Nested.customMutation,mu = 0, indpb=INDMUTP)
    toolbox.register("select", tools.selTournament, tournsize=TOURNSIZE)
    toolbox.register("varOr",eabc_Nested.varOr,cxpb= CXPROB, mutpb=MUTPROB)
    
    #Decorator bound    
    toolbox.decorate("mate", eabc_Nested.checkBounds(scaledQ))
    toolbox.decorate("mutate", eabc_Nested.checkBounds(scaledQ))
    
    LogAgents, LogPerf,ClassAlphabets,TRMat,VSMat,predictedVSmask,VSlabels,TSMat,predictedTS,TSlabels, ALPHABETS,ALPHABET,mask = main(dataTR,
                                                                                                                                      dataVS,
                                                                                                                                      dataTS,
                                                                                                                                      N_subgraphs,
                                                                                                                                      mu,
                                                                                                                                      lambda_,
                                                                                                                                      ngen,
                                                                                                                                      maxorder,
                                                                                                                                      CXPROB,
                                                                                                                                      MUTPROB)
    
    

    pickle.dump({'Name': name,
                 'Path': path,
                 'CrossOverPr':CXPROB,
                 'MutationPr':MUTPROB,
                 'IndXoverPr':INDCXP,
                 'IndMutPr':INDMUTP,
                 'TournamentSize':TOURNSIZE,
                 'Seed':seed,
                'Agents':LogAgents,
                'PerformancesTraining':LogPerf,
                'ClassAlphabets':ClassAlphabets,
                'TRMat':TRMat,
                'TRlabels':dataTR.labels,
                'VSMat':VSMat,
                'predictedVSmask':predictedVSmask,
                'VSlabels':VSlabels,
                'TSMat':TSMat,
                'predictedTS':predictedTS,
                'TSlabels':TSlabels,
                'ALPHABETS':ALPHABETS,
                'ALPHABET':ALPHABET,
                'mask':mask,
                'N_subgraphs':N_subgraphs,
                'N_gen':ngen,
                'Mu':mu,
                'lambda':lambda_,
                'max_order':maxorder
                },
                open(name+'.pkl','wb'))
    
    