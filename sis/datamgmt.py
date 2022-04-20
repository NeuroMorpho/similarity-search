from ntpath import join
import faiss
from numpy.core.arrayprint import printoptions
from numpy.lib.function_base import _digitize_dispatcher
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np
from . import com
import pyhash

depdate = com.todaysdate()
#depdate = '2020-06-30'

def genIndexData(datadict,npcs=None):
    # Takes dictionary with datarows and neuron_ids as input
    # and calculates principal components in order to explain for 
    # 99% of the variance
    # generates a tuple of: 
    # - index (no PCA and PCA) 
    # - datarows (no PCA and PCA)
    # - neuronids (only one version as it is the same for both)
    datamtrx = datadict['datarows']
    datamtrx = datamtrx.astype('float32')
    datamtrx[np.isnan(datamtrx)]=0 # hack to remove nan values
    neuronids = datadict['neuron_ids']
    if "neuron_names" in datadict.keys():
        neuronnames = datadict['neuron_names']
    else:
        neuronnames = {}
 
    d = datamtrx.shape[1]

    faiss.normalize_L2(datamtrx)
    index = []
    datarows= []

    # Generate non PCA index
    index.append(faiss.IndexFlatL2(d))
    index[0].add(datamtrx) 
    datarows.append(datamtrx)
    print('Before dimensionality reduction: {}'.format(d))

    # Set number of components to explain for 99% of variance unless provided
    pcaobj = PCA('mle')
    # if npcs is None:
    #     pcaobj = PCA()
    # else:
    #     pcaobj = PCA(npcs)
    pcaobj.fit(datamtrx)
    datamtrx = pcaobj.transform(datamtrx)
    ncomp = pcaobj.n_components_
    print('PCA performed using {} components'.format(ncomp))
    faiss.normalize_L2(datamtrx)
    #Use the number of components to explain 99 % variance
    # Generate PCA index
    index.append(faiss.IndexFlatL2(int(ncomp)))
    index[1].add(datamtrx) 
    datarows.append(datamtrx)
    
    return (index, datarows, neuronids,neuronnames)

def genMtrx(dataset, neuronix=4, dataixstart=5,nameix = 1):
    datarows = []
    neuron_ids = []
    neuron_names = []
    for line in dataset:
        datarows.append(line[dataixstart:])
        neuron_ids.append(line[neuronix]) 
        neuron_names.append(line[nameix])
    datadict = {
        'datarows': np.array(datarows),
        'neuron_ids': neuron_ids,
        'neuron_names': dict(zip(neuron_ids,neuron_names))
    }
    return datadict 


def generateMtrxFrData(whereclause=''):
    dataset = com.getfromdb(whereclause,depositiondate=depdate)
    print('Creating data matrix for pvec + measurements...')    
    return genMtrx(dataset)

def genMtrxPvec(*args,depositiondate=depdate):
    if len(args) > 0:
        neuronstring = 'WHERE p.neuron_id IN ({})'.format(args[0])
    else:
        neuronstring = ''
    dataset = com.getpvecfromdb(neuronstring,depositiondate=depdate)
    print('Creating data matrix for pvec...')
    return genMtrx(dataset,neuronix=2,dataixstart=3)

def genMtrxMeta(*args,depositiondate=depdate):
    if len(args) > 0:
        neuronstring = 'WHERE p.neuron_id = {}'.format(args[0])
    else:
        neuronstring = ''
    dataset = com.getfromdbmeta(neuronstring)
    hasher = pyhash.murmur3_32()
    dataset = [[line[0]] + [hasher(str(item)) for item in line[1:30]] for line in dataset]
    print('Creating data matrix for meta...')
    return genMtrx(dataset,0,1)

def genMtrxMes(postfix, domainids, *args,depositiondate=_digitize_dispatcher):
    if len(args) > 0:
        neuronstring = 'WHERE p.neuron_id = {}'.format(args[0])
    else:
        neuronstring = ''
    dataset = com.getfromdbmes(postfix,domainids,neuronstring)
    print('Creating data matrix for detailed...')
    return genMtrx(dataset,0,2)
    
def genResult(searchvector,n_neurons,indextouse, neuronids,theneuronid,simlimit=0):
    #TODO add thisneuronid as param in order to clear from data

    k = n_neurons
    k = k+1 # add 1 as the neuron itself weill be included.
 
    searchvector = searchvector.reshape((1,len(searchvector)))

    D, I = indextouse.search(searchvector, k*2) # sanity check

    # print('Generating dataframe using {} nearest neighbours...'.format(k))

    # generate a list of those neurons which is above the threshold
    
    result = {}
    s = []
    # for rowno in range(D.shape[0]):
    #     # loop over similar neurons for each neuron in query
    #     #TODO ugly hack not allowing more than one neuron in query in order to fit old solution
    #     s =  I[rowno].tolist()
    
    prev_val= -1
    indexes = I[0][1:k*2]
    #change place with first value if in later values

    for ix,val in enumerate(indexes):
        if 1-D[0][ix+1] >= simlimit:
            
            thisneuronid = neuronids[val]
            #TODO check this! should it be here 
            if thisneuronid == neuronids[prev_val]:
                continue
            # if the neuron not in first place, change place
            if theneuronid == thisneuronid:
                thisneuronid = neuronids[I[0][0]]

            s.append({
                "neuron_id": thisneuronid,
                "similarity": float('%.7f'%(1-D[0][ix+1]))
            })
            prev_val = val
            
    result["similarityresult"] = s
    result["status"] = 200
    return result

def genResultAll(datamtrx,indextouse,neuronids,simlimit=0.99999,upperlimit=1):
    """
    Method for finding all duplicates. Returns dictionary of duplicates, 
    """

    D, I = indextouse.search(datamtrx, 10)

    # print('Generating dataframe using {} nearest neighbours...'.format(k))

    # generate a list of those neurons which is above the threshold
 
    (nsearch,sd) = datamtrx.shape
    i = 0
    data = []
    for i in range(0,nsearch):
        # Second distance greater
        
        if I[i][1] < I[i][0] and (1-D[i][1]) > simlimit and (1-D[i][1]) < upperlimit:
            orgix = i
            orgid = neuronids[orgix]
            dupid = neuronids[I[i][1]]
            if orgid != dupid:
                data.append({
                    "orgid": orgid,
                    "dupid": dupid,
                    "level": (1-D[i][1])
                })
        i += 1
    data = com.getdataforids(data)
    return {'data': data}

def genResultPar(datamtrx,nnbs,indextouse,neuronids):
    """
    Method for parallel similarity search. Params:
    datamatrix -  matrix of vectors in rows 
    nnbs - number of neighbours
    index to use for search (e.g. pvec only)
    neuronids - neuronids of vectors 
    """
    datamtrx = np.array(datamtrx)
    D, I = indextouse.search(datamtrx, nnbs+1)     
    (nsearch,sd) = datamtrx.shape
    i = 0
    simres = []
    for i in range(0,nsearch):
        # Second distance greater
        data = []

        orgix = i
        for j in range(0,nnbs+1):
            thisid = I[i][j]
            if i != thisid:
                data.append({
                    "neuron_id": neuronids[thisid],
                    "similarity": (1-D[i][j])
                })
        simres.append({
            'neuron_id': neuronids[i],
            'similarityresult': data
        })

    return simres


def checkduplicatesinternal(prev_data,pvecmes,name_dict):
    neuron_names = []
    (pre_nneurons,pre_d) = (len(prev_data),len(prev_data[0]))
    datamtrx = np.ndarray((0,prev_data.shape[1]))
    
    neuron_names = [item['neuron_name'] for item in pvecmes['datarows']]
    datamtrx = np.asarray([item['data'] for item in pvecmes['datarows']],dtype =np.float32)

    jointdata = np.append(prev_data.astype(np.float32),datamtrx, axis=0)
    jointdata = np.ndarray(jointdata.shape,buffer=jointdata,dtype=np.float32)
    index = faiss.IndexFlatIP(d)
    #index = faiss.IndexFlatL2(d)
    index.add(jointdata) 
    D, I = index.search(searchmtrx, 2)

    i = 0
    duplicatejson = []
    orgdups = {}
    duporgs = {}
    for i in range(0,nsearch):
        # Second distance greater
        
        if I[i][1] < I[i][0] and (1-D[i][1]) > pvecmes["duplicatelevel"]:
            orgix = i
            orgname = neuron_names[orgix]
            if I[i][1] > pre_nneurons: # duplicate in new nuerons
                duploname  = neuron_names[I[i][1] - pre_nneurons]
            else:
                # duplicate in old neurons
                duploname = name_dict[I[i][1]+1]
            if duploname != orgname:
                if duploname in duporgs.keys():
                    orgname,duploname = duporgs[duploname],orgname
                else:
                    orgdups[orgname] = duploname
                duporgs[duploname] = orgname
                duplicatejson.append({
                    "Original": orgname,
                    "Original link": "http://cng.gmu.edu:8080/neuroMorphoDev/neuron_info.jsp?neuron_name={}".format(orgname),
                    "Duplicate": duploname,
                    "Duplicate link": "http://cng.gmu.edu:8080/neuroMorphoDev/neuron_info.jsp?neuron_name={}".format(duploname),
                    "Level": (1-D[i][1])
                })
        i += 1
    return duplicatejson

def checkscaled(prev_data,pvecmes,neuronids,n_neurons):

    neuronids.append(0)
    
    datamtrx = np.reshape(np.asarray([pvecmes],dtype =np.float32),(1,121))

    jointdata = np.append(prev_data.astype(np.float32),datamtrx, axis=0)
    jointdata = np.ndarray(jointdata.shape,buffer=jointdata,dtype=np.float32)
    jointdata = jointdata.astype(np.float32)
    jointdata[np.isnan(jointdata)]=0
    jointdata = preprocessing.normalize(jointdata, norm='l2')
    pcaobj = PCA('mle')
    pcaobj.fit(jointdata)
    jointdata = pcaobj.transform(jointdata)
    d = jointdata.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(jointdata) 
    D, I = index.search(np.reshape(jointdata[-1,:],(1,d)), n_neurons+1)

    result = {}
    s = []
    # for rowno in range(D.shape[0]):
    #     # loop over similar neurons for each neuron in query
    #     #TODO ugly hack not allowing more than one neuron in query in order to fit old solution
    #     s =  I[rowno].tolist()
    
    prev_val= -1
    indexes = I[0][0:(n_neurons+1)]
    #change place with first value if in later values

    for ix,val in enumerate(indexes):
            
        thisneuronid = neuronids[val]
        #TODO check this! should it be here 
        #if thisneuronid == neuronids[prev_val]:
        #    continue
        # if the neuron not in first place, change place

        s.append({
            "neuron_id": thisneuronid,
            "similarity": float('%.7f'%(1-D[0][ix]))
        })
        prev_val = val
        
    result["similarityresult"] = s
    result["status"] = 200
    return result


        

def genResultFromVec(searchvector,n_neurons,indextouse, neuronids,simlimit=0.95):
    #TODO add thisneuronid as param in order to clear from data

    k = n_neurons
    k = k+1 # add 1 as the neuron itself weill be included.
 
    searchvector = searchvector.astype(np.float32).reshape((1,len(searchvector)))
    faiss.normalize_L2(searchvector)

    D, I = indextouse.search(searchvector, k*2) # sanity check

    print('Generating dataframe using {} nearest neighbours...'.format(k))

    # generate a list of those neurons which is above the threshold
    
    result = {}
    s = {}
    # for rowno in range(D.shape[0]):
    #     # loop over similar neurons for each neuron in query
    #     #TODO ugly hack not allowing more than one neuron in query in order to fit old solution
    #     s =  I[rowno].tolist()
    

    response_index = 0
    prev_val= -1
    indexes = I[0][0:k*2]
    #change place with first value if in later values

    for ix,val in enumerate(indexes):
        if 1-D[0][ix] >= simlimit:
            
            thisneuronid = neuronids[val]
            #TODO check this! should it be here 
            if thisneuronid == neuronids[prev_val]:
                continue
            # if the neuron not in first place, change place
            response_index = response_index +1

            #TODO ugly hack
            if response_index >= k:
                break

            s[int(response_index)] = {
                "neuron_id": thisneuronid,
                "similarity": '%.7f'%(1-D[0][ix])
            }      
            prev_val = val
            
    result["similar_neuron_ids"] = s
    result["status"] = 200
    return result
