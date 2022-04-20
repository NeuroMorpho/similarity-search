# %%
from faiss.swigfaiss import InvertedLists
from numpy.lib.function_base import corrcoef
import numpy as np
import pandas as pd
#from sklearn.decomposition import PCA
from flask import jsonify
from flask_cors import CORS, cross_origin
from flask import request
import random, pickle,time,faiss, os,sis.com,sis.datamgmt,sis.cfg

from sklearn.preprocessing import scale

from flask import Flask
app = Flask(__name__)
CORS(app)

def vcorrcoef(X,y):
    """
    Correlation coefficient computation
    """
    Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    ym = np.mean(y)
    r_num = np.sum((X-Xm)*(y-ym),axis=1)
    r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
    r = r_num/r_den
    return r


def getrnddict(datadict):
    datadict['datarows'] = np.array([datadict['datarows'][item] for item in rixs]) 
    datadict['neuron_ids'] = [datadict['neuron_ids'][item] for item in rixs]
    return datadict
 
@app.before_first_request
@app.route('/initfaiss/<int:nrnd>', methods=['GET','POST'])
def init_faiss(nrnd=0):
    """
    Calculates the different indexes
    Parameters: 
    nnrnd: Number of random cells to generate a subset index from. 0 - generate fromm all.
    """

    # pvec + measurements
    cachefilename = 'pvecmes_cache.pkl'
    if os.path.isfile(cachefilename):
        datadict = pickle.load(open(cachefilename, "rb"))
    else:
        print('Loading from db...')
        datadict = sis.datamgmt.generateMtrxFrData()
        pickle.dump(datadict,open(cachefilename, "wb"))
    global g_datadict
    g_datadict = datadict
    global g_datamtrx
    global g_neuronids 
    global g_namedict
    global g_index
    print("Pvec and summary")   
    if nrnd != 0:
        global rixs
        rixs = random.sample(range(len(datadict['neuron_ids'])),k=nrnd)
        datadict = getrnddict(datadict)
    (g_index, g_datamtrx, g_neuronids,g_namedict) = sis.datamgmt.genIndexData(datadict,8)
    
    # pvec only 
    cachefilename = 'pvec_cache.pkl'
    if os.path.isfile(cachefilename):
        datadict = pickle.load(open(cachefilename, "rb"))
    else:
        print('Loading from db pvec...')
        datadict = sis.datamgmt.genMtrxPvec()
        pickle.dump(datadict,open(cachefilename, "wb"))
    global g_pvec_datamtrx
    global g_pvec_neuronids 
    global g_pvec_index

    print("Pvec only")
    if nrnd != 0:
        datadict = getrnddict(datadict)
    (g_pvec_index, g_pvec_datamtrx, g_pvec_neuronids,not_used) = sis.datamgmt.genIndexData(datadict,7)

    # summary measurements  
    cachefilename = 'sum_cache.pkl'
    if os.path.isfile(cachefilename):
        datadict = pickle.load(open(cachefilename, "rb"))
    else:
        print('Loading from db summary measurements...')
        datadict = sis.datamgmt.genMtrxMes('',[])
        pickle.dump(datadict,open(cachefilename, "wb"))
    global g_sum_datamtrx
    global g_sum_neuronids 
    global g_sum_index
    print("Summary only")
    if nrnd != 0:
        datadict = getrnddict(datadict)
    (g_sum_index, g_sum_datamtrx, g_sum_neuronids,not_used) = sis.datamgmt.genIndexData(datadict,4)

    # metadata
    cachefilename = 'meta_cache.pkl'
    if os.path.isfile(cachefilename):
        datadict = pickle.load(open(cachefilename, "rb"))
    else:
        print('Loading from db meta...')
        datadict = sis.datamgmt.genMtrxMeta()
        pickle.dump(datadict,open(cachefilename, "wb"))
    global g_meta_datamtrx
    global g_meta_neuronids 
    global g_meta_index
    print("Metadata only")
    if nrnd != 0:
        datadict = getrnddict(datadict)
    (g_meta_index, g_meta_datamtrx, g_meta_neuronids,not_used) = sis.datamgmt.genIndexData(datadict,9)


    #structural domain ids
    domain_ids = {
        'DAX': '(5,7)',
        "DEN": '(4,6)',
        "AX": "(1,3)",
        "NEU": "(8,9)",
        "PR": "(10,11)"
    }

    # detailed datastructures generation 
    global g_postfixes
    g_postfixes = ['DAX','DEN','AX','NEU','PR']
    cachefilename = 'detailed_cache.pkl'
    if os.path.isfile(cachefilename):
        datadict = pickle.load(open(cachefilename, "rb"))
    else:
        print('Loading from db detailed...')
        # concat Apical and Basal for dendrites only
        
        ap_den_dict = sis.datamgmt.genMtrxMes('AP',domain_ids['DEN'])
        bas_den_dict = sis.datamgmt.genMtrxMes('BS',domain_ids['DEN'])
        datadict['DEN']= {'datarows':  np.hstack((ap_den_dict['datarows'], bas_den_dict['datarows'])), 
            'neuron_ids': ap_den_dict['neuron_ids']} #enough with one since they are the same
        #concat apical and basal for dendrites and axons    
        ap_dax_dict = sis.datamgmt.genMtrxMes('AP',domain_ids['DAX'])
        bas_dax_dict = sis.datamgmt.genMtrxMes('BS',domain_ids['DAX'])
        ax_dax_dict = sis.datamgmt.genMtrxMes('AX',domain_ids['DAX'])
        datadict['DAX']= {'datarows':  np.hstack((ap_dax_dict['datarows'], bas_dax_dict['datarows'], ax_dax_dict['datarows'])), 
            'neuron_ids': ap_dax_dict['neuron_ids']} #enough with one since they are the same

        for item in g_postfixes[2:]: # Not first two
            datadict[item] = sis.datamgmt.genMtrxMes(item,domain_ids[item])
        
        # dump to datafile
        pickle.dump(datadict,open(cachefilename, "wb"))

    
    global g_detailed_datamtrx
    g_detailed_datamtrx = {}
    global g_detailed_neuronids
    g_detailed_neuronids = {} 
    global g_detailed_index
    g_detailed_index = {}
    
    for item in g_postfixes:
        print("Detailed only: {}".format(item))
        #if nrnd != 0:
        #    datadict[item] = getrnddict(datadict[item])

        (g_detailed_index[item], g_detailed_datamtrx[item], g_detailed_neuronids[item],not_used) = sis.datamgmt.genIndexData(datadict[item])
    
    # detailed + pvec datastructures generation 
    cachefilename = 'detailedpvec_cache.pkl'
    if os.path.isfile(cachefilename):
        datadict = pickle.load(open(cachefilename, "rb"))
    else:
        print('Loading from db detailed + pvec...')
        # concat pvec and detailed values

        for item in g_postfixes: 
            print("Detailed and pvec: {}".format(item))
            neuronstring = ",".join([str(item) for item in g_detailed_neuronids[item]])
            datadict[item]['datarows'] = np.hstack((datadict[item]['datarows'],sis.datamgmt.genMtrxPvec(neuronstring)['datarows']))
        
        # dump to datafile
        pickle.dump(datadict,open(cachefilename, "wb"))

    
    global g_detailedpvec_datamtrx
    g_detailedpvec_datamtrx = {}
    global g_detailedpvec_neuronids
    g_detailedpvec_neuronids = {} 
    global g_detailedpvec_index
    g_detailedpvec_index = {}
    for item in g_postfixes:
        print("Detailed + pvec: {}".format(item))
#        if nrnd != 0:
#            datadict[item] = getrnddict(datadict[item])

        (g_detailedpvec_index[item], g_detailedpvec_datamtrx[item], g_detailedpvec_neuronids[item],not_used) = sis.datamgmt.genIndexData(datadict[item])
    return jsonify({'neuronids': g_neuronids})

@app.route('/clearcache')
def clearcache():
    """
    Clear cache to enable indexing of new additions to database
    """
    files = os.listdir('/app')
    for item in files:
        if '.pkl' in item:
            os.remove('/app/{}'.format(item))
    return {"result": 'success'}

@app.route('/similarNeurons/<int:do_pca>/<int:neuron_id>/<int:n_neurons>/<int:cortype>', methods=['GET'])
def get(do_pca,neuron_id,n_neurons,cortype):
    """
    Searching using pvec and summary.
    Parameters: 
    do_pca: 0 for no pca, 1 for pca
    neuron_id: neuron id in NeuroMorpho db to search from

    """
    neuronIndex = g_neuronids.index(neuron_id)
    datadict = g_datamtrx[do_pca][neuronIndex]
    #TODO move these two rows to the if clause
    if cortype >0:
        corcoeffs = vcorrcoef(g_datamtrx[do_pca],datadict)
        c = np.argsort(-corcoeffs)
    if cortype !=1:
        result = sis.datamgmt.genResult(datadict,n_neurons,g_index[do_pca],g_neuronids,neuron_id)
        nsim = {item['neuron_id']: item["similarity"] for item in result['similarityresult']}

    if cortype==1:
        #similarity with correlation
        simsc = [{
            'neuron_id': g_neuronids[item],
            'similarity': float('%.7f'%corcoeffs[item])
            } for item in c[:n_neurons] if g_neuronids[item] != neuron_id]

        return jsonify({'similarityresult': simsc,'status': 200}) 
    elif cortype == 2:    
        print('c - {}'.format(c[100]))    
        simsc = [{
            'neuron_id': item,
            'similarity': float('%.7f'%(corcoeffs[g_neuronids.index(item)]*nsim[item])) 
            } for item in nsim][:n_neurons] 
        simsc = sorted(simsc, key=lambda k: -k['similarity']) 

        return jsonify({'similarityresult': simsc,'status': 200}) 

    else:
        #similarity with faiss
        return jsonify(result)

@app.route('/getallDuplicates/<int:do_pca>/<float:simlim>', methods=['GET'])
def getallDuplicates(do_pca,simlim):

    result = sis.datamgmt.genResultAll(g_datamtrx[do_pca],g_index[do_pca],g_neuronids,simlim,0.9999)


    return jsonify(result)

@app.route('/runexperimentpar/<int:nnbs>', methods=['GET'])
def runexperimentpar(nnbs):

    def searchallmethods(neuron_ids,dopca,ntosearch):
        #neuronIndexes = [g_neuronids.index(item) for item in neuron_ids]
        datadicts_smespvecs = g_datamtrx[dopca]
        smespvecs = sis.datamgmt.genResultPar(datadicts_smespvecs,ntosearch,g_index[dopca],g_neuronids)
        
        #neuronIndexes = [g_pvec_neuronids.index(item) for item in neuron_ids]
        datadicts_pvecs = g_pvec_datamtrx[dopca]
        pvecs = sis.datamgmt.genResultPar(datadicts_pvecs,ntosearch,g_pvec_index[dopca],g_pvec_neuronids)
        
        #neuronIndexes = [g_sum_neuronids.index(item) for item in neuron_ids]
        datadicts_smes = g_sum_datamtrx[dopca]
        smes = sis.datamgmt.genResultPar(datadicts_smes,ntosearch,g_sum_index[dopca],g_sum_neuronids)
       
        return (smespvecs,pvecs,smes)
        

    global g_trialids
    #g_trialids = sis.com.getrndneuronids(n_trials,seed,selectfrom=g_neuronids)
    g_trialids = g_neuronids
    
    #Search for #nnbs closest to get an approximation of which neurons that are suitable to use as benchmark
    # for earch neuron in the trial
    trials2results = []

    #nuniques = 0
    ntosearch = nnbs
    (smespvecspca,pvecspca,smespca) = searchallmethods(g_trialids,1,ntosearch)
    (smespvecs,pvecs,smes) = searchallmethods(g_trialids,0,ntosearch)

    # smesixs = {item['neuron_id']: smes.index(item) for item in smes}
    for ix in range(len(g_trialids)):
        adict = {}
        adict["neuron_id"] = g_trialids[ix]
        """adict["commonids"] = commonids[neuron_id]
        adict['smespvecspca'] = sortedsims(smespvecspca[smespvecspcaixs[neuron_id]],commonids[neuron_id])
        adict['pvecspca'] = sortedsims(pvecspca[pvecspcaixs[neuron_id]],commonids[neuron_id])
        adict['smespca'] = sortedsims(smespca[smespcaixs[neuron_id]],commonids[neuron_id])
        adict['smespvecs'] = sortedsims(smespvecs[smespvecsixs[neuron_id]],commonids[neuron_id])
        adict['pvecs'] = sortedsims(pvecs[pvecsixs[neuron_id]],commonids[neuron_id])
        adict['smes'] = sortedsims(smes[smesixs[neuron_id]],commonids[neuron_id])"""

        #adict['smespvecspca'] = smespvecspca[smespvecspcaixs[neuron_id]]
        adict['smespvecspca'] = smespvecspca[ix]
        adict['pvecspca'] = pvecspca[ix]
        adict['smespca'] = smespca[ix]
        adict['smespvecs'] = smespvecs[ix]
        adict['pvecs'] = pvecs[ix]
        adict['smes'] = smes[ix]

        # Take difference for each pair
        
        trials2results.append(adict)
    return jsonify(trials2results)


@app.route('/similarNeuronsPvec/<int:do_pca>/<int:neuron_id>/<int:n_neurons>/<int:cortype>', methods=['GET'])
def getPvec(do_pca,neuron_id,n_neurons,cortype):
    t = time.time()
    print('Searching using faiss and pvec...')
    neuronIndex = g_pvec_neuronids.index(neuron_id)
    datadict = g_pvec_datamtrx[do_pca][neuronIndex]
    if cortype:
        corcoeffs = vcorrcoef(g_pvec_datamtrx[do_pca],datadict)
        c = np.argsort(-corcoeffs)
        simsc = [{
            'neuron_id': g_pvec_neuronids[item],
            'similarity': float('%.7f'%corcoeffs[item])
            } for item in c if g_pvec_neuronids[item] != neuron_id][:n_neurons]

        return jsonify({'similarityresult': simsc,'status': 200}) 
    else:
 
        result = sis.datamgmt.genResult(datadict,n_neurons,g_pvec_index[do_pca],g_pvec_neuronids,neuron_id)
        result["response_time"] = time.time() - t
        return jsonify(result)

@app.route('/getDuplicatesExists/<int:do_pca>/<int:neuron_id>/<int:n_neurons>', methods=['GET'])
def getDuplicatesExists(do_pca,neuron_id,n_neurons): 
    neuronIndex = g_neuronids.index(neuron_id)
    datadict = g_datamtrx[do_pca][neuronIndex]
    result = sis.datamgmt.genResult(datadict,n_neurons,g_index[do_pca],g_neuronids,neuron_id, 0.999)
    
    return jsonify(result)
    
@app.route('/getDuplicates/', methods=['GET','POST'])
def getDuplicates(): 
    pvec = request.json["pvec"]
    measurements = request.json["measurements"]
    datadict = np.array(pvec + measurements)
    result = sis.datamgmt.genResultFromVec(datadict,10,g_index[0],g_neuronids, 0.95)
    return jsonify(result)

@app.route('/getDuplicatesfordata/', methods=['POST'])
def getDuplicatesfordata():
    #TODO Should also fetch data from db and add new data. 
    datadict = request.json

    result = sis.datamgmt.checkduplicatesinternal(g_datadict["datarows"],datadict,g_datadict["neuron_names"])
    return jsonify(result)

@app.route('/getscaled/<int:neuron_id>/<float:scalefactor>/<int:n_neurons>', methods=['GET'])
def getscaled(neuron_id,scalefactor,n_neurons):
    #TODO Should also fetch data from db and add new data. 
    neuronIndex = g_neuronids.index(neuron_id)
    datadict = sis.com.scalecell(g_datadict["datarows"][neuronIndex],100,scalefactor)

    result = sis.datamgmt.checkscaled(g_datadict["datarows"],datadict,g_neuronids,n_neurons)
    return jsonify(result)


@app.route('/getDupliMulti/<int:do_pca>/<int:neuron_id>/<int:n_neurons>', methods=['GET'])
def getDupliMulti(do_pca,neuron_id,n_neurons,cortype): 
    neuronIndex = g_neuronids.index(neuron_id)
    datadict = g_datamtrx[do_pca][neuronIndex]
    datamtrx = datadict['datarows']
    faiss.normalize_L2(datamtrx)
    result = sis.datamgmt.genResult(datadict,n_neurons,g_index[do_pca],g_neuronids,neuron_id, 0.999)

    duplodf = pd.DataFrame(columns=['neuron ID', 'neuron name', 'neuron link', 'neuron archive', 'duplicate ID',  'duplicate name', 'duplicate link', 'duplicate archive', 'PMID', 'Upload date'], index=['neuron ID'])
    for rowno in range(D.shape[0]):
        # loop over similar neurons for each neuron in query
        for ix in range(len(I[rowno])):
            item = I[rowno][ix]
            duploid = g_neuronids[item] 
            if duploid == q_neuronids[rowno]: 
                continue
            duploname = g_neuronnames[item] 
            duploarchive = g_neuronarchives[item] 
            duplolink = 'http://cng.gmu.edu:8080/neuroMorphoReview/neuron_info.jsp?neuron_name={}'.format(g_neuronnames[item])
            neuron_sim = 1 - D[rowno][ix]
            dfdata = {'neuron ID': q_neuronids[rowno], 'neuron name': q_neuronnames[rowno], 'neuron link': ' http://cng.gmu.edu:8080/neuroMorphoReview/neuron_info.jsp?neuron_name={}'.format(q_neuronnames[rowno]), 'neuron archive': q_neuronarchives[rowno], 'similarity': neuron_sim,  'duplicate ID': duploid, 'duplicate name': duploname, 'duplicate link': str(duplolink), 'duplicate archive': duploarchive, 'PMID': g_pmids[item], 'Upload date': g_uploaddates[item]}
            duplodf = duplodf.append(dfdata, ignore_index=True)

    return duplodf.to_html(render_links=True)


@app.route('/similarNeuronsMeta/<int:do_pca>/<int:neuron_id>/<int:n_neurons>/<int:cortype>', methods=['GET'])
def getMeta(do_pca,neuron_id,n_neurons,cortype):
    print('Searching using faiss...')
    neuronIndex = g_meta_neuronids.index(neuron_id)
    datadict = g_meta_datamtrx[do_pca][neuronIndex]
    if cortype:
        corcoeffs = vcorrcoef(g_meta_datamtrx[do_pca],datadict)
        c = np.argsort(-corcoeffs)
        simsc = [{
            'neuron_id': g_meta_neuronids[item],
            'similarity': float('%.7f'%corcoeffs[item])
            } for item in c if g_meta_neuronids[item] != neuron_id][:n_neurons]

        return jsonify({'similarityresult': simsc,'status': 200}) 
    else:
        result = sis.datamgmt.genResult(datadict,n_neurons,g_meta_index[do_pca],g_meta_neuronids,neuron_id)
        return jsonify(result)

@app.route('/similarNeuronsSum/<int:do_pca>/<int:neuron_id>/<int:n_neurons>/<int:cortype>', methods=['GET'])
def getSummary(do_pca,neuron_id,n_neurons,cortype):
    print('Searching using faiss...')
    neuronIndex = g_sum_neuronids.index(neuron_id)
    datadict = g_sum_datamtrx[do_pca][neuronIndex]
    if cortype:
        corcoeffs = vcorrcoef(g_sum_datamtrx[do_pca],datadict)
        c = np.argsort(-corcoeffs)
        simsc = [{
            'neuron_id': g_sum_neuronids[item],
            'similarity': float('%.7f'%corcoeffs[item])
            } for item in c if g_sum_neuronids[item] != neuron_id][:n_neurons]

        return jsonify({'similarityresult': simsc,'status': 200}) 
    else:
        result = sis.datamgmt.genResult(datadict,n_neurons,g_sum_index[do_pca],g_sum_neuronids,neuron_id)
        return jsonify(result)

@app.route('/similarNeuronsDetailed/<int:do_pca>/<int:neuron_id>/<int:n_neurons>/<int:cortype>', methods=['GET'])
def getDetailed(do_pca,neuron_id,n_neurons,cortype):
    domain = sis.com.getDomainId(neuron_id)
    if domain in (5,7):
        postfix = "DAX"
    elif domain in (4,6):
        postfix = "DEN"
    elif domain in (1,3):
        postfix = 'AX'
    elif domain in (8,9):
        postfix = "NEU"
    elif domain in (10,11):
        postfix = "PR"
    else:
        raise ValueError('Unknown structural domain of neuron')

    # dendrites apical, basal axonal
    
    print('Searching using faiss...')
    neuronIndex = g_detailed_neuronids[postfix].index(neuron_id)
    datadict = g_detailed_datamtrx[postfix][do_pca][neuronIndex]
    if cortype:
        corcoeffs = vcorrcoef(g_detailed_datamtrx[postfix][do_pca],datadict)
        c = np.argsort(-corcoeffs)
        simsc = [{
            'neuron_id': g_detailed_neuronids[postfix][item],
            'similarity': float('%.7f'%corcoeffs[item])
            } for item in c if g_detailed_neuronids[postfix][item] != neuron_id][:n_neurons]

        return jsonify({'similarityresult': simsc,'status': 200}) 
    else:
        result = sis.datamgmt.genResult(datadict,n_neurons,g_detailed_index[postfix][do_pca],g_detailed_neuronids[postfix],neuron_id)
        return jsonify(result)

@app.route('/similarNeuronsDetailedPvec/<int:do_pca>/<int:neuron_id>/<int:n_neurons>/<int:cortype>', methods=['GET'])
def getDetailedPvec(do_pca,neuron_id,n_neurons,cortype):
    domain = sis.com.getDomainId(neuron_id)
    if domain in (5,7):
        postfix = "DAX"
    elif domain in (4,6):
        postfix = "DEN"
    elif domain in (1,3):
        postfix = 'AX'
    elif domain in (8,9):
        postfix = "NEU"
    elif domain in (10,11):
        postfix = "PR"
    else:
        raise ValueError('Unknown structural domain of neuron')

    # dendrites apical, basal axonal
    
    print('Searching using faiss...')
    neuronIndex = g_detailedpvec_neuronids[postfix].index(neuron_id)
    datadict = g_detailedpvec_datamtrx[postfix][do_pca][neuronIndex]
    if cortype:
        corcoeffs = vcorrcoef(g_detailedpvec_datamtrx[postfix][do_pca],datadict)
        c = np.argsort(-corcoeffs)
        simsc = [{
            'neuron_id': g_detailedpvec_neuronids[postfix][item],
            'similarity': float('%.7f'%corcoeffs[item])
            } for item in c if g_detailedpvec_neuronids[postfix][item] != neuron_id][:n_neurons]

        return jsonify({'similarityresult': simsc,'status': 200}) 
    else:
        result = sis.datamgmt.genResult(datadict,n_neurons,g_detailedpvec_index[postfix][do_pca],g_detailedpvec_neuronids[postfix],neuron_id)
        return jsonify(result)    


@app.route('/similarNeuronsAll/<int:do_pca>/<int:neuron_id>/<int:n_neurons>', methods=['GET'])
def getMulti(do_pca,neuron_id,n_neurons):
    t = time.time()
    print('Searching using faiss...')
    neuronIndex = g_sum_neuronids.index(neuron_id)
    datadict = g_sum_datamtrx[do_pca][neuronIndex]
    result = sis.datamgmt.genResult(datadict,n_neurons,g_sum_index[do_pca],g_sum_neuronids,neuron_id)
    result["response_time"] = time.time() - t
    return jsonify(result)   



if __name__ == '__main__':
    app.run(host='0.0.0.0')