import paramiko,datetime,requests,filecmp,psycopg2
import mysql.connector as mysc
from . import cfg
from datetime import date
from sshtunnel import SSHTunnelForwarder

mypkey = paramiko.RSAKey.from_private_key_file(cfg.keyfile_path)
# if you want to use ssh password use - ssh_password='your ssh password', bellow

sql_ip = '1.1.1.1.1'

def todaysdate():
    return date.today().strftime("%Y-%m-%d")

def create_sftp_client(username, keyfilepath,host, port=22):
    """
    create_sftp_client(username, keyfilepath,host, port) -> SFTPClient
 
    Creates a SFTP client connected to the supplied host on the supplied port authenticating as the user with
    supplied username and supplied password or with the private key in a file with the supplied path.
    If a private key is used for authentication, the type of the keyfile needs to be specified as DSA or RSA.
    :rtype: SFTPClient object.
    """
    ssh = None
    sftp = None
    key = None
    try:
        key = paramiko.RSAKey.from_private_key_file(keyfilepath)
 
        # Connect SSH client accepting all host keys.
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, port, username, key)
 
        # Using the SSH client, create a SFTP client.
        sftp = ssh.open_sftp()
        # Keep a reference to the SSH client in the SFTP client as to prevent the former from
        # being garbage collected and the connection from being closed.
        sftp.sshclient = ssh
 
        return sftp
    except Exception as e:
        print('An error occurred creating SFTP client: %s: %s' % (e.__class__, e))
        if sftp is not None:
            sftp.close()
        if ssh is not None:
            ssh.close()
        pass


def pgconnect(f):
    #decorator for postgres operations
    def pgconnect_(*args, **kwargs):
        conn = psycopg2.connect(host="localhost",database="nmo", user="nmo", password="100neuralDB")
        conn.autocommit = True
        try:
            rv = f(conn, *args, **kwargs)
        except Exception:
            raise
        finally:
            conn.close()
        return rv
    return pgconnect_


def myconnect(f):
    #decorator for MySQL operations
    def myconnect_(*args, **kwargs):
        sshswitch = False

        if sshswitch:

            with SSHTunnelForwarder(
                    (cfg.host, cfg.port),
                    ssh_username=cfg.username,
                    ssh_pkey=mypkey,
                    remote_bind_address=(cfg.dbhost, 3306)) as tunnel:
                conn = mysc.connect(user=cfg.dbuser, password=cfg.dbpass, host='127.0.0.1', port=tunnel.local_bind_port, database=cfg.dbsel, connect_timeout=cfg.dbtimeout)
                conn.autocommit = True
                try:
                    rv = f(conn, *args, **kwargs)
                except Exception:
                    raise
                finally:
                    conn.close()
                return rv
        else:
            conn = mysc.connect(user=cfg.dbuser, password=cfg.dbpass, host=cfg.dbhost, database=cfg.dbsel)
            conn.autocommit = True
            try:
                rv = f(conn, *args, **kwargs)
            except Exception:
                raise
            finally:
                conn.close()
            return rv

    return myconnect_

@myconnect
def dumpduplos(cnx,jdata,trueids,filename):
    import json
    result = []
    cursor = cnx.cursor()
 
    duporgnames = {item["orgname"]: item["dupname"] for item in jdata if int(item["dupid"]) in trueids}
    for item in duporgnames:
        query = ("""SELECT PMID from neuron_article,neuron where 
        neuron_article.neuron_id = neuron.neuron_id AND neuron.neuron_name = '{}'""".format(item))

        cursor.execute(query)
        res = cursor.fetchall()
        pmids = ','.join([str(apm[0]) for apm in res])
        result.append({
            "duplicate": item,
            'PMID(s)': pmids,
            'original': duporgnames[item]
        })  
    jfile = open(filename,'w')
    json.dump(result,jfile)
    jfile.close()

@myconnect
def getneuronids(conn,nneurons,seed=1):
    cursor = conn.cursor(dictionary=True)
    stmt = """
SELECT DISTINCT 
    neuron_auxdata.png_url, 
    neuron.neuron_id,
    neuron.neuron_name,
    brainRegion.name as brainregion,
    cellType.name as celltype,
    species.species
FROM 
     neuron
INNER JOIN
        species
ON
        species.species_id = neuron.species_id

INNER JOIN 
     brainRegion_neuron
ON 
    ( 
        brainRegion_neuron.neuronId = neuron.neuron_id) 
INNER JOIN 
    cellType_neuron 
ON 
    ( 
        neuron.neuron_id = cellType_neuron.neuronId) 
INNER JOIN 
    cellType 
ON 
    ( 
        cellType_neuron.cellTypeId = cellType.id) 
INNER JOIN 
    brainRegion 
ON 
    ( 
        brainRegion_neuron.brainRegionId = brainRegion.id) 
INNER JOIN 
    neuron_auxdata 
ON 
    ( 
        neuron.neuron_id = neuron_auxdata.neuron_id) 
WHERE
    brainRegion_neuron.brainRegionLevel =1
AND cellType_neuron.cellTypeLevel  =2 
AND neuron.neuron_id < 155000
ORDER BY RAND({})
LIMIT {}
    """.format(seed,nneurons)
    cursor.execute(stmt)
    res = cursor.fetchall()
    return res


@myconnect
def getneuronfromname(conn,neuron_name):
    cursor = conn.cursor(dictionary=True)
    stmt = """
SELECT DISTINCT 
    neuron_auxdata.png_url, 
    neuron.neuron_id,
    neuron.neuron_name,
    brainRegion.name as brainregion,
    cellType.name as celltype,
    species.species
FROM 
     neuron
INNER JOIN
        species
ON
        species.species_id = neuron.species_id

INNER JOIN 
     brainRegion_neuron
ON 
    ( 
        brainRegion_neuron.neuronId = neuron.neuron_id) 
INNER JOIN 
    cellType_neuron 
ON 
    ( 
        neuron.neuron_id = cellType_neuron.neuronId) 
INNER JOIN 
    cellType 
ON 
    ( 
        cellType_neuron.cellTypeId = cellType.id) 
INNER JOIN 
    brainRegion 
ON 
    ( 
        brainRegion_neuron.brainRegionId = brainRegion.id) 
INNER JOIN 
    neuron_auxdata 
ON 
    ( 
        neuron.neuron_id = neuron_auxdata.neuron_id) 
WHERE
    brainRegion_neuron.brainRegionLevel =1
AND cellType_neuron.cellTypeLevel  =2 
AND neuron.neuron_name = '{}'
    """.format(neuron_name)
    cursor.execute(stmt)
    res = cursor.fetchall()
    return res[0]

@myconnect
def getneuronsfromids(conn,nids):

    cursor = conn.cursor(dictionary=True)
    stmt = """
SELECT DISTINCT 
    neuron_auxdata.png_url, 
    neuron.neuron_id,
    neuron.neuron_name,
    brainRegion.name as brainregion,
    cellType.name as celltype,
    species.species
FROM 
     neuron
INNER JOIN
        species
ON
        species.species_id = neuron.species_id

INNER JOIN 
     brainRegion_neuron
ON 
    ( 
        brainRegion_neuron.neuronId = neuron.neuron_id) 
INNER JOIN 
    cellType_neuron 
ON 
    ( 
        neuron.neuron_id = cellType_neuron.neuronId) 
INNER JOIN 
    cellType 
ON 
    ( 
        cellType_neuron.cellTypeId = cellType.id) 
INNER JOIN 
    brainRegion 
ON 
    ( 
        brainRegion_neuron.brainRegionId = brainRegion.id) 
INNER JOIN 
    neuron_auxdata 
ON 
    ( 
        neuron.neuron_id = neuron_auxdata.neuron_id) 
WHERE
    brainRegion_neuron.brainRegionLevel =1
AND cellType_neuron.cellTypeLevel  =2 
AND
    neuron.neuron_id in ({})
    """.format(','.join([str(item) for item in nids]))
    cursor.execute(stmt)
    res = cursor.fetchall()
    return res


@myconnect
def getpvecfromdb(cnx,neuronstring,depositiondate=todaysdate()):
    # Fetching detailed morphometrics + pvec
    cursor = cnx.cursor()
    #TODO rewrite query and method
    query = """SELECT
    archive.archive_name,
    neuron.neuron_name,
    neuron.neuron_id,
    p.*
FROM
    persistance_vector p
INNER JOIN
    neuron
ON
    (
        p.neuron_id = neuron.neuron_id)
INNER JOIN
    archive
ON
    (
        neuron.archive_id = archive.archive_id)
{}
ORDER BY
    p.neuron_id ASC ;""".format(neuronstring)


    cursor.execute(query)

    dbset = [(data[0],) + (data[1],) + (data[2],) + data[7:] for data in cursor]

    cnx.close()
    return dbset

@myconnect
def getfromdb(cnx,whereclause="",depositiondate=todaysdate()):
    # Fetching detailed morphometrics + pvec
    cnx = mysc.connect(user=cfg.dbuser, password=cfg.dbpass,
                            host=cfg.dbhost,
                            database=cfg.dbsel)

    cursor = cnx.cursor()
    #TODO rewrite query 
    query = """SELECT
    archive.archive_name,
    neuron.neuron_name,
    neuron.neuron_id,
    deposition.upload_date,
    p.*,
    IFNULL(NULLIF(m.Soma_Surface, ''), 0) AS Soma_Surface,
    m.N_stems                             AS M_N_stems,
    m.N_bifs                              AS M_N_bifs,
    m.N_branch                            AS M_N_branch,
    m.Width                               AS M_Width,
    m.Height                              AS M_Height,
    m.Depth                               AS M_Depth,
    m.Diameter                            AS M_Diameter,
    m.Length                              AS M_Length,
    m.Surface                             AS M_Surface,
    m.Volume                              AS M_Volume,
    m.EucDistance                         AS M_EucDistance,
    m.PathDistance                        AS M_PathDistance,
    m.Branch_Order                        AS M_Branch_Order,
    m.Contraction                         AS M_Contraction,
    m.Fragmentation                       AS M_Fragmentation,
    m.Partition_asymmetry                 AS M_Partition_asymmetry,
    m.Pk_classic                          AS M_Pk_classic,
    m.Bif_ampl_local                      AS M_Bif_ampl_local,
    m.Bif_ampl_remote                     AS M_Bif_ampl_remote,
    m.Fractal_Dim                         AS M_Fractal_Dim
FROM
    persistance_vector p
INNER JOIN
    measurements m
ON
    (
        p.neuron_id = m.neuron_id)
INNER JOIN
    neuron
ON
    (
        p.neuron_id = neuron.neuron_id)
INNER JOIN
    archive
ON
    (
        neuron.archive_id = archive.archive_id)
INNER JOIN
    deposition
ON
    (
        neuron.neuron_id = deposition.neuron_id)
WHERE
    deposition.upload_date < '{}'""".format(depositiondate) + whereclause + """
ORDER BY
    p.neuron_id ASC ;"""


    cursor.execute(query)

    dbset = [(data[0],) + (data[1],) + (data[2],) + (data[3],) + (data[5],) +  data[8:] for data in cursor]

    cnx.close()
    return dbset

@myconnect
def geturlstoswcs(cnx,archive_name):
    def nametourl(neuron_name):
        return 'http://neuromorpho.org/neuron_info.jsp?neuron_name={}'.format(neuron_name) 
    cnx = mysc.connect(user=cfg.dbuser, password=cfg.dbpass,
                            host=cfg.dbhost,
                            database=cfg.dbsel)

    cursor = cnx.cursor()
    query = ("""
SELECT
    neuron.neuron_id,
    neuron.neuron_name
FROM
    archive
INNER JOIN
    neuron
ON
    (
        archive.archive_id = neuron.archive_id)
WHERE
    archive.archive_name = '{}' ; 
    """).format(archive_name)

    cursor.execute(query)

    dbset = {data[1]: nametourl(data[1])  for data in cursor}

    cnx.close()
    return dbset

@myconnect
def getfromdbmeta(cnx,whereclause="",depositiondate=todaysdate()):
    # Fetching detailed morphometrics + pvec

    cursor = cnx.cursor()
    query = """SELECT
    neuron.neuron_id,
    neuron.species_id,
    neuron.strain_id,
    case 
    when age_scale = 'D' then round(max_age,1)
    when age_scale = 'M' then round(max_age * 30.417,1)
    when age_scale = 'Y' then round(max_age * 365,1)
    when max_age is null then 0
    else max_age END
    as max_ageno,
    case 
    when age_scale = 'D' then round(min_age,1)
    when age_scale = 'M' then round(min_age * 30.417,1)
    when age_scale = 'Y' then round(min_age * 365,1)
    when min_age is null then 0
    else min_age END
    as min_ageno,
    case 
    when min_weight is null then 0
    else min_weight END
    as min_weight,
    case 
    when max_weight is null then 0
    else max_weight END
    as max_weight,
    neuron.age_classification_id,
    case 
    when gender = 'F' THEN 1
    when gender = 'M' THEN 2
    when gender = 'H' THEN 3
    when gender = 'M/F' THEN 4
    when gender = 'NR' THEN 5
    ELSE 0 END
    AS genderno,
    neuron.format_id,
    neuron.protocol_id,
    neuron.thickness_id,
    neuron.slice_direction_id,
    neuron.stain_id,
    neuron.magnification_id,
    neuron.objective_id,
    neuron.reconstruction_id,
    neuron.expercond_id,
    domain_id,
    den_integrity_id,
    ax_integrity_id,
    den_ax_integrity_id,
    neu_integrity_id,
    pr_integrity_id,
    SUM(CASE WHEN brainRegionLevel = 1 THEN brainRegionId ELSE 0 END)/5 AS brainRegionLevel1,
    SUM(CASE WHEN brainRegionLevel = 2 THEN brainRegionId ELSE 0 END)/5 AS brainRegionLevel2,
    SUM(CASE WHEN brainRegionLevel = 3 THEN brainRegionId ELSE 0 END)/5 AS brainRegionLevel3,
    SUM(CASE WHEN brainRegionLevel = 4 THEN brainRegionId ELSE 0 END)/5 AS brainRegionLevel4,
    SUM(CASE WHEN brainRegionLevel = 5 THEN brainRegionId ELSE 0 END)/5 AS brainRegionLevel5,
    SUM(CASE WHEN cellTypeLevel = 1 THEN cellTypeId ELSE 0 END)/4 AS cellTypeLevel1,
    SUM(CASE WHEN cellTypeLevel = 2 THEN cellTypeId ELSE 0 END)/4 AS cellTypeLevel2,
    SUM(CASE WHEN cellTypeLevel = 3 THEN cellTypeId ELSE 0 END)/4 AS cellTypeLevel3,
    SUM(CASE WHEN cellTypeLevel = 4 THEN cellTypeId ELSE 0 END)/4 AS cellTypeLevel4,
    SUM(CASE WHEN cellTypeLevel = 5 THEN cellTypeId ELSE 0 END)/4 AS cellTypeLevel5
    
    FROM
        neuron
    INNER JOIN
        cellType_neuron
    ON
        (
            neuron.neuron_id = cellType_neuron.neuronId)
    INNER JOIN
        brainRegion_neuron
    ON
        (
            neuron.neuron_id = brainRegion_neuron.neuronId)
    RIGHT OUTER JOIN neuron_completeness
    ON (neuron.neuron_id = neuron_completeness.neuron_id)
    INNER JOIN
    deposition
ON
    (
        neuron.neuron_id = deposition.neuron_id)
WHERE
    deposition.upload_date < '{}'""".format(depositiondate) + """
    {}
    GROUP By neuron.neuron_id""".format(whereclause)

    cursor.execute(query)
    dbset = [data for data in cursor]
    cnx.close()
    return dbset

@myconnect
def getfromdbmes(cnx,postfix, domainid, whereclause="",depositiondate=todaysdate()):
    # Fetching detailed morphometrics + pvec
    cursor = cnx.cursor()
    tablename = "measurements{}".format(postfix)
    if postfix == '':
        domainquery = ''
        extratables = ''
    else:
        domainquery = 'WHERE nc.neuron_id = mes.neuron_id AND nc.domain_id = sd.domain_id AND sd.domain_id IN {}'.format(domainid)
        extratables = ',  neuron_completeness AS nc, structural_domain AS sd'
    query = """SELECT mes.* from {} AS mes{}
            {}  
            {}
            ORDER BY mes.neuron_id""".format(tablename,extratables,domainquery,whereclause)

    # query = """SELECT mes.* from {} AS mes{}
    #         INNER JOIN
    #         deposition
    #     ON
    #         (mes.neuron_id = deposition.neuron_id)
    #     WHERE
    #         deposition.upload_date < '{}'  
    #         {}  
    #         {}
    #         ORDER BY mes.neuron_id""".format(tablename,extratables,depositiondate,domainquery,whereclause)


    cursor.execute(query)
    dbset = [[0 if elem is None else elem for elem in data] for data in cursor]
    cnx.close()
    return dbset

@myconnect
def getDomainId(cnx,neuron_id):
    # Boolean vector: Dendrites, Axon, Neurites, Processes
    cursor = cnx.cursor()
    query = "SELECT domain_id from neuron_completeness where neuron_id = {}".format(neuron_id)
    cursor.execute(query)
    dbset = [data[0] for data in cursor]
    return dbset[0]

@myconnect
def getidfromname(conn,neuron_name):
    cur = conn.cursor()
    stmt = "SELECT neuron_id FROM neuron WHERE neuron_name = '{}'".format(neuron_name)
    cur.execute(stmt)
    res = cur.fetchone()
    return res[0]

@myconnect


@myconnect
def myinsert(conn,tablename,data):
    # takes data with fields as keys in dictionary, data as values
    cur = conn.cursor()
    data = {item: data[item] for item in data if data[item] is not None}
    fields = ",".join(data.keys())
    values = "','".join([str(item) for item in data.values()])
    statement = """INSERT INTO {}({}) VALUES ('{}') """.format(tablename,fields,values)
    cur.execute(statement)  
    cur.execute("select LAST_INSERT_ID()")
    result = cur.fetchone()
    inserted_id = result[0]
    return inserted_id

@myconnect
def getdataforoneid(conn,id):
    cur = conn.cursor()
    stmt = """SELECT neuron.neuron_name,archive.archive_name,deposition.upload_date,original_format.original_format 
    FROM neuron,archive,deposition,original_format 
    WHERE neuron.neuron_id = {} 
    AND neuron.archive_id = archive.archive_id
    AND neuron.format_id = original_format.original_format_id
    AND neuron.neuron_id = deposition.neuron_id""".format(id)
    cur.execute(stmt)
    res = cur.fetchone()
    return res

def getdataforids(data):
    versiondict = {"2006-08-01": 'alpha', "2006-09-02": 'beta', 
    "2006-11-20": '1.0', "2007-05-04": '1.1', "2007-08-29": '1.2', "2007-10-15": '1.3', "2007-12-05": '2.0', "2008-02-29": '2.1', 
    "2008-07-15": '3.0', "2008-10-01": '3.1', "2009-03-25": '3.2', "2009-09-04": '3.3', "2010-02-16": '4.0', "2010-11-05": '5.0', 
    "2011-03-10": '5.1', "2011-06-01": '5.2', "2011-11-08": '5.3', "2012-05-17": '5.4', "2013-01-15": '5.5', "2013-05-06": '5.6', 
    "2014-05-30": '5.7', "2014-12-11": '6.0', "2015-05-13": '6.1', "2015-10-06": '6.2', "2016-03-04": '6.3', "2016-09-01": '7.0', 
    "2017-03-30": '7.1', "2017-07-19": '7.2', "2017-11-28": '7.3', "2018-04-16": '7.4', "2018-08-02": '7.5', "2018-11-27": '7.6', 
    "2019-04-08": '7.7', "2019-08-19": '7.8', "2019-12-13": '7.9', "2020-06-29": '8.0'}
    orgids = [str(item['orgid']) for item in data]
    dupids = [str(item['dupid']) for item in data]
    levels = [str(item['level']) for item in data]
    orgstr = ','.join(orgids)
    dupstr = ','.join(dupids)
    #cur = conn.cursor()
    # stmt = """SELECT neuron.neuron_name,archive.archive_name,deposition.upload_date,original_format.original_format 
    # FROM neuron,archive,deposition,original_format 
    # WHERE neuron.neuron_id IN ({}) 
    # AND neuron.archive_id = archive.archive_id
    # AND neuron.format_id = original_format.original_format_id
    # AND neuron.neuron_id = deposition.neuron_id""".format(orgstr)
    # cur.execute(stmt)
    #orgres = cur.fetchall()
    newdata = []
    agiledate = datetime.datetime.strptime("2020-06-29", "%Y-%m-%d").date()
    for ix in range(0,len(data)):
        orgres = getdataforoneid(orgids[ix])
        if orgres is None:
            continue
        elif orgres[2] > agiledate:
            orgver = '8.0'
        else:
            orgver = versiondict[orgres[2].strftime('%Y-%m-%d')]
        dupres = getdataforoneid(dupids[ix])
        if dupres is None:
            continue
        if dupres[2] > agiledate:
            dupver = '8.0'
        else:
            dupver = versiondict[dupres[2].strftime('%Y-%m-%d')]
        
        orgext = orgres[3].split('.')
        dupext = dupres[3].split('.')
        srcfilesame = False
        if len(orgext) > 1 and len(dupext) > 1:
            orgsrcurl = 'http://neuromorpho.org/dableFiles/{}/Source-Version/{}.{}'.format(orgres[1].lower(),orgres[0],orgext[1])
            r = requests.get(orgsrcurl)
            open('orgsrcfile.temp','wb').write(r.content)
            dupsrcurl = 'http://neuromorpho.org/dableFiles/{}/Source-Version/{}.{}'.format(dupres[1].lower(),dupres[0],dupext[1])
            r = requests.get(dupsrcurl)
            open('dupsrcfile.temp','wb').write(r.content)
            srcfilesame = filecmp.cmp('orgsrcfile.temp','dupsrcfile.temp',shallow = False)

        if dupres[1] != orgres[1]:
            duparc = dupres[1] + "*"
        else:
            duparc = dupres[1]
        
        if dupver != orgver:
            dupver = dupver + "*"
        
        newdata.append({
            'orgid': orgids[ix],
            'orgname': orgres[0],
            'orgarchive': orgres[1],
            'orgver': orgver,
            'orgimg': 'http://neuromorpho.org/images/imageFiles/{}/{}.png'.format(orgres[1],orgres[0]),
            'orglink': 'http://neuromorpho.org/neuron_info.jsp?neuron_name={}'.format(orgres[0]),
            'dupid': dupids[ix],
            'dupname': dupres[0],
            'duparchive': duparc,
            'dupver': dupver,
            'dupimg': 'http://neuromorpho.org/images/imageFiles/{}/{}.png'.format(dupres[1],dupres[0]),
            'duplink': 'http://neuromorpho.org/neuron_info.jsp?neuron_name={}'.format(dupres[0]),
            'level': levels[ix],
            'srcfilesame': srcfilesame
        })
    return newdata

@pgconnect
def gettrueduplicates(conn):
    cur = conn.cursor()
    stmt = "SELECT neuron_name from duplicateactions where action = 'delete'"
    cur.execute(stmt)
    res = cur.fetchall()
    return [getidfromname(item[0]) for item in res]


@myconnect
def filternames(conn,namelist,idlist):
    cur = conn.cursor()
    namecommalist = '","'.join(namelist)
    idcommalist = ','.join([str(item) for item in idlist])
    stmt = 'SELECT neuron_name from neuron where neuron_name IN ("{}") AND neuron_id IN ({})'.format(namecommalist,idcommalist)
    cur.execute(stmt)
    res = cur.fetchall()
    return [item[0] for item in res]


@myconnect
def getrndneuronids(conn,nids,seed=0,exclude=[],selectfrom=[]):
    cur= conn.cursor()
    if len(exclude) > 0:
        excludeclause = 'AND neuron.neuron_id NOT IN ({})'.format(",".join(exclude))
    else:
        excludeclause = ''
    if len(selectfrom) > 0:
        selectfromstr = [str(item) for item in selectfrom]
        inclause = "neuron.neuron_id IN ({})".format(",".join(selectfromstr))
    else:
        inclause = ''
    stmt = """SELECT 
    neuron.neuron_id 
    FROM 
        neuron 
    WHERE 
    {}
    {} 
    ORDER BY RAND({})
    LIMIT {}
    """.format(excludeclause,inclause,seed,nids)
    cur.execute(stmt)
    result = cur.fetchall()
    return [item[0] for item in result]
