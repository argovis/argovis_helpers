import requests, datetime, copy, time

def argofetch(route, options={}, apikey='', apiroot='https://argovis-api.colorado.edu/'):
    # GET <apiroot>/<route>?<options> with <apikey> in the header.
    # raises on anything other than success or a 404.

    dl = requests.get(apiroot + route, params = options, headers={'x-argokey': apikey}).json()

    if 'code' in dl and dl['code'] != 404:
        raise Exception(dl['code'] + ': ' + dl['message'])

    if ('code' in dl[0] and dl[0]['code']==404) or ('code' in dl and dl['code']==404):
        return []

    return dl

def data_inflate(data_doc, metadata_doc=None):
    # given a single JSON <data_doc> downloaded from one of the standard data routes with compression=array,
    # return the data document with the data key reinflated to per-level dictionaries.

    data = data_doc['data']
    data_keys = find_key('data_keys', data_doc, metadata_doc)

    return [{data_keys[i]: v for i,v in enumerate(level)} for level in data]

def find_key(key, data_doc, metadata_doc):
    # some metadata keys, like data_keys and units, may appear on either data or metadata documents,
    # and if they appear on both, data_doc takes precedence.
    # given the pair, find the correct key assignment.

    if key in data_doc:
        return data_doc[key]
    else:
        if metadata_doc is None:
            raise Exception(f"Please provide metadata document _id {data_doc['metadata']}")
        if '_id' in metadata_doc and 'metadata' in data_doc and metadata_doc['_id'] != data_doc['metadata']:
            raise Exception(f"Data document doesn't match metadata document. Data document needs metadata document _id {data_doc['metadata']}, but got {metadata_doc['_id']}")

        return metadata_doc[key]

def parsetime(time):
    # time can be either an argopy-compliant datestring, or a datetime object; 
    # returns the opposite.

    if type(time) is str:
        if '.' not in time:
            time = time.replace('Z', '.000Z')
        return datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S.%fZ")
    elif type(time) is datetime.datetime:
        return time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    else:
        raise ValueError(time)

def query(route, options={}, apikey='', apiroot='https://argovis-api.colorado.edu/'):
    # middleware function between the user and a call to argofetch to make sure individual requests are reasonably scoped and timed.

    r = route.replace('/', '') # use this for lookups internally, allow user to put the leading / on the end of the api root or the start of the route, either or.
    data_routes = ['argo', 'cchdo', 'drifters', 'tc', 'grids/ohc_kg', 'grids/temperature_rg', 'grids/salinity_rg']
    scoped_parameters = {
        'argo': ['platform'],
        'cchdo': ['woceline', 'cchdo_cruise'],
        'drifters': ['wmo', 'platform'],
        'tc': ['name']
    }
    earliest_records = {
        'argo': parsetime("1997-07-27T20:26:20.002Z"),
        'cchdo': parsetime("1977-10-06T00:00:00.000Z"),
        'drifters': parsetime("1987-10-01T13:00:00.000Z"),
        'grids/ohc_kg': parsetime("2005-01-14T00:00:00.000Z"),
        'grids/temperature_rg': parsetime("2004-01-14T00:00:00.000Z"),
        'grids/salinity_rg': parsetime("2004-01-14T00:00:00.000Z"),
        'tc': parsetime("1851-06-24T00:00:00.000Z")
    }

    if r in data_routes:
        # these are potentially large requests that might need to be sliced up
        
        ## if a data query carries a scoped parameter, no need to slice up:
        if r in scoped_parameters and not set(scoped_parameters[r]).isdisjoint(options.keys()):
            return argofetch(route, options=options, apikey=apikey, apiroot=apiroot)

        ## slice up in time bins:
        start = None
        end = None
        if 'startDate' in options:
            start = parsetime(options['startDate'])
        else:
            start = earliest_records[r]
        if 'endDate' in options:
            end = parsetime(options['endDate'])
        else:
            end = datetime.datetime.now()

        delta = datetime.timedelta(days=30)
        times = [start]
        while times[-1] + delta < end:
            times.append(times[-1]+delta)
        times.append(end)
        times = [parsetime(x) for x in times]

        results = []
        ops = copy.deepcopy(options)
        for i in range(len(times)-1):
            ops['startDate'] = times[i]
            ops['endDate'] = times[i+1]
            results += argofetch(route, options=ops, apikey=apikey, apiroot=apiroot)
            if 'data' in 'options':
                time.sleep(1)
        return results

    else:
        return argofetch(route, options=options, apikey=apikey, apiroot=apiroot)

def units_inflate(data_doc, metadata_doc=None):
    # similar to data_inflate, but for units

    units = find_key('units', data_doc, metadata_doc)
    data_keys = find_key('data_keys', data_doc, metadata_doc)

    return {data_keys[i]:v for i,v in enumerate(units)}

