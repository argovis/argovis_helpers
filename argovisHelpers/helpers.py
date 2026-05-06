from __future__ import annotations
import requests, datetime, copy, time, re, area, math, urllib, json, xarray, numpy, scipy.interpolate, gsw
from shapely.geometry import shape, box, Polygon
from shapely.ops import orient
import pkg_resources
from pkg_resources import DistributionNotFound
from dataclasses import dataclass, field
from typing import Any
from dateutil import parser
from collections.abc import Sequence

_avhcache = {}
_CACHE_EXPIRY = 3600

def fetch_json(endpoint):
    """
    Fetches a JSON document from the given endpoint.
    Returns a cached version if the last fetch was within the last hour.
    """
    global _avhcache

    current_time = time.time()

    # Check if we have a cached version and if it's still valid
    if endpoint in _avhcache:
        cached_time, cached_data = _avhcache[endpoint]
        if current_time - cached_time < _CACHE_EXPIRY:
            return cached_data  # Return cached data

    # Fetch the data from the API
    try:
        response = requests.get(endpoint)
        response.raise_for_status()  # Raise an exception for HTTP errors
        json_data = response.json()

        # Cache the result with the current timestamp
        _avhcache[endpoint] = (current_time, json_data)
        return json_data
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch data from {endpoint}: {e}")

def get_timebound(dataset, bound):
    core_rl = fetch_json("https://argovis-api.colorado.edu/summary?id=ratelimiter")
    drifter_rl = fetch_json("https://argovis-drifters.colorado.edu/summary?id=ratelimiter")

    rl = core_rl[0]['metadata'] | drifter_rl[0]['metadata']

    keymap = {
        'argo': 'argo',
        'cchdo': 'cchdo',
        'drifters': 'drifters',
        'tc': 'tc',
        'argotrajectories': 'argotrajectories',
        'easyocean': 'easyocean',
        'grids/rg09': 'rg09',
        'grids/kg21': 'kg21',
        'grids/glodap': 'glodap',
        'timeseries/noaasst': 'noaasst',
        'timeseries/copernicussla': 'copernicussla',
        'timeseries/ccmpwind': 'ccmpwind',
        'extended/ar': 'ar'
    }

    return parsetime(rl[keymap[dataset]][bound])


def slice_timesteps(options, r):
    # given a qsr option dict and data route, return a list of reasonable time divisions

    maxbulk = 2000000 # should be <= maxbulk used in generating an API 413
    timestep = 30 # days
    extent = 360000000 / 13000 #// 360M sq km, all the oceans
    
    if 'polygon' in options:
        extent = area.area({'type':'Polygon','coordinates':[ options['polygon'] ]}) / 13000 / 1000000 # poly area in units of 13000 sq. km. blocks
    elif 'box' in options:
        extent = area.area({'type':'Polygon','coordinates':[[ options['box'][0], [options['box'][1][0], options['box'][0][0]], options['box'][1], [options['box'][0][0], options['box'][1][0]], options['box'][0]]]}) / 13000 / 1000000
        
    timestep = min(365*100,math.floor(maxbulk / extent))

    ## slice up in time bins:
    start = None
    end = None
    if 'startDate' in options:
        start = parsetime(options['startDate'])
    else:
        start = get_timebound(r, 'startDate')
    if 'endDate' in options:
        end = parsetime(options['endDate'])
    else:
        end = get_timebound(r, 'endDate')
        
    delta = datetime.datetime.timedelta(days=timestep)
    times = [start]
    while times[-1] + delta < end:
        times.append(times[-1]+delta)
    times.append(end)
    times = [parsetime(x) for x in times]
    
    return times
    
def data_inflate(data_doc, metadata_doc=None):
    # given a single JSON <data_doc> downloaded from one of the standard data routes,
    # return the data document with the data key reinflated to per-level dictionaries.

    data = data_doc['data']
    data_info = find_key('data_info', data_doc, metadata_doc)

    d = zip(*data) # per-variable becomes per-level 
    return [{data_info[0][i]: v for i,v in enumerate(level)} for level in d]

def find_key(key, data_doc, metadata_doc):
    # some metadata keys, like data_info, may appear on either data or metadata documents,
    # and if they appear on both, data_doc takes precedence.
    # given the pair, find the correct key assignment.

    if key in data_doc:
        return data_doc[key]
    else:
        if metadata_doc is None:
            raise Exception(f"Please provide metadata document _id {data_doc['metadata']}")
        if '_id' in metadata_doc and 'metadata' in data_doc and metadata_doc['_id'] not in data_doc['metadata']:
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
        t = time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        tokens = t.split('-')
        if len(tokens[0]) < 4:
            tokens[0] = ('000' + tokens[0])[-4:]
            t = '-'.join(tokens)
        return t
    else:
        raise ValueError(time)

def units_inflate(data_doc, metadata_doc=None):
    # similar to data_inflate, but for units

    data_info = find_key('data_info', data_doc, metadata_doc)
    uindex = data_info[1].index('units')

    return {data_info[0][i]: data_info[2][i][uindex] for i in range(len(data_info[0]))}


def combine_data_lists(lists):
    # given a list of data lists, concat them appropriately;
    # ie [[1,2],[3,4]] + [[5,6],[7,8]] = [[1,2,5,6], [3,4,7,8]]

    combined_list = []
    for sublists in zip(*lists):
        combined_sublist = []
        for sublist in sublists:
            combined_sublist.extend(sublist)
        combined_list.append(combined_sublist)
    return combined_list

def split_polygon(coords, max_lon_size=5, max_lat_size=5):
    # slice a geojson polygon up into a list of smaller polygons of maximum extent in lon and lat

    # if a polygon bridges the dateline and wraps its longitudes around, 
    # we need to detect this and un-wrap.
    coords = dont_wrap_dateline(coords)
        
    polygon = shape({"type": "Polygon", "coordinates": [coords]})
    smaller_polygons = []
    min_lon, min_lat, max_lon, max_lat = polygon.bounds

    lon = min_lon
    lat = min_lat
    while lon < max_lon:
        while lat < max_lat:
            # Create a bounding box for the current chunk
            bounding_box = box(lon, lat, lon + max_lon_size, lat + max_lat_size)

            # Intersect the bounding box with the original polygon
            chunk = polygon.intersection(bounding_box)

            # If the intersection is not empty, add it to the list of smaller polygons
            if not chunk.is_empty:
                # Convert the Shapely geometry to a GeoJSON polygon and add it to the list
                shapes = json.loads(gpd.GeoSeries([chunk]).to_json())
                if shapes['features'][0]['geometry']['type'] == 'Polygon':
                    smaller_polygons.append(shapes['features'][0]['geometry']['coordinates'][0])
                elif shapes['features'][0]['geometry']['type'] == 'MultiPolygon':
                    for poly in shapes['features'][0]['geometry']['coordinates']:
                        smaller_polygons.append(poly[0])

            lat += max_lat_size
        lat = min_lat
        lon += max_lon_size

    return smaller_polygons

def split_box(box, max_lon_size=5, max_lat_size=5):
    # slice a box up into a list of smaller boxes of maximum extent in lon and lat
    
    if box[0][0] > box[1][0]:
        # unwrap the dateline
        box[1][0] += 360
    
    smaller_boxes = []
    lon = box[0][0]
    lat = box[0][1]
    while lon < box[1][0]:
        while lat < box[1][1]:
            smaller_boxes.append([[lon, lat],[min(box[1][0], lon + max_lon_size), min(box[1][1], lat + max_lat_size)]])
            lat += max_lat_size
        lat = box[0][1]
        lon += max_lon_size
        
    return smaller_boxes

def dont_wrap_dateline(coords):
    # given a list of polygon coords, return them ensuring they dont modulo 360 over the dateline.
    
    for i in range(len(coords)-1):
        if coords[i][0]*coords[i+1][0] < 0 and abs(coords[i][0] - coords[i+1][0]) > 180:
            # ie if any geodesic edge crosses the dateline with a modulo, we must need to remap.
            return [[lon + 360 if lon < 0 else lon, lat] for lon, lat in coords]
    
    return coords

def generate_global_cells(lonstep=5, latstep=5):
    cells = []
    lon = -180
    lat = -90
    while lon < 180:
        while lat < 90:
            cells.append([[lon,lat],[lon+lonstep,lat],[lon+lonstep,lat+latstep],[lon,lat+latstep],[lon,lat]])

            lat += latstep
        lat = -90
        lon += lonstep
    return cells

def argofetch(route, options={}, apikey='', apiroot='https://argovis-api.colorado.edu/', suggestedLatency=0, verbose=False):
    # GET <apiroot>/<route>?<options> with <apikey> in the header.
    # raises on anything other than success or a 404.

    o = copy.deepcopy(options)
    for option in ['polygon', 'box']:
        if option in options:
            options[option] = str(options[option])

    try:
        version = pkg_resources.get_distribution('argovisHelpers').version
    except DistributionNotFound:
        version = '-1'
    dl = requests.get(apiroot.rstrip('/') + '/' + route.lstrip('/'), params = options, headers={'x-argokey': apikey, 'x-avh-telemetry': version})
    statuscode = dl.status_code
    if verbose:
        print(urllib.parse.unquote(dl.url))
    dl = dl.json()

    if statuscode==429:
        # user exceeded API limit, extract suggested wait and delay times, and try again
        wait = dl['delay'][0]
        latency = dl['delay'][1]
        time.sleep(wait*1.1)
        return argofetch(route, options=o, apikey=apikey, apiroot=apiroot, suggestedLatency=latency, verbose=verbose)

    if (statuscode!=404 and statuscode!=200) or (statuscode==200 and type(dl) is dict and 'code' in dl):
        if statuscode == 413:
            print('The temporospatial extent of your request is enormous! If you are using the query helper, it will now try to slice this request up for you. Try setting verbose=true to see how it is slicing this up.')
        elif statuscode >= 500 or (statuscode==200 and type(dl) is dict and 'code' in dl):
            print("Argovis' servers experienced an error. Please try your request again, and email argovis@colorado.edu if this keeps happening; please include the full details of the the request you made so we can help address.")
        raise Exception(statuscode, dl)

    # no special action for 404 - a 404 due to a mangled route will return an error, while a valid search with no result will return [].

    return dl, suggestedLatency

def query(route, options={}, apikey='', apiroot='https://argovis-api.colorado.edu/', verbose=False, slice=False):
    # middleware function between the user and a call to argofetch to make sure individual requests are reasonably scoped and timed.
    r = re.sub('^/', '', route)
    r = re.sub('/$', '', r)

    # start by just trying the request, to determine if we need to slice it
    if not slice:
        try:
            q = argofetch(route, options=copy.deepcopy(options), apikey=apikey, apiroot=apiroot, verbose=verbose)
            return q[0]
        except Exception as e:
            if e.args[0] == 413:
                # we need to slice
                return query(route=route, options=copy.deepcopy(options), apikey=apikey, apiroot=apiroot, verbose=verbose, slice=True)
            else:
                print(e)
                return e.args
        
    # slice request up into a series of requests
    
    ## identify timeseries, need to be recombined differently after slicing
    isTimeseries = r.split('/')[0] == 'timeseries'

    # should we slice by time or space?
    times = slice_timesteps(options, r)
    n_space = 2592 # number of 5x5 bins covering a globe 
    if 'polygon' in options:
        pgons = split_polygon(options['polygon'])
        n_space = len(pgons)
    elif 'box' in options:
        boxes = split_box(options['box'])
        n_space = len(boxes)
    
    if isTimeseries or n_space < len(times):
        ## slice up in space bins
        ops = copy.deepcopy(options)
        results = []
        delay = 0

        if 'box' in options:
            boxes = split_box(options['box'])
            for i in range(len(boxes)):
                ops['box'] = boxes[i]
                increment = argofetch(route, options=ops, apikey=apikey, apiroot=apiroot, suggestedLatency=delay, verbose=verbose)
                results += increment[0]
                delay = increment[1]
                time.sleep(increment[1]*0.8) # assume the synchronous request is supplying at least some of delay
        else:
            pgons = []
            if 'polygon' in options:
                pgons = split_polygon(options['polygon'])
            else:
                pgons = generate_global_cells()
            for i in range(len(pgons)):
                ops['polygon'] = pgons[i]
                increment = argofetch(route, options=ops, apikey=apikey, apiroot=apiroot, suggestedLatency=delay, verbose=verbose)
                results += increment[0]
                delay = increment[1]
                time.sleep(increment[1]*0.8) # assume the synchronous request is supplying at least some of delay
        # smaller polygons will trace geodesics differently than full polygons, need to doublecheck;
        # do it for boxes too just to make sure nothing funny happened on the boundaries
        ops = copy.deepcopy(options)
        ops['compression'] = 'minimal'
        true_ids = argofetch(route, options=ops, apikey=apikey, apiroot=apiroot, suggestedLatency=delay, verbose=verbose)
        true_ids = [x[0] for x in true_ids[0]]
        fetched_ids = [x['_id'] for x in results]
        if len(fetched_ids) != len(list(set(fetched_ids))):
            # deduplicate anything scooped up by multiple cells, like on cell borders
            r = {x['_id']: x for x in results}
            results = [r[i] for i in list(r.keys())]
            fetched_ids = [x['_id'] for x in results]
        to_drop = [item for item in fetched_ids if item not in true_ids]
        to_add = [item for item in true_ids if item not in fetched_ids]
        for id in to_add:
            p, delay = argofetch(route, options={'id': id}, apikey=apikey, apiroot=apiroot, suggestedLatency=delay, verbose=verbose)
            results += p
        results = [x for x in results if x['_id'] not in to_drop]

    else:
        ## slice up in time bins
        results = []
        ops = copy.deepcopy(options)
        delay = 0
        for i in range(len(times)-1):
            ops['startDate'] = times[i]
            ops['endDate'] = times[i+1]
            increment = argofetch(route, options=ops, apikey=apikey, apiroot=apiroot, suggestedLatency=delay, verbose=verbose)
            results += increment[0]
            delay = increment[1]
            time.sleep(increment[1]*0.8) # assume the synchronous request is supplying at least some of delay
        
    # slicing can end up duplicating results in batchmeta requests, deduplicate
    if 'batchmeta' in options:
        results = list({x['_id']: x for x in results}.values())

    return results

def sort_and_dedupe(data):
    # given a list <data> that may either be floats or lists of floats, 
    # deduplicate the outer list and sort it either by value or by first element as appropriate.
    def sort_key(x):
        return x[0] if isinstance(x, (list, tuple)) else x

    seen = set()
    out = []

    for x in data:
        dedupe_key = x if not isinstance(x, (list, tuple)) else tuple(x)
        if dedupe_key not in seen:
            seen.add(dedupe_key)
            out.append(x)

    return sorted(out, key=sort_key)

def queryGrid(route, options={}, apikey='', apiroot='https://argovis-api.colorado.edu/', verbose=False):
    # perform a search exactly as query(...) on a grid or timeseries route,
    # and munge the results into an xarray.Dataset
    
    ## fetch raw data from Argovis
    griddata = query(route, options=options, apikey=apikey, apiroot=apiroot, verbose=verbose)
    gridmeta = query(route, options={**options, 'batchmeta':True}, apikey=apikey, apiroot=apiroot, verbose=verbose)
    metalookup = {x['_id']: x for x in gridmeta}

    ## is this a grid or a timeseries?
    isGrid = False
    isTS = False
    if 'levels' in gridmeta[0]:
        isGrid = True
    elif 'timeseries' in gridmeta[0]:
        isTS = True
    if not isGrid and not isTS:
        raise Exception('Use this function to search for gridded and timeseries data only.')

    ## needs to be very rectangular - in levels, longitudes, latitudes, timestamps and variables
    if isGrid:
        if 'verticalRange' in options or 'presRange' in options:
            ### level subset
            levels = [p['levels'] for p in griddata]
        else:
            ### full level spectrum
            levels = [m['levels'] for m in gridmeta]
        levels = [j for i in levels for j in i] 
        levels = sort_and_dedupe(levels)
    elif isTS:
        levels = [0] # assume this is just a surface grid
        tslvls = [d.get('level', None) for d in griddata] # try and see if there are level annotations
        tslvls = [x for x in tslvls if x is not None]
        if len(tslvls) > 0:
            levels = tslvls
            levels.sort()

    locations = [d['geolocation']['coordinates'] for d in griddata]
    longitudes = list(set([x[0] for x in locations]))
    longitudes.sort()
    latitudes = list(set([x[1] for x in locations]))
    latitudes.sort()
    if isGrid:
        timestamps = list(set([x['timestamp'] for x in griddata]))
        timestamps.sort()
    elif isTS:
        if 'startDate' in options or 'endDate' in options:
            ### time subset
            timestamps = [p['timeseries'] for p in griddata]
        else:
            ### full time spectrum
            timestamps = [m['timeseries'] for m in gridmeta]
        timestamps = list({x for sub in timestamps for x in sub}) # no weird intervals like in levels
        timestamps.sort()
    timestamps = [parsetime(t) for t in timestamps]
    variables = [p['data_info'][0] for p in griddata]
    vars = list({x for sub in variables for x in sub})
    vars.sort()

    ## construct 4D data array
    darray = {}
    for v in vars:
        darray[v] = (('timestamp', 'longitude', 'latitude', 'level'), numpy.full((len(timestamps), len(longitudes), len(latitudes), len(levels)), numpy.nan, dtype=float))
    
    for p in griddata:
        m = metalookup[p['metadata'][0]]
        lvls = []
        if isGrid:
            if 'levels' in p:
                lvls = p['levels']
            else:
                lvls = m['levels']

        times = []
        if isTS:
            if 'timeseries' in p:
                times = p['timeseries']
            else:
                times = m['timeseries']
        times = [parsetime(t) for t in times]
        
        lon_idx = longitudes.index(p['geolocation']['coordinates'][0])
        lat_idx = latitudes.index(p['geolocation']['coordinates'][1])
        if isGrid:
            time_idx = timestamps.index(parsetime(p['timestamp']))
    
        for v in vars:
            if v in p['data_info'][0]:
                v_idx = p['data_info'][0].index(v)
                for i,val in enumerate(p['data'][v_idx]):
                    if isGrid:
                        lvl_idx = levels.index(lvls[i])
                    elif isTS:
                        time_idx = timestamps.index(times[i])
                        level = 0
                        if 'level' in p:
                            level = p['level']
                        lvl_idx = levels.index(level)
                    darray[v][1][time_idx][lon_idx][lat_idx][lvl_idx] = val
                    
    if isinstance(levels[0], list): # integral ranges have weird levels, label them with strings
        levels = ['_'.join([str(i) for i in x]) for x in levels]
    return xarray.Dataset(darray,coords = {'timestamp':timestamps, 'longitude':longitudes, 'latitude':latitudes, 'level':levels})

def queryProfile(route, options={}, apikey='', apiroot='https://argovis-api.colorado.edu/', verbose=False):
    # perform a search exactly as query(...) on a profile schema route,
    # and munge the results into a list of Profile objects

    ## fetch raw data from Argovis
    data = query(route, options=options, apikey=apikey, apiroot=apiroot, verbose=verbose)
    meta = query(route, options={**options, 'batchmeta':True}, apikey=apikey, apiroot=apiroot, verbose=verbose)
    metalookup = {x['_id']: x for x in meta}

    return [Profile(x, metalookup[x['metadata'][0]]) for x in data]

def profile_is_empty(data, data_info):
    # check if a profile is nothing but nan / none in every variable except pressure

    for i in range(len(data)):
        if data_info[0][i] != 'pressure' and any([v is not None and (type(v) not in [float, numpy.float64] or not math.isnan(v)) for v in data[i]]):
            return False

    return True

def rg_levels():
    # return the standard levels in dbar used in Roemmich-Gilson Argo climatology
    return [2.5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,182.5,200,220,240,260,280,300,320,340,360,380,400,420,440,462.5,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1412.5,1500,1600,1700,1800,1900,1975]

def build_dataset(interpolated_profiles, levels):
    # munge into an xarray dataset dimensioned by a profile index and levels, in analogy to Argo GDAC files
    # <interpolated_profiles> is a list of Argovis profile JSON or a list of Profile objects which must all have the same level spectrum
    # <levels> is a list of floats labeling the levels.

    # shred Profiles or json into the information we need
    nprofs = range(len(interpolated_profiles))
    darray = {}
    ids = []
    longitudes = []
    latitudes = []
    timestamps = []
    if isinstance(interpolated_profiles[0], Profile):
        ## Profile objects
        variables = [p.variable_names() for p in interpolated_profiles]
        vars = list({x for sub in variables for x in sub})
        vars.sort()

        ### empty block for variables
        for v in vars:
            if not v == 'pressure':
                darray[v] = (('nprof', 'level'), numpy.full((len(nprofs), len(levels)), numpy.nan, dtype=float))

        #### fill in variables
        for i, p in enumerate(interpolated_profiles):
            prof_idx = i
            for v in vars:
                if v in p.variable_names() and not v == 'pressure':
                    var = p.getvar(v)
                    if len(var) != len(levels):
                        raise Exception('all variables in all profiles must be interpolated to a consistent set of levels, as described by the <levels> argument.')
                    for j,val in enumerate(var):
                        lvl_idx = j
                        darray[v][1][prof_idx][lvl_idx] = val

        #### form coordinates
        ids = [p.id for p in interpolated_profiles]
        longitudes = [p.longitude for p in interpolated_profiles]
        latitudes = [p.latitude for p in interpolated_profiles]
        timestamps = [p.timestamp for p in interpolated_profiles]
        timestamps = [parsetime(t) for t in timestamps]
    else:
        ## JSON docs
        variables = [p['data_info'][0] for p in interpolated_profiles]
        vars = list({x for sub in variables for x in sub})
        vars.sort()

        ### empty block for variables
        for v in vars:
            if not v == 'pressure':
                darray[v] = (('nprof', 'level'), numpy.full((len(nprofs), len(levels)), numpy.nan, dtype=float))

        ### fill in variables
        for i, p in enumerate(interpolated_profiles):
            prof_idx = i
            for v in vars:
                if v in p['data_info'][0] and not v == 'pressure':
                    v_idx = p['data_info'][0].index(v)
                    if len(p['data'][v_idx]) != len(levels):
                        raise Exception('all variables in all profiles must be interpolated to a consistent set of levels, as described by the <levels> argument.')
                    for j,val in enumerate(p['data'][v_idx]):
                        lvl_idx = j
                        darray[v][1][prof_idx][lvl_idx] = val

        ### form coordinates
        ids = [p['_id'] for p in interpolated_profiles]
        longitudes = [p['geolocation']['coordinates'][0] for p in interpolated_profiles]
        latitudes = [p['geolocation']['coordinates'][1] for p in interpolated_profiles]
        timestamps = [p['timestamp'] for p in interpolated_profiles]
        timestamps = [parsetime(t) for t in timestamps]

    coords = {
        "id": ("nprof", ids),
        "longitude": ("nprof", longitudes),
        "latitude": ("nprof", latitudes),
        "timestamp": ("nprof", timestamps),
        "levels": ("level", levels),
    }

    attrs = {}
    return xarray.Dataset(darray,coords,attrs)

def setvar(profile, varname, values, data_info_meta=None):
    # set a new variable on a JSON profile, in analogy to Profile.setvar
    # data_info_meta should be a list specifying the metadata for this variable named in data_info[1]. 
    if not 'data_info' in profile:
        raise Exception('Profile must carry its data_info property, which it will if you included a "data" request in your query.')
    if varname in profile['data_info'][0]:
        raise Exception(f'Profile already has a variable {varname}.')

    p = copy.deepcopy(profile)
    p['data'].append(values)
    p['data_info'][0].append(varname)
    if data_info_meta is None:
        data_info_meta = ['']*len(p['data_info'][1])
    p['data_info'][2].append(data_info_meta)

    return p

def getvar(variable, data_doc, metadata_doc=None):
    # given a raw data document which includes data_info, try and extract variable as a list.
    # metadata_doc is required if data_doc doesn't have data_info

    data_info = find_key('data_info', data_doc, metadata_doc)
    try:
        var_idx = data_info[0].index(variable)
    except:
        print(variable, ' not found in this data document; available variables are ', data_info[0])
        return None
    
    return data_doc['data'][var_idx]

@dataclass
class Profile:
    rawdata: dict[str, Any] = field(default_factory=dict)
    rawmeta: dict[str, Any] = field(default_factory=dict)
    vars: dict[str, numpy.ndarray] = field(default_factory=dict, repr=False)

    def __init__(self, data, meta=None):
        self._rawdata = copy.deepcopy(data)
        self._rawmeta = copy.deepcopy(meta)
        
        data_info = [[],[],[]]
        if 'data_info' in data:
            data_info = data['data_info']
        elif 'data_info' in meta:
            data_info = meta['data_info']

        self.vars = {}
        for i, name in enumerate(data_info[0]):
            arr = numpy.ma.masked_invalid(numpy.array(data['data'][i], dtype=float))
            self.vars[name] = arr
        del self._rawdata['data']

        # dict for arbitrary annotations
        self.attrs = {}

    # ---- pretend like you're a dictionary for p['arbitrary_key'] ----
    def __getitem__(self, key):
        return self.attrs[key]

    def __setitem__(self, key, value):
        self.attrs[key] = value

    def get(self, key, default=None):
        return self.attrs.get(key, default)
        
    # ---- metadata sugar ----
    @property
    def id(self):
        return self.rawdata['_id']

    @property
    def timestamp(self):
        return parser.parse(self.rawdata['timestamp'], ignoretz=True)

    @property
    def longitude(self):
        return float(self.rawdata['geolocation']['coordinates'][0])

    @property
    def latitude(self):
        return float(self.rawdata['geolocation']['coordinates'][1])

    @property
    def rawdata(self):
        return self._rawdata

    @property
    def rawmeta(self):
        return self._rawmeta
    
    # ---- variable helpers ----
    def variable_names(self):
        return tuple(self.vars.keys())

    def hasvar(self, name):
        return name in self.vars

    def getvar(self, name, preserve_mask=False):
        if preserve_mask:
            return self.vars.get(name)
        else:
            return self.vars.get(name).filled(numpy.nan) if name in self.vars else None

    def setvar(self, name, values, mask=None):
        arr = numpy.ma.masked_array(values, mask=mask, dtype=float) 
        if arr.ndim != 1:
            raise ValueError(f"Variable {name!r} must be 1D, got shape {arr.shape}")
        self.vars[name] = arr

    def delvar(self, name):
        del self.vars[name]
