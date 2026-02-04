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

def datagrid(route, options={}, apikey='', apiroot='https://argovis-api.colorado.edu/', verbose=False):
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

def interpolate_to_levels(levels_raw, var_raw, levels_interp, pressure_buffer=-1, pressure_index_buffer=-1):
    # interpolate <var> to <levels> using PCHIP interpolation
    # keep <pressure_buffer> dbar on either side of the ROI and <pressure_index_buffer> points in the pressure buffer margins, at least.
    # flag 32 (little endian): ROI didn't contain enough info to interpolate

    flag = 0
    pressure, variable, flag = tidy_profile(levels_raw, var_raw, flag)

    # some truly pathological profiles will have no levels left at this point
    if len(pressure) == 0:
        interp = numpy.full(len(levels_interp), numpy.nan)
        flag = flag | 32
        return interp, flag

    # find indexes of ROI
    if pressure_buffer >= 0 and pressure_index_buffer >= 0:
        p_bracket = pad_bracket(pressure, levels_interp[0], levels_interp[-1], pressure_buffer, pressure_index_buffer)
    else:
        p_bracket = [0, len(pressure)-1]

    # ROI must contain at least two points for Pchip
    if len(pressure[p_bracket[0]:p_bracket[1]+1]) < 2:
        interp = numpy.full(len(levels_interp), numpy.nan)
        flag = flag | 32
        return interp, flag
    else:
        # interpolate; don't extrapolate to levels outside of measurement range
        interp = scipy.interpolate.PchipInterpolator(pressure[p_bracket[0]:p_bracket[1]+1], variable[p_bracket[0]:p_bracket[1]+1], extrapolate=False)(levels_interp)

        # if there wasn't a measured level within a certain radius of each level of interest, mask the interpolation at that level.
        interp = mask_far_interps(pressure[p_bracket[0]:p_bracket[1]+1], levels_interp, interp)

        return interp, flag

def tidy_profile(pressure, var, flag):
    # pchip needs pressures to be monotonically increasing; also need the dependent variable to always be defined
    # flags (little endian):
    # 1: degenerate adjacent levels
    # 2: levels in reverse order
    # 4: variable of interest was NaN, masked
    # 8: levels non-monotonic, had to sort
    # 16: pressure was NaN, masked

    ## dependent variable must be defined
    mask = [0]*len(var)
    for i in range(len(var)):
        if var[i] is None or math.isnan(var[i]):
            mask[i] = 1
            flag = flag | 4
    p = [pressure[i] for i in range(len(mask)) if mask[i]==0]
    v = [var[i] for i in range(len(mask)) if mask[i]==0]

    ## pressure must be defined
    mask = [0]*len(p)
    for i in range(len(p)):
        if p[i] is None or math.isnan(p[i]):
            mask[i] = 1
            flag = flag | 16
    p = [p[i] for i in range(len(mask)) if mask[i]==0]
    v = [v[i] for i in range(len(mask)) if mask[i]==0]

    ## drop degenerate levels and flag
    mask = [0]*len(p)
    for i in range(len(p)-1):
        if p[i] == p[i+1]:
            mask[i] = 1
            mask[i+1] = 1
            flag = flag | 1
    p = [p[i] for i in range(len(mask)) if mask[i]==0]
    v = [v[i] for i in range(len(mask)) if mask[i]==0]

    if all(p[i] < p[i + 1] for i in range(len(p) - 1)):
        # pressure is monotonically increasing, return
        return p, v, flag

    if all(p[i] > p[i + 1] for i in range(len(p) - 1)):
        # pressure is monotonically decreasing, reverse and return
        flag = flag | 2
        return p[::-1], v[::-1], flag

    # pressure is non-monotonic, sort and try again
    x = sorted(zip(p,v))
    p = [element[0] for element in x]
    v = [element[1] for element in x]
    flag = flag | 8
    return tidy_profile(p,v,flag)

def pad_bracket(lst, low_roi, high_roi, buffer, places):
    # returns the indexes of the last element below and first element above an ROI padded with <buffer>, and containing at least <places> elements in the padding.

    tight_bracket = find_bracket(lst, low_roi, high_roi)
    buffer_bracket = find_bracket(lst, low_roi - buffer, high_roi + buffer)

    low = buffer_bracket[0]
    if tight_bracket[0] - buffer_bracket[0] < places-1: # -1 since find_bracket gives the first bound in the wing, so there's already one point in the wing even for tight_bracket
        low = max(0, tight_bracket[0] - places+1)

    high = buffer_bracket[1]
    if buffer_bracket[1] - tight_bracket[1] < places-1:
        high = min(len(lst)-1, tight_bracket[1] + places-1)

    return low, high

def mask_far_interps(measured_pressures, interp_levels, interp_values):
    # mask interpolated values that are too far from the nearest measured level

    for i, level in enumerate(interp_levels):
        ## determine how far is too far:
        radius = 0
        if level < 100:
            radius = 10
        elif level < 150:
            radius = 20
        elif level < 250:
            radius = 40
        elif level < 350:
            radius = 60
        elif level < 450:
            radius = 80
        else:
            radius = 100

        i_below = 0
        i_above = len(measured_pressures)-1
        for j in range(len(measured_pressures)):
            if measured_pressures[j] <= level:
                i_below = j
            else:
                i_above = j
                break
        if abs(measured_pressures[i_below] - level) > radius or abs(measured_pressures[i_above] - level) > radius:
            interp_values[i] = numpy.nan

    return interp_values

def find_bracket(lst, low_roi, high_roi):
    # lst: ordered list of floats
    # low_roi: lower bound of region of interest
    # high_roi: upper bound "
    # returns the indexes of the last element below and first element above the ROI, without running off ends of list

    if low_roi <= lst[0]:
        low_index = 0
    else:
        low = 0
        high = len(lst) - 1
        low_index = -1

        while low <= high:
            mid = (low + high) // 2

            if lst[mid] < low_roi:
                low_index = mid
                low = mid + 1
            else:
                high = mid - 1

    if high_roi >= lst[-1]:
        high_index = len(lst) - 1
    else:
        low = 0
        high = len(lst) - 1
        high_index = -1

        while low <= high:
            mid = (low + high) // 2

            if lst[mid] > high_roi:
                high_index = mid
                high = mid - 1
            else:
                low = mid + 1

    return low_index, high_index

def interpolate_all(profile, levels, pressure_buffer=-1, pressure_index_buffer=-1):
    # interpolate all variables in a profile to a common set of levels
    # assumes profile has its 'data_info' key present and that 'pressure' is one of the variables, to be interpolated on.
    
    ## remove any QC vectors
    variables = profile['data_info'][0]
    qcvecs = ['qc' in x for x in variables] # a bit duck-typie...
    data = [x for i,x in enumerate(profile['data']) if not qcvecs[i]]
    data_info = [None, None, None]
    data_info[0] = [x for i,x in enumerate(profile['data_info'][0]) if not qcvecs[i]]
    data_info[1] = profile['data_info'][1]
    data_info[2] = [x for i,x in enumerate(profile['data_info'][2]) if not qcvecs[i]]

    level_idx = data_info[0].index('pressure')
    raw_levels = data[level_idx]

    for i in range(len(data_info[0])):
        if i == level_idx:
            data[i] = levels
        else:
            data[i], _ = interpolate_to_levels(raw_levels, data[i], levels, pressure_buffer, pressure_index_buffer)
            data[i] = list(data[i])

    return {**profile, 'data':data, 'data_info':data_info}

def profile_is_empty(data, data_info):
    # check if a profile is nothing but nan / none in every variable except pressure

    for i in range(len(data)):
        if data_info[0][i] != 'pressure' and any([v is not None and (type(v) not in [float, numpy.float64] or not math.isnan(v)) for v in data[i]]):
            return False

    return True

def rg_levels():
    # return the standard levels in dbar used in Roemmich-Gilson Argo climatology
    return [2.5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,182.5,200,220,240,260,280,300,320,340,360,380,400,420,440,462.5,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1412.5,1500,1600,1700,1800,1900,1975]

def query_interpolated(route, options={}, apikey='', apiroot='https://argovis-api.colorado.edu/', levels=None, verbose=False, pressure_buffer=-1, pressure_index_buffer=-1, suppress_no_interps=True, audit_all=False):
    # fetch profiles from Argovis and interpolate all variables to a common set of levels.

    # route [string]: any argovis route
    # options [dict]: dict of corresponding query options
    # apikey [string]: your argovis apikey
    # apiroot [string]: the root url of the argovis api
    # levels [float list]: list of levels to interpolate to; if None, use Roemmich-Gilson levels
    # verbose [bool]: print out each individual request made to Argovis
    # pressure_buffer [float]: dbar of extra pressure range to include on either side of the ROI for interpolation; if -1, don't use a buffer
    # pressure_index_buffer [int]: minimum number of extra points to include in the buffer on either side of the ROI for interpolation; if -1, don't use a buffer
    # suppress_no_interps [bool]: drop profiles that are nothing but NaN in every variable except pressure after interpolation
    # audit_all [bool]: return a tuple of (interpolated_profiles, raw_profiles) for auditing purposes

    ## request validation
    if 'data' not in options:
        raise Exception('Interpolated queries require some level data to interpolate on; please add a data request to your query options.')
        return
    if levels is None:
        print("warning: you didn't provide a level spectrum to interpolate on; using the Roemmich-Gilson levels by default.")
        levels = rg_levels()

    ## fetch raw data from Argovis
    rawdata = query(route, options=options, apikey=apikey, apiroot=apiroot, verbose=verbose)
    interpolated_profiles = [interpolate_all(x, levels, pressure_buffer, pressure_index_buffer) for x in rawdata]
    
    ## intended for sanity checking
    if audit_all:
        return interpolated_profiles, rawdata

    ## dump profiles that are nothing but None / NaN in every variable except pressure
    if suppress_no_interps:
        interpolated_profiles = [x for x in interpolated_profiles if not profile_is_empty(x['data'], x['data_info'])]
        
    return interpolated_profiles

def build_dataset(interpolated_profiles, levels):
    # munge into an xarray dataset dimensioned by a profile index and levels, in analogy to Argo GDAC files
    # <interpolated_profiles> is a list of Argovis profile objects which must all have the same level spectrum
    # <levels> is a list of floats labeling the levels.

    # make sure the user at least appears to have respected the standard-levels requirement
    for p in interpolated_profiles:
        for v in p['data']:
            if not len(v) == len(levels):
                raise Exception('all variables in all profiles must be interpolated to a consistent set of levels, as described by the <levels> argument.')
    
    nprofs = range(len(interpolated_profiles))
    variables = [p['data_info'][0] for p in interpolated_profiles]
    vars = list({x for sub in variables for x in sub})
    vars.sort()

    darray = {}
    for v in vars:
        if not v == 'pressure':
            # pressures must be interpolated to a standard spectrum captured as the levels coord, no need to repeat them for every profile
            darray[v] = (('nprof', 'level'), numpy.full((len(nprofs), len(levels)), numpy.nan, dtype=float))

    for i, p in enumerate(interpolated_profiles):
        prof_idx = i
        for v in vars:
            if v in p['data_info'][0] and not v == 'pressure':
                v_idx = p['data_info'][0].index(v)
                for j,val in enumerate(p['data'][v_idx]):
                    lvl_idx = j
                    darray[v][1][prof_idx][lvl_idx] = val

    # form coordinates
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

def append_gsw(profiles, gsw_variables, temperature_key='temperature', salinity_key='salinity', pressure_key='pressure'):
    def _clean(v): 
        return numpy.array([numpy.nan if x is None else x for x in v])

    # validate every profile
    for p in profiles:
        if not 'data_info' in p:
            raise Exception('All profile must carry their data_info property, which they will if you included a "data" request in your query.')
        if not pressure_key in p['data_info'][0]:
            raise Exception('All GSW parameters require a pressure to calculate, and the value "'+pressure_key+'" you provided for pressure_key is not found in at least one of your profiles.')
        if not salinity_key in p['data_info'][0]:
            raise Exception('All GSW parameters require a salinity to calculate, and the value "'+salinity_key+'" you provided for salinity_key is not found in at least one of your profiles.')
        if not temperature_key in p['data_info'][0] and any(item in gsw_variables for item in ['conservative_temperature', 'potential_density', 'Nsquared']):
            raise Exception('The GSW parameters you requested require a temperature to calculate, and the value "'+temperature_key+'" you provided for temperature_key is not found in at least one of your profiles.')

    munged = []
    for profile in profiles:
        prof = copy.deepcopy(profile)
        longitude = prof['geolocation']['coordinates'][0]
        latitude = prof['geolocation']['coordinates'][1]
        temperature_idx = prof['data_info'][0].index(temperature_key)
        salinity_idx = prof['data_info'][0].index(salinity_key)
        pressure_idx = prof['data_info'][0].index(pressure_key)
        units_idx = prof['data_info'][1].index('units')

        t = _clean(prof['data'][temperature_idx])
        s = _clean(prof['data'][salinity_idx])
        p = _clean(prof['data'][pressure_idx])
        
        SA = gsw.SA_from_SP(s, p, longitude, latitude)
        CT = gsw.CT_from_t(SA, t, p)
        sigma0 = gsw.sigma0(SA, CT) + 1000
        N2, _ = gsw.Nsquared(SA, CT, p, lat=latitude)

        if 'absolute_salinity' in gsw_variables:
            prof['data'].append(SA.tolist())
            prof['data_info'][0].append('gsw_absolute_salinity')
            prof['data_info'][2].append(['']*len(prof['data_info'][1]))
            prof['data_info'][2][-1][units_idx] = 'g/kg'
        if 'conservative_temperature' in gsw_variables:
            prof['data'].append(CT.tolist())
            prof['data_info'][0].append('gsw_conservative_temperature')
            prof['data_info'][2].append(['']*len(prof['data_info'][1]))
            prof['data_info'][2][-1][units_idx] = 'degC'
        if 'potential_density' in gsw_variables:
            prof['data'].append(sigma0.tolist())
            prof['data_info'][0].append('gsw_potential_density')
            prof['data_info'][2].append(['']*len(prof['data_info'][1]))
            prof['data_info'][2][-1][units_idx] = 'kg/m^3'
        if 'Nsquared' in gsw_variables:
            prof['data'].append(N2.tolist())
            prof['data_info'][0].append('gsw_Nsquared')
            prof['data_info'][2].append(['']*len(prof['data_info'][1]))

            
        munged.append(prof)

    return munged

def regional_mean(dxr, form='area'):
    # given an xarray dataset <dxr> with latitudes and longitudes as dimensions,
    # calculate the mean of all data variables, weighted by grid cell area
    weights = numpy.cos(numpy.deg2rad(dxr.latitude))
    weights.name = "weights"
    dxr_weighted = dxr.weighted(weights)
    
    if form =='area':
        return dxr_weighted.mean(("longitude", "latitude"))
    elif form == 'meridional':
        return dxr_weighted.mean(("latitude"))
    elif form == 'zonal':
        return dxr_weighted.mean(("longitude"))

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
        
        data_info = []
        if 'data_info' in data:
            data_info = data['data_info']
        elif 'data_info' in meta:
            data_info = meta['data_info']
        else:
            raise ValueError("no data_info property found")

        self.vars = {}
        for i, name in enumerate(data_info[0]):
            arr = numpy.asarray(data['data'][i])
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
        return parser.parse(self.rawdata['timestamp'])

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

    def getvar(self, name):
        return self.vars.get(name)

    def setvar(self, name, values):
        arr = numpy.asarray(values)
        if arr.ndim != 1:
            raise ValueError(f"Variable {name!r} must be 1D, got shape {arr.shape}")
        self.vars[name] = arr