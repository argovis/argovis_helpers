import requests, datetime, copy, time, re, area, math, urllib, json
from shapely.geometry import shape, box, Polygon
from shapely.ops import orient
import geopandas as gpd

# networking helpers

def argofetch(route, options={}, apikey='', apiroot='https://argovis-api.colorado.edu/', suggestedLatency=0, verbose=False):
    # GET <apiroot>/<route>?<options> with <apikey> in the header.
    # raises on anything other than success or a 404.

    o = copy.deepcopy(options)
    for option in ['polygon', 'box']:
        if option in options:
            options[option] = str(options[option])

    dl = requests.get(apiroot.rstrip('/') + '/' + route.lstrip('/'), params = options, headers={'x-argokey': apikey})
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
            print('The temporospatial extent of your request is enormous! Consider using the `query` helper in this package to split it up into more manageable chunks.')
        elif statuscode >= 500 or (statuscode==200 and type(dl) is dict and 'code' in dl):
            print("Argovis' servers experienced an error. Please try your request again, and email argovis@colorado.edu if this keeps happening; please include the full details of the the request you made so we can help address.")
        raise Exception(statuscode, dl)

    # no special action for 404 - a 404 due to a mangled route will return an error, while a valid search with no result will return [].

    return dl, suggestedLatency

def query(route, options={}, apikey='', apiroot='https://argovis-api.colorado.edu/', verbose=False):
    # middleware function between the user and a call to argofetch to make sure individual requests are reasonably scoped and timed.
    r = re.sub('^/', '', route)
    r = re.sub('/$', '', r)

    data_routes = ['argo', 'cchdo', 'drifters', 'tc', 'argotrajectories', 'easyocean', 'grids/rg09', 'grids/kg21', 'grids/glodap', 'timeseries/noaasst', 'timeseries/copernicussla', 'timeseries/ccmpwind', 'extended/ar']
    
    scoped_parameters = {
        'argo': ['id','platform', 'metadata'],
        'cchdo': ['id', 'woceline', 'cchdo_cruise', 'metadata'],
        'drifters': ['id', 'wmo', 'platform', 'metadata'],
        'tc': ['id', 'name', 'metadata'],
        'argotrajectories': ['id', 'platform', 'metadata'],
        'easyocean': ['id', 'woceline', 'metadata'],
        'grids/rg09': ['id'],
        'grids/kg21': ['id'],
        'grids/glodap': ['id'],
        'timeseries/noaasst': ['id'],
        'timeseries/copernicussla': ['id'],
        'timeseries/ccmpwind': ['id'],
        'extended/ar': ['id']
    }
    
    if r in data_routes and (not 'compression' in options or options['compression']!='minimal'):
        # these are potentially large requests that might need to be sliced up

        ## identify timeseries, need to be recombined differently after slicing
        isTimeseries = r.split('/')[0] == 'timeseries'

        ## if a data query carries a scoped parameter, no need to slice up:
        if r in scoped_parameters and not set(scoped_parameters[r]).isdisjoint(options.keys()):
            return argofetch(route, options=options, apikey=apikey, apiroot=apiroot, verbose=verbose)[0]

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

    else:
        return argofetch(route, options=options, apikey=apikey, apiroot=apiroot, verbose=verbose)[0]

def slice_timesteps(options, r):
    # given a qsr option dict and data route, return a list of reasonable time divisions

    earliest_records = {
        'argo': parsetime("1997-07-27T20:26:20.002Z"),
        'cchdo': parsetime("1972-07-23T09:11:00.000Z"),
        'drifters': parsetime("1987-10-01T13:00:00.000Z"),
        'tc': parsetime("1851-06-24T00:00:00.000Z"),
        'argotrajectories': parsetime("2001-01-03T22:46:33.000Z"),
        'easyocean': parsetime("1983-10-08T00:00:00.000Z"),
        'grids/rg09': parsetime("2004-01-14T00:00:00.000Z"),
        'grids/kg21': parsetime("2005-01-14T00:00:00.000Z"),
        'grids/glodap': parsetime("1000-01-01T00:00:00.000Z"),
        'timeseries/noaasst': parsetime("1989-12-30T00:00:00.000Z"),
        'timeseries/copernicussla': parsetime("1993-01-02T00:00:00Z"),
        'timeseries/ccmpwind': parsetime("1993-01-02T00:00:00Z"),
        'extended/ar': parsetime("2000-01-01T00:00:00Z")
    }

    # plus a day vs the API, just to make sure we don't artificially cut off 
    last_records = {
        'argo': datetime.datetime.now(),
        'cchdo': parsetime("2023-03-10T17:48:00.000Z"),
        'drifters': parsetime("2020-07-01T23:00:00.000Z"),
        'tc': parsetime("2020-12-26T12:00:00.000Z"),
        'argotrajectories': parsetime("2021-01-02T01:13:26.000Z"),
        'easyocean': parsetime("2022-10-17T00:00:00.000Z"),
        'grids/rg09': parsetime("2022-05-16T00:00:00.000Z"),
        'grids/kg21': parsetime("2020-12-16T00:00:00.000Z"),
        'grids/glodap': parsetime("1000-01-02T00:00:00.000Z"),
        'timeseries/noaasst': parsetime("2023-01-30T00:00:00.000Z"),
        'timeseries/copernicussla': parsetime("2022-08-01T00:00:00.000Z"),
        'timeseries/ccmpwind': parsetime("2019-12-30T00:00:00Z"),
        'extended/ar': parsetime("2022-01-01T21:00:00Z")
    }

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
        start = earliest_records[r]
    if 'endDate' in options:
        end = parsetime(options['endDate'])
    else:
        end = last_records[r]
        
    delta = datetime.timedelta(days=timestep)
    times = [start]
    while times[-1] + delta < end:
        times.append(times[-1]+delta)
    times.append(end)
    times = [parsetime(x) for x in times]
    
    return times
    
# data munging helpers

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
