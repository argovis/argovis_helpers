from argovisHelpers import helpers
import datetime, pytest

@pytest.fixture
def apiroot():
    return 'http://api:8080'

@pytest.fixture
def apikey():
    return 'developer'

def test_argofetch(apiroot, apikey):
    '''
    check basic behavior of argofetch
    '''

    profile = helpers.argofetch('/argo', options={'id': '13857_068'}, apikey=apikey, apiroot=apiroot)[0]
    assert len(profile) == 1, 'should have returned exactly one profile'
    assert profile[0]['geolocation'] == { "type" : "Point", "coordinates" : [ -26.257, 3.427 ] }, 'fetched wrong profile'

    profile = helpers.argofetch('argo', options={'id': '13857_068'}, apikey=apikey, apiroot=apiroot)[0]
    assert len(profile) == 1, 'leading / on route shouldnt affect results'
    profile = helpers.argofetch('/argo', options={'id': '13857_068'}, apikey=apikey, apiroot=apiroot+'/')[0]
    assert len(profile) == 1, 'extra slashes betwen apiroot and route shouldnt matter'

def test_argofetch_404(apiroot, apikey):
    '''
    check various flavors of 404
    '''

    # typoed route should give an error
    profile = helpers.argofetch('/agro', options={'startDate':'2022-02-01T00:00:00Z', 'endDate':'2022-02-02T00:00:00Z'}, apikey=apikey, apiroot=apiroot)[0]
    assert profile['message'] == 'not found'

    # valid search with no results should give an empty list
    profile = helpers.argofetch('/argo', options={'startDate':'2072-02-01T00:00:00Z', 'endDate':'2072-02-02T00:00:00Z'}, apikey=apikey, apiroot=apiroot)[0]
    assert profile == []

def test_bulky_fetch(apiroot, apikey):
    '''
    make sure argofetch handles rapid requests for the whole globe reasonably
    '''

    result = []
    delay = 0
    for i in range(3):
        request = helpers.argofetch('/grids/rg09', options={'startDate': '2004-01-01T00:00:00Z', 'endDate': '2004-02-01T00:00:00Z', 'data':'rg09_temperature'}, apikey='regular', apiroot=apiroot)
        result += request[0]
        delay += request[1]
    assert len(result) == 60, 'should have found 20x3 grid docs'
    assert delay > 0, 'should have experienced at least some rate limiter delay'

def test_polygon(apiroot, apikey):
    '''
    make sure polygons are getting handled properly
    '''

    profile = helpers.argofetch('/argo', options={'polygon': [[-26,3],[-27,3],[-27,4],[-26,4],[-26,3]]}, apikey=apikey, apiroot=apiroot)[0]
    assert len(profile) == 1, 'polygon encompases exactly one profile'

def test_data_inflate(apiroot, apikey):
    '''
    check basic behavior of data_inflate
    '''

    data_doc = {
        'data': [[1,2,3],[4,5,6]],
        'data_info': [['a','b'],[],[]]
    }
    inflate = helpers.data_inflate(data_doc)
    print(inflate)
    assert inflate == [{'a':1, 'b':4}, {'a':2, 'b':5}, {'a':3, 'b':6}], f'simple array didnt inflate correctly, got {inflate}'

def test_find_key(apiroot, apikey):
    '''
    check basic behavior of find_key
    '''

    data = {'metadata': ['meta'], 'a': 1, 'b':2, 'c':3}
    meta = {'_id': 'meta', 'a': 4, 'd':5}

    assert helpers.find_key('a', data, meta) == 1, 'find_key should select the entry from data_doc if key appears in both data and metadata'
    assert helpers.find_key('d', data, meta) == 5, 'find_key should look in meta doc'


def test_parsetime(apiroot, apikey):
    '''
    check basic behavior of parsetime
    '''

    datestring = '1999-12-31T23:59:59.999999Z'
    dtime = datetime.datetime(1999, 12, 31, 23, 59, 59, 999999)

    assert helpers.parsetime(datestring) == dtime, 'date string should have been converted to datetime.datetime'
    assert helpers.parsetime(helpers.parsetime(datestring)) == datestring, 'parsetime should be its own inverse'

def test_parsetime(apiroot, apikey):
    '''
    check small-year behavior of parsetime
    '''

    datestring = '0001-12-31T23:59:59.999999Z'
    dtime = datetime.datetime(1, 12, 31, 23, 59, 59, 999999)

    assert helpers.parsetime(datestring) == dtime, 'date string should have been converted to datetime.datetime'
    assert helpers.parsetime(helpers.parsetime(datestring)) == datestring, 'parsetime should be its own inverse'

def test_query(apiroot, apikey):
    '''
    check basic behavior of query
    '''

    response = helpers.query('/tc', options={'startDate': '1851-05-26T00:00:00Z', 'endDate': '1852-01-01T00:00:00Z'}, apikey=apikey, apiroot=apiroot)
    assert len(response) == 9, f'should be able to query entire globe for 6 months, with time divisions landing exactly on one timestamp, and get back 9 tcs, instead got {response}'

def test_big_poly(apiroot, apikey):
    '''
    query with polygon big enough to trigger lune slices behind the scenes
    note  TC ID AL041851_18510816000000 is fudged to sit on longitude 45, right on a lune boundary
    '''

    response = helpers.query('/tc', options={'startDate': '1851-05-26T00:00:00Z', 'endDate': '1852-01-01T00:00:00Z', 'polygon': [[-40,60],[-100,60],[-100,-60],[-40,-60],[-40,60]]}, apikey=apikey, apiroot=apiroot)
    assert len(response) == 9, f'should be able to query entire globe for 6 months, with time divisions landing exactly on one timestamp, and get back 9 tcs, instead got {len(response)}'


def test_query_vocab(apiroot, apikey):
    '''
    check basic behavior of vocab query
    '''

    response = helpers.query('/cchdo/vocabulary', options={'parameter': 'woceline',}, apikey=apikey, apiroot=apiroot)
    assert response == ["A12", "AR08", "SR04"], f'should be able to query woceline vocab, instead got {response}'

def test_units_inflate(apiroot, apikey):
    '''
    check basic behavior of units_inflate
    '''

    data = {'metadata': ['meta'], 'data_info': [['a', 'b', 'c'],['x', 'units'],[[0, 'dbar'],[1, 'kelvin'],[2, 'psu']]]}
    units = helpers.units_inflate(data) 

    assert units == {'a': 'dbar', 'b': 'kelvin', 'c': 'psu'}, f'failed to reconstruct units dict, got {units}'

def test_combine_data_lists(apiroot, apikey):
    '''
    check basic behavior of combine_data_lists
    '''

    a = [[1,2],[3,4]]
    b = [[5,6],[7,8]]
    c = [[10,11],[12,13]]
    assert helpers.combine_data_lists([a]) == [[1,2],[3,4]], 'failed to combine a single data list'
    assert helpers.combine_data_lists([a,b]) == [[1,2,5,6],[3,4,7,8]], 'failed to combine two data lists'
    assert helpers.combine_data_lists([a,b,c]) == [[1,2,5,6,10,11],[3,4,7,8,12,13]], 'failed to combine three data lists'


def test_timeseries_recombo(apiroot, apikey):
    '''
    make sure a timeseries request that gets forcibly sliced is recombined correctly
    '''

    slice_response = helpers.query('/timeseries/ccmpwind', options={'startDate':'1995-01-01T00:00:00Z', 'endDate':'2019-01-01T00:00:00Z', 'polygon': [[-10,-10],[10,-10],[10,10],[-10,10],[-10,-10]], 'data':'all'}, apikey=apikey, apiroot=apiroot)
    noslice_response = helpers.query('/timeseries/ccmpwind', options={'startDate':'1995-01-01T00:00:00Z', 'endDate':'2019-01-01T00:00:00Z', 'id': '0.125_0.125', 'data':'all'}, apikey=apikey, apiroot=apiroot)

    assert slice_response[0]['data'] == noslice_response[0]['data'], 'mismatch on data recombination'
    assert slice_response[0]['timeseries'] == noslice_response[0]['timeseries'], 'mismatch on timestamp recombination'

def test_timeseries_recombo_edges(apiroot, apikey):
    '''
    check some edgecases of timeseries recombo
    '''

    response = helpers.query('/timeseries/ccmpwind', options={'startDate':'1995-01-01T00:00:00Z', 'endDate':'2019-01-01T00:00:00Z', 'polygon': [[-10,-10],[10,-10],[10,10],[-10,10],[-10,-10]]}, apikey=apikey, apiroot=apiroot)
    assert 'data' not in response[0], 'make sure timeseries recombination doesnt coerce a data key onto a document that shouldnt have one'
    response = helpers.query('/timeseries/ccmpwind', options={'polygon': [[-10,-10],[10,-10],[10,10],[-10,10],[-10,-10]]}, apikey=apikey, apiroot=apiroot)
    assert 'timeseries' not in response[0], 'make sure timeseries recombination doesnt coerce a timeseries key onto a document that shouldnt have one'

def test_generate_global_cells(apiroot, apikey):
    '''
    check basic behavor of generate_global_cells
    '''

    assert len(helpers.generate_global_cells()) == 2592, 'global 5x5 grid generated wrong number of cells'
    assert helpers.generate_global_cells()[0] == [[-180,-90],[-175,-90],[-175,-85],[-180,-85],[-180,-90]], 'first cell of globabl 5x5 grid generated incorrectly'

def test_dont_wrap_dateline(apiroot, apikey):
    '''
    check basic behavior of dont_wrap_dateline
    '''

    assert helpers.dont_wrap_dateline([[-175,0],[-175,10],[175,10],[175,0],[-175,0]]) == [[185,0],[185,10],[175,10],[175,0],[185,0]], 'basic dateline unwrap failed'
    assert helpers.dont_wrap_dateline([[-175,0],[175,0],[175,10],[-175,10],[-175,0]]) == [[185,0],[175,0],[175,10],[185,10],[185,0]], 'unwrap cw'
    assert helpers.dont_wrap_dateline([[5,0],[-5,0],[-5,5],[5,5],[5,0]]) == [[5,0],[-5,0],[-5,5],[5,5],[5,0]], 'unwrap shoudnt affect meridian crossing'



