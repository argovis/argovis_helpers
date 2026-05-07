from argovisHelpers import helpers
from argovisHelpers import analysis
import datetime, pytest, numpy, scipy, xarray, gsw

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

def test_big_time_slice(apiroot, apikey):
    '''
    check that slicing in a query with a long time range and polygon is working correctly
    '''

    natlantic = [[-52.91015625000001,57.57635026510582],[-47.4609375,58.59841337380398],[-41.13281250000001,58.96285043960036],[-36.12304687500001,58.73552560169896],[-28.828125000000004,58.781109991263875],[-24.433593750000004,58.91750479454867],[-17.929687500000004,58.82663462015099],[-12.216796875000002,58.50670551226914],[-12.392578125,41.263494202188674],[-20.126953125000004,41.06499545917395],[-28.388671875000004,41.592988409051024],[-37.44140625000001,40.865895731685946],[-45.87890625,41.78988186577712],[-52.11914062500001,41.724317678639935],[-52.91015625000001,57.57635026510582]]
    options = {
        'startDate': '2000-01-01T00:00:00Z',
        'endDate': '2021-01-01T00:00:00Z',
        'polygon': natlantic,
        'presRange': '0,100',
        'compression': 'minimal',
        'data': 'doxy,1'
    }
    response = helpers.query('/argo', options=options, apikey='regular', apiroot=apiroot)
    print(response)
    assert len(response) == 0, 'query should run to completion with no result'

def test_mask_far_interps():

    insitu_pres = numpy.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.,15.])
    interp_pres = numpy.array([4.5, 25, 25.1]) # note its not this function's job to disqualify levels outside of the range of measurements, only interpolated levels that don't have a close neighbor.
    interp_vals = numpy.array([0.,1.,2.])

    assert numpy.allclose(analysis.mask_far_interps(insitu_pres, interp_pres, interp_vals), [0,1,numpy.nan],equal_nan=True), 'basic mask'

def test_interpolate_to_levels():

    profile_levels = [1,2,3,4,5]
    profile_temperature = [10,20,30,40,50]
    degen_levels = [1,1,3,4,5]

    assert numpy.allclose(analysis.interpolate_to_levels(profile_levels, profile_temperature, [1.5,2.5,3.5,4.5])[0], [15,25,35,45]), 'basic interp'
    assert numpy.allclose(analysis.interpolate_to_levels(profile_levels, profile_temperature, [1.5,2.5,3.5,4.5])[1], 0), 'basic interp flagging'
    assert numpy.allclose(analysis.interpolate_to_levels(profile_levels, profile_temperature, [2,4,6])[0], [20,40, numpy.nan], equal_nan=True), 'dont run off end of insitu data'
    assert numpy.allclose(analysis.interpolate_to_levels(profile_levels, profile_temperature, [0.9999,2,4])[0], [numpy.nan,20,40], equal_nan=True), 'dont run off start of insitu data'
    assert numpy.allclose(analysis.interpolate_to_levels(degen_levels, profile_temperature, [2,4,6])[0], [numpy.nan,40,numpy.nan], equal_nan=True), 'degenerate profile'
    assert numpy.allclose(analysis.interpolate_to_levels(degen_levels, profile_temperature, [2,4,6])[1], 1), 'degenerate profile flagging'

def test_profile_is_empty():
    mock_data_info = [['a', 'pressure', 'b'],[],[]]

    assert helpers.profile_is_empty([[],[],[]], mock_data_info) == True, 'profile empty'
    assert helpers.profile_is_empty([[],[1,2,3],[]], mock_data_info) == True, 'profile empty other than pressure'
    assert helpers.profile_is_empty([[numpy.nan,numpy.nan,numpy.nan],[1,2,3],[]], mock_data_info) == True, 'nans count as empty'
    assert helpers.profile_is_empty([[None,None,None],[1,2,3],[]], mock_data_info) == True, 'Nones count as empty'
    assert helpers.profile_is_empty([[numpy.nan, numpy.nan],[1,2],[10,20]], mock_data_info) == False, 'data should pass even if some vectors are empty'
    assert helpers.profile_is_empty([['xyz', numpy.nan],[1,2],[10,20]], mock_data_info) == False, 'strings count as not empty'
    assert helpers.profile_is_empty([[numpy.nan, 2], [1,2], [None,None]], mock_data_info) == False, 'partially nan vector should pass'

def test_tidy_profile():
    assert analysis.tidy_profile([1,2,3,3,4], [6,7,8,9,10], 0) == ([1,2,4], [6,7,10], 1), 'mask degen neighbors'
    assert analysis.tidy_profile([6,5,4,3],[2,5,3,4], 0) == ([3,4,5,6], [4,3,5,2], 2), 'levels in reverse order'
    assert analysis.tidy_profile([1,2,4,3,5], [6,1,4,2,9], 0) == ([1,2,3,4,5], [6,1,2,4,9], 8), 'levels out of order'

def test_interpolate_all(apiroot, apikey):

    # basics
    p = {'data': [[1,2,3,4,5], [10,20,30,40,50], [100,200,300,400,500]], 'data_info': [['temperature','pressure','salinity'],[],[]]}
    interpolated_profile = analysis.interpolate_all(p, [15,25,35,45])
    assert numpy.allclose(interpolated_profile['data'][0], [1.5,2.5,3.5,4.5]), 'interpolated temperature'
    assert numpy.allclose(interpolated_profile['data'][1], [15,25,35,45]), 'unchanged pressure'
    assert numpy.allclose(interpolated_profile['data'][2], [150,250,350,450]), 'interpolated salinity'

    # on a more realistic profile
    p = helpers.queryProfile('/argo', options={'id':'13857_068', 'data':'pressure,temperature'}, apikey=apikey, apiroot=apiroot)
    interp_p = analysis.interpolate_all(p[0], helpers.rg_levels())
    assert numpy.ma.allequal(interp_p.getvar('temperature')[0:5], numpy.ma.masked_array([None, None, 27.97794242903443, 27.969, 27.969], [True, True, False, False, False]))
    assert 'pressure' in interp_p.variable_names(), 'pressure should still be a variable after interpolation'
    assert numpy.ma.allequal(interp_p.getvar('pressure')[0:5], helpers.rg_levels()[0:5]), 'pressure should be unchanged by interpolation'


def test_queryGrid(apiroot, apikey):
    datagrid = helpers.queryGrid('/grids/rg09', options={'startDate': '2004-01-01T00:00:00Z', 'endDate': '2004-02-01T00:00:00Z', 'data':'rg09_temperature'}, apikey=apikey, apiroot=apiroot)

    assert datagrid.sizes['longitude'] == 20, 'test data has 20 longitude points'
    assert datagrid.sizes['latitude'] == 1, 'test data has 1 latitude point'
    assert numpy.allclose(datagrid['rg09_temperature'].sel(longitude=20.5).data, [-0.033,-0.076,-0.21,-0.593,-1.201,-1.598,-1.663,-1.569,-1.15,-0.603,-0.134,0.262,0.6,0.861,1.057,1.184,1.249,1.259,1.279,1.307,1.315,1.329,1.342,1.349,1.345,1.33,1.309,1.284,1.263,1.245,1.224,1.203,1.175,1.127,1.07,1.013,0.955,0.896,0.843,0.792,0.748,0.7,0.661,0.621,0.584,0.549,0.52,0.489,0.464,0.441,0.415,0.383,0.356,0.309,0.265,0.219,0.175,0.128]), 'should be able to extract expected data'

def test_sort_and_dedupe(apiroot, apikey):
    assert helpers.sort_and_dedupe([5,4,3,2,1]) == [1,2,3,4,5], 'should sort list'
    assert helpers.sort_and_dedupe([1,2,3,4,5]) == [1,2,3,4,5], 'shouldnt mess with already sorted list'
    assert helpers.sort_and_dedupe([1,2,2,3,4]) == [1,2,3,4], 'should remove duplicates'
    assert helpers.sort_and_dedupe([5,5,5,5]) == [5], 'should handle all duplicates'
    assert helpers.sort_and_dedupe([[1,2],[1,2],[2,5],[0,4]]) == [[0,4],[1,2],[2,5]], 'should sort by first element in list of lists'

def test_queryProfile(apiroot, apikey):
    profiles = helpers.queryProfile('/argo', options={'id': '13857_068', 'data':'pressure,temperature'}, apikey=apikey, apiroot=apiroot)

    assert profiles[0].longitude == -26.257, 'longitude should be -26.257'
    assert profiles[0].latitude == 3.427, 'latitude should be 3.427'
    assert profiles[0].timestamp == datetime.datetime(2022, 2, 1, 19, 3, 42, 2000), 'timestamp incorrect'
    assert profiles[0].variable_names() == ('pressure','temperature'), 'variable names incorrect'
    assert numpy.all(profiles[0].getvar('temperature')[0:5] == [28.021,28,27.969,27.969,27.969]), 'temperature data should be correct'
    
def test_build_dataset(apiroot,apikey):
    # on Profile objects:
    profiles = helpers.queryProfile('/cchdo', options={'startDate':'1996-01-01T00:00:00Z', 'endDate':'1997-01-01T00:00:00Z', 'data':'pressure,doxy'}, apikey=apikey, apiroot=apiroot)
    interp_profiles = [analysis.interpolate_all(p, helpers.rg_levels()) for p in profiles]
    ds = helpers.build_dataset(interp_profiles, helpers.rg_levels())
    assert len(interp_profiles) == ds.sizes['nprof']
    assert numpy.allclose(ds.isel(nprof=0)['doxy'].data, interp_profiles[0].getvar('doxy'), equal_nan=True), 'data shouldnt get mangled on conversion to dataset'

    # on raw docs:
    profiles = helpers.query('/cchdo', options={'startDate':'1996-01-01T00:00:00Z', 'endDate':'1997-01-01T00:00:00Z', 'data':'pressure,doxy'}, apikey=apikey, apiroot=apiroot)
    interp_profiles = [analysis.interpolate_all(p, helpers.rg_levels()) for p in profiles]
    ds = helpers.build_dataset(interp_profiles, helpers.rg_levels())
    assert len(interp_profiles) == ds.sizes['nprof']
    assert numpy.allclose(ds.isel(nprof=0)['doxy'].data, helpers.getvar('doxy', interp_profiles[0]), equal_nan=True), 'data shouldnt get mangled on conversion to dataset'

def test_setgetvar(apiroot, apikey):
    p = {'data': [[1,2,3,4,5], [10,20,30,40,50], [100,200,300,400,500]], 'data_info': [['temperature','pressure','salinity'],['units'],[['C','dbar','psu']]]}
    assert helpers.getvar('temperature', p) == [1,2,3,4,5], 'getvar should extract temperature'
    assert helpers.getvar('pressure', p) == [10,20,30,40,50], 'getvar should extract pressure'
    assert helpers.getvar('salinity', p) == [100,200,300,400,500], 'getvar should extract salinity'
    p = helpers.setvar(p, 'doxy', [6,7,8,9,10], ['umol/kg'])
    assert helpers.getvar('doxy', p) == [6,7,8,9,10], 'setvar should have successfully posted doxy'

def test_Profile(apiroot, apikey):
    p = helpers.query('/argo', options={'id': '13857_068', 'data':'pressure,temperature'}, apikey=apikey, apiroot=apiroot)[0]
    p = helpers.Profile(p)
    assert p.id == '13857_068', 'Profile should have correct id'
    assert p.longitude == -26.257, 'Profile should have correct longitude'
    assert p.latitude == 3.427, 'Profile should have correct latitude'
    assert p.timestamp == datetime.datetime(2022, 2, 1, 19, 3, 42, 2000), 'Profile should have correct timestamp'
    assert numpy.allclose(p.getvar('pressure')[0:5], [11.9, 17, 22.1, 27.200001, 32.299999]), 'getvar should extract pressure'
    assert numpy.allclose(p.getvar('temperature')[0:5], [28.021, 28, 27.969, 27.969, 27.969]), 'getvar should extract temperature'
    p.setvar('salinity', [100,200,300,400,500])
    assert numpy.allclose(p.getvar('salinity'), [100,200,300,400,500]), 'setvar should have successfully posted salinity'

def test_MLD_estimate(apiroot, apikey):
    x = [0,1,2,3,4,5,6,7]
    y = [6.25,2.25,0.25,0.25,2.25,6.25,12.25,20.25]

    root = analysis.MLD_estimate(x, y, threshold_delta=0.03, reference_pressure=3)
    pchip = scipy.interpolate.PchipInterpolator(x, y, extrapolate=False)
    assert numpy.isclose(pchip(root)[0], 0.28), 'MLD should be inverting pchip at the right point'

def test_MLD_mask(apiroot, apikey):
    x = [0,1,2,3,4,5,6,7]
    y = numpy.ma.masked_array([6.25,2.25,0.25,0.25,2.25,6.25,12.25,20.25], [False, False, False, False, False, False, False, True])

    root = analysis.MLD_estimate(x, y, threshold_delta=0.03, reference_pressure=3)
    pchip = scipy.interpolate.PchipInterpolator(x, y, extrapolate=False)
    assert numpy.isclose(pchip(root)[0], 0.28), 'MLD shouldnt be thrown off by a masked value far away'

def test_AOU_estimate(apiroot, apikey):
    SA = [34.7118, 34.8915, 35.0256, 34.8472, 34.7366, 34.7324]
    CT = [28.8099, 28.4392, 22.7862, 10.2262, 6.8272, 4.3236]
    p =  [10, 50, 125, 250, 600, 1000]
    lat =  [4, 4, 4, 4, 4, 4]
    long = [188, 188, 188, 188, 188, 188]
    
    potential_temperature = gsw.pt_from_CT(SA, CT)
    salinity = gsw.SP_from_SA(SA, p, long, lat)
    density = gsw.rho(SA, CT, p)
    oxygen = [0,0,0,0,0,0]

    ref, _ = analysis.AOU_estimate(SA, CT, p, long, lat, oxygen)

    # simple stationary test
    o2sol = gsw.O2sol(SA, CT, p, long, lat)
    O2_eq_umol_per_kg = o2sol * gsw.rho(SA, CT, p) / 1000

    assert numpy.allclose(ref, O2_eq_umol_per_kg)

def test_regional_mean_area_constant():
    lat = [0, 30, 60]
    lon = [10, 20]
    data = numpy.full((3, 2), 5.0)

    ds = xarray.Dataset(
        {"temp": (("latitude", "longitude"), data)},
        coords={"latitude": lat, "longitude": lon},
    )

    out = analysis.regional_mean(ds, form="area")
    assert numpy.isclose(out["temp"].item(), 5.0)


def test_regional_mean_area_manual():
    lat = numpy.array([0.0, 60.0])
    lon = numpy.array([0.0, 1.0])

    ds = xarray.Dataset(
        {"temp": (("latitude", "longitude"), numpy.array([[10.0, 11.0], [20.0, 21.0]]))},
        coords={"latitude": lat, "longitude": lon},
    )

    out = analysis.regional_mean(ds, form="area")

    weights = numpy.cos(numpy.deg2rad(lat))
    expected = (10.0 * weights[0] + 11.0 * weights[0] + 20.0 * weights[1] + 21.0 * weights[1]) / (2*weights[0] + 2*weights[1])

    assert numpy.isclose(out["temp"].item(), expected)


def test_regional_mean_meridional():
    lat = numpy.array([0.0, 60.0])
    lon = numpy.array([0.0, 1.0])

    ds = xarray.Dataset(
        {"temp": (("latitude", "longitude"), numpy.array([[10.0, 100.0], [20.0, 200.0]]))},
        coords={"latitude": lat, "longitude": lon},
    )

    out = analysis.regional_mean(ds, form="meridional")

    weights = numpy.cos(numpy.deg2rad(lat))
    expected = numpy.array([
        (10.0 * weights[0] + 20.0 * weights[1]) / (weights[0] + weights[1]),
        (100.0 * weights[0] + 200.0 * weights[1]) / (weights[0] + weights[1]),
    ])

    assert out["temp"].dims == ("longitude",)
    assert numpy.allclose(out["temp"].values, expected)


def test_regional_mean_zonal():
    lat = numpy.array([0.0, 60.0])
    lon = numpy.array([0.0, 1.0])

    ds = xarray.Dataset(
        {"temp": (("latitude", "longitude"), numpy.array([[10.0, 30.0], [20.0, 40.0]]))},
        coords={"latitude": lat, "longitude": lon},
    )

    out = analysis.regional_mean(ds, form="zonal")

    expected = numpy.array([20.0, 30.0]) # no weights since we're averaging inside individual latitude bands

    assert out["temp"].dims == ("latitude",)
    assert numpy.allclose(out["temp"].values, expected)


def test_regional_mean_multiple_variables():
    lat = [0.0, 60.0]
    lon = [0.0, 1.0]

    ds = xarray.Dataset(
        {
            "temp": (("latitude", "longitude"), numpy.array([[1.0, 2.0], [3.0, 4.0]])),
            "salt": (("latitude", "longitude"), numpy.array([[10.0, 11.0], [30.0, 31.0]])),
        },
        coords={"latitude": lat, "longitude": lon},
    )

    out = analysis.regional_mean(ds, form="area")

    weights = numpy.cos(numpy.deg2rad(lat))
    expected_temp = (1.0 * weights[0] + 2.0 * weights[0] + 3.0 * weights[1] + 4.0 * weights[1]) / (2*weights[0] + 2*weights[1])
    expected_salt = (10.0 * weights[0] + 11.0 * weights[0] + 30.0 * weights[1] + 31.0 * weights[1]) / (2*weights[0] + 2*weights[1])

    assert numpy.isclose(out["temp"].item(), expected_temp)
    assert numpy.isclose(out["salt"].item(), expected_salt)