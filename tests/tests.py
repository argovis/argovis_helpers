from helpers import helpers
import datetime

class TestClass():
    def setUp(self):   
        self.apiroot = 'http://api:8080'

        return

    def tearDown(self):
        return

    def test_argofetch(self):
        '''
        check basic behavior of argofetch
        '''

        profile = helpers.argofetch('/argo', options={'id': '4901283_021'}, apikey='', apiroot=self.apiroot)
        assert len(profile) == 1, 'should have returned exactly one profile'
        assert profile[0]['geolocation'] == { "type" : "Point", "coordinates" : [ -35.430227, 1.315393 ] }, 'fetched wrong profile'

    def test_polygon(self):
        '''
        make sure polygons are getting handled properly
        '''

        profile = helpers.argofetch('/argo', options={'polygon': [[-34,2],[-35,2],[-35,3],[-34,3],[-34,2]]}, apikey='', apiroot=self.apiroot)
        assert len(profile) == 1, 'polygon encompases exactly one profile'

    def test_data_inflate(self):
        '''
        check basic behavior of data_inflate
        '''

        data_doc = {
            'data': [[1,2,3],[4,5,6]],
            'data_keys': ['a','b','c']
        }
        inflate = helpers.data_inflate(data_doc)
        assert inflate == [{'a':1, 'b':2, 'c':3}, {'a':4, 'b':5, 'c':6}], f'simple array didnt inflate correctly, got {inflate}'

    def test_find_key(self):
        '''
        check basic behavior of find_key
        '''

        data = {'metadata': 'meta', 'a': 1, 'b':2, 'c':3}
        meta = {'_id': 'meta', 'a': 4, 'd':5}

        assert helpers.find_key('a', data, meta) == 1, 'find_key should select the entry from data_doc if key appears in both data and metadata'
        assert helpers.find_key('d', data, meta) == 5, 'find_key should look in meta doc'

    
    def test_parsetime(self):
        '''
        check basic behavior of parsetime
        '''

        datestring = '1999-12-31T23:59:59.999999Z'
        dtime = datetime.datetime(1999, 12, 31, 23, 59, 59, 999999)

        assert helpers.parsetime(datestring) == dtime, 'date string should have been converted to datetime.datetime'
        assert helpers.parsetime(helpers.parsetime(datestring)) == datestring, 'parsetime should be its own inverse'


    def test_query(self):
        '''
        check basic behavior of query
        '''

        response = helpers.query('/tc', options={'startDate': '1851-05-26T00:00:00Z', 'endDate': '1852-01-01T00:00:00Z'}, apiroot=self.apiroot)
        assert len(response) == 9, f'should be able to query entire globe for 6 months, with time divisions landing exactly on one timestamp, and get back 9 tcs, instead got {response}'
        
    def test_units_inflate(self):
        '''
        check basic behavior of units_inflate
        '''

        data = {'metadata': 'meta', 'data_keys': ['a', 'b', 'c']}
        meta = {'_id': 'meta', 'data_keys': ['a', 'd', 'c'], 'units': ['kg', 's', 'm']}
        units = helpers.units_inflate(data, meta) 

        assert units == {'a': 'kg', 'b': 's', 'c': 'm'}, f'failed to reconstruct units dict, got {units}'
