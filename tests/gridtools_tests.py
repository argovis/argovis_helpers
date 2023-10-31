from argovisHelpers import gridtools
import numpy, pytest, copy
from functools import partial

@pytest.fixture
def index_transform():
	def index2coords(longitudes, latitudes, index):
	    # index [lat_idx, lon_idx]; return [lon, lat]
	    lon = longitudes[index[1]] - 22.5
	    if lon < -180:
	        lon += 360.

	    if index[0] == 0:
	        lat = -90
	    elif index[0] == 8:
	        lat = 90
	    else:
	        lat = latitudes[index[0]] - 11.25

	    return [lon, lat]

	return index2coords

# basic feature labeling --------------------------------------------------

def test_label_features_basic():
	# simple check for two distinct regions
	binary_features = [
		[0,0,0,0,0,0,0,0],
		[0,1,1,0,0,0,0,0],
		[0,1,1,0,0,0,0,0],
		[0,0,0,1,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,1,1,0],
		[0,0,0,0,0,1,0,0],
		[0,0,0,0,0,0,0,0]
	]

	correct_labels = numpy.array([
		[0,0,0,0,0,0,0,0],
		[0,1,1,0,0,0,0,0],
		[0,1,1,0,0,0,0,0],
		[0,0,0,1,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,2,2,0],
		[0,0,0,0,0,2,0,0],
		[0,0,0,0,0,0,0,0]
	])

	labeled_map = gridtools.label_features(binary_features)
	assert correct_labels.tolist() == labeled_map.tolist()

def test_label_features_first_pole():
	# check pole features are getting grouped by default
	binary_features = [
		[0,1,1,0,0,1,0,0],
		[0,1,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	]

	correct_labels = numpy.array([
		[0,1,1,0,0,1,0,0],
		[0,1,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	])

	labeled_map = gridtools.label_features(binary_features)
	assert correct_labels.tolist() == labeled_map.tolist()

def test_label_features_first_pole_no_connection():
	# check pole features are not grouped when grouping turned off
	binary_features = [
		[0,1,1,0,0,1,0,0],
		[0,1,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	]

	correct_labels = numpy.array([
		[0,1,1,0,0,2,0,0],
		[0,1,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	])

	labeled_map = gridtools.label_features(binary_features, connected_poles=False)
	assert correct_labels.tolist() == labeled_map.tolist()

def test_label_features_last_pole():
	# check pole features are getting grouped by default
	binary_features = [
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,1,0,0,0,0,0,0],
		[0,1,0,0,0,1,0,0]
	]

	correct_labels = numpy.array([
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,1,0,0,0,0,0,0],
		[0,1,0,0,0,1,0,0]
	])

	labeled_map = gridtools.label_features(binary_features)
	assert correct_labels.tolist() == labeled_map.tolist()

def test_label_features_last_pole_no_connection():
	# check pole features are not grouped when grouping turned off
	binary_features = [
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,1,0,0,0,0,0,0],
		[0,1,0,0,0,1,0,0]
	]

	correct_labels = numpy.array([
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,1,0,0,0,0,0,0],
		[0,1,0,0,0,2,0,0]
	])

	labeled_map = gridtools.label_features(binary_features, connected_poles=False)
	assert correct_labels.tolist() == labeled_map.tolist()

def test_label_features_dateline():
	# check dateline features are getting grouped by default
	binary_features = [
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[1,1,0,0,0,0,0,0],
		[1,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,1],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	]

	correct_labels = numpy.array([
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[1,1,0,0,0,0,0,0],
		[1,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,1],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	])

	labeled_map = gridtools.label_features(binary_features)
	assert correct_labels.tolist() == labeled_map.tolist()

def test_label_features_dateline_no_connection():
	# check dateline features are not grouped when grouping turned off
	binary_features = [
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[1,1,0,0,0,0,0,0],
		[1,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,1],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	]

	correct_labels = numpy.array([
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[1,1,0,0,0,0,0,0],
		[1,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,2],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	])

	labeled_map = gridtools.label_features(binary_features, periodic_dateline=False)
	assert correct_labels.tolist() == labeled_map.tolist()

def test_label_features_dateline_first_pole():
	# check dateline features grouped correctly at first pole
	binary_features = [

		[1,1,0,0,0,0,0,0],
		[1,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,1],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	]

	correct_labels = numpy.array([

		[1,1,0,0,0,0,0,0],
		[1,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,1],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	])

	labeled_map = gridtools.label_features(binary_features)
	assert correct_labels.tolist() == labeled_map.tolist()

def test_label_features_dateline_last_pole():
	# check dateline features are not grouped when grouping turned off
	binary_features = [
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[1,1,0,0,0,0,0,0],
		[1,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,1]

	]

	correct_labels = numpy.array([
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[1,1,0,0,0,0,0,0],
		[1,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,2]
	])

	labeled_map = gridtools.label_features(binary_features, periodic_dateline=False)
	assert correct_labels.tolist() == labeled_map.tolist()

# holes ------------------------------------------------------

def test_label_holes_basic():
	# simple check for two distinct regions
	binary_features = [
		[0,0,0,0,0,0,0,0],
		[0,1,1,1,0,0,0,0],
		[0,1,0,1,0,0,0,0],
		[0,1,1,1,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	]

	correct_labels = numpy.array([
		[0,0,0,0,0,0,0,0],
		[0,1,1,1,0,0,0,0],
		[0,1,0,1,0,0,0,0],
		[0,1,1,1,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	])

	labeled_map = gridtools.label_features(binary_features)
	assert correct_labels.tolist() == labeled_map.tolist()

def test_label_holes_external_diagonal():
	# diagonally connected external hole
	binary_features = [
		[0,0,0,0,0,0,0,0],
		[0,1,1,1,0,0,0,0],
		[0,1,1,1,0,0,0,0],
		[0,1,0,1,0,0,0,0],
		[0,1,1,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	]

	correct_labels = numpy.array([
		[0,0,0,0,0,0,0,0],
		[0,1,1,1,0,0,0,0],
		[0,1,1,1,0,0,0,0],
		[0,1,0,1,0,0,0,0],
		[0,1,1,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	])

	labeled_map = gridtools.label_features(binary_features)
	assert correct_labels.tolist() == labeled_map.tolist()

def test_label_holes_internal_diagonal():
	# diagonally connected internal hole
	binary_features = [
		[0,0,0,0,0,0,0,0],
		[0,1,1,1,1,0,0,0],
		[0,1,1,0,1,0,0,0],
		[0,1,0,1,1,0,0,0],
		[0,1,1,1,1,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	]

	correct_labels = numpy.array([
		[0,0,0,0,0,0,0,0],
		[0,1,1,1,1,0,0,0],
		[0,1,1,0,1,0,0,0],
		[0,1,0,1,1,0,0,0],
		[0,1,1,1,1,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	])

	labeled_map = gridtools.label_features(binary_features)
	assert correct_labels.tolist() == labeled_map.tolist()

def test_label_holes_dateline():
	# hole on the dateline
	binary_features = [
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[1,1,0,0,0,0,1,1],
		[0,1,0,0,0,0,1,0],
		[1,1,0,0,0,0,1,1],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	]

	correct_labels = numpy.array([
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[1,1,0,0,0,0,1,1],
		[0,1,0,0,0,0,1,0],
		[1,1,0,0,0,0,1,1],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	])

	labeled_map = gridtools.label_features(binary_features)
	assert correct_labels.tolist() == labeled_map.tolist()

def test_label_holes_firstpole():
	# hole on the low-index pole
	binary_features = [
		[0,0,1,0,1,0,0,0],
		[0,0,1,1,1,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	]

	correct_labels = numpy.array([
		[0,0,1,0,1,0,0,0],
		[0,0,1,1,1,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	])

	labeled_map = gridtools.label_features(binary_features)
	assert correct_labels.tolist() == labeled_map.tolist()

def test_label_holes_lastpole():
	# hole on the high-index pole
	binary_features = [
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,1,1,1,0],
		[0,0,0,0,1,0,1,0]
	]

	correct_labels = numpy.array([
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,1,1,1,0],
		[0,0,0,0,1,0,1,0]
	])

	labeled_map = gridtools.label_features(binary_features)
	assert correct_labels.tolist() == labeled_map.tolist()

def test_label_holes_internal_island():
	# diagonally connected feature inside hole in larger feature
	binary_features = [
		[0,0,0,0,0,0,0,0],
		[0,1,1,1,1,1,0,0],
		[0,1,0,0,1,1,0,0],
		[0,1,0,1,0,1,0,0],
		[0,1,0,0,0,1,0,0],
		[0,1,1,1,1,1,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	]

	correct_labels = numpy.array([
		[0,0,0,0,0,0,0,0],
		[0,1,1,1,1,1,0,0],
		[0,1,0,0,1,1,0,0],
		[0,1,0,1,0,1,0,0],
		[0,1,0,0,0,1,0,0],
		[0,1,1,1,1,1,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	])

	labeled_map = gridtools.label_features(binary_features)
	assert correct_labels.tolist() == labeled_map.tolist()

# shape tracing -------------------------------------------

def isCircular(arr1, arr2, reverse=False):
	a1 = copy.deepcopy(arr1)
	a2 = copy.deepcopy(arr2)
	if reverse:
		a1.reverse()

	if len(a1) != len(a2):
		return False

	str1 = ' '.join(map(str, a1))
	str2 = ' '.join(map(str, a2))
	if len(str1) != len(str2):
		return False

	return str1 in str2 + ' ' + str2

def test_trace_shape_basic():
	# simple check for two distinct regions
	binary_features = [
		[0,0,0,0,0,0,0,0],
		[0,1,1,0,0,0,0,0],
		[0,1,1,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,1,1,0],
		[0,0,0,0,0,1,0,0],
		[0,0,0,0,0,0,0,0]
	]
	labeled_map = gridtools.label_features(binary_features)
	
	correct_vertexes = [[1,1],[2,1],[3,1],[3,2],[3,3],[2,3],[1,3],[1,2],[1,1]] # should be CCW
	vertexes = gridtools.trace_shape(labeled_map, 1, nlatsteps=8)[0]
	assert isCircular(correct_vertexes[:-1], vertexes[:-1]) # note slice off the last polygon-closing vertex, will be arbitraily different depending on starting point

	correct_vertexes = [[5,5],[6,5],[7,5],[7,6],[6,6],[6,7],[5,7],[5,6],[5,5]]
	vertexes = gridtools.trace_shape(labeled_map, 2, nlatsteps=8)[0]
	assert isCircular(correct_vertexes[:-1], vertexes[:-1]) 

def test_trace_shape_dateline():
	# trace a shape bridging the dateline
	binary_features = [
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[1,0,0,0,0,0,0,1],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	]
	labeled_map = gridtools.label_features(binary_features)

	correct_vertexes = [[3,0],[3,7],[4,7],[4,0],[4,1],[3,1],[3,0]]
	vertexes = gridtools.trace_shape(labeled_map, 1, nlatsteps=8)[0]
	assert isCircular(correct_vertexes[:-1], vertexes[:-1])

# geojson generation ------------------------------------------

def test_generate_geojson_first_pole(index_transform):
	# generate geojson for pathological shape at pole
	binary_features = [
		[0,0,1,0,0,1,0,0],
		[0,0,1,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0],
		[0,0,0,0,0,0,0,0]
	]
	labeled_map = gridtools.label_features(binary_features)

	geo = gridtools.generate_geojson(labeled_map, 1, partial(index_transform,[0,45,90,135,180,225,270,315],[-90,-67.5,-45,-22.5,0,22.5,45,67.5,90]))
	correct_geo = ({'type': 'MultiPolygon', 'coordinates': [[[[67.5, -90], [112.5, -90], [112.5, -56.25], [67.5, -56.25], [67.5, -90]]], [[[202.5, -90], [247.5, -90], [247.5, -78.75], [202.5, -78.75], [202.5, -90]]]]}, {'first_pole'})
	print(geo)
	print(correct_geo)
	# correct_geo logic:
	# latitudes: top bound is -90, bottom of first row is half a latitude step, bottom of second row is 1.5 lat steps
	# longitudes: center of third column is 90; center of 6th is 225
	assert geo == correct_geo















