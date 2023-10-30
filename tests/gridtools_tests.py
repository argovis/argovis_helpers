from argovisHelpers import gridtools
import numpy, pytest

@pytest.fixture
def index_transform():
	def index2coords(longitudes, latitudes, index):
	    # index [lat_idx, lon_idx]; return [lon, lat]
	    lon = longitudes[index[1]] - 5./16.
	    if lon < -180:
	        lon += 360.

	    if index[0] == 0:
	        lat = -90
	    elif index[0] == 361:
	        lat = 90
	    else:
	        lat = latitudes[index[0]] - 0.25

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

# winding helpers ----------------------------------------

def test_isccw_basic():
	# basic test of winding checker

	loop = [[0,0],[1,0],[1,1],[0,1],[0,0]]
	assert gridtools.is_ccw_winding(loop)

def test_isccw_dateline():
	# winding checker on dateline

	loop = [[-179, 0], [179, 0], [179, 1], [-179, 1], [-179, 0]]
	assert not gridtools.is_ccw_winding(loop)