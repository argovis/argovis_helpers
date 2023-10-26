from argovisHelpers import gridtools
import numpy

class TestClass():
    def setUp(self):   
        return

    def tearDown(self):
        return

    def test_label_features_basic(self):
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