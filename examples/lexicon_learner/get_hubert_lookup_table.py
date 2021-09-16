#from dan lyth

import joblib
kmeans_model_path = '../../fairseq/examples/textless_nlp/gslm/speech2unit/pretrained_models/hubert/km100/hubert_km100.bin'
kmeans_model = joblib.load(open(kmeans_model_path, "rb")) # this is just a sklearn model
centroids = kmeans_model.cluster_centers_
