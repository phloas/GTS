# GST
Source codes for GST

## Required Packages:
Users need to install the exact trajectory similarity computing package for various trajectory similarity metrics named 'trajectory_distance' which can be downloaded from https://github.com/maikol-solis/trajectory_distance.

## Running Procedures:
1. Download data files from the following websites:  
    Geolife: https://www.microsoft.com/en-us/download/details.aspx?id=52367  
    Porto: https://archive.ics.uci.edu/ml/datasets/Taxi+Service+Trajectory+Prediction+Challenge+ECML+PKDD+2015
2. If you exploit Geolife or Porto dataset, you can use 'preprocess_geolife.py' or 'preprocess_porto.py' to obtain the data.
3. Run 'preprocessing.py' to preprocess the coordinate tuples and generate the ground truth distances between trajectories.
4. Run 'generate_graph.py' to get the graph.
5. Run 'generate_trajectory' to divide the datasets.
6. Run 'train.py' to train the GST. To test GST, you can load the saved model and use the 'matching_test' function in the 'train.py'. 