import pickle
import os

"""

NOTE: This will need to change as there are other options for specifying weight files
i.e., in order to support very large weight files 

"""

# Searches for weights that contain the name arg as a substring in the project
# Args: substring for search, project directory, weights filename (optional)
# Returns dictionary of potential matches with real name -> value
def find_params(name, proj_dir, weights_fname="weights.pkl"):
    weights_file = os.path.join(proj_dir, weights_fname)
    matches = dict()
    with open(weights_file, "rb") as fhand:
        weight_dict = pickle.load(fhand)
        for key in weight_dict:
            if name in key:
                matches[key] = weight_dict[key]
    return matches
