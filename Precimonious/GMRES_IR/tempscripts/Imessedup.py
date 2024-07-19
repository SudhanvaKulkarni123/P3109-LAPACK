import os

directory = "matrix_collection_999"

for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)):
        new_filename = filename.split(".")[0] + ".csv"
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))