import os
import shutil


# Path to the source file
source_file = "../scripts/GMRES_IR.cpp"

# Path to the directory containing the header files
header_files_dir = "/root/home/Precimonious/tlapack/include"

# Path to the destination directory
destination_dir = "../scripts"

# List to store the imported header files
imported_files = []

# Read the source file
with open(source_file, "r") as file:
    lines = file.readlines()

    # Iterate over each line
    for line in lines:        
        if line.startswith("#include"):
            # Extract the imported file name
            imported_file = line.split()[1]

            # Check if the imported file ends with .hpp
            if imported_file.endswith(".hpp>"):
                # Add the imported file to the list
                imported_files.append(imported_file[1:-1])

# Copy the imported files to the destination directory
for imported_file in imported_files:
    source_path = os.path.join(header_files_dir, imported_file)
    destination_path = os.path.join(destination_dir, imported_file)
    if not os.path.exists(destination_path):
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        open(destination_path, 'a').close()
    shutil.copyfile(source_path, destination_path)