This folder will eventually include the functions required to simulate a segmented DM.
It is currently organized as follows:

- The .yaml files contain the configuration parameters corresponding to the [TN] used for the simulation
All data generated during a simulation is saved in a folder with foldername corresponding to the configuration [TN]
If the same configuration file is used, data is READ (not COMPUTED) directly from the .fits files in the corresponding folder.
Thus, if you wish to change a parameter in a configuration file after running the code, you should either create a new .yaml file or delete the
corresponding folder before running the simulation

- main_script.py is where all the fun is happening

- utilities.py contains some high-level wrapper functions for ease of use

- hexagonal_geometry.py contains a class definining the hexagonal segments' geometry (masks, initial actuator location, segment disposition)
  from the data read in the given .yaml configuration file associated to the [TN]

- matrix_calculator.py contains all the functions to deal build different matrices (Zernike, influence functions, reconstructor)

- deformable_mirror.py defined the deformable mirror (DM) class, inherited by both the segmented DM and the single segment

- segment_mirror.py contains the (single) Segment() class

- segmented_deformable_mirror.py contains the SegmentedMirror() class, building the mirror as an array of Segment() objects

- all other files are quite self explanatory (I'm sure you can figure out what read_configuration.py does)
