# jPPA

This repository contains code which computes jPPA and probability of being on a bus between a person's GPS data and a bus's.
Participant data will have PPA computed for gaps between data points, as well as linear path interpolation (using a modified version of the Astar pathfinding algorithm) for gaps when points are too far apart for PPA to exist (vmax is set to 2m/s - approximate walking speed). 
- Note: points that are further apart than 5000m will not have the path that unites them interpolated. This can be changed by setting a different astar_threshold parameter in ppa_person from ppa_jppa_stable_trip_det.py.
Bus data will have its gaps filled with linear path interpolation along the road network graph.

Tasks:
1. review code in astar.py.
2. Test validity of the modified Astar pathfinding method. One way to do so is as follows:
	- get dataframe containing visit counts for each cell for minute-by-minute SenseDoc data
	- get dataframe containing visit counts for each cell for every 5th minute in the minute-by-minute SenseDoc data, with gaps filled in with Astar
	- compute entry-wise mean squared error

Bugs:
1. running ppa_jppa_stable_trip_det.py eventually runs in the situation in which a memory error is raised when computing the shortest path between two points that are far enough apart. To handle this problem:
	- address task 1
	- a threshold was imposed for how far points can be before the path between them is no longer computed.
2. running bus_work with ProcessPoolExecutor from concurrent.futures appears to run normally until a permission error is raised when trying to write a file in the 10-Victoria directory. 
	- unknown reason

Workflow -- after completing task 1!
- Set up pandas dataframe representing city using city_matrix from city_matrix.py and using shapefiles with geographical data in the LCC coordinate system (from StatCa)
- Set up networkx graph of city road network using pickle_utm_graph from street_network.py.
- Process all participant data (using pre_trip_detection.py) and save results in csv files on a per participant basis (path of directory where processed participant data will be saved is defined in the global_histogram function variable "path")
	- when using SenseDoc data, take every 5th entry in processed dataframe
	- global_histogram returns a dataframe containing the shape of the city, with entries as integers indicating how many times that cell was visited by study participants
- raw_histogram_processing from pre_trip_detection.py generates the dataframe that Astar will use to interpolate path -- this dataframe contains the costs of the Astar pathfinding algorithm stepping onto a certain cell in the city
- Use get_stable_trip_dwells.py to extract trips and dwells of all participants
- run_all_PPA.py can be used to compute PPA and path interpolation of participant data in a parallelized manner.
- Process all bus data with bus_df from bus_df_extraction.py.
- Use bus_work.py to fill in gaps in bus GPS data in a parallelized manner.
- (not tested on more than one trip at a time) Run jppa_per_trip from jppa_analysis.py to get probability of a participant being on a bus during a trip.
	- returns a one row dataframe for each participant's trip
	- merge rows with headers ['interact_id', 'trip_start_time', 'probability'] for all trips of a participant

