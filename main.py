# This script simulates slingshots for a spacecraft within the Solar System using multiple numerical methods for solving
# differential equations. The simulated data is then plotted with real planet ephemeride data generated by NASA to portray
# the effects of the slingshot maneuver used for spacecrafts like Voyager 2.

import numpy as np
import spiceypy as spice
import func as f # Functions created for this project is in this module

spice.furnsh('C:\VSCode\Python\SOP_Voyager2\ASC_FILES\simulation.mk') # Open metakernel (ephemeride data) from path

frame = 'ECLIPJ2000' # Data reference frame
step = 864 # Stepsize for data seconds: 86400 [s] = 1 [day]
initial_id = 10 # Sun reference
ids = [8] # Chosen planet ids where Mercury ID: 1 
utc = ['Jun 27, 1989', 'Nov 17, 1989'] # MMM DD, YYYY [start, end]

times = f.times(utc, step) # Defines data times

# Acquiring planet data
object_pos = f.get_obj_pos(ids, times, frame, initial_id)
object_vel = f.get_obj_vel(ids, times, frame, initial_id)
voy2 = f.get_voy2(times, frame, initial_id)

# Simulation
# Define arrays
r = np.zeros([len(times), 2])
v = np.zeros([len(times), 2])

# Ephemeride Voy2 Initial Data
r[0] = [voy2[0, 0], voy2[0, 1]]
v[0] = [voy2[0, 2], voy2[0, 3]]

#exit()

for n in range(len(r) - 1): # Fills arrays with data
    object_points = f.get_obj_index(object_pos, n) # Get planet points for the chosen index
    checklist = f.SOI_numcheck(ids, object_points, r[n]) # Check for Patched Conics
    f.generate_rk4_points(checklist, r, v, n, step, object_points, ids) # Generate points through simulation [euler, heun, rk4]

# Plotting and saving data
#f.plot_everything(voy2, object_pos, object_vel, r, v, step, ids)

#exit()

# Calculating and comparing theory: Only works with one planet id where spacecraft is within SOI
calculations = f.theory_calculate(r, v, object_pos, object_vel, ids, voy2)
print(f'''
    {'NM_Delta-V:':<25}{round(calculations[0], 1):>7}
    {'NM_Planetcentric_Delta-V:':<25}{round(calculations[1], 1):>7}
    {'R_Delta-V:':<25}{round(calculations[2], 1):>7}
    {'R_Theoretical_Delta-V:':<25}{round(calculations[3], 1):>7}
    {'R_Planetcentric_Delta-V:':<25}{round(calculations[4], 1):>7}
    ''') # Prints theoretical and simulated datapoints for analysis