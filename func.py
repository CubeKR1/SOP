# This file contains all necessary functions that are used for the simulation of gravitational slingshots.
# Within are also functions that allow the plotting of the simulated data

import numpy as np
import matplotlib.pyplot as plt
import spiceypy as spice
import matplotlib.backends.backend_pdf as mpdf


def km_to_AU(distance): # Turns kilometers into Astronomical Units
    return distance / 149597871


def km_to_m(distance): # Turns kilometers into meters
    return distance * 1000


def AU_to_m(distance): # Turns Astronomical Units into meters
    return distance * 149597870700


def sscalc_escape(r): # Calculates Solar System escape velocity for given radius
    mu = 132712440042e9
    return np.sqrt((2 * mu) / r)


def max(arr): # Returns max value of object_pos/planets position list
    max_value = 0
    for n in range(len(arr)):
        for i in range(len(arr[n])):
            for s in range(len(arr[n, i])):
                if np.abs(arr[n, i, s]) > max_value:
                    max_value = np.abs(arr[n, i, s])
    return max_value


def times(arr_time, step): # Returns time array with step for chosen UTC min and max time
    time1 = spice.str2et(arr_time[0])
    time2 = spice.str2et(arr_time[1])
    return np.arange(time1, time2, step)


def planetid_to_string(id):
    dict = {
        1: "Mercury",
        2: "Venus",
        3: "Earth",
        4: "Mars",
        5: "Jupiter",
        6: "Saturn",
        7: "Uranus",
        8: "Neptune"
    }
    return dict[id]


def planetid_to_colorstr(id):
    dict = {
        1: "cornflowerblue",
        2: "goldenrod",
        3: "mediumblue",
        4: "orangered",
        5: "darkorange",
        6: "darkkhaki",
        7: "mediumturquoise",
        8: "lightskyblue"
    }
    return dict[id]


def mu(ids):
    mu = {
        -1: 132712440042e9,
        1: 22032e9,
        2: 324859e9,
        3: 398600e9,
        4: 42828e9,
        5: 126686531e9,
        6: 37931206e9,
        7: 5793951e9,
        8: 6835100e9
    }
    return [mu[n] for n in ids]


def get_obj_pos(ids, times, frame, initial_id): # updated planet data with np.arrays using SPICE kernels from NASA
    obj_pos = []
    for n in range(len(ids)): # for every given id of object
        # get initial data
        rs = spice.spkezr(spice.bodc2n(ids[n]), times, frame, 'NONE', spice.bodc2n(initial_id))
    
        # define cartesian [x, y]
        cart_pos = np.zeros([len(rs[0]), 2])
        for s in range(len(rs[0])):
            cart_pos[s, 0] = km_to_m(rs[0][s][0]) # x-pos
            cart_pos[s, 1] = km_to_m(rs[0][s][1]) # y-pos
        
        obj_pos.append(cart_pos)
    return np.array(obj_pos)


def get_obj_vel(ids, times, frame, initial_id):
    obj_vel = []
    for n in range(len(ids)):
        vs = spice.spkezr(spice.bodc2n(ids[n]), times, frame, 'NONE', spice.bodc2n(initial_id))

        cart_vel = np.zeros([len(vs[0]), 2])
        for s in range(len(vs[0])):
            cart_vel[s, 0] = km_to_m(vs[0][s][3]) # x-vel
            cart_vel[s, 1] = km_to_m(vs[0][s][4]) # y-vel

        obj_vel.append(cart_vel)
    return np.array(obj_vel)


def get_voy2(times, frame, initial_id): # Get ephemeride Voyager 2 data from SPICE Kernel
    rs = spice.spkezr(spice.bodc2n(-32), times, frame, 'NONE', spice.bodc2n(initial_id))
    data = np.zeros([len(rs[0]), 4])
    for n in range(len(rs[0])):
        data[n] = [km_to_m(rs[0][n][0]), km_to_m(rs[0][n][1]), km_to_m(rs[0][n][3]), km_to_m(rs[0][n][4])]
    return data


def get_voy2_pos(): # Get real Voyager 2 data from .txt-file
    with open("C:\VSCode\Python\Soyager\Files\LST_Files\helios_KzQCHADCZp.lst.txt", "r") as f:
        lines_after_1 = f.readlines()[1:]
    data = np.zeros([len(lines_after_1), 2])
    for line in range(len(lines_after_1)):
        data[line] = [AU_to_m(float(lines_after_1[line].split()[2])), AU_to_m(float(lines_after_1[line].split()[3]))]
    return data


def get_obj_index(data, n): # Returns list of planet positions for chosen time/index
    index_list = np.zeros([len(data), 2])
    for id in range(len(data)):
        index_list[id] = data[id, n]
    return index_list


def dist_to_sun(ids): # Returns the average distance of chosen planets to the sun
    sun_dict = { # Dictionary of average distances to sun from planets
        1: 58e9,
        2: 108e9,
        3: 150e9,
        4: 228e9,
        5: 779e9,
        6: 1434e9,
        7: 2873e9,
        8: 4495e9
    }
    return [sun_dict[ids[n]] for n in range(len(ids))]


def SOI(ids): # Returns SOIs for chosen planets
    M = 1.989e30 # Mass of Sun
    dists = dist_to_sun(ids)
    mass_dict = { # Dictionary of masses of planets
        1: 0.330e24,
        2: 4.87e24,
        3: 5.97e24,
        4: 0.642e24,
        5: 1898e24,
        6: 568e24,
        7: 86.8e24,
        8: 102e24
    }
    return [dists[n] * (mass_dict[ids[n]] / M)**0.4 for n in range(len(ids))] # List comprehension


def SOI_numcheck(ids, obj_data, ship_pos): # Checks for a spacecraft data point if it is inside a planet SOI
    checklist = []
    SOIs = SOI(ids)
    for n in range(len(ids)):
        ship2obj = abs(np.linalg.norm(ship_pos - obj_data[n]))
        if ship2obj <= SOIs[n]:
            checklist.append(True) # Spacecraft in planet SOI
        else:
            checklist.append(False) # Spacecraft not in planet SOI
    return checklist


def a(r, index_id, planets_points, ids): # Calculates spacecraft acceleration based on current SOI
    mu = {
        -1: 132712440042e9,
        1: 22032e9,
        2: 324859e9,
        3: 398600e9,
        4: 42828e9,
        5: 126686531e9,
        6: 37931206e9,
        7: 5793951e9,
        8: 6835100e9
    }
    if index_id != -1: # Spacecraft witin planet SOI
        #if ids[index_id] == 5 and abs(np.linalg.norm(r - planets_points[index_id])) < 10000000000:
            #print(index_id, abs(np.linalg.norm(r - planets_points[index_id])))
        return -mu[ids[index_id]] / abs(np.linalg.norm(r - planets_points[index_id]))**3 * (r - planets_points[index_id])
    else: # Spacecraft within Sun SOI
        return -mu[index_id] / abs(np.linalg.norm(r))**3 * r


# Euler method
def euler_point(r, v, n, index_id, h, planet_points, ids):
    v[n + 1] = v[n] + (h * a(r[n], index_id, planet_points, ids))
    r[n + 1] = r[n] + (h * v[n])


def generate_euler_points(checklist, r, v, n, h, planet_points, ids):
    for index_id in range(len(checklist)):
        if checklist[index_id]:
            euler_point(r, v, n, index_id, h, planet_points, ids)
            break
        else:
            euler_point(r, v, n, -1, h, planet_points, 0)


# Improved Euler Method / Heun's Method
def heun_point(r, v, n, index_id, h, planet_points, ids):
    kv1 = a(r[n], index_id, planet_points, ids)
    kr1 = v[n]
    kv2 = a(r[n] + kr1 * h, index_id, planet_points, ids)
    kr2 = v[n] + kv1 * h

    v[n + 1] = v[n] + ((h / 2) * (kv1 + kv2))
    r[n + 1] = r[n] + ((h / 2) * (kr1 + kr2))


def generate_heun_points(checklist, r, v, n, h, planet_points, ids):
    for index_id in range(len(checklist)):
        if checklist[index_id]:
            heun_point(r, v, n, index_id, h, planet_points, ids)
            break
        else:
            heun_point(r, v, n, -1, h, planet_points, 0)


# Runge-Kutta-4 Method
def rk4_point(r, v, n, index_id, h, planet_points, ids): # Calculates velocity and position numerically for spacecraft using RK4-method
    kv1 = a(r[n], index_id, planet_points, ids)
    kr1 = v[n]
    kv2 = a(r[n] + kr1 * h/2, index_id, planet_points, ids)
    kr2 = v[n] + kv1 * h/2
    kv3 = a(r[n] + kr2 * h/2, index_id, planet_points, ids)
    kr3 = v[n] + kv2 * h/2
    kv4 = a(r[n] + kr3 * h, index_id, planet_points, ids)
    kr4 = v[n] + kv3 * h

    v[n + 1] = v[n] + ((h / 6) * (kv1 + 2 * kv2 + 2 * kv3 + kv4))
    r[n + 1] = r[n] + ((h / 6) * (kr1 + 2 * kr2 + 2 * kr3 + kr4))


def generate_rk4_points(checklist, r, v, n, h, planet_points, ids): # Generates points for spacecraft based on SOI
    for index_id in range(len(checklist)):
        if checklist[index_id]:
            rk4_point(r, v, n, index_id, h, planet_points, ids) # New point in terms of planet that spacecraft is within SOI of
            break
        else:
            rk4_point(r, v, n, -1, h, planet_points, 0) # New point in terms of sun gravity


# Calculations and theory
def do_calculate(checklist, planet_centric, in_v, ids, obj_vel): # Calculating steps for theory
    print(f'Start: {checklist[0]}, End: {checklist[-1]}, Length: {len(obj_vel[0])}')
    periapsis = min([abs(np.linalg.norm(planet_centric[0, n])) for n in range(len(planet_centric[0]))])
    eccentricity = 1 + ((abs(np.linalg.norm(planet_centric[1, checklist[0]]))**2) * periapsis) / mu(ids)[0]
    deflection_angle = 2 * np.arcsin(1 / eccentricity)
    in_angle = np.arctan(planet_centric[1, checklist[0], 1] / planet_centric[1, checklist[0], 0])
    print(in_angle / (2 * np.pi) * 360)
    out_angle = in_angle - deflection_angle
    print(out_angle / (2 * np.pi) * 360)
    out_velprr = np.array([abs(np.linalg.norm(planet_centric[1, checklist[0]])) * np.cos(out_angle), abs(np.linalg.norm(planet_centric[1, checklist[0]])) * np.sin(out_angle)])
    out_velhrr = out_velprr + obj_vel[0, checklist[-1]]
    deltaV = abs(np.linalg.norm(out_velhrr)) - abs(np.linalg.norm(in_v))
    
    return deltaV


def theory_calculate(r, v, obj_pos, obj_vel, ids, voy2): # Only works if there is one planet-id
    # Setting up values
    planet_centric = np.zeros([2, len(v), 2])
    voy2_centric = np.zeros([2, len(voy2), 2])
    for n in range(len(planet_centric[0])):
        planet_centric[0, n] = r[n] - obj_pos[0, n]
        planet_centric[1, n] = v[n] - obj_vel[0, n]
        voy2_centric[0, n] = voy2[n, [0, 1]] - obj_pos[0, n]
        voy2_centric[1, n] = voy2[n, [2, 3]] - obj_vel[0, n]
    checklist_true = [index for index in range(len(planet_centric[0])) if abs(np.linalg.norm(planet_centric[0, index])) < SOI(ids)[0]]
    voy2_true = [index for index in range(len(voy2_centric[0])) if abs(np.linalg.norm(voy2_centric[0, index])) < SOI(ids)[0]]

    # Calculating steps for theory
    deltaV_theory_voy2 = do_calculate(voy2_true, voy2_centric, voy2[voy2_true[0], [2, 3]], ids, obj_vel)

    # Calculating observed
    deltaV_num = abs(np.linalg.norm(v[checklist_true[-1]])) - abs(np.linalg.norm(v[checklist_true[0]])) # NM-theory
    deltaV_planetcentric = abs(np.linalg.norm(planet_centric[1, checklist_true[-1]])) - abs(np.linalg.norm(planet_centric[1, checklist_true[0]]))
    deltaV_voy2 = abs(np.linalg.norm(voy2[voy2_true[-1], [2, 3]])) - abs(np.linalg.norm(voy2[voy2_true[0], [2, 3]]))
    deltaV_voy2_planetcentric = abs(np.linalg.norm(voy2_centric[1, voy2_true[-1]])) - abs(np.linalg.norm(voy2_centric[1, voy2_true[0]]))

    # Return chosen values
    return deltaV_num, deltaV_planetcentric, deltaV_voy2, deltaV_theory_voy2, deltaV_voy2_planetcentric


def plot_everything(voy2, object_pos, object_vel, r, v, h, ids): # Plots and saves data for spacecraft (velocity and position) and planets (position)
    #plt.style.use('dark_background')
    pdf = mpdf.PdfPages('numerical_plots.pdf')
    max_value = max(object_pos)

    # Plot trajectories
    fig1 = plt.figure(1, figsize=(6, 6))
    for n in range(len(object_pos)):
        plt.plot(object_pos[n, :,0], object_pos[n, :,1], color=f'{planetid_to_colorstr(ids[n])}') # Planets

    plt.plot(voy2[:,0], voy2[:,1], color='maroon', label='Voyager2')
    plt.plot(r[:,0], r[:,1], color='indigo', label='Spacecraft Trajectory') # Spacecraft

    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.xlim(-max_value, max_value)
    plt.ylim(-max_value, max_value)
    #plt.xlim(-6e11, -5e11)
    #plt.ylim(5e11, 6e11)
    plt.title('2D-trajectories of planets and spacecraft')
    plt.legend()
    
    # Plot velocity of spacecraft
    fig2 = plt.figure(2, figsize=(8, 5))
    velocity = np.zeros([len(v), 2])
    for q in range(len(v)):
        velocity[q] = [h * (q + 1), abs(np.linalg.norm(v[q]))]

    # Plot escape velocity of solar system in same figure
    escape = np.zeros([len(r), 1])
    for s in range(len(r)):
        escape[s] = [sscalc_escape(abs(np.linalg.norm(r[s])))]

    plt.plot(velocity[:,0], velocity[:,1], color='indigo', label='Spacecraft Velocity')
    plt.plot(velocity[:,0], [abs(np.linalg.norm(voy2[i, [2, 3]])) for i in range(len(voy2))], color='maroon', label='Voyager2 Velocity')
    plt.plot(velocity[:,0], escape[:,0], color='crimson', label='Solar System Escape Velocity')

    plt.ylabel("Y [m/s]")
    plt.title('Spacecraft Heliocentric Velocity')
    plt.legend()

    # Plot planetcentric velocities
    fig3 = plt.figure(3, figsize=(8, 5))
    difference = np.zeros(len(v))
    for planet in range(len(object_vel)):
        for n in range(len(difference)):
            difference[n] = abs(np.linalg.norm(v[n] - object_vel[planet, n]))
        plt.plot(velocity[:,0], difference[:], label=f'{planetid_to_string(ids[planet])}centric', color=f'{planetid_to_colorstr(ids[planet])}')

    plt.ylabel("Y [m/s]")
    plt.title('Spacecraft Planetcentric Velocities')
    plt.legend()

    # Show and save plot(s) as pdf
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
    pdf.close()
    plt.show()


def plot_all3(voy2, object_pos, r, v, h, ids):
    #plt.style.use('dark_background')
    max_value = max(object_pos)

    # Plot all three numerical methods and planets
    fig1 = plt.figure(1, figsize=(6, 6))

    # Plot planets
    for n in range(len(object_pos)):
        plt.plot(object_pos[n, :,0], object_pos[n, :,1], color=f'{planetid_to_colorstr(ids[n])}') # Planets
    plt.plot(voy2[:,0], voy2[:,1], color='maroon', label='Voyager2')

    # Euler Method
    for n in range(len(r) - 1):
        object_points = get_obj_index(object_pos, n)
        checklist = SOI_numcheck(ids, object_points, r[n])
        generate_euler_points(checklist, r, v, n, h, object_points, ids)
    plt.plot(r[:,0], r[:,1], label='Euler', color='blue')
    velocity = [[h * (q + 1), abs(np.linalg.norm(v[q]))] for q in range(len(v))]
    euler_velocity = np.array(velocity)

    # Heun's Method
    for n in range(len(r) - 1):
        object_points = get_obj_index(object_pos, n)
        checklist = SOI_numcheck(ids, object_points, r[n])
        generate_heun_points(checklist, r, v, n, h, object_points, ids)
    plt.plot(r[:,0], r[:,1], label='Heun', color='red')
    velocity = [[h * (q + 1), abs(np.linalg.norm(v[q]))] for q in range(len(v))]
    heun_velocity = np.array(velocity)

    # RK4 Method
    for n in range(len(r) - 1):
        object_points = get_obj_index(object_pos, n)
        checklist = SOI_numcheck(ids, object_points, r[n])
        generate_rk4_points(checklist, r, v, n, h, object_points, ids)
    plt.plot(r[:,0], r[:,1], label='RK4', color='purple')
    velocity = [[h * (q + 1), abs(np.linalg.norm(v[q]))] for q in range(len(v))]
    rk4_velocity = np.array(velocity)

    #plt.xlim(-max_value, max_value)
    #plt.ylim(-max_value, max_value)
    plt.xlim(-6.2e11, -5.6e11)
    plt.ylim(5e11, 5.6e11)
    plt.legend()

    # Plot velocities
    fig2 = plt.figure(2, figsize=(8, 5))

    plt.plot(euler_velocity[:,0], euler_velocity[:,1], label='Euler', color='blue')
    plt.plot(heun_velocity[:,0], heun_velocity[:,1], label='Heun', color='red')
    plt.plot(rk4_velocity[:,0], rk4_velocity[:,1], label='RK4', color='purple')
    plt.plot(rk4_velocity[:,0], [abs(np.linalg.norm(voy2[i, [2, 3]])) for i in range(len(voy2))], color='maroon', label='Voyager2 Velocity')

    plt.legend()
    plt.show()


#if __name__ == "__main__":
    