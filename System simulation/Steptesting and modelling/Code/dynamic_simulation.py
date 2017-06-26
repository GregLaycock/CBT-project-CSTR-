from math import pi, exp
from scipy.optimize import fsolve
import numpy
from steady_state_values import steady_state

# A+B-C
# Propylene Oxide+ H2O ---> Propylene glycol
# Methanol-Solvent

# Parameters (do not change these unless specifically asked)
Ko = 2.941e8  # m^3/(s*kmol)
Ea = 75360  # KJ/kmol
R = 8.313  # KJ/(kmol K)
rho_a = 801.2  # kg/m^3
rho_b = 966.2  # kg/m^3
rho_c = 1004   # kg/m^3
rho_solvent = 764.6  # kg/m^3
rhocw = 966.2  # kg/m^3 Water density
MM_A = 58.1  # kg/kmol
MM_B = 18  # kg/kmol
MM_C = 76.09  # kg/kmol
Cv1 = 3.089e-3  # m^2
alpha1 = 2
delta_Pv1 = 703.6  # Pa
delta_Pv3 = 1000  #Pa
alpha3 = 2
Cv2 = 1.33e-3  # m^2
Cv3 = (2.360e-3/(2**(-0.5)))/(numpy.sqrt(delta_Pv3/966.2))  #m^2
alpha2 = 2
g = 9.807  # m/s^2
tau_v1 = 36
zeta_v1 = 0.35
Kv1 = 1/80  # 1/Kpa
tau_v2 = 36
tau_v3 = 36
Kv2 = 1/80  # 1/Kpa
Kv3 = 1/80
Cp1 = 2.522  # KJ/kg degC
Cp2 = 4.187  # KJ/kg degC
Cp3 = 2.531  # KJ/kg degC
Cpcw = 4.187  # BTU/lbm/degF
delta_H = -84660  # Kj/kmol A
Uj = 4.750  # kW/(m^2 degC)
D = 1  # m
U_air = 44.5/1000  # kW/(m^2 degC)
L = 1.5*D  # m
A = (pi/4)*(D**2)  # m^2
To = 21.1  # degC
theta = 90  # s
Vj = ((pi/4)*(D*1.1)**2 - A)*L/3


def differential_equations(x):
    """
    This function will evaluate all the differential equation in the system
    :param x : (list)
                all the parameters, inputs and variables which are required by the differential equations
    :return: (list)
             differential of all the equations
    """

    # the parameters, inputs and variables
    F1, F2, F3,F_cw, Cao, Cbo, Ca, Cb, Cc, Na, V, rho1, rho2, rho3, z, xv1, xv2, xv3, Ps1, Ps2,Ps3, Tao, Tbo, T, Qj_tot, Qj1, Qj2, \
    Qj3, Tcwo, Tcw1, Tcw2, Tcw3, Qair1, Qair2, Qair3 = x

    # Differential equation from equation 1, 2, 3 and 6
    dVCa_dt = F1*Cao - (F3*Ca + Na*V)
    dVCb_dt = F2*Cbo - (F3*Cb + Na*V)
    dVCc_dt = Na*V - F3*Cc
    dVrho3_dt = rho1*F1 + rho2*F2 - (rho3*F3)

    # Differential equation from equation 19 with the second order differential rewritten as 2 first order differentials
    dz_dt = (-(2*zeta_v1*tau_v1*z + xv1) + Kv1*(Ps1 - 20))/tau_v1**2
    dxv1_dt = z

    # Differential equation from equation 20  (1st order valve)
    dxv2_dt = (Kv2*(Ps2 - 20) - xv2)/tau_v2
    dxv3_dt = (Kv3*(Ps3 - 20) - xv3)/tau_v3
    # Differential equation from equation 21
    drho3VT_dt = (rho1*F1*Cp1*Tao + rho2*F2*Cp2*Tbo - delta_H*Na*V - (rho3*F3*Cp3*T + Qj_tot))/Cp3

    # Differential equation from equation 22, 23 and 24
    dTcw1_dt = (F_cw*rhocw*Cpcw*Tcwo + Qj1 - (F_cw*rhocw*Cpcw*Tcw1 + Qair1))/(rhocw * Vj * Cpcw)
    dTcw2_dt = (F_cw*rhocw*Cpcw*Tcw1 + Qj2 - (F_cw*rhocw*Cpcw*Tcw2 + Qair2))/(rhocw * Vj * Cpcw)
    dTcw3_dt = (F_cw*rhocw*Cpcw*Tcw2 + Qj3 - (F_cw*rhocw*Cpcw*Tcw3 + Qair3))/(rhocw * Vj * Cpcw)

    return [dVCa_dt, dVCb_dt, dVCc_dt, dVrho3_dt, dz_dt, dxv1_dt, dxv2_dt,dxv3_dt, drho3VT_dt, dTcw1_dt, dTcw2_dt, dTcw3_dt]


# def calculate_new_states(x, old_values, step_size):
#     """
#     The new states are calculated from the state values in the previous time step and the differential of the current
#      time step and the integration step size
#     :param x: (list)
#               differentials of all the states
#
#     :param old_values: (list)
#                         states of the previous time step
#
#     :param step_size: (float)
#                       integration step size
#
#     :return: (list)
#               newly calculated states
#     """
#     # annotate the differentials and previous time step values to variables
#     dVCa_dt, dVCb_dt, dVCc_dt, dVrho3_dt, dz_dt, dxv1_dt, dxv2_dt, drho3VT_dt, dTcw1_dt, dTcw2_dt, dTcw3_dt = x
#     VCa, VCb, VCc, Vrho3, z, xv1, xv2, rho3VT, Tcw1, Tcw2, Tcw3 = old_values
#
#     # calculate the new states by (new_state = old_state + differential_state*step_size
#     VCa_new = VCa + dVCa_dt*step_size
#     VCb_new = VCb + dVCb_dt*step_size
#     VCc_new = VCc + dVCc_dt*step_size
#     Vrho3_new = Vrho3 + dVrho3_dt*step_size
#     z_new = z + dz_dt*step_size
#     xv1_new = xv1 + dxv1_dt*step_size
#     xv2_new = xv2 + dxv2_dt*step_size
#     rho3VT_new = rho3VT + drho3VT_dt*step_size
#     Tcw1_new = Tcw1 + dTcw1_dt*step_size
#     Tcw2_new = Tcw2 + dTcw2_dt*step_size
#     Tcw3_new = Tcw3 + dTcw3_dt*step_size
#
#     return [VCa_new, VCb_new, VCc_new, Vrho3_new, z_new, xv1_new, xv2_new, rho3VT_new, Tcw1_new, Tcw2_new, Tcw3_new]


def simultaneous_states(variables_to_solve, states):
    """
    The states in the differential equations are products of multiple variables
    This function contains residual equations (LHS - RHS = error) to be used in fsolve to simultaneous solve for all the variables which
    which make up all the states

    :param variables_to_solve: (list)
                                variables to be solved

    :param states: (list)
                    list of all the newly calculated states

    :return: (list)
              the residuals of all the equations
    """

    # annotate the variables to solve and states to variables
    Ca, Cb, Cc, V, rho3, T = variables_to_solve
    VCa, VCb, VCc, Vrho3, rho3VT,  = states

    # residual equations
    eq1 = VCa - Ca*V
    eq2 = VCb - Cb*V
    eq3 = VCc - Cc*V
    eq4 = Vrho3 - rho3*V
    eq5 = rho3VT - T*rho3*V

    # this equation is the residual of rho3
    # to calculate rho3 equations 9, 12, 13 and 14 was pre-solved using sympy
    eq6 = Ca * MM_A - (Ca * MM_A * rho_solvent) / rho_a + Cb * MM_B - (Cb * MM_B * rho_solvent) / rho_b + Cc * MM_C - \
          (Cc * MM_C * rho_solvent) / rho_c + rho_solvent - rho3

    return [eq1, eq2, eq3, eq4, eq5, eq6]


def states(x, old):
    """
    Calculate all the state variables using fsolve

    :param x: (list)
              newly calculated states

    :param old: (list)
                all the previous time step's state variables (used as the initial guess in fsolve)

    :return: (list)
             calculated state variables
    """

    # annotate states and previous time step's state variables to variables
    VCa, VCb, VCc, Vrho3, rho3VT  = x
    Ca_old, Cb_old, Cc_old, V_old, rho3_old, T_old = old

    # calculate the state variables using fsolve
    Ca, Cb, Cc, V, rho3, T = fsolve(simultaneous_states, [Ca_old, Cb_old, Cc_old, V_old, rho3_old, T_old],
                                    args=([VCa, VCb, VCc, Vrho3, rho3VT]))

    return [Ca, Cb, Cc, V, rho3, T]


def algebraic_equations_kinetics(x):
    """
    Equations used to calculate the kinetic parameters

    :param x: (list)
               variables needed to solve the equations

    :return: (list)
             kinetic parameters
    """

    Ca, Cb, T = x

    # Equations 4 and 5
    K = Ko*exp(-Ea/(R*(T + 273)))
    Na = K*Ca*Cb

    return [K, Na]


def algebraic_equations_density(x):
    """
    Equations used to calculate the densities and mass fractions

    :param x:  (list)
               variables needed to solve the equations

    :return: (list)
             calculated densities and mass fractions
    """

    Cao, Cbo, Ca, Cb, Cc, rho3 = x

    # mass fractions of stream 1 and 2
    # Equations 10 and 11
    rao = 1/(rho_solvent*(1/(Cao*MM_A) - 1/rho_a + 1/rho_solvent))
    rbo = 1/(rho_solvent*(1/(Cbo*MM_B) - 1/rho_b + 1/rho_solvent))

    # densities of stream 1 and stream 2
    # Equations 7 and 8
    rho1 = 1/(rao/rho_a + (1 - rao)/rho_solvent)
    rho2 = 1/(rbo/rho_b + (1 - rbo)/rho_solvent)

    # mass fractions of components in the reactor
    # Equations 12, 13, and 14 (these equations are not strictly necessary to be solved as they are not used in any
    # equation note that these equations were already used to solve rho3 but are calculated here for the value to be
    # known)
    ra = Ca*MM_A/rho3
    rb = Cb*MM_B/rho3
    rc = Cc*MM_C/rho3
    rsolvent = 1 - (ra + rb + rc)

    return [rao, rbo, rho1, rho2, ra, rb, rc, rsolvent]


def algebraic_equations_h(x):
    """
    calculate the reactor height

    :param x: (list)
               variables needed to solve the equations

    :return: (list)
             liquid level in  reactor
    """

    V, = x

    # Equation 32
    h = V/A

    return [h]


def algebraic_equations_valve(x):
    """
    Calculate the flow through the valves

    :param x: (list)
               variables needed to solve the equations

    :return: (list)
             volumetric flow through valves as well as pressure drop over valve 3
    """

    xv1, xv2, xv3, rho2, rho3,rhocw, h = x

    # Equation 16
    F2 = Cv1*alpha1**(xv1 - 1)*(delta_Pv1/rho2)**0.5

    # Equations 17 and 18
    delta_Pv2 = rho3*g*h
    F3 = Cv2*alpha2**(xv2 - 1)*(delta_Pv2/rho3)**0.5
    F_cw = Cv3*alpha3**(xv3 - 1)*(delta_Pv3/rhocw)**0.5
    return [F2, F3,F_cw, delta_Pv2]


def algebraic_equations_section_height(x):
    """
    Calculate the liquid height as seen from each of the segments

    :param x: (list)
               variables needed to solve the equations

    :return: (list)
             liquid height as seen from each of the segment
    """

    h, = x

    # Equations 33, 34 and 35
    if h - L/3 <= 0:
        h1 = h
    else:
        h1 = L/3

    if h - 2*L/3 <= 0 and h > L/3:
        h2 = h - L/3
    elif h <= L/3:
        h2 = 0
    else:
        h2 = L/3

    if h - L <= 0 and h > 2*L/3:
        h3 = h - 2*L/3
    else:
        h3 = 0

    return [h1, h2, h3]


def algebraic_equations_heat_transfer(x):
    """
    Calculate the heat transfer between the cooling water, reaction mixture and air

    :param x: (list)
               variables needed to solve the equations

    :return: (list)
             Heat transfer rates
    """
    h1, h2, h3, T, Tcw1, Tcw2, Tcw3 = x

    # Equations 25, 26, 27, 28, 29 and 30
    Qj1 = (Uj*pi*D*(h1*(T - Tcw1)))
    Qj2 = (Uj*pi*D*(h2 * (T - Tcw2)))
    Qj3 = (Uj*pi*D*(h3 * (T - Tcw3)))
    Qj_tot = Qj1 + Qj2 + Qj3
    Qair1 = -(U_air*pi*D*((L / 3 - h1)*(To - Tcw1)))
    Qair2 = -(U_air*pi*D*((L / 3 - h2)*(To - Tcw2)))
    Qair3 = -(U_air*pi*D*((L / 3 - h3)*(To - Tcw3)))

    return [Qj1, Qj2, Qj3, Qair1, Qair2, Qair3, Qj_tot]


def Cc_with_dead_time(t, cc, tspan,theta):
    """
    calculate the delayed measured concentration of C

    :param t: (float)
              time of current step

    :param cc: (array)
               array containing concentration of C in the reactor at each of the time steps

    :param tspan: (array)
                  array containing the time corresponding to a particular time step

    :return: (float)
             measured concentration of C
    """
    return numpy.interp(t - theta, tspan, cc)


# def single_iteration(inputs, previous, step_size, t, n, Cc_list, tspan):
#     """
#     Calculate the variables for a single time step
#
#     :param inputs: (list)
#                    all the exogenous inputs
#
#     :param previous: (list)
#                      values of the previous time step
#
#     :param step_size: (float)
#                       step size
#
#     :param t: (float)
#               time of current iteration
#
#     :param n: (int)
#               the step number
#
#     :param Cc_list: (array)
#                     stored values of the C concentration in the reactor
#
#     :param tspan: (array)
#                   array containing the time corresponding to a particular time step
#
#     :return: (list)
#               all calculated variables
#     """
#     # exogenous inputs
#     F1, Cao, Cbo, Ps1, Ps2, Tao, Tbo, Fcw, Tcwo = inputs
#
#     # values of the previous time step
#     F2, F3, Ca, Cb, Cc, Na, V, rho1, rho2, rho3, z, xv1, xv2, T, Qj_tot, Tcw1, Tcw2, Tcw3, Qj1, Qj2, Qj3, Qair1, Qair2, Qair3, rao, rbo, ra, rb, rc, rsolvent, h1, h2, h3, h, delta_Pv2, _ = previous
#
#     # list of all the variables needed in differential_equations
#     diff_eq_inputs = [F1, F2, F3, Cao, Cbo, Ca, Cb, Cc, Na, V, rho1, rho2, rho3, z, xv1, xv2, Ps1, Ps2, Tao, Tbo, T,
#                       Qj_tot, Qj1, Qj2, Qj3, Fcw, Tcwo, Tcw1, Tcw2, Tcw3, Qair1, Qair2, Qair3]
#
#     # calculated differentials
#     dVCa_dt, dVCb_dt, dVCc_dt, dVrho3_dt, dz_dt, dxv1_dt, dxv2_dt, drho3VT_dt, dTcw1_dt, dTcw2_dt, dTcw3_dt = \
#         differential_equations(diff_eq_inputs)
#
#     # list of all the previous time step's differentials to calculate the new states
#     new_states_inputs_x = [dVCa_dt, dVCb_dt, dVCc_dt, dVrho3_dt, dz_dt, dxv1_dt, dxv2_dt, drho3VT_dt, dTcw1_dt,
#                            dTcw2_dt, dTcw3_dt]
#
#     # list containing all the previous time step's state variables
#     new_states_inputs_old_values = [V*Ca, V*Cb, V*Cc, V*rho3, z, xv1, xv2, rho3*V*T, Tcw1, Tcw2, Tcw3]
#
#     # newly calculated states
#     VCa, VCb, VCc, Vrho3, z, xv1, xv2, rho3VT, Tcw1, Tcw2, Tcw3 = calculate_new_states(new_states_inputs_x,
#                                                                                        new_states_inputs_old_values,
#
#                                                                                        step_size)
#     # list of all the states containing products of state variables
#     states_inputs = [VCa, VCb, VCc, Vrho3, rho3VT]
#
#     # list of all the previous time step's states containing products of state variables
#     states_old = [Ca, Cb, Cc, V, rho3, T]
#
#     # newly calculated state variables
#     Ca, Cb, Cc, V, rho3, T = states(states_inputs, states_old)
#
#     # add the newly calculated Cc to the Cc_list
#     Cc_list[n] = Cc
#
#     # list of the inputs to solve the kinetic parameters
#     kinetics_inputs = [Ca, Cb, T]
#     K, Na = algebraic_equations_kinetics(kinetics_inputs)
#
#     # list of the inputs to solve the densities and mass fractions
#     density_inputs = [Cao, Cbo, Ca, Cb, Cc, rho3]
#     rao, rbo, rho1, rho2, ra, rb, rc, rsolvent = algebraic_equations_density(density_inputs)
#
#     # calculate tank level
#     h_inputs = [V]
#     h, = algebraic_equations_h(h_inputs)
#
#     # calculate flow though valves
#     valve_inputs = [xv1, xv2,  rho2, rho3, h]
#     F2, F3, delta_Pv2 = algebraic_equations_valve(valve_inputs)
#
#     # calculate the liquid height as experienced by each segment
#     height_inputs = [h]
#     h1, h2, h3 = algebraic_equations_section_height(height_inputs)
#
#     # calculate the heat transfer rates
#     heat_transfer_inputs = [h1, h2, h3, T, Tcw1, Tcw2, Tcw3]
#     Qj1, Qj2, Qj3, Qair1, Qair2, Qair3, Qj_tot= algebraic_equations_heat_transfer(heat_transfer_inputs)
#
#     # calculate the measured C concentration
#     # Todo: add white noise to this variable
#     Cc_measured = Cc_with_dead_time(t, Cc_list, tspan,theta)
#
#     return [F2, F3, Ca, Cb, Cc, Na, V, rho1, rho2, rho3, z, xv1, xv2, T, Qj_tot, Tcw1, Tcw2, Tcw3, Qj1, Qj2, Qj3, Qair1,
#             Qair2, Qair3, rao, rbo, ra, rb, rc, rsolvent, h1, h2, h3, h, delta_Pv2, Cc_measured]
#
#
# def run_simulation(Tao_in, Cao_in, Tbo_in, Cbo_in, F1_in, Ps1_in, Ps2_in, Tcwo_in, Fcw_in, tspan, manual):
#     """
#     Euler integration of the DAE system
#
#     :param Tao_in: Temperature of stream 1
#     :param Cao_in: Concentration of A in stream 1
#     :param Tbo_in: Temperature of stream 2
#     :param Cbo_in: Concentration of B in stream 2
#     :param F1_in: Volumetric flow of stream 1
#     :param Ps1_in: Pressure to valve 1
#     :param Ps2_in: Pressure to valve 2
#     :param Tcwo_in: Temperature of cooling water
#     :param Fcw_in: Volumetric flow of cooling water
#     :param tspan: array containing the time corresponding to a particular time step
#
#     :return: (dict)
#              dictionary with  lists of the solved system variables
#     """
#     ss_values = steady_state()
#     # initial steady state values
#     ca = ss_values['Ca']
#     cb = ss_values['Cb']
#     cc = ss_values['Cc']
#     rho_1 = ss_values['rho1']
#     rho_2 = ss_values['rho2']
#     rho_3 = ss_values['rho3']
#     v = ss_values['V']
#     Q_j_tot = ss_values['Qj_tot']
#     Temp = ss_values['T']
#     na = ss_values['Na']
#     F_2 = ss_values['F2']
#     F_3 = ss_values['F3']
#     Z = ss_values['z']
#     x_v1 = ss_values['xv1']
#     x_v2 = ss_values['xv2']
#     T_cw1 = ss_values['T_cw1']
#     T_cw2 = ss_values['T_cw2']
#     T_cw3 = ss_values['T_cw3']
#     Q_j1 = ss_values['Qj1']
#     Q_j2 = ss_values['Qj2']
#     Q_j3 = ss_values['Qj3']
#     Q_air1 = ss_values['Qair1']
#     Q_air2 = ss_values['Qair2']
#     Q_air3 = ss_values['Qair3']
#     dt = tspan[1] - tspan[0]
#     rao = ss_values['rao']
#     rbo = ss_values['rbo']
#     ra = ss_values['ra']
#     rb = ss_values['rb']
#     rc = ss_values['rc']
#     rsolvent = ss_values['rsolvent']
#     h_1 = ss_values['h1']
#     h_2 = ss_values['h2']
#     h_3 = ss_values['h3']
#     H = ss_values['H']
#     d_Pv2 = ss_values['d_Pv2']
#
#     # create an array to store the values of Cc
#     Cc_array = numpy.zeros(tspan.shape)
#
#     # list of all the out put variables
#     outputs = [F_2, F_3, ca, cb, cc, na, v, rho_1, rho_2, rho_3, Z, x_v1, x_v2, Temp,  Q_j_tot, T_cw1, T_cw2, T_cw3, Q_j1, Q_j2, Q_j3, Q_air1, Q_air2, Q_air3, rao, rbo, ra, rb, rc, rsolvent, h_1, h_2, h_3, H, d_Pv2, cc]
#
#     # list to store all the calculated variables
#     out_list = []
#
#     # dictionary to store all the calculated variables (for easy retrieval of responses)
#     solution = {}
#
#     # dictionary keys
#     names = ['F2', 'F3', 'Ca', 'Cb', 'Cc', 'Na', 'V', 'rho1', 'rho2', 'rho3', 'z', 'xv1', 'xv2', 'T', 'Qj_tot', 'T_cw1',
#              'T_cw2', 'T_cw3', 'Qj1', 'Qj2', 'Qj3', 'Qair1', 'Qair2', 'Qair3', 'rao', 'rbo', 'ra', 'rb', 'rc',
#              'rsolvent', 'h1', 'h2', 'h3', 'H', 'd_Pv2', 'Cc_measured']
#
#     # create the save dictionary
#     for key in names[: -1]:
#         solution[key] = [ss_values[key]]
#
#     solution['Cc_measured'] = [cc]
#
#     # begin Euler integration
#     for n, t in enumerate(tspan[1:]):
#
#         if not manual:
#
#             F1 = F1_in(t)
#             Cao = Cao_in(t)
#             Cbo = Cbo_in(t)
#             Ps1 = Ps1_in(t)
#             Ps2 = Ps2_in(t)
#             Tao = Tao_in(t)
#             Tbo = Tbo_in(t)
#             Fcw = Fcw_in(t)
#             Tcwo = Tcwo_in(t)
#
#             # add your controller here and remember to remove the input above which your controller is the output
#             # you can get access to any variable by calling solution['variable name'][-1] which is the value of the
#             # previous time step
#
#         else:
#
#             F1 = F1_in(t)
#             Cao = Cao_in(t)
#             Cbo = Cbo_in(t)
#             Ps1 = Ps1_in(t)
#             Ps2 = Ps2_in(t)
#             Tao = Tao_in(t)
#             Tbo = Tbo_in(t)
#             Fcw = Fcw_in(t)
#             Tcwo = Tcwo_in(t)
#
#         # list of all the time dependent inputs
#         inputs = [F1,
#                   Cao,
#                   Cbo,
#                   Ps1,
#                   Ps2,
#                   Tao,
#                   Tbo,
#                   Fcw,
#                   Tcwo]
#
#         # append the outputs to out_list
#         outputs = single_iteration(inputs, outputs, dt, t, n, Cc_array, tspan)
#         out_list.append(outputs)
#
#         # save the responses to dictionary
#         for key, value in zip(names, outputs):
#             solution[key].append(value)
#
#     return solution
