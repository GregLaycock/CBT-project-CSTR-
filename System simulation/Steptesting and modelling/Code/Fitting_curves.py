from Stepping_all import *
from Fitting_module import *
from steady_state_values import steady_state
import numpy
import scipy.optimize



 # there are now 18 data sets

def get_type(name):                                                                  # based on intuition after seeing curves
    if name == 'F1T' or name == 'Ps3T' or name == 'Ps3Cc_measured':
        fit_type = 'SOPTD'

    elif name == 'Ps2T' or name == 'Ps2Cc_measured':
        fit_type = 'SOZPTD'

    else:
        fit_type = 'FOPTD'

    return fit_type


def run_all_fits(names,fit_types,initials,yo_vals,u_vals,data):
    print('Initializing')
    fitted_params = []
    reserror = numpy.zeros(len(names))
    print('Running')
    count = 1
    for i, name in enumerate(names):
        print('current fit is '+str(count)+' of '+str(len(names)))

        model = fit_types[i]
        parameters = initials[i]
        u = u_vals[i]
        t = tspan
        if name == 'F1T':
            opt = scipy.optimize.minimize(error_func,parameters,args = (model,u,t,yo_vals[i],data[name]), bounds = [[10e3,10e4],[50,500],[0,1],[0,200]])
        else:
            opt = scipy.optimize.minimize(error_func,parameters,args = (model,u,t,yo_vals[i],data[name]))
        fitted_params.append(opt.x)
        residual = opt.fun
        reserror[i] = residual

        print('completed fit '+str(count) + ' with residual sum abs error of '+str(residual))
        count += 1

    fitting_results = {'fit': names, 'type': fit_types, 'initial': initials, 'optimal_parameters': fitted_params,'residuals':[reserror]}
    return fitting_results


def get_initials(types):
    initials = []
    for i, typ in enumerate(types):
        if typ == 'FOPTD':
            initials.append([0.1,100,50])
        elif typ == 'SOPTD':
            initials.append([-0.01,100,0.7,50])
        elif typ == 'SOZPTD':
            initials.append([-1,0.5,100,100,50])

    if len(types) == 18:
        initials[-2] = [15e3,100,0.65,50]
    return initials


################### initializing and running fits############
ss_values = steady_state()
ccss = ss_values['Cc']
tss = ss_values['T']
hss = ss_values['H']


stepped_var = ['Ps1','Ps1','Ps1','Ps2','Ps2','Ps2','Ps3','Ps3','Ps3','Cao','Cao','Cao','Tbo','Tbo','Tbo','F1','F1','F1']
tspan = numpy.linspace(0, 2000, 1000)
data = get_results()


stepped_vars = ['Ps1','Ps2','Ps3','Cao','Tbo','F1']
outputs = ['Cc_measured', 'T', 'H']
names = []
for i, input in enumerate(stepped_vars):
    for j, output in enumerate(outputs):
        names.append(str(input) + str(output))




data_sets = [data[key] for key in names]
fit_types = [get_type(name) for name in names]
yo_vals = [ccss,tss,hss,ccss,tss,hss,ccss,tss,hss,ccss,tss,hss,ccss,tss,hss,ccss,tss,hss]
initials = get_initials(fit_types)

u_vals = [20,20,20,20,20,20,20,20,20,0.2*7.4,0.2*7.4,0.2*7.4,0.2*24,0.2*24,0.2*24,0.2*7.334e-4,0.2*7.334e-4,0.2*7.334e-4]

results = run_all_fits(names,fit_types,initials,yo_vals,u_vals,data)

import csv
params = results['optimal_parameters']
with open (r'Fit_results.csv', 'w', newline='') as write_file:
    write = csv.writer(write_file)
    write.writerows(fit for fit in params)