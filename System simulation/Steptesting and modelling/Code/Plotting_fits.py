import csv
from matplotlib import pyplot as plot
import numpy
from Fitting_curves import get_type
from steady_state_values import steady_state


filename = 'fit_results.csv'
with open(filename, 'rU') as p:
    #reads csv into a list of lists
    my_list = [rec for rec in csv.reader(p, delimiter=',')]

all_params = [[float(i) for i in my_list[j]] for j,lis in enumerate(my_list)]

from Stepping_all import run_sim,get_results
import Fitting_module

tspan = numpy.linspace(0, 2000, 1000)
output_vars = ['Cc_measured','T','H','Cc_measured','T','H','Cc_measured','T','H','Cc_measured','T','H','Cc_measured','T','H','Cc_measured','T','H']
stepped_vars = ['Ps1','Ps2','Ps3','Cao','Tbo','F1']
outputs = ['Cc_measured', 'T', 'H']
names = []
stepped = ['Ps1','Ps1','Ps1','Ps2','Ps2','Ps2','Ps3','Ps3','Ps3','Cao','Cao','Cao','Tbo','Tbo','Tbo','F1','F1','F1']
for i, input in enumerate(stepped_vars):
    for j, output in enumerate(outputs):
        names.append(str(input) + str(output))
# print(names)
data = get_results()
ss_values = steady_state()
ccss = ss_values['Cc']
tss = ss_values['T']
hss = ss_values['H']
types = [get_type(name) for name in names]
# print(types)
u_vals = [20,20,20,20,20,20,20,20,20,0.2*7.4,0.2*7.4,0.2*7.4,0.2*24,0.2*24,0.2*24,0.2*7.334e-4,0.2*7.334e-4,0.2*7.334e-4]
yo_vals = [ccss,tss,hss,ccss,tss,hss,ccss,tss,hss,ccss,tss,hss,ccss,tss,hss,ccss,tss,hss]
simdata = data

# getting data from fits
fitted = []

for i,fit in enumerate(names):
    parameters = all_params[i]
    model = types[i]
    u = u_vals[i]
    t = tspan
    yo = yo_vals[i]
    vals = Fitting_module.get_model_vals(parameters,model,u,t,yo)
    fitted.append(vals)




for i, name in enumerate(names):
    plot.figure()
    plot.plot(tspan,simdata[name],'b-',label = 'simulated')
    plot.plot(tspan,fitted[i],'r-',label='fitted')
    plot.title("Results of fitting a "+str(types[i]) + " model to step resonse of "+str(output_vars[i])+" to "+str(stepped[i]))
    plot.axis()
    plot.legend(loc=4)
    plot.xlabel('Time in sec')
    plot.ylabel('value of output in SI units')

plot.show()