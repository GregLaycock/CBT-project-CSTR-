import csv
from matplotlib import pyplot as plot
import numpy
import Fitting_curves
from Stepping_all import run_sim,get_results()
import Fitting_module
with open('fit_results.csv', 'rb') as csv_file:
    reader = csv.reader(csv_file)
    results = dict(reader)

tspan = numpy.linspace(0, 2000, 1000)
stepped_vars = Fitting_curves.stepped_var
output_vars = ['Cc_measured','T','H','Cc_measured','T','H','Cc_measured','T','H','Cc_measured','T','H','Cc_measured','T','H','Cc_measured','T','H']
data = Fitting_curves.data
types = Fitting_curves.fit_types
u_vals = [20,20,20,20,20,20,20,20,20,0.2*7.4,0.2*7.4,0.2*7.4,0.2*24,0.2*24,0.2*24,0.2*7.334e-4,0.2*7.334e-4,0.2*7.334e-4]
yo_vals = Fitting_curves.yo_vals
simdata = get_results()

# getting data from fits
fitted = []

for i,fit in enumerate(results['fit']):
    parameters = results['optimal_parameters'][i]
    model = results['type'][i]
    u = u_vals[i]
    t = tspan
    yo = yo_vals[i]
    vals = Fitting_module.get_model_vals(parameters,model,u,t,yo)
    fitted.append(vals)




for i, name in enumerate(simdata.keys()):
    plot.figure()
    plot.plot(tspan,simdata[name],label = 'simulated')
    plot.plot(tspan,fitted[i],label='fitted')
    plot.title("Results of fitting a "+str(types[i]) + " model to step resonse of "+str(output_vars[i])+" to "+str(stepped_vars[i]))
    plot.axis()
    plot.legend(loc=4)
    plot.xlabel('Time in sec')
    plot.ylabel('value of output in SI units')

plot.show()