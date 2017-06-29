import numpy
from scipy.signal import lti

def FOPTD(parameters,u,tspan,yo):  # form is k/(ts+1)

    kc, tau, theta = parameters
    G1 = lti([kc], [tau, 1])
    dt = tspan[1] - tspan[0]
    results = numpy.zeros(len(tspan))
    yvals = numpy.zeros(len(tspan))
    y = yo
    x = numpy.zeros([G1.A.shape[0], 1])

    for i, t in enumerate(tspan):
        yvals[i] = y
        t_interp = t - theta
        ydelayed = numpy.interp(t_interp, tspan, yvals)
        results[i] = ydelayed
        dx = G1.A.dot(x) + G1.B.dot(u)
        x += dx * dt
        y_prime = G1.C.dot(x) + G1.D.dot(u)
        y = y_prime[0][0] + yo

    return numpy.array(results)

def SOPTD(parameters,u,tspan,yo):    # form is k/(t1s+1)(t2s+1)

    k,tau,zeta,theta = parameters

    G1 = lti([k], [tau**2,2*zeta*tau, 1])
    dt = tspan[1] - tspan[0]
    results = numpy.zeros(len(tspan))
    yvals = numpy.zeros(len(tspan))
    y = yo
    x = numpy.zeros([G1.A.shape[0], 1])

    for i, t in enumerate(tspan):
        yvals[i] = y
        t_interp = t - theta
        ydelayed = numpy.interp(t_interp, tspan, yvals)
        results[i] = ydelayed
        dx = G1.A.dot(x) + G1.B.dot(u)
        x += dx * dt
        y_prime = G1.C.dot(x) + G1.D.dot(u)
        y = y_prime[0][0] + yo

    return numpy.array(results)

def SOZPTD(parameters,u,tspan,yo):          # for a 2nd order response with a process zero (inverse responses) form is (c1*s+c2)/(t1s+1)(t2s+1)

    c1,c2,tau1,tau2,theta = parameters

    G1 = lti([c1,c2], [tau1*tau2, (tau1+tau2), 1])
    dt = tspan[1] - tspan[0]
    results = numpy.zeros(len(tspan))
    yvals = numpy.zeros(len(tspan))
    y = yo
    x = numpy.zeros([G1.A.shape[0], 1])

    for i, t in enumerate(tspan):
        yvals[i] = y
        t_interp = t - theta
        ydelayed = numpy.interp(t_interp, tspan, yvals)
        results[i] = ydelayed
        dx = G1.A.dot(x) + G1.B.dot(u)
        x += dx * dt
        y_prime = G1.C.dot(x) + G1.D.dot(u)
        y = y_prime[0][0] + yo

    return numpy.array(results)

def error_func(parameters,model,u,t,yo,simdata):

    if model == 'FOPTD':
        results = FOPTD(parameters,u,t,yo)
    elif model == 'SOPTD':
        results = SOPTD(parameters, u, t, yo)
    elif model == 'SOZPTD':
        results = SOZPTD(parameters,u,t,yo)

    abs_error = abs(simdata - results)
    err = sum(abs_error)
    return err

def get_model_vals(parameters,model,u,t,yo):
    if model == 'FOPTD':
        results = FOPTD(parameters,u,t,yo)
    elif model == 'SOPTD':
        results = SOPTD(parameters, u, t, yo)
    elif model == 'SOZPTD':
        results = SOZPTD(parameters,u,t,yo)

    return results

