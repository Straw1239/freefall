def square(x):
    return x * x

def kalman_update(mean, var, proc_var, obs_var, obs, proc_mult=0.9):
    mean *= proc_mult
    var *= proc_mult * proc_mult
    var += proc_var

    y = obs - mean
    s = obs_var + var
    k = var / s
    #print(k, end='k')
    mean += y * k
    var *= (1 - k)


