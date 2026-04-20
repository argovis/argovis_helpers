from .helpers import Profile
import scipy.interpolate
import numpy, numbers, math, copy

def MLD_estimate(pressure, var, threshold_delta, reference_pressure=10):
    # simple mixed layer depth estimator based on an absolute change in a variable relative to a reference pressure (default 10 dbar)
    # suggested variable / threshold deltas:
    # potential density / 0.03 kg/m3
    reference_val = interpolate_to_levels(pressure, var, [reference_pressure])[0][0]
    if numpy.isnan(reference_val):
        return None
    threshold_val = reference_val + threshold_delta
    print(threshold_val)
    ## assume the MLD is in the first 1000 dbar and only look there
    low_i = 0
    high_i = 0
    while pressure[high_i] < 1000:
        high_i += 1
        if high_i == len(pressure)-1:
            break
    print(low_i, high_i)
    # inverse pchip interp    
    pchip = scipy.interpolate.PchipInterpolator(pressure[low_i:high_i], var[low_i:high_i], extrapolate=False)
    roots = numpy.asarray(pchip.solve(threshold_val, extrapolate=False), dtype=float)
    print(roots)
    return roots[0]

def AOU_estimate(potential_temperature, salinity, density, oxygen):
    # function to compute AOU in umol/kg
    # based on Sarmiento & Gruber (Garcia and Gordon, 1992)
    # all inputs can be either scalar or 1D iterables of the same length
    # potential_temperature in degrees Celsius
    # salinity in situ in PSU
    # density in situ in kg/m3
    # oxygen in situ in umol/kg

    # convert to arrays; None -> nan
    pt = numpy.asarray(potential_temperature, dtype=float)
    sa = numpy.asarray(salinity, dtype=float)
    ro = numpy.asarray(density, dtype=float)
    o2 = numpy.asarray(oxygen, dtype=float)

    scalar = (pt.ndim == sa.ndim == ro.ndim == o2.ndim == 0)

    # force 1D so we only write the algebra once
    pt = numpy.atleast_1d(pt)
    sa = numpy.atleast_1d(sa)
    ro = numpy.atleast_1d(ro)
    o2 = numpy.atleast_1d(o2)

    # enforce equal shapes (no broadcasting surprises)
    if not (pt.shape == sa.shape == ro.shape == o2.shape):
        raise ValueError(f"Inumpyuts must have the same shape; got {pt.shape}, {sa.shape}, {ro.shape}, {o2.shape}")

    # elementwise validity mask
    valid = numpy.isfinite(pt) & numpy.isfinite(sa) & numpy.isfinite(ro) & numpy.isfinite(o2)

    out = numpy.full(pt.shape, numpy.nan, dtype=float)

    # constants
    A0 = 2.00907
    A1 = 3.22014
    A2 = 4.05010
    A3 = 4.94457
    A4 = -0.256847
    A5 = 3.88767
    B0 = -6.24523e-3
    B1 = -7.37614e-3
    B2 = -1.03410e-2
    B3 = -8.17083e-3
    C0 = -4.88682e-7

    # compute only on valid positions
    ptv, sav, rov, o2v = pt[valid], sa[valid], ro[valid], o2[valid]

    Ts = numpy.log((298.15 - ptv) / (273.15 + ptv))
    l  = (A0
          + A1*Ts + A2*(Ts**2) + A3*(Ts**3) + A4*(Ts**4) + A5*(Ts**5)
          + sav*(B0 + B1*Ts + B2*(Ts**2) + B3*(Ts**3))
          + C0*(sav**2))

    O2_eq_mmol_per_m3 = (1000/22.3916) * numpy.exp(l)
    O2_eq_umol_per_kg = O2_eq_mmol_per_m3 * rov / 1000
    out[valid] = O2_eq_umol_per_kg - o2v

    if scalar:
        return float(out[0])
    return out

def interpolate_to_levels(levels_raw, var_raw, levels_interp):
    # interpolate <var> to <levels> using PCHIP interpolation
    # flag 32 (little endian): ROI didn't contain enough info to interpolate

    flag = 0
    pressure, variable, flag = tidy_profile(levels_raw, var_raw, flag)

    # some truly pathological profiles will have no levels left at this point
    if len(pressure) == 0:
        interp = numpy.ma.masked_array(numpy.full(len(levels_interp), numpy.nan), mask=True)
        flag = flag | 32
        return interp, flag

    # interpolate
    interp = scipy.interpolate.PchipInterpolator(pressure, variable, extrapolate=True)(levels_interp)

    # mask levels that fall outside the insitu range, or which are too far from an insitu measurement
    interp = mask_far_interps(pressure, levels_interp, interp)

    return interp, flag

def is_numeric(x):
    return isinstance(x, numbers.Real) and not math.isnan(x) and math.isfinite(x) and not isinstance(x, bool)

def tidy_profile(pressure, var, flag):
    # pchip needs pressures to be monotonically increasing; also need the dependent variable to always be defined
    # flags (little endian):
    # 1: degenerate adjacent levels
    # 2: levels in reverse order
    # 4: variable of interest was NaN, masked
    # 8: levels non-monotonic, had to sort
    # 16: pressure was NaN, masked

    ## dependent variable must be defined
    mask = [0]*len(var)
    for i in range(len(var)):
        if not is_numeric(var[i]):
            mask[i] = 1
            flag = flag | 4
    p = [pressure[i] for i in range(len(mask)) if mask[i]==0]
    v = [var[i] for i in range(len(mask)) if mask[i]==0]

    ## pressure must be defined
    mask = [0]*len(p)
    for i in range(len(p)):
        if not is_numeric(p[i]):
            mask[i] = 1
            flag = flag | 16
    p = [p[i] for i in range(len(mask)) if mask[i]==0]
    v = [v[i] for i in range(len(mask)) if mask[i]==0]

    ## drop degenerate levels and flag
    mask = [0]*len(p)
    for i in range(len(p)-1):
        if p[i] == p[i+1]:
            mask[i] = 1
            mask[i+1] = 1
            flag = flag | 1
    p = [p[i] for i in range(len(mask)) if mask[i]==0]
    v = [v[i] for i in range(len(mask)) if mask[i]==0]

    if all(p[i] < p[i + 1] for i in range(len(p) - 1)):
        # pressure is monotonically increasing, return
        return p, v, flag

    if all(p[i] > p[i + 1] for i in range(len(p) - 1)):
        # pressure is monotonically decreasing, reverse and return
        flag = flag | 2
        return p[::-1], v[::-1], flag

    # pressure is non-monotonic, sort and try again
    x = sorted(zip(p,v))
    p = [element[0] for element in x]
    v = [element[1] for element in x]
    flag = flag | 8
    return tidy_profile(p,v,flag)

def mask_far_interps(measured_pressures, interp_levels, interp_values):
    # mask interpolated values that are too far from the nearest measured level
    # or which fall outside range of measured levels

    mask = [False for x in interp_levels]

    for i, level in enumerate(interp_levels):
        ## mask out anything that was extrapolated:
        if interp_levels[i] < measured_pressures[0] or interp_levels[i] > measured_pressures[-1]:
            mask[i] = True
            continue

        ## determine how far is too far when interpolating to interiror holes:
        radius = 0
        if level < 50:
            radius = 50
        elif level < 150:
            radius = 150
        else:
            radius = 500

        i_below = 0
        i_above = len(measured_pressures)-1
        for j in range(len(measured_pressures)):
            if measured_pressures[j] <= level:
                i_below = j
            else:
                i_above = j
                break
        if abs(measured_pressures[i_below] - level) > radius or abs(measured_pressures[i_above] - level) > radius:
            mask[i] = True

    return numpy.ma.masked_array(interp_values, mask=mask)

def interpolate_all(profile, levels):
    # interpolate all variables in a profile to a common set of levels
    # profile is either a json profile schema or a Profile object.
    # assumes json profile has its 'data_info' key present and that 'pressure' is one of the variables, to be interpolated on.
    
    if isinstance(profile, Profile):
        variables = profile.variable_names()
        datavecs = [x for x in variables if 'qc' not in x] # don't interpolate QC; a bit duck-typie...
        pressure = profile.getvar('pressure')
        p = copy.deepcopy(profile)
        p.delvar('pressure')
        p.setvar('pressure', levels)
        for v in variables:
            if v in datavecs and v != 'pressure':
                i = interpolate_to_levels(pressure, profile.getvar(v), levels)
                p.delvar(v)
                p.setvar(v, i[0])
            else:
                p.delvar(v)
        return p       
    else:
        ## remove any QC vectors
        variables = profile['data_info'][0]
        qcvecs = ['qc' in x for x in variables] # a bit duck-typie...
        data = [x for i,x in enumerate(profile['data']) if not qcvecs[i]]
        data_info = [None, None, None]
        data_info[0] = [x for i,x in enumerate(profile['data_info'][0]) if not qcvecs[i]]
        data_info[1] = profile['data_info'][1]
        data_info[2] = [x for i,x in enumerate(profile['data_info'][2]) if not qcvecs[i]]

        level_idx = data_info[0].index('pressure')
        raw_levels = data[level_idx]

        for i in range(len(data_info[0])):
            if i == level_idx:
                data[i] = levels
            else:
                data[i], _ = interpolate_to_levels(raw_levels, data[i], levels)
                data[i] = list(data[i])

        return {**profile, 'data':data, 'data_info':data_info}

def regional_mean(dxr, form='area'):
    # given an xarray dataset <dxr> with latitudes and longitudes as dimensions,
    # calculate the mean of all data variables, weighted by grid cell area
    weights = numpy.cos(numpy.deg2rad(dxr.latitude))
    weights.name = "weights"
    dxr_weighted = dxr.weighted(weights)
    
    if form =='area':
        return dxr_weighted.mean(("longitude", "latitude"))
    elif form == 'meridional':
        return dxr_weighted.mean(("latitude"))
    elif form == 'zonal':
        return dxr_weighted.mean(("longitude"))