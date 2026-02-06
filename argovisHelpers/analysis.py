from .helpers import interpolate_to_levels, pchip_search
import numpy

def MLD_estimate(pressure, var, threshold_delta, reference_pressure=10):
    # simple mixed layer depth estimator based on an absolute change in a variable relative to a reference pressure (default 10 dbar)
    # suggested variable / threshold deltas:
    # potential density / 0.03 kg/m3
    reference_val = interpolate_to_levels(pressure, var, [reference_pressure])[0][0]
    if numpy.isnan(reference_val):
        return None
    threshold_val = reference_val + threshold_delta

    return pchip_search(pressure, var, threshold_val, 0, 1000, 1)

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