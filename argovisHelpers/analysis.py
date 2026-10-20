# flag table:
# 1: degenerate adjacent levels
# 2: levels in reverse order
# 4: variable of interest was NaN, masked
# 8: levels non-monotonic, had to sort
# 16: pressure was NaN, masked
# 32: insufficient levels for cacluation (interpolation or otherwise)
# 512: no threshold crossing found in MLD estimation

from .helpers import Profile
import scipy.interpolate
import numpy, numbers, math, copy, gsw, xarray, warnings

def MLD_estimate(pressure, var, threshold_delta, reference_pressure=10):
    # simple mixed layer depth estimator based on an absolute change in a variable relative to a reference pressure (default 10 dbar)
    # suggested variable / threshold deltas:
    # potential density / 0.03 kg/m3
    # note this implementation assumes there is only one threshold crossing; if there are multiple, it will return the shallowest one

    reference_val, flag = interpolate_to_levels(pressure, var, [reference_pressure])
    if numpy.isnan(reference_val):
        return None, flag
    threshold_val = reference_val[0] + threshold_delta

    # clean up input data
    pressure, variable, flag = tidy_profile(pressure, var, flag)
    # ROI must contain at least two points for Pchip
    if len(pressure) < 2:
        flag = flag | 32
        return None, flag

    # inverse pchip interp    
    pchip = scipy.interpolate.PchipInterpolator(pressure, variable, extrapolate=False)
    roots = numpy.asarray(pchip.solve(threshold_val, extrapolate=False), dtype=float)

    # it's on you to make sure you're giving it a search range without a zillion roots
    if len(roots) > 0 and not numpy.isnan(roots[0]):
        # return the first root > reference pressure
        roots.sort()
        for r in roots:
            if r > reference_pressure:
                return r, flag
        return None, 512 # roots exist but none of them physical
    else:
        return None, 512

def AOU_estimate(SA, CT, p, lon, lat, oxygen):
    # apparent oxygen utilization
    # oxygen in umol/kg
    # p in dbar

    o2sol = gsw.O2sol(SA, CT, p, lon, lat)
    O2_eq_umol_per_kg = o2sol * gsw.rho(SA, CT, p) / 1000

    return O2_eq_umol_per_kg - oxygen, 0 # no flagging implemented but let's match MLD


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

def interpolate_to_levels(levels_raw, var_raw, levels_interp):
    # interpolate <var> to <levels> using PCHIP interpolation
    # flag 32 (little endian): ROI didn't contain enough info to interpolate

    flag = 0
    pressure, variable, flag = tidy_profile(levels_raw, var_raw, flag)

    # some truly pathological profiles will have 0 or 1 levels left at this point, interpolation impossible by construction
    if len(pressure) < 2:
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
        for v in variables:
            if v in datavecs and v != 'pressure':
                i, _ = interpolate_to_levels(pressure, profile.getvar(v), levels)
                p.delvar(v)
                p.setvar(v, i)
            else:
                p.delvar(v)
        p.setvar('pressure', levels)
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
                data[i] = data[i].filled(numpy.nan)

        return {**profile, 'data':data, 'data_info':data_info}

def bin_profiles(
    profiles,
    levels,
    lat_bin,
    lon_bin,
    varname='temperature',
    lat_range=None,
    lon_range=None,
    lat_step=None,
    lon_step=None,
):
    """
    Average profiles into rectangular latitude/longitude cells.

    This is a plain bin mean, NOT a mapped/gridded product: there is no
    weighting by distance or time, no mapping error, and no correction for
    uneven sampling in space, season, or year. Each cell is simply the mean
    of whichever profiles fall in it.

    By default, cells tile the requested region without overlap:

        lat_step = lat_bin
        lon_step = lon_bin

    Smaller step sizes produce overlapping cells. For example,

        lon_bin=0.5, lon_step=0.25

    gives 0.5-degree-wide longitude bins every 0.25 degrees.

    A section is simply a grid containing one bin in one horizontal
    dimension. For example,

        lat_range=(-5, 5), lat_bin=10,
        lon_range=None,   lon_bin=1

    gives a longitude-depth equatorial section, averaging profiles between
    5 S and 5 N in 1-degree longitude bins.

    The bins must fit the requested range exactly:
      * the low edge of the first bin equals the range minimum
      * the high edge of the last bin equals the range maximum

    If lat_range or lon_range is None, the complete domain is used:
      * latitude:  -90 to 90
      * longitude:   0 to 360

    Parameters
    ----------
    profiles : list of Profile
        Profiles already interpolated onto `levels`.

    levels : array-like
        Levels onto which all profiles have been interpolated.

    lat_bin : float
        Latitude bin width in degrees.

    lon_bin : float
        Longitude bin width in degrees.

    varname : str, optional
        Variable to average. Default is 'temperature'.

    lat_range : (min, max), optional
        Latitude range covered by the bins. Default is (-90, 90).

    lon_range : (min, max), optional
        Longitude range covered by the bins. Longitudes use a 0-360
        convention internally. Dateline-crossing ranges such as (350, 20)
        are allowed. Default is the full globe.

    lat_step : float, optional
        Distance between adjacent latitude bin centers. Defaults to
        `lat_bin`, giving non-overlapping bins.

    lon_step : float, optional
        Distance between adjacent longitude bin centers. Defaults to
        `lon_bin`, giving non-overlapping bins.

    Returns
    -------
    xarray.Dataset
        `<varname>` has dimensions (latitude, longitude, level).
        `n_profiles` has dimensions (latitude, longitude).

        Empty cells contain NaN and have n_profiles = 0.

    Notes
    -----
    If a step is smaller than its corresponding bin width, bins overlap and
    an individual profile may therefore contribute to more than one cell.

    Steps larger than bin widths are not allowed, since they would leave
    gaps between cells.
    """

    levels = numpy.asarray(levels, dtype=float)

    lat_bin = float(lat_bin)
    lon_bin = float(lon_bin)

    # ------------------------------------------------------------------
    # Validate basic bin/step arguments
    # ------------------------------------------------------------------

    if lat_bin <= 0:
        raise ValueError("lat_bin must be positive")

    if lon_bin <= 0:
        raise ValueError("lon_bin must be positive")

    if lat_bin > 180:
        raise ValueError("lat_bin cannot exceed 180 degrees")

    if lon_bin > 360:
        raise ValueError("lon_bin cannot exceed 360 degrees")

    # Default to ordinary non-overlapping tiling.
    if lat_step is None:
        lat_step = lat_bin
    else:
        lat_step = float(lat_step)

    if lon_step is None:
        lon_step = lon_bin
    else:
        lon_step = float(lon_step)

    if lat_step <= 0:
        raise ValueError("lat_step must be positive")

    if lon_step <= 0:
        raise ValueError("lon_step must be positive")

    if lat_step > lat_bin:
        raise ValueError(
            "lat_step cannot exceed lat_bin; that would leave gaps "
            "between latitude bins"
        )

    if lon_step > lon_bin:
        raise ValueError(
            "lon_step cannot exceed lon_bin; that would leave gaps "
            "between longitude bins"
        )

    # ------------------------------------------------------------------
    # Resolve ranges.
    #
    # Do this before constructing any bins, so everything below works
    # with concrete numeric bounds regardless of whether the caller
    # supplied a range.
    # ------------------------------------------------------------------

    if lat_range is None:
        lat_min = -90.0
        lat_max = 90.0
    else:
        if len(lat_range) != 2:
            raise ValueError("lat_range must be (min, max)")

        lat_min, lat_max = map(float, lat_range)

    if not (-90 <= lat_min < lat_max <= 90):
        raise ValueError(
            "lat_range must satisfy -90 <= min < max <= 90"
        )

    if lon_range is None:
        # Full globe, represented internally as the continuous interval
        # 0 ... 360.
        lon_min = 0.0
        lon_max = 360.0

    else:
        if len(lon_range) != 2:
            raise ValueError("lon_range must be (min, max)")

        raw_min, raw_max = map(float, lon_range)
        raw_span = raw_max - raw_min

        if numpy.isclose(raw_span, 0):
            raise ValueError("lon_range has zero width")

        if abs(raw_span) > 360 and not numpy.isclose(abs(raw_span), 360):
            raise ValueError(
                "lon_range cannot span more than 360 degrees"
            )

        lon_min = raw_min % 360

        if numpy.isclose(abs(raw_span), 360):
            # Examples:
            #   (0, 360)
            #   (-180, 180)
            #
            # Both explicitly request the whole globe.
            lon_span = 360.0
        else:
            # This also handles ranges crossing the longitude seam:
            #
            #   (350, 20) -> span 30
            #
            lon_span = raw_span % 360

        if numpy.isclose(lon_span, 0):
            raise ValueError("lon_range has zero width")

        # Use an unwrapped longitude coordinate internally.
        #
        #   (350, 20) becomes 350 ... 380
        #
        lon_max = lon_min + lon_span

    # ------------------------------------------------------------------
    # Construct bin centers.
    #
    # This is deliberately written in a fairly pedestrian way:
    #
    #   1. Put the first bin against the low edge.
    #   2. Determine whether an integer number of steps reaches a final
    #      bin whose high edge hits the upper range limit.
    #   3. Explicitly verify both outer edges.
    #
    # This makes the intended geometry obvious when somebody reads this
    # code six months from now.
    # ------------------------------------------------------------------

    def make_centers(range_min, range_max, bin_width, step, name):

        range_width = range_max - range_min

        if bin_width > range_width and not numpy.isclose(
            bin_width, range_width
        ):
            raise ValueError(
                f"{name}_bin ({bin_width} degrees) is wider than the "
                f"requested {name} range ({range_width} degrees)"
            )

        # The first bin is anchored directly against the lower range edge.
        first_center = range_min + bin_width / 2.0

        # If one bin exactly fills the range, we're done.
        if numpy.isclose(bin_width, range_width):
            centers = numpy.array([first_center])

        else:
            # How far must the bin centers travel from the first bin to
            # the last bin?
            #
            # Example:
            #
            # range:     0 -------------------- 20
            # bin width: |---------|
            #             center=5
            #
            # For a 10-degree bin, the final center must be at 15 so its
            # high edge lands on 20. Thus the centers need to travel
            # 10 degrees total.
            distance_to_last_center = range_width - bin_width

            n_steps_float = distance_to_last_center / step
            n_steps = int(round(n_steps_float))

            if not numpy.isclose(n_steps_float, n_steps):
                raise ValueError(
                    f"{name} bins do not fit the requested range exactly: "
                    f"range width is {range_width} degrees, "
                    f"bin width is {bin_width} degrees, and "
                    f"step is {step} degrees. The first bin can begin at "
                    f"the range minimum, but no bin lands exactly on the "
                    f"range maximum."
                )

            centers = (
                first_center
                + numpy.arange(n_steps + 1, dtype=float) * step
            )

        # Explicitly check the geometry we actually care about.
        first_low_edge = centers[0] - bin_width / 2.0
        last_high_edge = centers[-1] + bin_width / 2.0

        if not numpy.isclose(first_low_edge, range_min):
            raise RuntimeError(
                f"internal error constructing {name} bins: "
                f"first bin begins at {first_low_edge}, "
                f"expected {range_min}"
            )

        if not numpy.isclose(last_high_edge, range_max):
            raise RuntimeError(
                f"internal error constructing {name} bins: "
                f"last bin ends at {last_high_edge}, "
                f"expected {range_max}"
            )

        return centers

    lat_centers = make_centers(
        lat_min,
        lat_max,
        lat_bin,
        lat_step,
        "latitude",
    )

    lon_centers_unwrapped = make_centers(
        lon_min,
        lon_max,
        lon_bin,
        lon_step,
        "longitude",
    )

    # Present longitude coordinates conventionally as 0 ... 360,
    # even if the internal geometry crossed the seam.
    lon_centers = lon_centers_unwrapped % 360

    # ------------------------------------------------------------------
    # Collect profile positions and data
    # ------------------------------------------------------------------

    lon = numpy.asarray(
        [p.longitude % 360 for p in profiles],
        dtype=float,
    )

    lat = numpy.asarray(
        [p.latitude for p in profiles],
        dtype=float,
    )

    data = numpy.full(
        (len(profiles), len(levels)),
        numpy.nan,
        dtype=float,
    )

    for i, p in enumerate(profiles):

        v = p.getvar(varname)

        if v is None:
            continue

        v = numpy.asarray(v, dtype=float)

        if len(v) != len(levels):
            raise ValueError(
                f"profile {p.id}: '{varname}' has {len(v)} levels, "
                f"expected {len(levels)} — interpolate profiles onto "
                "`levels` first"
            )

        data[i] = v

    # Ignore profiles with no usable value at any level.
    keep = ~numpy.all(numpy.isnan(data), axis=1)

    # Put every longitude into the same continuous coordinate system used
    # above for the longitude bins.
    #
    # For example, with lon_range=(350, 20):
    #
    #     355 -> 355
    #       5 -> 365
    #
    lon_unwrapped = lon_min + ((lon - lon_min) % 360)

    # ------------------------------------------------------------------
    # Allocate output arrays
    # ------------------------------------------------------------------

    mean = numpy.full(
        (
            len(lat_centers),
            len(lon_centers_unwrapped),
            len(levels),
        ),
        numpy.nan,
        dtype=float,
    )

    count = numpy.zeros(
        (
            len(lat_centers),
            len(lon_centers_unwrapped),
        ),
        dtype=int,
    )

    # ------------------------------------------------------------------
    # Average profiles in each cell
    # ------------------------------------------------------------------

    for j, lat_center in enumerate(lat_centers):

        lat_low = lat_center - lat_bin / 2.0
        lat_high = lat_center + lat_bin / 2.0

        # Normally bins are half-open [low, high). For the final latitude
        # bin, include the upper edge too so a profile exactly at +90
        # degrees is not lost.
        if j == len(lat_centers) - 1:
            in_lat = (
                (lat >= lat_low)
                & (lat <= lat_high)
            )
        else:
            in_lat = (
                (lat >= lat_low)
                & (lat < lat_high)
            )

        for i, lon_center in enumerate(lon_centers_unwrapped):

            lon_low = lon_center - lon_bin / 2.0
            lon_high = lon_center + lon_bin / 2.0

            # Longitude stays half-open. Because profile longitudes have
            # been normalized to [0, 360), 360 itself is represented by 0.
            in_lon = (
                (lon_unwrapped >= lon_low)
                & (lon_unwrapped < lon_high)
            )

            in_cell = keep & in_lat & in_lon

            count[j, i] = numpy.count_nonzero(in_cell)

            if count[j, i]:
                with warnings.catch_warnings():
                    # Some vertical levels may be NaN for every profile
                    # in an otherwise populated cell.
                    warnings.simplefilter(
                        'ignore',
                        RuntimeWarning,
                    )

                    mean[j, i] = numpy.nanmean(
                        data[in_cell],
                        axis=0,
                    )

    # ------------------------------------------------------------------
    # Package as xarray
    # ------------------------------------------------------------------

    return xarray.Dataset(
        {
            varname: (
                ('latitude', 'longitude', 'level'),
                mean,
            ),
            'n_profiles': (
                ('latitude', 'longitude'),
                count,
            ),
        },
        coords={
            'latitude': lat_centers,
            'longitude': lon_centers,
            'level': levels,
        },
        attrs={
            'method': (
                'bin mean of individual profiles in rectangular '
                'latitude/longitude cells — not a mapped product'
            ),
            'lat_bin': lat_bin,
            'lon_bin': lon_bin,
            'lat_step': lat_step,
            'lon_step': lon_step,
            'lat_range': (lat_min, lat_max),
            'lon_range': (
                lon_min,
                lon_max,
            ),
        },
    )