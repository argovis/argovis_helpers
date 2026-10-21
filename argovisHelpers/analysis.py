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
import numpy, numbers, math, copy, gsw, xarray, warnings, pandas

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
    time_bin=None,
    time_range=None,
    time_step=None,
):
    """
    Average profiles into rectangular latitude/longitude cells, optionally
    also binned in time.

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

    Time binning is off unless `time_bin` is given. When it is, profiles
    are additionally sorted into time bins, and the result gains a leading
    `time` dimension. Two kinds of time bin are supported:

      * fixed-width bins: `time_bin` is anything `pandas.Timedelta`
        understands, e.g. '30D', '7D', '12h', or a `pandas.Timedelta`.
        These behave exactly like the spatial bins: `time_step` defaults
        to `time_bin`, and a smaller `time_step` gives overlapping bins
        (e.g. time_bin='30D', time_step='10D' is a 30-day running mean
        sampled every 10 days).

      * calendar bins: `time_bin` is a pandas offset alias anchored to a
        calendar boundary, e.g. 'MS' (calendar months), '3MS' (calendar
        quarters), 'QS-DEC' (quarters starting in December, i.e. DJF,
        MAM, JJA, SON), 'YS' (calendar years), 'W' (weeks ending Sunday).
        Calendar bins have unequal widths, so `time_step` is not allowed
        and bins never overlap. Note that '7D' is a fixed-width bin while
        'W' is a calendar bin.

    If `time_range` is None, the bins cover the span of the profiles'
    timestamps: the first bin begins at the earliest timestamp (fixed
    width) or at the calendar boundary on or before it (calendar), and
    bins are added until the latest timestamp is covered. An explicit
    `time_range` must fit the bins exactly, as for latitude and
    longitude; for calendar bins both ends must lie on the calendar
    boundary (e.g. the first of a month for 'MS').

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

    time_bin : str or pandas.Timedelta, optional
        Time bin width: a fixed duration ('30D') or a calendar period
        ('MS', '3MS', 'YS'). Default None, meaning no time binning.

    time_range : (start, end), optional
        Time range covered by the bins; anything `pandas.Timestamp`
        accepts. Default is the span of the profiles' timestamps.

    time_step : str or pandas.Timedelta, optional
        Distance between adjacent time bin centers, for fixed-width time
        bins only. Defaults to `time_bin`, giving non-overlapping bins.

    Returns
    -------
    xarray.Dataset
        Without time binning:
          `<varname>` has dimensions (latitude, longitude, level).
          `n_profiles` has dimensions (latitude, longitude).

        With time binning:
          `<varname>` has dimensions (time, latitude, longitude, level).
          `n_profiles` has dimensions (time, latitude, longitude).
          `time_bnds` has dimensions (time, bnds) and holds the low and
          high edge of each time bin; the `time` coordinate is the bin
          center.

        Empty cells contain NaN and have n_profiles = 0.

    Notes
    -----
    If a step is smaller than its corresponding bin width, bins overlap and
    an individual profile may therefore contribute to more than one cell.

    Steps larger than bin widths are not allowed, since they would leave
    gaps between cells.

    Bins are half-open [low, high) in every dimension, except that the
    final latitude bin and the final time bin also include their upper
    edge, so a profile exactly at the end of the range is not lost.
    Longitude stays half-open because 360 is the same meridian as 0.

    Every profile must carry a timestamp when time binning is requested.
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
    #
    # The same routine serves fixed-width time bins, with everything
    # expressed in seconds instead of degrees.
    # ------------------------------------------------------------------

    def make_centers(range_min, range_max, bin_width, step, name, units='degrees'):

        range_width = range_max - range_min

        if bin_width > range_width and not numpy.isclose(
            bin_width, range_width
        ):
            raise ValueError(
                f"{name}_bin ({bin_width} {units}) is wider than the "
                f"requested {name} range ({range_width} {units})"
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
                    f"range width is {range_width} {units}, "
                    f"bin width is {bin_width} {units}, and "
                    f"step is {step} {units}. The first bin can begin at "
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
    # Time bins.
    #
    # Everything below produces two arrays of bin edges, `time_low` and
    # `time_high` (datetime64[ns]), one entry per time bin. When no time
    # binning was requested there is a single bin spanning all time, and
    # the time dimension is squeezed out of the result at the end.
    # ------------------------------------------------------------------

    if time_bin is None:

        if time_step is not None:
            raise ValueError("time_step requires time_bin")

        if time_range is not None:
            raise ValueError("time_range requires time_bin")

        # One bin covering everything; no timestamps needed.
        in_time_bins = [numpy.ones(len(profiles), dtype=bool)]
        time_low = time_high = None

    else:

        # Every profile must have a timestamp for time binning.
        timestamps = []
        for p in profiles:
            if p.rawdata.get('timestamp') is None:
                raise ValueError(
                    f"profile {p.id}: no timestamp — every profile must "
                    "carry one when time_bin is set"
                )
            timestamps.append(numpy.datetime64(p.timestamp, 'ns'))

        timestamps = numpy.asarray(timestamps, dtype='datetime64[ns]')

        # Classify time_bin: fixed-width (Timedelta) or calendar (offset).
        time_bin_delta = None
        time_bin_offset = None

        #
        # Strings go through to_offset rather than Timedelta: Timedelta
        # happily reads '3MS' as three milliseconds, whereas to_offset
        # reads it as three calendar months, which is what anyone
        # writing '3MS' means.
        if isinstance(time_bin, str):
            try:
                time_bin_offset = pandas.tseries.frequencies.to_offset(time_bin)
            except (ValueError, TypeError):
                raise ValueError(
                    f"time_bin={time_bin!r} is neither a duration "
                    "('30D', pandas.Timedelta) nor a calendar offset "
                    "alias ('MS', '3MS', 'YS')"
                )

            if isinstance(time_bin_offset, pandas.tseries.offsets.Tick):
                # e.g. '30D', '12h': a fixed duration after all.
                time_bin_delta = pandas.Timedelta(time_bin_offset)
                time_bin_offset = None
        else:
            try:
                time_bin_delta = pandas.Timedelta(time_bin)
            except (ValueError, TypeError):
                raise ValueError(
                    f"time_bin={time_bin!r} is neither a duration "
                    "('30D', pandas.Timedelta) nor a calendar offset "
                    "alias ('MS', '3MS', 'YS')"
                )

        if time_range is not None:
            if len(time_range) != 2:
                raise ValueError("time_range must be (start, end)")
            t_min = pandas.Timestamp(time_range[0])
            t_max = pandas.Timestamp(time_range[1])
            if t_min.tzinfo is not None or t_max.tzinfo is not None:
                raise ValueError(
                    "time_range must be timezone-naive, like profile "
                    "timestamps"
                )
            if not t_min < t_max:
                raise ValueError("time_range must satisfy start < end")
        elif len(timestamps) == 0:
            raise ValueError(
                "time_range must be given explicitly when there are no "
                "profiles to infer it from"
            )
        else:
            t_min = pandas.Timestamp(timestamps.min())
            t_max = pandas.Timestamp(timestamps.max())

        if time_bin_delta is not None:
            # --- fixed-width time bins, same geometry as lat/lon -------

            if time_bin_delta <= pandas.Timedelta(0):
                raise ValueError("time_bin must be positive")

            if time_step is None:
                time_step_delta = time_bin_delta
            else:
                time_step_delta = pandas.Timedelta(time_step)

            if time_step_delta <= pandas.Timedelta(0):
                raise ValueError("time_step must be positive")

            if time_step_delta > time_bin_delta:
                raise ValueError(
                    "time_step cannot exceed time_bin; that would leave "
                    "gaps between time bins"
                )

            bin_s = time_bin_delta.total_seconds()
            step_s = time_step_delta.total_seconds()

            if time_range is None:
                # Anchor the first bin at the earliest profile and add
                # bins until the latest profile is covered.
                span_s = (t_max - t_min).total_seconds()
                n_extra = max(0, math.ceil((span_s - bin_s) / step_s))
                t_max = t_min + time_bin_delta + n_extra * time_step_delta

            centers_s = make_centers(
                0.0,
                (t_max - t_min).total_seconds(),
                bin_s,
                step_s,
                "time",
                units="seconds",
            )

            time_low = numpy.asarray(
                [t_min + pandas.Timedelta(seconds=c - bin_s / 2.0) for c in centers_s],
                dtype='datetime64[ns]',
            )
            time_high = numpy.asarray(
                [t_min + pandas.Timedelta(seconds=c + bin_s / 2.0) for c in centers_s],
                dtype='datetime64[ns]',
            )

            time_bin_attr = str(time_bin_delta)
            time_step_attr = str(time_step_delta)

        else:
            # --- calendar time bins --------------------------------------

            if time_step is not None:
                raise ValueError(
                    "time_step is only supported for fixed-width time "
                    "bins; calendar bins such as 'MS' never overlap"
                )

            if time_range is None:
                # Start at the calendar boundary on or before the earliest
                # profile, then step forward one bin at a time until the
                # latest profile is covered. Stepping (rather than
                # rolling t_max forward) keeps multi-period bins such as
                # '3MS' on the lattice defined by the first edge.
                edges = [time_bin_offset.rollback(t_min)]
                while edges[-1] <= t_max:
                    edges.append(edges[-1] + time_bin_offset)
                edges = pandas.DatetimeIndex(edges)

            else:
                edges = pandas.date_range(t_min, t_max, freq=time_bin_offset)

                if (
                    len(edges) < 2
                    or edges[0] != t_min
                    or edges[-1] != t_max
                ):
                    raise ValueError(
                        f"time bins do not fit the requested range exactly: "
                        f"with time_bin={time_bin!r}, both ends of time_range "
                        f"must lie on the calendar boundary, and the range "
                        f"must contain a whole number of bins "
                        f"(got {t_min} to {t_max})"
                    )

            time_low = numpy.asarray(edges[:-1], dtype='datetime64[ns]')
            time_high = numpy.asarray(edges[1:], dtype='datetime64[ns]')

            time_bin_attr = time_bin_offset.freqstr
            time_step_attr = time_bin_offset.freqstr

        # Half-open [low, high), with the final bin closed on top so a
        # profile exactly at the end of the range is not lost.
        in_time_bins = []
        for k in range(len(time_low)):
            if k == len(time_low) - 1:
                in_time_bins.append(
                    (timestamps >= time_low[k]) & (timestamps <= time_high[k])
                )
            else:
                in_time_bins.append(
                    (timestamps >= time_low[k]) & (timestamps < time_high[k])
                )

    n_time = len(in_time_bins)

    # ------------------------------------------------------------------
    # Allocate output arrays
    # ------------------------------------------------------------------

    mean = numpy.full(
        (
            n_time,
            len(lat_centers),
            len(lon_centers_unwrapped),
            len(levels),
        ),
        numpy.nan,
        dtype=float,
    )

    count = numpy.zeros(
        (
            n_time,
            len(lat_centers),
            len(lon_centers_unwrapped),
        ),
        dtype=int,
    )

    # ------------------------------------------------------------------
    # Average profiles in each cell
    # ------------------------------------------------------------------

    for k, in_time in enumerate(in_time_bins):

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

                in_cell = keep & in_time & in_lat & in_lon

                count[k, j, i] = numpy.count_nonzero(in_cell)

                if count[k, j, i]:
                    with warnings.catch_warnings():
                        # Some vertical levels may be NaN for every profile
                        # in an otherwise populated cell.
                        warnings.simplefilter(
                            'ignore',
                            RuntimeWarning,
                        )

                        mean[k, j, i] = numpy.nanmean(
                            data[in_cell],
                            axis=0,
                        )

    # ------------------------------------------------------------------
    # Package as xarray
    # ------------------------------------------------------------------

    attrs = {
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
    }

    if time_bin is None:

        # No time binning requested: drop the singleton time axis so the
        # result is identical to what this function has always returned.
        return xarray.Dataset(
            {
                varname: (
                    ('latitude', 'longitude', 'level'),
                    mean[0],
                ),
                'n_profiles': (
                    ('latitude', 'longitude'),
                    count[0],
                ),
            },
            coords={
                'latitude': lat_centers,
                'longitude': lon_centers,
                'level': levels,
            },
            attrs=attrs,
        )

    # Bin centers as the time coordinate; edges kept alongside, CF-style,
    # since calendar bins have unequal widths and the centers alone would
    # not recover them.
    time_centers = time_low + (time_high - time_low) / 2

    attrs.update(
        {
            'method': (
                'bin mean of individual profiles in rectangular '
                'latitude/longitude/time cells — not a mapped product'
            ),
            'time_bin': time_bin_attr,
            'time_step': time_step_attr,
            'time_range_start': str(pandas.Timestamp(time_low[0])),
            'time_range_end': str(pandas.Timestamp(time_high[-1])),
        }
    )

    return xarray.Dataset(
        {
            varname: (
                ('time', 'latitude', 'longitude', 'level'),
                mean,
            ),
            'n_profiles': (
                ('time', 'latitude', 'longitude'),
                count,
            ),
            'time_bnds': (
                ('time', 'bnds'),
                numpy.stack([time_low, time_high], axis=1),
            ),
        },
        coords={
            'time': time_centers,
            'latitude': lat_centers,
            'longitude': lon_centers,
            'level': levels,
        },
        attrs=attrs,
    )

def remove_seasonal_and_trend(da, time_dim='timestamp', seasonal='climatology',
                              n_harmonics=2, trend=None, fit_period=None,
                              smooth_window=None, return_components=False):
    """
    Remove the seasonal cycle and (optionally) a trend from a time series or a
    time × space field, returning the anomaly.

    The climatology (time mean + seasonal cycle) and the trend are fit together
    by least squares, separately at every grid point, NaN-safe. Fitting them
    jointly avoids the bias of doing them one after the other when the record
    is short or starts/ends in a particular season.

    Parameters
    ----------
    da            : xarray.DataArray – series or field with a datetime `time_dim`
    time_dim      : str          – name of the time dimension (default 'timestamp')
    seasonal      : str or None  – 'climatology': one mean per calendar month
                                   'harmonic'   : annual harmonics (smoother; better
                                                  for short or gappy records)
                                   None         : remove the time mean only
    n_harmonics   : int          – number of harmonics if seasonal='harmonic'
                                   (1 = annual, 2 = annual + semi-annual, ...)
    trend         : str or None  – None, 'linear', or 'quadratic'
    fit_period    : (start, end) – optional; estimate climatology and trend from
                                   this period only (e.g. ('1993-01-01', '2012-12-31')),
                                   then remove them from the whole record
    smooth_window : int or None  – optional centred running mean of the anomaly,
                                   in time steps
    return_components : bool     – if True, return a Dataset with `anomaly`,
                                   `climatology` and `trend`

    Returns
    -------
    xarray.DataArray anomaly (same shape as `da`), or an xarray.Dataset of components.
    With seasonal='climatology' and trend=None this matches subtracting the
    monthly climatology.
    """
    if seasonal not in ('climatology', 'harmonic', None):
        raise ValueError("seasonal must be 'climatology', 'harmonic' or None")
    if trend not in (None, 'linear', 'quadratic'):
        raise ValueError("trend must be None, 'linear' or 'quadratic'")

    times = da[time_dim].to_index()
    if fit_period is None:
        in_fit = numpy.ones(len(times), dtype=bool)
    else:
        start, end = pandas.Timestamp(fit_period[0]), pandas.Timestamp(fit_period[1])
        in_fit = numpy.asarray((times >= start) & (times <= end))
        if not in_fit.any():
            raise ValueError(f'no time steps inside fit_period {fit_period}')

    # time in years, centred on the fit period (keeps the quadratic well-conditioned)
    t_fit = times[in_fit]
    t0 = t_fit.min() + (t_fit.max() - t_fit.min()) / 2
    t_yr = numpy.asarray((times - t0) / pandas.Timedelta(days=365.25), dtype=float)

    # design matrix: climatology columns first, then trend columns
    if seasonal == 'climatology':
        clim_cols = [(times.month == m).astype(float) for m in range(1, 13)]
    else:
        clim_cols = [numpy.ones_like(t_yr)]
        if seasonal == 'harmonic':
            for k in range(1, n_harmonics + 1):
                clim_cols += [numpy.cos(2 * numpy.pi * k * t_yr), numpy.sin(2 * numpy.pi * k * t_yr)]
    degree = {None: 0, 'linear': 1, 'quadratic': 2}[trend]
    trend_cols = [t_yr ** d for d in range(1, degree + 1)]
    X = numpy.column_stack(clim_cols + trend_cols)
    n_clim = len(clim_cols)

    def _components(coef, used):
        # coef: (n_columns, n_series); rows of unused columns are ignored
        clim = X[:, :n_clim] @ numpy.where(used[:n_clim, None], coef[:n_clim], 0)
        clim[X[:, :n_clim][:, ~used[:n_clim]].any(axis=1)] = numpy.nan  # unsampled months
        return clim, X[:, n_clim:] @ coef[n_clim:]

    # one row per grid point, time last
    da_t = da.astype(float).transpose(..., time_dim)
    y = da_t.values.reshape(-1, len(times))
    clim = numpy.full(y.shape, numpy.nan)
    trnd = numpy.full(y.shape, numpy.nan)

    finite = numpy.isfinite(y[:, in_fit])
    complete = finite.all(axis=1)
    if complete.any():
        # all grid points without gaps share the same fit: solve them in one call
        used = X[in_fit].any(axis=0)
        if in_fit.sum() > used.sum():
            coef = numpy.zeros((X.shape[1], complete.sum()))
            coef[used] = numpy.linalg.lstsq(X[in_fit][:, used], y[complete][:, in_fit].T, rcond=None)[0]
            c, t = _components(coef, used)
            clim[complete], trnd[complete] = c.T, t.T
    for i in numpy.flatnonzero(~complete & finite.any(axis=1)):
        # grid points with gaps: fit each one on its own valid samples
        ok = in_fit & numpy.isfinite(y[i])
        used = X[ok].any(axis=0)
        if ok.sum() <= used.sum():
            continue                                  # not enough points to fit
        coef = numpy.zeros((X.shape[1], 1))
        coef[used, 0] = numpy.linalg.lstsq(X[ok][:, used], y[i, ok], rcond=None)[0]
        c, t = _components(coef, used)
        clim[i], trnd[i] = c[:, 0], t[:, 0]

    clim = da_t.copy(data=clim.reshape(da_t.shape)).transpose(*da.dims)
    trnd = da_t.copy(data=trnd.reshape(da_t.shape)).transpose(*da.dims)

    anomaly = da - clim - trnd
    if smooth_window is not None:
        anomaly = anomaly.rolling({time_dim: smooth_window}, center=True).mean()

    if return_components:
        return xarray.Dataset({'anomaly': anomaly, 'climatology': clim, 'trend': trnd})
    return anomaly