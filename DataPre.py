import pandas as pd
from rasterio import features, Affine
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.enums import Resampling

# Load data
data = pd.read_csv("data/data.txt", sep="")

# Number of damaged cells per year
outbr_yr = data[data['bb.kill'] > 0].groupby('year').size().reset_index(name='n')

# Convert data to grids for all years of the outbreak
all_grids = []
for year in outbr_yr['year']:
    tst = data[data['year'] == year]
    cdf = pd.DataFrame({'x': tst['x.coord'], 'y': tst['y.coord'], 'z': tst['bb.kill']})
    r = rasterio.features.rasterize(
        ((x, y) for x, y in zip(cdf['x'], cdf['y'])),
        out_shape=(len(all_grids[-1][0]) if all_grids else None, len(all_grids[-1][1]) if all_grids else None),
        transform=all_grids[-1][2] if all_grids else None,
        fill=0,
        all_touched=True
    )
    if all_grids:
        r = r + all_grids[-1][3]
    else:
        r = extend(r, Affine.identity())
    all_grids.append((r, year))

# Add information of damages of previous years
all_grids_prep = all_grids.copy()
for i in range(1, len(all_grids_prep)):
    all_grids_prep[i] = (np.maximum(all_grids[i][0] * 1, all_grids[i - 1][0] * 2), all_grids[i][1])

# Prepare climate proxies
tst = data[data['year'] == 1990]
cdf = pd.DataFrame({'x': tst['x.coord'], 'y': tst['y.coord'], 'z': tst['S3.temp.yr']})
temp_grid = rasterio.features.rasterize(
    ((x, y) for x, y in zip(cdf['x'], cdf['y'])),
    out_shape=(len(all_grids[0][0]), len(all_grids[0][1])),
    transform=all_grids[0][2],
    fill=0,
    all_touched=True
)

# There are only minimal spatial differences in precipitation pattern
# Use a single value for the whole area for every year
prec_anomaly = data.groupby('year')['S1.JJA.prec'].mean().to_dict()

# Extract examples
dimpx = 9
indices = [(x, y) for x in range(-dimpx, dimpx + 1) for y in range(-dimpx, dimpx + 1)]

all_points = pd.DataFrame()
for year in outbr_yr['year']:
    print(year)
    r, _ = all_grids_prep[year - outbr_yr['year'].min()]
    pts = np.where(r == 1)
    no_pts = np.where(r == 0)

    tst = np.array([r[p[0] + x, p[1] + y] for p in pts for x, y in indices])
    pts_df = pd.DataFrame(tst)

    tst = np.array([r[p[0] + x, p[1] + y] for p in no_pts for x, y in indices])
    pts_df = pd.concat([pts_df, pd.DataFrame(tst)])

    # Add proxies
    pts_df['temp'] = temp_grid[pts[0], pts[1]].tolist() + temp_grid[no_pts[0], no_pts[1]].tolist()
    pts_df['damage'] = [1] * len(pts[0]) + [0] * len(no_pts[0])
    pts_df['year'] = year
    pts_df['precanomaly'] = prec_anomaly[year]
    pts_df['pts'] = list(pts[0]) + list(no_pts[0])
    all_points = pd.concat([all_points, pts_df])

all_points['year'] = all_points['year'] - 1989
all_points['V181'] = 0

# Examples used for training should only contain damages from previous years
all_points.iloc[:, 0:361][all_points.iloc[:, 0:361] == 0] = 1
all_points = all_points.fillna(0)

# Fit into 8-bit integer
all_points['temp'] = (all_points['temp'] * 10).astype(int)
all_points['precanomaly'] = (all_points['precanomaly'] / 5).astype(int)

# Add the outbreak level as a numeric value
outbreak_level = data[data['ID'] == 1][['year', 'S1.outbreak']]
outbreak_level['year'] = outbreak_level['year'] - 1989
outbreak_code = outbreak_level['S1.outbreak'].astype(int).tolist()
all_points['code'] = outbreak_code

# Save data sets
all_points_shf = all_points.sample(frac=1)

# Experiment 1
all_points_shf[all_points_shf['year'] + 1989].isin([1993, 1997, 2005]).to_csv("bbyearseval.txt", index=False, header=False, sep=" ")
all_points_shf[~(all_points_shf['year'] + 1989).isin([1993, 1997, 2005])].to_csv("bbyearstrain.txt", index=False, header=False, sep=" ")

# Experiment 2
train_size = int(len(all_points_shf) * 0.8)
all_points_shf.iloc[:train_size, :].to_csv("bbtrainall.txt", index=False, header=False, sep=" ")
all_points_shf.iloc[train_size:, :].to_csv("bbevalall.txt", index=False, header=False, sep=" ")
