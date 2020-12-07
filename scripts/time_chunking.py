import parcels
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4
from datetime import timedelta
import numpy as np
import xarray as xr
import time
from glob import glob
import psutil
import os

p_version = parcels.version[:5]
print(p_version)
runtime = timedelta(days=30)
lonmin = -30
lonmax = 20
latmin = -40
latmax = -20

directory = '/data/oceanparcels/input_data/CMEMS/GLOBAL_ANALYSIS_FORECAST_PHY_001_024_SMOC/'
filelist = sorted(glob(directory+'SMOC_201701*'))

n_particles = [1000]

chunksize = [128, 256, 'auto', False, 'indices']

func_time = np.zeros((len(chunksize),len(n_particles)))
mem_used_GB = np.zeros((len(chunksize),len(n_particles)))
loaded_chunks = np.zeros((len(chunksize),len(n_particles),2))

variables = {'U': 'utotal',
             'V': 'vtotal'}
dimensions = {'lon': 'longitude',
              'lat': 'latitude',
              'depth': 'depth',
              'time': 'time'}


for i, cs in enumerate(chunksize):
    if cs == 'indices':
        fielddata = filelist[0]
        ds = xr.open_dataset(fielddata)

        iy_min = np.argmin(np.abs(ds['latitude']-latmin).values)
        print(iy_min)
        iy_max = np.argmin(np.abs(ds['latitude']-latmax).values)
        print(iy_max)
        ix_min = np.argmin(np.abs(ds['longitude']-lonmin).values)
        print(ix_min)
        ix_max = np.argmin(np.abs(ds['longitude']-lonmax).values)
        print(ix_max)
        indices = {'lat': range(iy_min, iy_max), 'lon': range(ix_min, ix_max)}

        fieldset = FieldSet.from_netcdf(filelist, variables, dimensions, indices=indices)
    else:
        if p_version in ['2.2.0','2.2.1']:
            if cs not in ['auto', False]:
                cs = (1, 1, cs, cs)
            fieldset = FieldSet.from_netcdf(filelist, variables, dimensions, field_chunksize=cs)
        else:
            if cs not in ['auto', False]:
                cs = {'time':('time',1), 'depth':('depth',1),'lat':('latitude',cs),'lon':('longitude',cs)}
            fieldset = FieldSet.from_netcdf(filelist, variables, dimensions, chunksize=cs)

    for j, npart in enumerate(n_particles):
        lons = np.linspace(-10, 10, npart)
        lats = np.linspace(-30, -35, npart)
        pset = ParticleSet(fieldset = fieldset,
                           pclass = JITParticle,
                           lon = lons,
                           lat = lats)
        
        #output_file = pset.ParticleFile(name=f'/scratch/rfischer/chunking_{npart}_{runtime.days}.nc', outputdt=timedelta(hours=1))

        tic = time.time()
        pset.execute(AdvectionRK4, runtime=runtime, dt=timedelta(hours=1))
        func_time[i,j] = time.time()-tic
        print(func_time[i,j])
        process = psutil.Process(os.getpid())
        mem_B_used = process.memory_info().rss
        mem_used_GB[i,j] = mem_B_used / (1024*1024)
        print(fieldset.U.grid.load_chunk)
        loaded_chunks[i,j,0]=len(fieldset.U.grid.load_chunk[fieldset.U.grid.load_chunk>0])
        loaded_chunks[i,j,1]=len(fieldset.U.grid.load_chunk)
np.save(f'n_particles', n_particles)
np.save(f'chunksizes', chunksize)
np.save(f'execute_time_v{p_version}_{runtime.days}d', func_time)
np.save(f'execute_mem_v{p_version}_{runtime.days}d', mem_used_GB)
np.save(f'execute_chunks_v{p_version}_{runtime.days}d', loaded_chunks)
