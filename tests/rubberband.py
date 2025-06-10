'''
Test planktos on various rubberband simulations
'''

import numpy as np
import planktos

rndState = np.random.default_rng(1)

envir = planktos.environment(1,1)

###### Poroelastic rubberband #######
envir.read_IB2d_fluid_data('poroelastic_rb/viz_IB2d', 1e-4, 20)
envir.read_IB2d_mesh_data('poroelastic_rb/viz_IB2d', 1e-4, 20, periodic=True)

start_positions = rndState.random((100,2))*0.2 + 0.4
swrm = planktos.swarm(envir=envir, init=start_positions, 
                      ib_condition='sliding', seed=1)
swrm.shared_props['cov'] *= 0.01
# swrm.props['color'] = np.full(100, swrm.shared_props['color'])
# swrm.plot()

# agent 56 is an escapee!
# swrm.props.loc[56, 'color'] = 'orange'
for ii in range(200): #200
    swrm.move(0.001)
swrm.plot_all('mvib2d_sliding.mp4')

###### Damped springs rubberband ######
# envir.read_IB2d_fluid_data('damped_rb/viz_IB2d', 1e-3, 20)
# envir.read_IB2d_mesh_data('damped_rb/viz_IB2d', 1e-3, 20, periodic=True)

# start_positions = rndState.random((100,2))*0.2 + 0.4
# swrm = planktos.swarm(envir=envir, init=start_positions, 
#                       ib_condition='sliding', seed=1)
# swrm.shared_props['cov'] *= 0.005
# # swrm.plot()
# for ii in range(150):
#     swrm.move(0.01)
# swrm.plot_all('mvib2d_slide_damped.mp4')

###### Moving rubberband ######
# envir.read_IB2d_fluid_data('moving_rb/viz_IB2d', 1e-4, 5)
# envir.read_IB2d_mesh_data('moving_rb/viz_IB2d', 1e-4, 5, periodic=True)

# start_positions = rndState.random((100,2))*0.3 + 0.1
# swrm = planktos.swarm(envir=envir, init=start_positions, 
#                       ib_condition='sliding', seed=1)
# swrm.shared_props['cov'] *= 0.05
# # swrm.plot()
# for ii in range(100):
#     swrm.move(0.0005)
# swrm.plot_all('mvib2d_slide_moving.mp4')