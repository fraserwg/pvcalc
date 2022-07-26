from xgcm import Grid

boundz = 'fill'

def create_xgcm_grid(ds):
    grid = Grid(ds,
            periodic=[],
            coords={'X': {'left': 'XG', 'center': 'XC'},
                    'Y': {'left': 'YG', 'center': 'YC'},
                    'Z': {'right': 'Zl', 'center': 'Z', 'outer': 'Zp1', 'left': 'Zu'}})
    return grid

def create_drL_from_dataset(ds):
    drL = - ds['drC'].isel(Zp1=slice(0, -1)).rename({'Zp1': 'Zl'})
    return drL


def create_drU_from_dataset(ds):
    drU = - ds['drC'].isel(Zp1=slice(1, None)).rename({'Zp1': 'Zu'})
    

def calculate_density(da_rho_anom, da_rho_ref):
    da_rho = da_rho_anom + da_rho_ref
    return da_rho


def calculate_buoyancy(da_rho, rho_0=1000, g=9.81):
    da_b = - g * da_rho / rho_0
    return da_b


def calculate_grad_buoyancy(da_b, ds_grid, grid):
    dbdz = grid.diff(da_b, axis='Z', boundary='extend', to='right') / ds_grid['drL']
    dbdx = grid.diff(da_b, axis='X', boundary='extend') / ds_grid['dxC']
    dbdy = grid.diff(da_b, axis='Y', boundary='extend') / ds_grid['dyC']
    return dbdx, dbdy, dbdz


def calculate_curl_velocity(da_u, da_v, da_w, ds_grid, grid, no_slip_bottom, no_slip_sides):
    
    if no_slip_bottom:
        bottom_boundary_kwargs = {'boundary': 'fill',
                                   'fill_value': 0}
    else:
        bottom_boundary_kwargs = {'boundary': 'extend'}
        
    if no_slip_sides:
        lateral_boundary_kwargs = {'boundary': 'fill',
                                   'fill_value': 0}
    else:
        lateral_boundary_kwargs = {'boundary': 'extend'}
        
    dwdy = grid.diff(da_w, axis='Y', **lateral_boundary_kwargs) / ds_grid['dyC']
    dvdz = grid.diff(da_v, axis='Z', **bottom_boundary_kwargs, to='right') / ds_grid['drL']
    zeta_x = dwdy - dvdz

    dudz = grid.diff(da_u, axis='Z', **bottom_boundary_kwargs, to='right') / ds_grid['drL']
    dwdx = grid.diff(da_w, axis='X', **lateral_boundary_kwargs) / ds_grid['dxC']
    zeta_y = dudz - dwdx

    dvdx = grid.diff(da_v, axis='X', **lateral_boundary_kwargs) / ds_grid['dxV']
    dudy = grid.diff(da_u, axis='Y', **lateral_boundary_kwargs) / ds_grid['dyU']
    zeta_z = dvdx - dudy

    return zeta_x, zeta_y, zeta_z


def calculate_C_potential_vorticity(zeta_x, zeta_y, zeta_z, b, ds_grid, grid, beta, f0, fprime=0):
    """ clauclates the potential vorticity using the C-grid formula
    
    Notes
    -----
    * See Morel et al. (Ocean Modelling, 2019) for full details of
        the algorithm employed here.
    """
    
    zi_x = zeta_x
    zi_y = zeta_y + fprime
    zi_z = zeta_z + f0 + beta * ds_grid['YG']
    
    b_x = grid.interp(b, to={'Z': 'right', 'Y': 'left'}, axis=['Y', 'Z'], boundary='extend')
    b_y = grid.interp(b, to={'Z': 'right', 'X': 'left'}, axis=['X', 'Z'], boundary='extend')
    b_z = grid.interp(b, axis=['X', 'Y'], boundary='extend')
    
    zi_b_x = zi_x * b_x
    zi_b_y = zi_y * b_y
    zi_b_z = zi_z * b_z

    Q_x = grid.diff(zi_b_x, axis='X', boundary='extend') / ds_grid['dxV']
    Q_y = grid.diff(zi_b_y, axis='Y', boundary='extend') / ds_grid['dyU']
    Q_z = grid.diff(zi_b_z, to='right', axis='Z', boundary='extend') / ds_grid['drL']
    
    Q = Q_x + Q_y + Q_z
    return Q


def calculate_potential_vorticity(zeta_x, zeta_y, zeta_z, dbdx, dbdy, dbdz, ds_grid, grid, beta, f0):
    zeta_x_interp = grid.interp(zeta_x, axis=['Y', 'Z'], boundary=boundz)
    zeta_y_interp = grid.interp(zeta_y, axis=['X', 'Z'], boundary=boundz)
    zeta_z_interp = grid.interp(zeta_z.chunk({'Z': -1}), axis=['X', 'Y'], boundary=boundz)
    
    f = f0 + beta * ds_grid['YC']

    dbdx_interp = grid.interp(dbdx, axis=['X'], boundary=boundz)
    dbdy_interp = grid.interp(dbdy, axis=['Y'], boundary=boundz)
    dbdz_interp = grid.interp(dbdz, axis=['Z'], boundary=boundz)
    
    # I have this here only for debugging
    # Q is a scalar and does NOT have components
    Q_x = dbdx_interp * zeta_x_interp
    Q_y = dbdy_interp * zeta_y_interp
    Q_z = dbdz_interp * (zeta_z_interp + f)
    Q = Q_x + Q_y + Q_z
    return Q