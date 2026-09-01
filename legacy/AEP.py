import numpy as np
from floris.tools import FlorisInterface, WindRose
from floris.simulation import flow_field
from floris.tools.optimization.layout_optimization.layout_optimization_scipy import (
    LayoutOptimizationScipy,
)
from scipy.interpolate import NearestNDInterpolator,griddata
from scipy.spatial.distance import cdist
import pandas as pd
from time import perf_counter as timerpc
import floris.tools.visualization as wakeviz
import matplotlib.pyplot as plt
import utilities as uti
from floris.tools.visualization import plot_rotor_values
from floris.tools.optimization.yaw_optimization.yaw_optimizer_scipy import YawOptimizationScipy
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import YawOptimizationSR
import time
import copy
import configBinary as cf
from sklearn.cluster import KMeans
from floris.tools import ParallelComputingInterface
import os
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.ops import unary_union
from math import acos, cos

np.random.seed(42)

# def mooring_costs(nturb,nlines,bathyMap,xCoords,yCoords):
#     for x,y in zip(xCoords,yCoords):
#
#         total=nturb*nlines*(c_anchor + 1.5*(aep.getBathy(x,y) + 410)*c_line + 50*c_chain)
#     return total

def calc2DShort(flint, Xcoords, Ycoords, freq, wd_array, ws_array, parallel=False, suffixW="", viz=(False,0,7)):
    fi = copy.deepcopy(flint)

    # Inizializzazione e calcolo AEP
    fi.reinitialize(layout_x=Xcoords, layout_y=Ycoords, wind_directions=wd_array, wind_speeds=ws_array)
    AEP = fi.get_farm_AEP(freq=freq)

    if viz[0]:
        plotDir = int(viz[1])
        plotSpeed = int(viz[2])
        hubh = fi.floris.farm.hub_heights[0]

        MIN_WS = 1.0
        MAX_WS = plotSpeed

        # Calcolo del piano orizzontale
        horizontal_plane = fi.calculate_horizontal_plane(x_resolution=200, y_resolution=200, height=hubh, wd=[plotDir], ws=[plotSpeed])
        fig, ax = plt.subplots(figsize=(11, 9))

        # Visualizzazione del piano di taglio e delle turbine
        wakeviz.visualize_cut_plane(horizontal_plane, ax=ax, color_bar=True, min_speed=MIN_WS, max_speed=MAX_WS)
        wakeviz.plot_turbines_with_fi(fi, ax, color="c", wd=[plotDir])

        # Calcolo dei tick e delle etichette degli assi in diametri
        D = 240  # Diametro della turbina in metri
        tick_values_x = np.linspace(min(Xcoords), max(Xcoords), num=10)  # Adattare il numero di tick se necessario
        tick_values_y = np.linspace(min(Ycoords), max(Ycoords), num=10)  # Adattare il numero di tick se necessario
        tick_labels_x = [str(int(x / D)) for x in tick_values_x]
        tick_labels_y = [str(int(y / D)) for y in tick_values_y]

        # Applica i nuovi valori dei tick e le etichette agli assi
        ax.set_xticks(tick_values_x)
        ax.set_xticklabels(tick_labels_x)
        ax.set_yticks(tick_values_y)
        ax.set_yticklabels(tick_labels_y)

        # Aggiorna le etichette degli assi per indicare che la misura è in diametri
        ax.set_xlabel("X (D)", fontsize=18)
        ax.set_ylabel("Y (D)", fontsize=18)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        # Salva il grafico
        plt.savefig(os.path.join(cf.plotsDir, f"wakes_{suffixW}.png"))
        plt.close()
        plt.clf()

    return AEP
def get_farm_LSEP_slow(flint, farm, freq, wd_array, ws_array):
    fi = copy.deepcopy(flint)
    fi.reinitialize(layout_x=farm.x.values, layout_y=farm.y.values, wind_directions=wd_array, wind_speeds=ws_array)
    AEPs_actualized=[]
    for v in range (1,(cf.lifetime+1)):
        de_rating= (cf.MAX_Derating / cf.lifetime) * v
        weights=np.ones((len(wd_array), len(ws_array), len(farm.x.values)))
        for ws_index, ws in enumerate(ws_array):
            if ws <= cf.ratedSpeed:
                weights[:, ws_index, :] *= (1 - de_rating)
        AEPs_actualized.append(fi.get_farm_AEP(freq=freq,turbine_weights=weights) *((1+cf.r)**(-v)/1e9))
    LSEP=np.sum(AEPs_actualized)
    return LSEP



def get_farm_LSEP(flint,farm,freq,wd_array,ws_array):
    fi = copy.deepcopy(flint)
    fi.reinitialize(layout_x=farm.x.values, layout_y=farm.y.values, wind_directions=wd_array, wind_speeds=ws_array)
    weights = np.ones((len(wd_array), len(ws_array), len(farm.x.values)))
    AEPs_actualized = []
    fi.calculate_wake()
    de_rating_annual_increment = cf.MAX_Derating / cf.lifetime
    turbine_powers_NO_weights = fi.get_turbine_powers()
    for v in range(1, (cf.lifetime+1)):
        de_rating = de_rating_annual_increment * v
        for ws_index, ws in enumerate(ws_array):
            if ws <= cf.ratedSpeed:
                weights[:, ws_index, :] *= (1 - de_rating)
        turbine_powers_weights = np.multiply(weights, turbine_powers_NO_weights)
        farm_power=np.sum(turbine_powers_weights, axis=2)
        aep = np.sum(np.multiply(freq, farm_power) * 365 * 24)
        AEPs_actualized.append(aep*((1+cf.r)**(-v)/1e9))
    LSEP = np.sum(AEPs_actualized)
    return LSEP



def calc2DShort1(flint,Xcoords,Ycoords,freq,wd_array,ws_array,parallel=False,suffixW="",viz=(False,0,7)):
    fi=copy.deepcopy(flint)

    # Pour this into a parallel computing interface
    parallel_interface = "concurrent"
    fi.reinitialize(layout_x=Xcoords,layout_y=Ycoords, wind_directions=wd_array, wind_speeds=ws_array)
    #if parallel:
    #    fi_aep_parallel = ParallelComputingInterface(
    #        fi=fi,
    #        max_workers=cf.maxworkers,
    #        n_wind_direction_splits=cf.ndirs,
    #        n_wind_speed_splits=cf.ndirs,
    #        interface=parallel_interface,
    #        print_timings=False,
    #    )
    #    #yawangles = yaws.reshape((len(wd_array), len(ws_array), len(Ycoords)))
    #    AEP=fi_aep_parallel.get_farm_AEP(freq=freq)
    #else:
    AEP=fi.get_farm_AEP(freq=freq)
    if viz[0]:
        #plotDir = wd_array[int(viz[1])]
        plotDir = int(viz[1])
        plotSpeed = int(viz[2])
        #plotSpeed = ws_array[int(viz[2])]
        #plotYaw=yawangles[plotDirIndex,plotSpeedIndex,:].reshape((1,1,len(Ycoords)))
        #timestr = time.strftime("%Y%m%d-%H%M%S")
        #print("current time:-", timestr)

        #uti.fig_horizontal(fi, plotDir, plotSpeed, plotDirIndex, plotSpeedIndex, "test", f"Plots/layout_{suffixW}.png")

        #fi.reinitialize(wind_directions=[plotDir], wind_shear=0.2,
        #                wind_speeds=[plotSpeed])

        # configurazione baseline, rispetto la direzione preponderante del vento di 300°

        hubh = fi.floris.farm.hub_heights[0]

        MIN_WS = 1.0
        MAX_WS = plotSpeed

        horizontal_plane = fi.calculate_horizontal_plane(x_resolution=200, y_resolution=200, height=hubh,wd=[plotDir],ws=[plotSpeed])
        fig, ax = plt.subplots(figsize=(11, 9))
        ax.set_xlabel("X (m)", fontsize=18)
        ax.set_ylabel("Y (m)",fontsize=18)
        plt.yticks(fontsize=16)
        plt.xticks(fontsize=16)
        wakeviz.visualize_cut_plane(horizontal_plane, ax=ax, color_bar=True, min_speed=MIN_WS,
                                    max_speed=MAX_WS)

        wakeviz.plot_turbines_with_fi(fi, ax, color="c",wd=[plotDir])
        # wakeviz.plot_turbines_with_fi_no_rot(fi, ax, color="c", wd=[plotDir])
        # wakeviz.add_turbine_id_labels(fi, ax, color="w", backgroundcolor="k")
        # wakeviz.add_turbine_id_labels_no_rot(fi, ax, color="w", backgroundcolor="k")

        # plt.show()
        plt.savefig(os.path.join(cf.plotsDir,f"wakes_{suffixW}.png"))
        plt.close()
        plt.clf()
    return AEP

def calc2DFarm(flint,farm,freq,wd_array,ws_array,parallel=False,suffixW="",viz=(False,0,7)):
    fi=copy.deepcopy(flint)
    AEP=0
    AEP_nowake=0
    parallel_interface = "concurrent"
    fi.reinitialize(layout_x=farm.x.values,layout_y=farm.y.values,wind_speeds=ws_array,wind_directions=wd_array)
    #if parallel:
    #    fi_aep_parallel = ParallelComputingInterface(
    #        fi=fi,
    #        max_workers=cf.maxworkers,
    #        n_wind_direction_splits=cf.maxworkers,
    #        n_wind_speed_splits=cf.maxworkers,
    #        interface=parallel_interface,
    #        print_timings=False,
    #    )
        #yawangles = yaws.reshape((len(wd_array), len(ws_array), len(Ycoords)))
    #    AEP=fi_aep_parallel.get_farm_AEP(freq=freq)
    #    AEP_nowake=AEP
    #    #AEP_nowake=fi_aep_parallel.get_farm_AEP(freq=freq,no_wake=True)
    #else:
    AEP=fi.get_farm_AEP(freq=freq)
    #AEP_nowake=AEP
    AEP_nowake = fi.get_farm_AEP(freq=freq,no_wake=True)

    if viz[0]:
        plotDir = wd_array[int(viz[1])]
        plotSpeed = ws_array[int(viz[2])]
        #plotYaw=yawangles[plotDirIndex,plotSpeedIndex,:].reshape((1,1,len(Ycoords)))
        timestr = time.strftime("%Y%m%d-%H%M%S")
        #print("current time:-", timestr)

        #uti.fig_horizontal(fi, plotDir, plotSpeed, plotDirIndex, plotSpeedIndex, "test", f"Plots/layout_{suffixW}.png")

        fi.reinitialize(wind_directions=[plotDir], wind_shear=0.2,
                        wind_speeds=[plotSpeed])

        # configurazione baseline, rispetto la direzione preponderante del vento di 300°

        hubh = fi.floris.farm.hub_heights[0]

        MIN_WS = 1.0
        MAX_WS = plotSpeed

        horizontal_plane = fi.calculate_horizontal_plane(x_resolution=200, y_resolution=200, height=hubh,wd=[plotDir],ws=[plotSpeed])
        fig, ax = plt.subplots(figsize=(11, 9))
        wakeviz.visualize_cut_plane(horizontal_plane, ax=ax, title="test", color_bar=True, min_speed=MIN_WS,
                                    max_speed=MAX_WS)

        wakeviz.plot_turbines_with_fi(fi, ax, color="c",wd=[plotDir])
        # wakeviz.plot_turbines_with_fi_no_rot(fi, ax, color="c", wd=[plotDir])
        # wakeviz.add_turbine_id_labels(fi, ax, color="w", backgroundcolor="k")
        # wakeviz.add_turbine_id_labels_no_rot(fi, ax, color="w", backgroundcolor="k")

        # plt.show()
        plt.savefig(os.path.join(cf.plotsDir,f"wakes_{suffixW}.png"))
        plt.close()
        plt.clf()

    return AEP,AEP_nowake

def get_farm_VI(farm,wd_array,obs_coords,obs_coords_weights,obs_heights,freq,return_list=False):
    #Ipotesi: dSoP è sempre 1
    R = cf.Earth_radius
    ht = cf.hubHeight
    D = cf.rotorDiameter
    dsop = 1
    xfov = 120*np.pi/180
    zfov = 40*np.pi/180
    layout_x = farm.x.values
    layout_y = farm.y.values
    VI_list = []
    VI_final = []
    for cord, weight, height in zip(obs_coords, obs_coords_weights, obs_heights):
        x_obs = cord[0]
        y_obs = cord[1]
        gamma_hor = acos(R/(R+height))
        horizon_distance = gamma_hor * R
        li = calcDistancesFarm(farm, fromPoint=[x_obs,y_obs]) #shape=(n_turb) #distanze turbina-obspoint
        gamma_i = li/R
        hdi = np.where(li <= horizon_distance, 0, (R / np.cos(gamma_i - gamma_hor)) - R)
        zi = (ht-hdi)/li #shape=(n_turb) altezze hub in funzione di l
        #zi=ht/li
        VI_dir = np.ones(len(wd_array)) #shape=(n_wd)
        num_dir = np.ones(len(wd_array)) #shape=(n_wd)
        ziD = D/li #shape=(n_turb) ingombro verticale rotore in funzione di l
        ziTot = zi+(ziD/2) #shape=(n_turb) ingombro verticale totale in funzione di l
        xc = (layout_x.max()+layout_x.min())/2 #coordinate centro farm
        yc = (layout_y.max()+layout_y.min())/2
        theta_i = np.arctan2(layout_x-x_obs,layout_y-y_obs) #shape=(n_turb)
        theta_fv = np.arctan2(xc-x_obs,yc-y_obs) #shape=(n_turb)
        delta_i = theta_i-theta_fv #shape=(n_turb)
        xi = dsop*delta_i #shape=(n_turb) proiezioni delle posizioni dell'albero delle turbine
        #sort the array xi
        #xi=xi[np.argsort(xi)] #shape=(n_turb) ordinate da sinistra a destra
        indices = np.argsort(xi)
        xi = xi[indices]
        ziTot = ziTot[indices]
        freq_dir = np.sum(freq, axis=1)
        rettangoli=[]
        VI_dir_list = []
        for dir, theta_rot in enumerate(wd_array):
            theta_rot_radians = np.deg2rad(theta_rot)
            #theta_i_radians = np.deg2rad(theta_i)
            
            phi = theta_rot_radians - theta_i
            #phi = theta_rot - theta_i
            cos_phi = np.abs(np.cos(phi))
            xiD = D / li * cos_phi #ingombro orizzontale rotore
            xiL = xi - xiD / 2
            xiR = xi + xiD / 2
            rettangoli=[]       ##########correction for rebuttal!!!!!!!!!!!!!!
            for i in range(len(layout_x)):
                minx = xiL[i]
                maxx = xiR[i]
                miny = 0  
                maxy = ziTot[i]
                rettangoli.append(box(minx, miny, maxx, maxy))
                
            # Create a mask for the conditions
            # mask = np.zeros((len(xi), len(xi)), dtype=bool)
            # for i in range(len(xi)):
            #     for j in range(len(xi)):
            #         if j != i and ziTot[j] > ziTot[i]: #va corretto credo perchè zi non è ordinato e serve ziTot
            #             mask[i, j] = (xiL[i] < xiL[j] < xiR[i]) or (xiL[i] < xiR[j] < xiR[i])
            # # Apply the mask to xiR and xiL
            # xiR_masked = np.where(mask, xiR[:, None], np.inf)
            # xiL_masked = np.where(mask, xiL[:, None], -np.inf)
            # Calculate IiR and Iil
            # IiR = np.min(xiR_masked, axis=1)
            # Iil = np.max(xiL_masked, axis=1)
            # Ii = Iil - IiR
            num_dir[dir]= (unary_union(rettangoli).area)
            VI_dir[dir]= ((unary_union(rettangoli).area) / (xfov * zfov)) 
            VI_dir_list.append(VI_dir[dir])
            VI_dir[dir] = VI_dir[dir] * freq_dir[dir]
            #VI_dir[dir] = (np.sum(Ii * ziTot) / xfov * zfov) * freq_dir[dir]
        VI_final.append(VI_dir_list)
        VI_obs = np.sum(VI_dir) * weight 
        VI_list.append(VI_obs)
    VI=np.sum(VI_list)
    print("")
    if return_list:
        return VI,VI_final
    else:
        return VI


def calc2DFarmNOwake(flint,farm,freq,wd_array,ws_array):
    fi = copy.deepcopy(flint)

    fi.reinitialize(layout_x=farm.x.values, layout_y=farm.y.value)#,wind_speeds=ws_array,wind_directions=wd_array)
    # yawangles = yaws.reshape((len(wd_array), len(ws_array), len(Ycoords)))
    #fi.calculate_wake()
    AEP = fi.get_farm_AEP(freq=freq,no_wake=True)
    return AEP

def calc2DYawShort(fi,Xcoords,Ycoords,yaws,freq,wd_array,ws_array,xu,xl,suffixW="",viz=False):

    fi.reinitialize(layout_x=Ycoords,layout_y=Xcoords, wind_directions=wd_array, wind_speeds=ws_array)
    yawangles = yaws.reshape((len(wd_array), len(ws_array), len(Ycoords)))
    fi.calculate_wake(yaw_angles=yawangles)
    AEP=fi.get_farm_AEP(freq=freq)

    if viz:
        plotDirIndex = 0
        plotSpeedIndex = 7
        plotDir = wd_array[plotDirIndex]
        plotSpeed = ws_array[plotSpeedIndex]
        plotYaw=yawangles[plotDirIndex,plotSpeedIndex,:].reshape((1,1,len(Ycoords)))
        timestr = time.strftime("%Y%m%d-%H%M%S")
        #print("current time:-", timestr)
        fi.reinitialize(wind_directions=[plotDir], wind_shear=0.2,
                        wind_speeds=[plotSpeed])
        uti.fig_horizontalYaw(fi, plotDir, plotSpeed,plotDirIndex,plotSpeedIndex, "test", f"Plots/test_{suffixW}.png",yawAngles=plotYaw)

    return AEP

def calcDistances(Xcoords):
    pa=np.array([Xcoords , np.zeros_like(Xcoords)])
    pb= np.array([Xcoords, np.zeros_like(Xcoords)])
    d=cdist(pa.T,pb.T,metric="euclidean")
    return d

def calcDistancesNew(Xcoords,Ycoords,fromPoint=None):
    pa=np.array([Xcoords , Ycoords])
    pb= np.array([Xcoords , Ycoords]).T
    if fromPoint is not None:
        pb=fromPoint
    d=cdist(pa.T,pb,metric="euclidean")
    return d

def calcDistancesBinary(X,fromPoint=None):
    Xcoords,Ycoords=get_coords_from_binary_layout(X)
    pa=np.array([Xcoords , Ycoords])
    pb= np.array([Xcoords , Ycoords]).T
    if fromPoint is not None:
        pb=fromPoint
    d=cdist(pa.T,pb,metric="euclidean")
    return d

def calcDistancesFarm(farm,fromPoint=None):
    # pa=np.array([farm.x , farm.y])
    # pb= np.array([farm.x , farm.y]).T
    pa=farm[["x","y"]].values
    pb=pa
    if fromPoint is not None:
        #pb=np.tile(fromPoint,(len(farm),1)).T
        d = np.sqrt(((pa - fromPoint) ** 2).sum(axis=1))
    else:
        d=cdist(pa,pb,metric="euclidean")
    return d

def iniFloriswithroseclass():
    wind_rose = WindRose()
    wind_rose.read_wind_rose_csv("Rosa_dei_venti_Stretto_di_SIcilia.csv")

    # Show the wind rose
    wind_rose.plot_wind_rose()


def initFloris(farmfile,windfile):
    # case = "Sicilia_15"# case = "Sicilia_15"
    #fi = FlorisInterface(farmfile)
    # D = fi.floris.farm.rotor_diameters[0]
    df_wr = pd.read_csv(windfile)
    # bins=[i for i in range(26)]+[50]
    # dflist=[]
    # for b in bins:
    #     cp=df_wr.copy()
    #     sub=cp.query("ws==@b")
    #     sub["freq_val"]=sub["freq_val"].values[::-1]
    #     dflist.append(sub)
    # df_wrw=pd.concat(dflist,axis=0)
    # df_wr=df_wrw
    wd_array = np.array(df_wr["wd"].unique(), dtype=float)
    ws_array = np.array(df_wr["ws"].unique(), dtype=float)
    wd_grid, ws_grid = np.meshgrid(wd_array, ws_array, indexing="ij")
    freq_interp = NearestNDInterpolator(df_wr[["wd", "ws"]], df_wr["freq_val"])
    freq = freq_interp(wd_grid, ws_grid)
    freq = freq / np.sum(freq)
    fi = FlorisInterface(farmfile)
    fi.reinitialize(wind_speeds=ws_array, wind_directions=wd_array)
    
    # print(cf.casename)
    # print(wd_array[7])
    # print(ws_array[7])
    # print(freq[7, 7])

    # nturbs = n_turbs
    # yaw_cols = ["yaw_{:03d}".format(ti) for ti in range(nturbs)]
    #
    # if "yaw_000" not in df_wr.columns:
    #     df_wr[yaw_cols] = 0.0  # Add zeros
    # # Map angles from dataframe onto floris wind direction/speed grid
    # yaw_angles = np.array(df_wr[yaw_cols], dtype=float)
    # yaw_interp = NearestNDInterpolator(df_wr[["wd", "ws"]], yaw_angles)
    # yaw_angles_floris = yaw_interp(wd_grid, ws_grid)

    return fi,freq,wd_array,ws_array#,yaw_angles_floris

def loadBathyMap(xmin,xmax,ymin,ymax):
    from PIL import Image
    im = Image.open('area_1_elevation_w_bathymetry.tif')
    imarray = np.array(im)
    img_x = np.linspace(xmin, xmax, imarray.shape[0])
    img_y = np.linspace(ymin, ymax, imarray.shape[1])
    # x_array=np.arange(xmin, xmax, 1)
    # y_array = np.arange(ymin, ymax, 1)
    x_grid, y_grid = np.meshgrid(img_x, img_y, indexing="ij")
    bathy_interp = NearestNDInterpolator(list(zip(img_x, img_y)),imarray)
    bathyMap = bathy_interp(img_x, img_x)

    center=bathy_interp(0, 0)

    return bathyMap

def loadBathyMapBinary():
    from PIL import Image
    im = Image.open('area_1_elevation_w_bathymetry(1).tif')
    imarray = np.array(im)
    img_x = np.linspace(0, cf.xLen, imarray.shape[0])
    img_y = np.linspace(0, cf.yLen, imarray.shape[1])
    # x_array=np.arange(xmin, xmax, 1)
    # y_array = np.arange(ymin, ymax, 1)

    bathy_interp = NearestNDInterpolator(list(zip(img_x, img_y)),imarray)
    #bathyMap = bathy_interp((0,0))

    return bathy_interp
def LCOE():
    return

def cantor_pairing(x, y):
    return (x + y) * (x + y + 1) // 2 + y


def inverse_cantor_pairing(z):
    w = int((8 * z + 1) ** 0.5 - 1) // 2
    t = (w * w + w) // 2
    y = z - t
    x = w - y
    return x, y
def inverse_pairing(z):
    y = z // (cf.y_points)
    x = z % (cf.y_points)
    return x, y
def convert_ids_to_coordinates(ids_set):
  coordinates = []
  for unique_id in ids_set:
    x, y = inverse_pairing(unique_id)
    coordinates.append((x, y))
  return coordinates

def convert_ids_to_grid_indices(ids_set):
  indices = []
  for unique_id in ids_set:
    x, y = inverse_cantor_pairing(unique_id)
    indices.append((int(x), int(y)))
  return indices

# def pick_coordinates(ids,xcoords_mesh,ycoords_mesh):
#     x_grid, y_grid = np.meshgrid(xcoords_mesh, ycoords_mesh)
#     x_flatten=x_grid.flatten()
#     y_flatten=y_grid.flatten()

def convert_coordinates_to_ids(coordinates):
    ids_set = []
    for coord in coordinates:
        x, y = coord[0], coord[1]
        unique_id = cantor_pairing(x, y)
        ids_set.append(unique_id)
    return ids_set



def interpGridData(filename):
    from PIL import Image
    im = Image.open(filename)
    imarray = np.array(im)
    img_x = np.linspace(0, cf.xLen, imarray.shape[0])
    img_y = np.linspace(0, cf.yLen, imarray.shape[1])


    ximg_grid, yimg_grid = np.meshgrid(img_x, img_y)

    a=(ximg_grid.T.flatten(),yimg_grid.T.flatten())
    b=imarray.flatten()
    c=(cf.x_grid.T.flatten(),cf.y_grid.T.flatten())
    ass=a[0].shape, a[1].shape
    bss=b.shape
    css=c[0].shape, c[1].shape
    z_gridded=griddata(a,
                       b,
                       c,
                       method="linear")
    return z_gridded


def len_intcab(n_turb, farm, pos_sub):
    pos_T = farm[["x", "y"]].to_numpy()
    best_length = float('inf')
    best_solution = None
    if n_turb % 3 == 2:
        k= (n_turb // 3) + 1
    else:
        k= n_turb // 3
    modello_kmeans = KMeans(n_clusters=k,n_init="auto",random_state=0)
    modello_kmeans.fit(pos_T)
    total_cable_length = 0
    for i in range(k):
        indici_cluster_corrente = np.where(modello_kmeans.labels_ == i)[0]
        if len(indici_cluster_corrente) >= 1:
            cluster_points = pos_T[indici_cluster_corrente]
            distance_to_substation = np.linalg.norm(cluster_points - pos_sub, axis=1)
            closest_point_index = np.argmin(distance_to_substation)
            closest_point = cluster_points[closest_point_index]
            total_cable_length += distance_to_substation[closest_point_index]
            if len(cluster_points) >= 2:
                internal_cable_length = np.linalg.norm(cluster_points - closest_point, axis=1).sum()
                total_cable_length += internal_cable_length
    if total_cable_length < best_length:
        best_length = total_cable_length
        best_solution = modello_kmeans.labels_.copy()
    return best_length / 1e3, np.mean(distance_to_substation)

def get_coords_from_binary_layout(x):
    ones_indices = np.where(x == 1)[0]
    # le coordinate si ottengono invertendo la cantor pairing function
    indices = convert_ids_to_coordinates(ones_indices)
    xcoords = [x * cf.dx for x, y in indices]
    ycoords = [y * cf.dy for x, y in indices]
    return xcoords,ycoords

