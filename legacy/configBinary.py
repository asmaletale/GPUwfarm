import numpy as np
import AEP as aep
import os
optLayBool=True
optYawBool=True
#projectDir="."
projectDir="."
#projectDir="C://Users//alari//PycharmProjects//florisSdewes//sdewes//test"
#casename="SDEWES_NEAR_lcoe_VI_1500x2000_MAIN_V0"   #main case V0
#casename="SDEWES_NEAR_lcoe_VI_1500x2000_MAIN_R1_mutation0.01"  ##correzione al main per rebuttal mut 0.01 
#casename="SDEWES_NEAR_lcoe_VI_1500x2000_MAIN_R1" ##correzione al main per rebuttal
#casename="SDEWES_NEAR_lcoe_VI_doublearea_samedensity_1500x2000"
#casename="SDEWES_NEAR_lcoe_VI_halfarea_samedensity_1500x2000"


#########    R1     ##############
casename="SDEWES_NEAR_lcoe_VI_MAIN_ADOPTION"  #DONE
#casename="SDEWES_NEAR_lcoe_VI_ERA5_12bin_ADOPTION" #DONE   
#casename="SDEWES_NEAR_lcoe_VI_ERA5_24bin_ADOPTION"  #DONE
#casename = "SDEWES_NEAR_lcoe_VI_Jensen_ADOPTION"    #DONE
#casename = "SDEWES_NEAR_lcoe_VI_5nrel_ADOPTION"    #
#casename="SDEWES_NEAR_lcoe_VI_doublearea_samedensity_ADOPTION" #DONE
#casename="SDEWES_NEAR_lcoe_VI_halfarea_samedensity_ADOPTION" #DONE
#casename="SDEWES_NEAR_lcoe_VI_tinyarea_samedensity_ADOPTION"    #DONE



resultsDir=os.path.join(projectDir,"Results","R1",casename)
plotsDir=os.path.join(projectDir,"Plots","R1",casename)
appendixDir=os.path.join(projectDir,"Plots","R1","Appendix")

if not os.path.exists(resultsDir):
    os.makedirs(resultsDir)
if not os.path.exists(plotsDir):
    os.makedirs(plotsDir)
if not os.path.exists(appendixDir):
    os.makedirs(appendixDir)

dataDir=os.path.join(projectDir,"data","SDEWES")
wind_file=os.path.join(dataDir,"SDEWES_NEAR.lib")
bathy_file= os.path.join(dataDir,"SDEWES_NEAR.tif")

lifetime=25
MAX_Derating = 0.05
rotorDiameter=240
hubHeight=150
ratedPower=15
ratedSpeed=10.59 #IEA 15MW
if "Jensen" in casename:
    floris_file= os.path.join(projectDir,"jensen_iea_15MW.yaml")
elif "5nrel" in casename:
    wind_file=os.path.join(dataDir,"SDEWES_NEAR_100.lib")
    floris_file = os.path.join(projectDir,"gch_nrel_5MW.yaml")
    rotorDiameter=126
    hubHeight=90
    ratedPower=5
    ratedSpeed=11.4
else:
    floris_file= os.path.join(projectDir,"gch_iea_15MW.yaml") 

ndirs = 12
if "ERA5" in casename:
    wind_file=None
    if "12" in casename:
        wind_rose_file= os.path.join(dataDir,"ERA5_12.csv")
        ndirs = 12
        color_rose="greens"
    elif "24" in casename:
        wind_rose_file= os.path.join(dataDir,"ERA5_24.csv")
        ndirs = 24
        color_rose="oranges"
elif "5nrel" in casename:
    wind_rose_file= os.path.join(dataDir,"SDEWES_NEAR_100.csv")

else:
    wind_rose_file= os.path.join(dataDir,"SDEWES_NEAR.csv") 
    color_rose="reds"
jensen_path = os.path.join(projectDir,"Results","R1","SDEWES_NEAR_lcoe_VI_Jensen_ADOPTION")
nrel_5MW_path = os.path.join(projectDir,"Results","R1","SDEWES_NEAR_lcoe_VI_5nrel_ADOPTION")
ERA5_24_path = os.path.join(projectDir,"Results","R1","SDEWES_NEAR_lcoe_VI_ERA5_24bin_ADOPTION")
ERA_12_path = os.path.join(projectDir,"Results","R1","SDEWES_NEAR_lcoe_VI_ERA5_12bin_ADOPTION")
tiny_path = os.path.join(projectDir,"Results","R1","SDEWES_NEAR_lcoe_VI_tinyarea_samedensity_ADOPTION")
half_path = os.path.join(projectDir,"Results","R1","SDEWES_NEAR_lcoe_VI_halfarea_samedensity_ADOPTION")
double_path = os.path.join(projectDir,"Results","R1","SDEWES_NEAR_lcoe_VI_doublearea_samedensity_ADOPTION")
main_path= os.path.join(projectDir,"Results","R1","SDEWES_NEAR_lcoe_VI_MAIN_ADOPTION")



popsize=10#    1500
offspring=popsize
n_gen= 10#4000
prob=0.5            
prob_var= 0.01     #tried 0.01: not converging HV but better LCOE
save_check= n_gen
florisParallel=False
pymooParallel=True
maxworkers=64




#########################
if "tinyarea_samedensity" in casename: #casename=="SDEWES_NEAR_lcoe_VI_tinyarea_samedensity_1500x2000" or casename=="SDEWES_NEAR_lcoe_VI_tinyarea_samedensity_1500x2000_OLD_REPAIR_mut0.1":
    xLen = 1697     #half=3394; normal=4800, double= 6780
    yLen = 1697    
    nrows = 7
    ncols = 7
    mindiam = 3
    nturbs_up = 18
    nturbs_down = 2
    nturbs_atteso= 8
#########################
elif "halfarea_samedensity" in casename: #casename=="SDEWES_NEAR_lcoe_VI_halfarea_samedensity_1500x2000":
    xLen = 3394     #half=3394; normal=4800, double= 6780
    yLen = 3394    
    nrows = 14
    ncols = 14
    mindiam = 3
    nturbs_up = 18
    nturbs_down = 2
    nturbs_atteso= 8
#########################
###SDEWES_NEAR_lcoe_VI_doublearea_samedensity_1500x2000
elif "doublearea_samedensity" in casename:#casename=="SDEWES_NEAR_lcoe_VI_doublearea_samedensity_1500x2000":
    xLen = 6780  
    yLen = 6780    
    nrows = 29
    ncols = 29
    mindiam = 3
    nturbs_up = 30
    nturbs_down = 10
    nturbs_atteso= 64
#########################
else:
    xLen = 4800     #half=3394; normal=4800, double= 6780
    yLen = 4800    
    nrows = 20
    ncols = 20
    mindiam = 3
    nturbs_up = 36
    nturbs_down = 5
    nturbs_atteso= 15



Earth_radius=6.371*1e6
obs_coords=[[9200,11000],[14300,4800],[17600,-3100]]    
obs_coords_weights=[0.333,0.333,0.333]
obs_heights=[1.77,1.77,1.77]



x_points=nrows+1
y_points=ncols+1
ncell= nrows * ncols
#A=701 km2 calabria
gridPoints= x_points * y_points
dx=xLen/ncols
dy=yLen/nrows

x_discretization=np.arange(0,xLen+dx/2,dx)
y_discretization=np.arange(0,yLen+dy/2,dy)
x_grid,y_grid=np.meshgrid(x_discretization,y_discretization)


nspeed=25
#Plots
most_frequent_wd=120               
most_frequent_ws=7                     
powerMax=8
powerMin=3
# LCOE from 
# "Mapping of the levelised cost of energy for floating offshore wind in the European Atlantic"
# https://doi.org/10.1016/j.rser.2021.111889

r=0.05 # discount rate
dAndC_total=0.21 #[millionseuro/mwh]
# dAndC_costs={
#     "Environmental Survey":0.07,
#     "Seabed survey":0.15,
#     "Met mast":0.08,
#     "Dev services":0.8,
# }
#mooring_total=800/1e6 #euro/m, multyply by 3*bathy*nmoorings
#turbine_cost=12.5 #millions euro
floater_turb=16 #millions euro
anchoring_turb=3*0.4 #millions euro

electric_sub=25000.*ratedPower/1e6 #euro/mw
floater_sub=28.8 #millions euro
anchoring_sub=4*0.4 #millions euro

inter_array=0.9 #millions euro/km
export_cable=1 #millions euro/km

station_onshore = 1.65*ratedPower #millions euro/mw
junction=2 #millions euro
onshore_cables=1.5 #millions euro/km

DevCons_1WT = 0.21*ratedPower  # millions
TurbSubstr_1WT = (1.6*ratedPower)+8  # millions

# Trasmissione
pos_sub=[4800+720,2400]  
D_cluster_intcabMAX=5
D_cluster_intcabMin=3
AC_DC_threeshold = 55
n_expcablesAC = 1    # n/300MW
n_expcablesDC = 1    # n/300MW
C_expcableAC = 2.336  # millions/km
C_expcableDC = 1.168  # millions/km
n_offsub=1
C_offsubAC = 39  # millions
C_offsubDC = 142.75  # millions
C_onsubDC = 84.35  # millions
C_intcab = 0.3035  # millions/km

len_expcab = 9  

#mooring
n_lines = 3
#C_anchor= 0.123 #milions
#C_line= 48/10e6  #milions/m
#extra_line= 410 #m
#C_chain=270/10e6  #milions/m
#chain_len= 50 #m
MBL_chain = 22286  # kN per turbina da 15MW [Beiter et al.]
MBL_DEA = 9800  # kN [Giglio et al.]
f_USD_E = 0.92  # fattore di conversione E/USD [European Central Bank]

#Installazione ref Cavazzi e Martinez
T_inst = 48  # h
V_AHTS = 10  # km/h
V_PSV = 61.7  # km/h
C_boat = 0.011979  # millions /h affitto [Cavazzi]
n_turtrip = 3  # turbine che la nave porta per volta [Myhr]
n_fltrip = 2  # floaters che la nave porta per volta [Myhr]

D_port = 10
C_inst_intcab = 0.115  # millions/km cavazzi
C_inst_expcab = 0.637  # millions/km    # martinez
C_inst_offsub = 20  # martinez
C_inst_moo_per_turb = 0.24  # millions/turbine

# Decommissioning
R_dec = 0.23  # millions/MW  [Bjerkseter]

#OPEX
opex_1WT_martinez=0.138*ratedPower #millions euro/year      #Martinez, da rivedere Cavazzi
#cavazzi
fix_cost_cav=20.61 #euro/MWh
port_fees_cav= 3.44 #euro/MWh
var_cost_cav= 6.87 #euro/100km
#refisa
fix_refisa_1WT = 0.07 * ratedPower #milions
var_refisa = 8 / 1e6 #milions / MWh
#myhr
opex_1WT_myhr = 0.131 * ratedPower
# def mooring_costs(nturb,nlines,bathyMap,xCoords,yCoords):
#     for x,y in zip(xCoords,yCoords):
#
#         total=nturb*nlines*(c_anchor + 1.5*(aep.getBathy(x,y) + 410)*c_line + 50*c_chain)
#     return total
