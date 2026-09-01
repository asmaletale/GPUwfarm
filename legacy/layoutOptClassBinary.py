
import pandas as pd
from pymoo.core.problem import ElementwiseProblem,LoopedElementwiseEvaluation
import configBinary as cf
import numpy as np
import numpy.ma as ma
import AEP
from pymoo.core.repair import Repair
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import utilities as uti
import random
import csv
from icecream import ic
np.random.seed(42)

def invert_unfeasible_individuals(arr, n, value):
    """

    Args:
        arr:
        n:
        value: valore da invertire, 0 se devo aggiungere turbine, 1 se devo togliere

    Returns:

    """

    zero_indices = [i for i in range(len(arr)) if arr[i] == value]
    # if len(zero_indices) < n:
    #     print(f"Not enough zeroes in the array to replace with ones.")
    #     return arr
    # random.seed(42)
    indices_to_replace = random.sample(zero_indices, n)

    for index in indices_to_replace:
        arr[index] = np.logical_not(arr[index])
    return arr

class Consider_ntur_repair(Repair):
    def _do(self, problem, X, **kwargs):
        ntur = np.sum(X.astype(int), axis=1)
        for i in range(len(X)):
            if ntur[i] < cf.nturbs_down:
                diff=abs(ntur[i] - cf.nturbs_down)
                X[i,:]=invert_unfeasible_individuals(X[i,:],diff,0)

                # dists = AEP.calcDistancesBinary(X)
                # # calcolare dist per distanze dalla sottostazione
                # dists=dists[np.nonzero(dists)]
                # mindist = min(dists)
                # diam = problem.fi.floris.farm.rotor_diameters_sorted[0][0][0]
                # if mindist<cf.mindiam*diam:
                    
            elif ntur[i] > cf.nturbs_up:
                diff=abs(ntur[i] - cf.nturbs_up)
                X[i,:]=invert_unfeasible_individuals(X[i,:],diff,1)
        return X

def create_random_individual(n_avg):
    prob = n_avg / cf.gridPoints

    val = np.random.random((1,cf.gridPoints))
    sample = (val < prob).astype(bool)
    #ntur = np.sum(sample.astype(int), axis=1)

    return sample

def add_n_turbines(arr, n):
    # add 2 turbines at 2 random positions
    # get indices of 0s
    zero_indices = [i for i in range(len(arr)) if arr[i] == 0]
    indices_to_replace = random.sample(zero_indices, n)
    for index in indices_to_replace:
        arr[index] = np.logical_not(arr[index])
    return arr

class adoption_repair(Repair):
    def _do(self, problem, X, **kwargs):
        ntur = np.sum(X.astype(int), axis=1)
        for i in range(len(X)):
            dists = AEP.calcDistancesBinary(X[i])
            # calcolare dist per distanze dalla sottostazione
            dists=dists[np.nonzero(dists)]
            if len(dists)==0:
                X[i,:]=add_n_turbines(X[i],2)
                dists = AEP.calcDistancesBinary(X[i])
                dists=dists[np.nonzero(dists)]

            mindist = min(dists)
            diam = problem.fi.floris.farm.rotor_diameters_sorted[0][0][0]
            ntur_check = np.sum(X[i].astype(int))
            #ic(" 0 ", i,ntur_check)  
            if mindist<cf.mindiam*diam:
                X[i,:]=create_random_individual(ntur[i])
                ntur_check = np.sum(X[i].astype(int))
                #ic(" 1 ", i,ntur_check)  
            if ntur_check < cf.nturbs_down:
                diff=abs(ntur_check - cf.nturbs_down)
                X[i,:]=invert_unfeasible_individuals(X[i,:],diff,0)
                ntur_check = np.sum(X[i,:].astype(int))
                #ic(" 2 ", i,ntur_check)    
            elif ntur_check > cf.nturbs_up:
                diff=abs(ntur_check - cf.nturbs_up)
                X[i,:]=invert_unfeasible_individuals(X[i,:],diff,1)
                ntur_check = np.sum(X[i,:].astype(int))
                #ic(" 3 ", i,ntur_check)   
            # dists = AEP.calcDistancesBinary(X[i])
            # # calcolare dist per distanze dalla sottostazione
            # dists=dists[np.nonzero(dists)]
            # mindist = min(dists)
        return X


class layoutOpt(ElementwiseProblem):
    # myxl=np.array([-2500 for i in range (10)])
    # myxu = np.array([2500 for i in range(10)])
    def __init__(self,farmfile,windfile,bathyfile,myRunner=LoopedElementwiseEvaluation()):
        super().__init__(n_var=cf.gridPoints,
                         n_obj=2,
                         n_ieq_constr=3,
                         #xl=self.myxl,#x,y,yaw_angle,nturb
                         #xu=self.myxu)
                         xl=np.zeros(cf.gridPoints),  # n_turb, turb_size, layout_id
                         xu=np.ones(cf.gridPoints),
                         elementwise_runner=myRunner)

        self.fi, self.freq, self.wd_array, self.ws_array=AEP.initFloris(farmfile,windfile)
        #self.bathyMap=AEP.loadBathyMap(cf.x_down,cf.x_up,cf.y_down,cf.y_up)
        bathy_grid = AEP.interpGridData(bathyfile)
        self.bathy_grid = bathy_grid.reshape(cf.x_points,cf.y_points)
        self.lim_up=cf.nturbs_up
        self.lim_down=cf.nturbs_down
        self.n_atteso=cf.nturbs_atteso
        self.ncell= cf.gridPoints
        self.results=np.zeros((1,11))
        self.resultslay=np.zeros(cf.gridPoints)

    def _evaluate(self, x, out, *args, **kwargs):
        # La soluzione del layout sarà un vettore binario di lunghezza prefissata pari al numero di posizioni possibili
        # l'indice del vettore binario in cui i==1 corrisponde all'id della cantor pairing function
        
        #self.resultslay = np.vstack((self.resultslay, x.astype(int)))

        xcoords,ycoords=AEP.get_coords_from_binary_layout(x)

        #bathy=self.bathy_grid.compressed()
        bathy = ma.array(self.bathy_grid, mask=-(x - 1)).compressed()  #

        nturbs = sum(x)
        # if nturbs > cf.nturbs_up or nturbs < cf.nturbs_down:
        #     return

        farm=pd.DataFrame(data={"x":xcoords,"y":ycoords,"h":bathy})

        # print(xcoords)
        # print(ycoords)
        # print("nturbs",nturbs)

        # if nturbs<cf.nturbs_down or nturbs>cf.nturbs_up:
        #     return
                ###dists = AEP.calcDistancesNew(xcoords,ycoords)
        dists = AEP.calcDistancesFarm(farm)
        # calcolare dist per distanze dalla sottostazione
        dists=dists[np.nonzero(dists)]
        mindist = min(dists)
        diam = self.fi.floris.farm.rotor_diameters_sorted[0][0][0]
        g1 = cf.mindiam*diam-mindist
        if mindist < cf.mindiam*diam:
            out["F"] = [0,0] #,f3,f4]
            out["G"] = [np.inf,np.inf,np.inf]
            out["nturbs"] = np.nan
            out["AEP"] = np.nan
            out["Losses"] = np.nan
            out["AEPS"] = np.nan
            out["Costs"] = np.nan
            out["LCOE_NOdr"] = np.nan
            out["LCOE_DR"]=np.nan
            out["CF"] = np.nan
            out["capex"] = np.nan
            out["opex"] = np.nan
            out["Tras"] = np.nan
            out["Moo"] = np.nan
            out["Inst"] = np.nan
            out["avg_dist_sub"] = np.nan
            out["len_intcab"] = np.nan
            out["H_av"] = np.nan
            out["VI"] = np.nan

            return
        
        LS_production=AEP.get_farm_LSEP(self.fi,farm,self.freq,self.wd_array,self.ws_array)
        aep,aep_NO_wake = AEP.calc2DFarm(self.fi,
                              farm,
                              self.freq,self.wd_array, self.ws_array,parallel=cf.florisParallel,viz=(False,0,7))
        
        aep,aep_NO_wake = aep/1e9, aep_NO_wake/1e9
        VI = AEP.get_farm_VI(farm,self.wd_array,cf.obs_coords,cf.obs_coords_weights,cf.obs_heights,self.freq)



        # # f2 = np.sum(dists)
        #distsfromSub = AEP.calcDistancesFarm(farm,fromPoint=np.array([[0,0]]))
        #f2 = np.sum(distsfromSub)

        #
        #hull = ConvexHull(tuple(zip(xcoords,ycoords)))
        #area=hull.area


        len_intcab,avg_dist_sub= AEP.len_intcab(nturbs,farm,cf.pos_sub)
        #len_intcab=3
        #len_intcab=3
        
        #turb_float=nturbs*(cf.turbine_cost+cf.floater_turb+cf.anchoring_turb)
        #opex_martinez = cf.opex_1WT_martinez * nturbs
        opex_myhr = cf.opex_1WT_myhr * nturbs
        #opex_cavazzi =( (cf.fix_cost_cav + cf.port_fees_cav) * aep * 1e3 + (cf.len_expcab * cf.var_cost_cav/100) )/ 1e6
        #fix = cf.fix_refisa_1WT * nturbs
        #var = cf.var_refisa * aep * 1e3
        #opex_refisa = fix + var
        # aep_NO_wake = AEP.calc2DFarmNOwake(self.fi,
        #                       farm,
        #                       self.freq,self.wd_array, self.ws_array) / 1e9
        #capex = turb_float+cf.electric_sub+cf.floater_sub+cf.anchoring_sub

        #CAPEX
        #DevCons: Develompment and Consenting
        DevCons=cf.DevCons_1WT*nturbs

        #TurbSubstr: Turbina and substructure
        TurbSubstr= cf.TurbSubstr_1WT*nturbs

        #Tras: Trasmissions
        k = (nturbs * cf.ratedPower  // 330) +1
        if cf.len_expcab < cf.AC_DC_threeshold:
            Tras_subon=0
            Tras_suboff=cf.C_offsubAC
            C_expcable=cf.C_expcableAC
            n_expcab=cf.n_expcablesAC * k
        else:
            Tras_subon=cf.C_onsubDC
            Tras_suboff=cf.C_offsubDC
            C_expcable=cf.C_expcableDC
            n_expcab=cf.n_expcablesDC * k
        Tras_exp= cf.len_expcab * n_expcab * C_expcable
        Tras_int=len_intcab* cf.C_intcab
        Tras= Tras_int+Tras_exp+Tras_suboff+Tras_subon

        #Moo: Mooring
        H_av=sum(farm.h)/nturbs
        H_av=H_av*(-1)
        #Moo_1WT=cf.n_lines*(cf.C_anchor+(1.5*H_av+cf.extra_line)*cf.C_line+cf.chain_len*cf.C_chain)
        #Moo=Moo_1WT*nturbs
        Moo_1WT = (cf.n_lines*((0.0591*cf.MBL_chain-87.69)*H_av+10.198*cf.MBL_DEA)*cf.f_USD_E) /1E6
        Moo = Moo_1WT*nturbs

        #Inst: Installation
        if nturbs % cf.n_turtrip == 0:
            n_trips_tur = nturbs / cf.n_turtrip
        else:
            n_trips_tur = (nturbs // cf.n_turtrip) + 1
        if nturbs % cf.n_fltrip == 0:
            n_trips_fl = nturbs / cf.n_fltrip
        else:
            n_trips_fl = (nturbs // cf.n_fltrip) + 1
        #D_center = cf.D_port + (((cf.xLen / 2) + 240) / 1000)
        Inst_tur = cf.C_boat * (nturbs * cf.T_inst + (2 * cf.D_port / cf.V_AHTS) * n_trips_fl + (2 * cf.D_port / cf.V_PSV) * n_trips_tur)
        Inst_intcab = cf.C_inst_intcab * len_intcab
        Inst_expcab = cf.C_inst_expcab * cf.len_expcab * n_expcab
        Inst_offsub = cf.C_inst_offsub
        Inst_mooring = cf.C_inst_moo_per_turb * nturbs
        Inst = Inst_tur+Inst_intcab+Inst_expcab+Inst_offsub+Inst_mooring

        # Decommissioning
        Dec = -cf.R_dec * cf.ratedPower * nturbs

        # Total CAPEX
        capex = DevCons+TurbSubstr+Tras+Moo+Inst+Dec

        # OPEX
        TOT_OPEX_act = [opex_myhr*((1+cf.r)**(-i)) for i in range(1, (cf.lifetime+1))]

        # TOTAL COSTS
        costs = capex + sum(TOT_OPEX_act)

        aeps = [(aep * 1e3) * ((1+cf.r)**(-i)) for i in range(1, (cf.lifetime+1))]
        aeps = sum(aeps)
        #aeps= cf.lifetime*aep * 1e3
        lcoe_NOdr = costs*1e6 / aeps
        lcoe_DR = costs*1e6 / (LS_production * 1e3)
        CF = aep * 1e3 / (cf.ratedPower * nturbs * 8760)     #no derating
        Losses = ((aep_NO_wake - aep) / aep_NO_wake) *  100   #no derating

        #bathyCosts=sum(farm.h)*10000
        diam = self.fi.floris.farm.rotor_diameters_sorted[0][0][0]
        f1 = lcoe_DR
        f2 = VI    #VI
        g2 = cf.nturbs_down-nturbs
        g3 = nturbs-cf.nturbs_up
        out["F"] = [f1,f2]
        out["G"] = [g1,g2,g3]
        out["nturbs"] = nturbs
        out["AEP"] = aep
        out["Losses"] = Losses
        out["AEPS"] = aeps
        out["Costs"] = costs
        out["LCOE_NOdr"] = lcoe_NOdr
        out["LCOE_DR"] = lcoe_DR
        out["CF"] = CF
        out["capex"] = capex
        out["opex"] = opex_myhr
        out["Tras"] = Tras
        out["Moo"] = Moo
        out["Inst"] = Inst
        out["avg_dist_sub"] = avg_dist_sub
        out["len_intcab"] = len_intcab
        out["H_av"] = H_av
        out["VI"] = VI