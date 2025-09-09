import pandas as pd
import numpy as np
import h5py

class EnergyStorage:
    def __init__(self, index, bus, Pdmax, Pcmax, Emax, Emin, Hemax, cd, cc):
        self.index = index
        self.bus = bus
        self.Pdmax = Pdmax
        self.Pcmax = Pcmax
        self.Emax = Emax
        self.Emin = Emin
        self.Hemax = Hemax
        self.cd = cd
        self.cc = cc


class Farm:
    def __init__(self, mu, sigma, bus, Pwmax):
        self.mu = mu
        self.sigma = sigma
        self.bus = bus
        self.Pwmax = Pwmax


class Load:
    def __init__(self, bus):
        self.bus = bus


class Bus:
    def __init__(self, nodeID, kind, Pd, Qd, Vmax, Vmin, Gs, Bs, Vm, gr = 0.0, ga = 0.0, Pg = 0.0, Qg = 0.0, Pgmax = 0.0, Qgmax = 0.0,
        Pgmin = 0.0, Qgmin = 0.0, pi1 = 0.0, pi2 = 0.0, qobjcoeff = 0.0,
        Pmgcost = 0.0, Ji = 0.0, coord = None, genids =None, farmids =None,
        outlist = None, inlist = None):
        if str(kind[0].decode('utf-8') ) == 'PQ':
            self.kind = 'PQ'
        elif str(kind[0].decode('utf-8') ) == 'PV':
            self.kind = 'PV'
        elif str(kind[0].decode('utf-8') ) == 'Ref':
            self.kind = 'Ref'
        else:
            raise ValueError("Invalid kind for Bus")

        self.nodeID = nodeID
        self.Pd = Pd
        self.Qd = Qd
        self.gr = gr
        self.ga = ga
        self.Pg = Pg
        self.Qg = Qg
        self.Pgmax = Pgmax
        self.Qgmax = Qgmax
        self.Pgmin = Pgmin
        self.Qgmin = Qgmin
        self.pi1 = pi1
        self.pi2 = pi2
        self.qobjcoeff=qobjcoeff
        self.Pmgcost = Pmgcost
        self.Vmax = Vmax
        self.Vmin = Vmin
        self.Ji = Ji
        self.coord = coord if (isinstance(coord, np.ndarray) ) else [0,0]
        self.genids = genids if (isinstance(genids, np.ndarray) ) else []
        self.farmids = []
        self.outlist = outlist
        self.inlist = inlist
        self.Gs = Gs
        self.Bs = Bs
        self.Vm = Vm

    def setg(self, genidx, Pg, Qg, Pgmax, Pgmin, Qgmax, Qgmin):
        self.Pg += Pg
        self.Qg += Qg
        self.Pgmax += Pgmax
        self.Pgmin += Pgmin
        self.Qgmax += Qgmax
        self.Qgmin += Qgmin
        if self.kind == 'PQ':
            print(f"Warning: Generator {genidx} was assigned to bus {self.nodeID}, but this bus has type PV")
        self.genids.append(genidx)


class Generator:
    def __init__(self, genID, busidx, Pg, Qg, Pgmax, Pgmin, Qgmax, Qgmin, pi1, pi2, pi3):
        self.genID = genID
        self.busidx = busidx
        self.Pg = Pg
        self.Qg = Qg
        self.Pgmax = Pgmax
        self.Pgmin = Pgmin
        self.Qgmax = Qgmax
        self.Qgmin = Qgmin
        self.pi1 = pi1
        self.pi2 = pi2
        self.pi3 = pi3


class Line:
    def __init__(self, arcID, tail, head, r, x, u, turns, d, Imax, Imin, b_charge):
        self.arcID = arcID
        self.tail = tail
        self.head = head
        self.r = r
        self.x = x
        self.gamma = r / (r ** 2 + x ** 2)
        self.beta = -x / (r ** 2 + x ** 2)
        self.u = u
        self.ratio = turns
        self.distance_scale = d
        self.Imax = Imax
        self.Imin = Imin
        self.a = 1 / r
        self.Jij = 0
        self.Iij = Imax
        self.b_charge = b_charge


def get_thermal_capacity(line, mva_base):
    return line.u  # / mva_base  # line limits


def get_sync_capacity(line, mva_base):
    return line.gamma


class TransLine:
    def __init__(self, translineID, arcID, tail, head, r, x, u, turns, d, Imax, Imin, b_charge):
        self.translineID = translineID
        self.arcID = arcID
        self.tail = tail
        self.head = head
        self.r = r
        self.x = x
        self.gamma = r / (r ** 2 + x ** 2)
        self.beta = -x / (r ** 2 + x ** 2)
        self.u = u
        self.ratio = turns
        self.distance_scale = d
        self.Imax = Imax
        self.Imin = Imin
        self.a = 1 / r
        self.Jij = 0.0
        self.Iij = Imax
        self.b_charge = b_charge


class Scenario:
    def __init__(self, lineIDs):
        self.lineIDs = lineIDs


def get_line_id(lines, head, tail):
    line_ids = []
    for line in lines:
        if (line.head == head and line.tail == tail) or (line.head == tail and line.tail == head):
            line_ids.append(line.arcID)
    return line_ids


def load_ES(datadir):
    print(f">>>>> Reading feeder data from {datadir}")
    ES_raw = pd.read_csv(f"{datadir}/ES.csv")
    ESs = []
    for _, row in ES_raw.iterrows():
        new_e = EnergyStorage(row['index'], row['node'], row['pdmax'], row['pcmax'], row['Emax'], row['Emin'],
                              row['Hemax'], row['cd'], row['cc'])
        ESs.append(new_e)
    return ESs


def load_timeseries(datadir):
    print(f">>>>> Reading Timeseries data from {datadir}")
    # wind_data = pd.read_csv(f"{datadir}/wind_data.csv", header=None).values
    load_data = pd.read_csv(f"{datadir}/load_data.csv", header=None).values
    Hw = pd.read_csv(f"{datadir}/Hw.csv", header=None).values
    return load_data, Hw

# 读取 JLD 文件并构建对象列表
def load_jld_data(file_path):
    with h5py.File(file_path, 'r') as f:
        # 读取 Line 数据并构建 Line 对象列表
        lines_dataset = f['lines']
        lines_list = []
        for i in range(len(lines_dataset)):
            line_ref = lines_dataset[i]
            line_obj = f[line_ref]  # 解引用
            line_data = line_obj[()]
            line_obj = Line(
                arcID=line_data[0]-1,
                tail=line_data[1]-1,
                head=line_data[2]-1,
                r=line_data[3],
                x=line_data[4],
                u=line_data[7],
                turns=line_data[8],
                d=line_data[9],
                Imax=line_data[10],
                Imin=line_data[11],
                b_charge=line_data[-1]
            )
            lines_list.append(line_obj)

        # 读取 Generator 数据并构建 Generator 对象列表
        generators_dataset = f['generators']
        generators_list = []
        for i in range(len(generators_dataset)):
            gen_ref = generators_dataset[i]
            gen_obj = f[gen_ref]
            gen_data = gen_obj[()]
            generator_obj = Generator(
                genID=gen_data[0]-1,
                busidx=gen_data[1]-1,
                Pg=gen_data[2],
                Qg=gen_data[3],
                Pgmax=gen_data[4],
                Pgmin=gen_data[5],
                Qgmax=gen_data[6],
                Qgmin=gen_data[7],
                pi1=gen_data[8],
                pi2=gen_data[9],
                pi3=gen_data[10]
            )
            generators_list.append(generator_obj)

        # 读取 Bus 数据并构建 Bus 对象列表
        buses_dataset = f['buses']
        buses_list = []
        for i in range(len(buses_dataset)):
            bus_ref = buses_dataset[i]
            bus_obj = f[bus_ref]
            bus_data = bus_obj[()]
            bus_obj = Bus(
                nodeID=bus_data[0]-1,
                kind=bus_data[1],
                Pd=bus_data[2],
                Qd=bus_data[3],
                gr=bus_data[4], # grounding resistance
                ga=bus_data[5], # inverse of grounding resistance
                Pg=bus_data[6],
                Qg=bus_data[7],
                Pgmax=bus_data[8],
                Qgmax=bus_data[9],
                Pgmin=bus_data[10],
                Qgmin=bus_data[11],
                pi2=bus_data[12], # Objective coefficient
                pi1=bus_data[13], # Objective coefficient
                qobjcoeff=bus_data[14],
                Pmgcost=bus_data[15],
                Vmax=bus_data[16],
                Vmin=bus_data[17],
                Ji=bus_data[18], # DC induced by GIC voltagex
                coord= f[bus_data[19]][()],
                genids=f[bus_data[20]][()] - 1 if (isinstance(f[bus_data[20]][()], np.ndarray) ) else f[bus_data[20]][()],
                outlist = [k - 1 for k in f[bus_data[22]][()] if isinstance(f[bus_data[22]][()], np.ndarray) and f[bus_data[22]][()].size > 0] if isinstance(f[bus_data[22]][()], np.ndarray) else [],
                inlist = [k - 1 for k in f[bus_data[23]][()] if isinstance(f[bus_data[23]][()], np.ndarray) and f[bus_data[23]][()].size > 0] if isinstance(f[bus_data[23]][()], np.ndarray) else [],
                Gs=bus_data[24],
                Bs=bus_data[25],
                Vm=bus_data[26]
            )
            buses_list.append(bus_obj)

    return lines_list, generators_list, buses_list