
"""Multi‑echelon supply‑chain network optimizer (mini‑SAILS)."""
import pandas as pd, pulp
from utils import haversine, ROAD_FACTOR

def _prep_lanes(nodes: pd.DataFrame, lanes: pd.DataFrame) -> pd.DataFrame:
    # Ensure From/To columns exist
    lanes = lanes.copy()
    # Merge coordinates for distance calc
    lanes = lanes.merge(nodes[['Latitude','Longitude']], left_on='FromNode', right_index=True)                     .rename(columns={'Latitude':'fr_lat','Longitude':'fr_lon'})
    lanes = lanes.merge(nodes[['Latitude','Longitude']], left_on='ToNode', right_index=True)                     .rename(columns={'Latitude':'to_lat','Longitude':'to_lon'})
    lanes['Dist_mi'] = lanes.apply(lambda r: haversine(r.fr_lon,r.fr_lat,r.to_lon,r.to_lat)*ROAD_FACTOR, axis=1)
    lanes.set_index(['FromNode','ToNode'], inplace=True)
    return lanes

def build_model(nodes, demand, lanes, products,
                service_days=7, safety_stock_pct=0.10,
                carbon_price=0.0, shut_nodes=None):
    """Return (PuLP model, variable dicts)."""
    shut_nodes = set(shut_nodes or [])
    # Prep data
    lanes = _prep_lanes(nodes, lanes)
    N   = nodes.index.tolist()
    SUP = nodes.query('Type=="Supplier"').index.tolist()
    PLT = nodes.query('Type=="Plant"').index.tolist()
    DC  = nodes.query('Type=="DC"').index.tolist()
    CST = nodes.query('Type=="Customer"').index.tolist()
    P   = products.index.tolist()
    T   = sorted(demand['Period'].unique())

    # Decision variables
    mdl = pulp.LpProblem('NetworkDesign', pulp.LpMinimize)
    open_dc = pulp.LpVariable.dicts('Open', DC, lowBound=0, upBound=1, cat='Binary')
    flow = pulp.LpVariable.dicts('F',
                ((i,j,p,t) for i,j in lanes.index for p in P for t in T),
                lowBound=0)
    make = pulp.LpVariable.dicts('Make',
                ((pl,p,t) for pl in PLT for p in P for t in T),
                lowBound=0)
    inv  = pulp.LpVariable.dicts('Inv',
                ((n,p,t) for n in DC+PLT for p in P for t in T),
                lowBound=0)

    # Cost components
    def node_fixed(n):  return nodes.loc[n,'FixedCost'] if 'FixedCost' in nodes.columns else 0
    def node_var(n):    return nodes.loc[n,'VarCost_per_lb'] if 'VarCost_per_lb' in nodes.columns else 0
    def node_cap(n):    return nodes.loc[n,'Cap_Lbs_Per_Period'] if 'Cap_Lbs_Per_Period' in nodes.columns else None

    # Objective
    prod_cost = pulp.lpSum(make[pl,p,t]*node_var(pl) for pl in PLT for p in P for t in T)
    wh_fixed  = pulp.lpSum(open_dc[d]*node_fixed(d) for d in DC)
    wh_hold   = pulp.lpSum(inv[n,p,t]*node_var(n) for n in DC+PLT for p in P for t in T)
    trans_cost = pulp.lpSum(flow[i,j,p,t]*lanes.loc[(i,j),'Cost_per_lb_mi']*lanes.loc[(i,j),'Dist_mi']
                            for (i,j) in lanes.index for p in P for t in T)
    duty_cost = pulp.lpSum(flow[i,j,p,t]*lanes.loc[(i,j)].get('DutyRate',0)
                           for (i,j) in lanes.index for p in P for t in T)
    carbon = pulp.lpSum(flow[i,j,p,t]*lanes.loc[(i,j),'Dist_mi']*lanes.loc[(i,j)].get('CO2_per_lb_mi',0)*carbon_price/2204.62
                        for (i,j) in lanes.index for p in P for t in T)
    mdl += prod_cost + wh_fixed + wh_hold + trans_cost + duty_cost + carbon

    # Shutdown constraints
    for n in shut_nodes:
        if n in DC:
            mdl += open_dc[n] == 0
        for (i,j) in lanes.index:
            if i==n or j==n:
                for p in P:
                    for t in T:
                        mdl += flow[i,j,p,t] == 0

    # Capacity
    for n in PLT+DC:
        cap = node_cap(n)
        if cap:
            for t in T:
                mdl += pulp.lpSum(flow[i,n,p,t] for i in N if (i,n) in lanes.index for p in P) <= cap

    # Demand fulfilment
    for (_, row) in demand.iterrows():
        c,p,t,dem = row['CustomerID'], row['ProductID'], row['Period'], row['DemandLbs']
        inflow = pulp.lpSum(flow[i,c,p,t] for i in N if (i,c) in lanes.index)
        mdl += inflow >= dem

    # Service time
    for (_, row) in demand.iterrows():
        c,p,t,dem = row['CustomerID'], row['ProductID'], row['Period'], row['DemandLbs']
        mdl += pulp.lpSum(flow[i,c,p,t]*lanes.loc[(i,c),'TransitDays'] for i in N if (i,c) in lanes.index)                    <= dem*service_days

    # Mass balance at DCs & Plants
    for n in DC+PLT:
        for p in P:
            for ti,t in enumerate(T):
                inflow = pulp.lpSum(flow[i,n,p,t] for i in N if (i,n) in lanes.index)
                outflow = pulp.lpSum(flow[n,j,p,t] for j in N if (n,j) in lanes.index)
                if n in PLT:
                    outflow += make[n,p,t]
                inv_prev = inv[n,p,T[ti-1]] if ti>0 else 0
                mdl += inflow + inv_prev == outflow + inv[n,p,t]

    # BOM (simple one-level parent/child)
    if 'BOM_Component' in products.columns:
        for comp,row in products.iterrows():
            if pd.notna(row['BOM_Component']):
                parent = row['BOM_Component']
                ratio  = row['InputLB_per_OutputLB']
                for pl in PLT:
                    for t in T:
                        mdl += make[pl,parent,t]*ratio >= make[pl,comp,t]

    return mdl, dict(flow=flow, open=open_dc, inv=inv, make=make)

def solve_and_extract(nodes, demand, lanes, products, **kwargs):
    mdl, var = build_model(nodes,demand,lanes,products, **kwargs)
    solver = pulp.PULP_CBC_CMD(msg=False)
    mdl.solve(solver)
    status = pulp.LpStatus[mdl.status]
    if status != 'Optimal':
        return status, None

    # Flow table
    rows = []
    for (i,j,p,t),v in var['flow'].items():
        if v.varValue and v.varValue>1e-6:
            rows.append(dict(From=i,To=j,Product=p,Period=t,FlowLbs=v.varValue))
    flow_df = pd.DataFrame(rows)

    # Facility open
    open_dc = [d for d,v in var['open'].items() if v.varValue>0.5]
    return 'Optimal', dict(flow=flow_df, open_dc=open_dc, objective=pulp.value(mdl.objective))