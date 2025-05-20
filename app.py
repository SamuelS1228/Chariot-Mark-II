
import streamlit as st, pandas as pd
from optimization import solve_and_extract

st.title('Mini‑SAILS Network Optimizer')

st.sidebar.header('Upload CSVs')
nodes_file  = st.sidebar.file_uploader('nodes.csv', type='csv')
demand_file = st.sidebar.file_uploader('demand.csv', type='csv')
lanes_file  = st.sidebar.file_uploader('lanes.csv', type='csv')
prod_file   = st.sidebar.file_uploader('products.csv', type='csv')

if all([nodes_file,demand_file,lanes_file,prod_file]):
    nodes   = pd.read_csv(nodes_file).set_index('NodeID')
    demand  = pd.read_csv(demand_file)
    lanes   = pd.read_csv(lanes_file)
    products= pd.read_csv(prod_file).set_index('ProductID')
    svc_days = st.sidebar.slider('Max customer transit days',1,15,7)
    carbon  = st.sidebar.number_input('Carbon price ($/ton)',0.0,500.0,0.0,step=5.0)

    if st.button('Optimize'):
        status,res = solve_and_extract(nodes,demand,lanes,products,service_days=svc_days,carbon_price=carbon)
        if status!='Optimal':
            st.error(f'Solver ended with status: {status}')
        else:
            st.success(f'Optimal cost = ${res["objective"]:,.0f}')
            st.subheader('Open DCs')
            st.write(res['open_dc'])
            st.subheader('Material flows')
            st.dataframe(res['flow'])
else:
    st.info('Upload all four CSV files to begin.')
