#MOCK ANALYTICS WEB APP
#imports
import altair as alt
import pydeck as pdk
import pandas as pd
from datetime import datetime, timedelta, date
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

#set up page configuration
st.set_page_config(page_title="Maxwell Knowles Mock Analytics Web App", page_icon=":apple:", layout="wide",initial_sidebar_state="expanded")

#import data sets from github repo
merch_data_all = pd.read_csv("https://raw.githubusercontent.com/maxwellknowles/analytics_web_app/main/Analytics%20Web%20App%20-%20Orders-Items.csv")
orders = pd.read_csv("https://raw.githubusercontent.com/maxwellknowles/analytics_web_app/main/Analytics%20Web%20App%20-%20Orders.csv")
merch_data = pd.read_csv("https://raw.githubusercontent.com/maxwellknowles/analytics_web_app/main/Analytics%20Web%20App%20-%20Vendors-Category-Item.csv")
orders_items_data = pd.read_csv("https://raw.githubusercontent.com/maxwellknowles/analytics_web_app/main/Analytics%20Web%20App%20-%20Count%20Items%20Orders%20Market.csv")
refunds_today = pd.read_csv("https://raw.githubusercontent.com/maxwellknowles/analytics_web_app/main/Analytics%20Web%20App%20-%20Refunds.csv")
#refunds_items_7days
#refunds_items_location_7days
lat_long = pd.read_csv("https://raw.githubusercontent.com/maxwellknowles/analytics_web_app/main/LatLon.csv")

#data selection
year, month, day = date.today().year, date.today().month, date.today().day
fromDate = datetime(year, month, day) - timedelta(days=30)
toDate = datetime(year, month, day)

fromDate = str(fromDate)
toDate = str(toDate)

#sidebar menu
st.sidebar.title('Grocery Delivery Startup Analytics')
st.sidebar.subheader("An Interactive Home for A Grocery Delivery Startup's Data")
choose = st.sidebar.selectbox(
    "What data would you like to see?",
    ("ML: Item, Cart, and Customer Clustering", "Merchandising Data", "Order Distribution & Metrics")
)

if choose == "ML: Item, Cart, and Customer Clustering":
    st.title("ML: Item, Cart, and Customer Clustering")
    st.write('Data starting from:', fromDate)
    st.write("The following graphs and tables are built using k-means clustering, an unsupervised machine learning algorithm that clusters data points based on their relative locations along given dimensions.")

    st.subheader("Classifying Items by Cart Presence")
    #getting data
    kmeans_items = merch_data_all[['Order ID', 'Customer', 'Total', 'Item', 'Quantity', 'Category', 'Share of Order']]
    kmeans_items_1 = kmeans_items.groupby(['Item', 'Category']).agg({'Quantity':['sum'], 'Share of Order':['mean'], 'Total':['mean']})
    kmeans_items_1 = kmeans_items_1.reset_index(drop=False, inplace=False)
    kmeans_items_1.columns = kmeans_items_1.columns.map(lambda x:x[0])
    kmeans_items_1['Average Share of Order']=kmeans_items_1['Share of Order']
    kmeans_items_1['Average Order Total When Present']=kmeans_items_1['Total']
    kmeans_items_1_1 = kmeans_items_1[['Quantity', 'Average Share of Order']]
    k_means_items_scaled = StandardScaler().fit_transform(kmeans_items_1_1)
    k_means_items_scaled = pd.DataFrame(k_means_items_scaled, columns = kmeans_items_1_1.columns)

    #kmeans clustering using average share of order and quantity sold by item
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(k_means_items_scaled)
        sse.append(kmeans.inertia_)
    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    n = kl.elbow
    kmeans = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=10, random_state=42)
    pred_y = kmeans.fit_predict(k_means_items_scaled)
    kmeans_items_1['Item Classification'] = pd.Series(pred_y, index=kmeans_items_1.index)

    #graphing customer clusters, observe qualities of customers
    c = alt.Chart(kmeans_items_1).mark_circle().encode(
        x='Quantity', y='Average Share of Order', color='Item Classification', tooltip=['Item', 'Category', 'Item Classification'])

    st.altair_chart(c, use_container_width=True)

    #getting data
    kmeans_items_1_1 = kmeans_items_1[['Quantity', 'Average Order Total When Present']]
    k_means_items_scaled = StandardScaler().fit_transform(kmeans_items_1_1)
    k_means_items_scaled = pd.DataFrame(k_means_items_scaled, columns = kmeans_items_1_1.columns)

    #kmeans clustering using average order total when present and quantity sold by item
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(k_means_items_scaled)
        sse.append(kmeans.inertia_)
    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    n = kl.elbow
    kmeans = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=10, random_state=42)
    pred_y = kmeans.fit_predict(k_means_items_scaled)
    kmeans_items_1['Item Classification'] = pd.Series(pred_y, index=kmeans_items_1.index)

    #graphing customer clusters, observe qualities of customers
    c = alt.Chart(kmeans_items_1).mark_circle().encode(
        x='Quantity', y='Average Order Total When Present', color='Item Classification', tooltip=['Item', 'Category', 'Item Classification'])

    st.altair_chart(c, use_container_width=True)

    kmeans_items_1_results = kmeans_items_1[['Item','Item Classification','Total']]
    kmeans_items_1_results=kmeans_items_1_results.groupby(["Item", "Item Classification"]).agg({'Total':['mean']})
    kmeans_items_1_results = kmeans_items_1_results.reset_index(drop=False, inplace=False)
    kmeans_items_1_results.columns = kmeans_items_1_results.columns.map(lambda x:x[0])
    sorted_kmeans_items_1_results = kmeans_items_1_results.sort_values("Total", ascending=False)
    sorted_kmeans_items_1_results = sorted_kmeans_items_1_results.reset_index(drop=True, inplace=False)
    sorted_kmeans_items_1_results['Average Order Total When Present'] = sorted_kmeans_items_1_results['Total']
    sorted_kmeans_items_1_results = sorted_kmeans_items_1_results[['Item','Item Classification','Average Order Total When Present']]
    best_item_classification = sorted_kmeans_items_1_results.head(n=20)
    best_item_classification = best_item_classification[['Item Classification']].mode()
    best_item_classification = best_item_classification['Item Classification'][0]
    st.write("The best classification for an item, from a revenue perspective, is:",best_item_classification)
    sorted_kmeans_items_1_results

    st.subheader("Classifying Carts by Composition")
    kmeans_orders = pd.merge(kmeans_items, sorted_kmeans_items_1_results, on="Item", how="left")
    l = []
    for i in kmeans_orders.iterrows():
        order_id = i[1]['Order ID']
        customer = i[1]['Customer']
        item = i[1]['Item']
        item_classification = i[1]['Item Classification']
        if item_classification==best_item_classification:
            count_best=1 
        else:
            count_best=0
        tup = (order_id, customer, item, item_classification, count_best)
        l.append(tup)
    kmeans_orders = pd.DataFrame(l, columns=['Order ID', 'Customer', 'Item', 'Item Classification', 'Count of Top Item Classification'])
    kmeans_orders = kmeans_orders.groupby(['Order ID', 'Customer']).agg({'Count of Top Item Classification':['sum'], 'Item':['count']})
    kmeans_orders = kmeans_orders.reset_index(drop=False, inplace=False)
    kmeans_orders.columns = kmeans_orders.columns.map(lambda x:x[0])
    kmeans_orders['Item Count'] = kmeans_orders['Item']
    kmeans_orders_1 = kmeans_orders[['Count of Top Item Classification', 'Item Count']]
    kmeans_orders_1_scaled = StandardScaler().fit_transform(kmeans_orders_1)
    kmeans_orders_1_scaled = pd.DataFrame(kmeans_orders_1_scaled, columns = kmeans_orders_1.columns)

    #kmeans clustering using count of top classification items (by revenue) and count of items in order
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(kmeans_orders_1_scaled)
        sse.append(kmeans.inertia_)
    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    n = kl.elbow
    kmeans = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=10, random_state=42)
    pred_y = kmeans.fit_predict(kmeans_orders_1_scaled)
    kmeans_orders['Cart Classification'] = pd.Series(pred_y, index=kmeans_orders.index)

    #graphing customer clusters, observe qualities of customers
    c = alt.Chart(kmeans_orders).mark_circle().encode(
        x='Count of Top Item Classification', y='Item Count', color='Cart Classification', tooltip=['Customer', 'Count of Top Item Classification', 'Item Count', 'Cart Classification'])

    st.altair_chart(c, use_container_width=True)

    customer_classification = pd.merge(kmeans_items, kmeans_orders, on="Customer", how="left")
    customer_classification = customer_classification[["Customer", 'Item Count', 'Count of Top Item Classification', "Total", "Cart Classification"]]
    customer_classification = customer_classification.drop_duplicates()
    customer_classification = customer_classification.reset_index(drop=True, inplace=False)
    customer_classification

    #kmeans clustering using order count and AOV by customer
    st.subheader("Clustering Customer Types By Order History")
    k_means = orders.groupby(['Customer']).agg({'Order Count':['sum'], 'Order Value':['mean']})
    k_means = k_means.reset_index(drop=False, inplace=False)
    k_means.columns = k_means.columns.map(lambda x:x[0])
    k_means2 = k_means[['Order Count', 'Order Value']]
    k_means_scaled = StandardScaler().fit_transform(k_means2)
    k_means_scaled = pd.DataFrame(k_means_scaled, columns = k_means2.columns)

    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(k_means_scaled)
        sse.append(kmeans.inertia_)
    kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
    n = kl.elbow
    kmeans = KMeans(n_clusters=n, init='k-means++', max_iter=300, n_init=10, random_state=42)
    pred_y = kmeans.fit_predict(k_means_scaled)
    k_means['Classification'] = pd.Series(pred_y, index=k_means.index)

    #graphing customer clusters, observe qualities of customers
    c = alt.Chart(k_means).mark_circle().encode(
        x='Order Count', y='Order Value', color='Classification', tooltip=['Customer', 'Order Count', 'Order Value', 'Classification'])

    st.altair_chart(c, use_container_width=True)

    k_means

if choose == "Merchandising Data":
    st.title('Merchandising Data')

    option = st.selectbox(
        'Select a view',
        ('All', 'Item Popularity', 'Breakdown by Vendor', 'Breakdown by Category'))

    if option == 'All':
        st.write('Data starting from:')
        st.write(str(fromDate))
        merch_data

    elif option == 'Item Popularity':
        st.write('Data starting from:')
        st.write(str(fromDate))

        orders_items_data2 = orders_items_data
        orders_items_data2['Orders with Item'] = orders_items_data2['Order ID']
        orders_items_data2['Order Prolificity Rank'] = orders_items_data2['Orders with Item'].rank(method='dense', ascending=False)
        orders_items_data2 = orders_items_data2[['Item', 'Market', 'Orders with Item', 'Order Prolificity Rank']]
        st.subheader("Top Items by Number of Orders")
        orders_items_data2

        c = alt.Chart(orders_items_data.head(n=20)).mark_circle().encode(
        x='Item', y='Orders with Item', size='Orders with Item', color='Market', tooltip=['Item', 'Orders with Item', 'Market'])
        st.altair_chart(c, use_container_width=True)

        inFlow_orders_item = merch_data.groupby(['Item']).agg({'Quantity':['sum']})
        inFlow_orders_item = inFlow_orders_item.reset_index(drop=False, inplace=False)
        inFlow_orders_item.columns = inFlow_orders_item.columns.map(lambda x:x[0])
        inFlow_orders_item = inFlow_orders_item.sort_values("Quantity", ascending=False)
        inFlow_orders_item = inFlow_orders_item.reset_index(drop=True, inplace=False)
        inFlow_orders_item = inFlow_orders_item.set_index('Item')
        inFlow_orders_item['Item Prolificity Rank'] = inFlow_orders_item['Quantity'].rank(method='dense', ascending=False)
        st.subheader("Top Items by Total Quantity Purchased")
        inFlow_orders_item

        orders_items_data = orders_items_data[['Item', 'Orders with Item']]
        orders_items_data_agg = orders_items_data.groupby(['Item']).agg({'Orders with Item':['sum']})
        orders_items_data_agg = orders_items_data_agg.reset_index(drop=False, inplace=False)
        orders_items_data_agg.columns = orders_items_data_agg.columns.map(lambda x:x[0])
        orders_items_data_agg = orders_items_data_agg.sort_values("Orders with Item", ascending=False)
        orders_items_data_agg = orders_items_data_agg.reset_index(drop=True, inplace=False)
        orders_items_data_agg = orders_items_data_agg.set_index('Item')
        orders_items_data_agg['Order Prolificity Rank'] = orders_items_data_agg['Orders with Item'].rank(method='dense', ascending=False)

        rank_compare = orders_items_data_agg.merge(inFlow_orders_item, on='Item', how='left')
        rank_compare['Avg Quantity When Ordered'] = rank_compare['Quantity']/rank_compare['Orders with Item']
        rank_compare['Composite Score'] = rank_compare['Order Prolificity Rank'] + rank_compare['Item Prolificity Rank']
        rank_compare = rank_compare.sort_values("Composite Score", ascending=True)

        st.subheader("Top Items Ranked by Composite Score")
        st.write("The composite score is simply an item's rank based on quantity sold + its rank by number of orders in which it's included...")
        rank_compare

    elif option == 'Breakdown by Vendor':
        inFlow_orders_vendor = merch_data.groupby(['Vendor']).agg({'Quantity':['sum']})
        inFlow_orders_vendor = inFlow_orders_vendor.reset_index(drop=False, inplace=False)
        inFlow_orders_vendor.columns = inFlow_orders_vendor.columns.map(lambda x:x[0])
        inFlow_orders_vendor = inFlow_orders_vendor.sort_values("Quantity", ascending=False)
        inFlow_orders_vendor = inFlow_orders_vendor.reset_index(drop=True, inplace=False)
        inFlow_orders_vendor = inFlow_orders_vendor.set_index('Vendor')
        st.write('Data starting from:')
        st.write(str(fromDate))
        inFlow_orders_vendor
        st.subheader("Top 20 Vendors")
        st.bar_chart(inFlow_orders_vendor.head(n=20))

    elif option == 'Breakdown by Category':
        inFlow_orders_category = merch_data.groupby(['Category']).agg({'Quantity':['sum']})
        inFlow_orders_category = inFlow_orders_category.reset_index(drop=False, inplace=False)
        inFlow_orders_category.columns = inFlow_orders_category.columns.map(lambda x:x[0])
        inFlow_orders_category = inFlow_orders_category.sort_values("Quantity", ascending=False)
        inFlow_orders_category = inFlow_orders_category.reset_index(drop=True, inplace=False)
        inFlow_orders_category = inFlow_orders_category.set_index('Category')
        st.write('Data starting from:')
        st.write(str(fromDate))
        inFlow_orders_category
        st.bar_chart(inFlow_orders_category)

elif choose == 'Refunds & Unpicked Items':

    st.title('Unpicked Items (Refunds) Data')

    st.header("Outstanding Refunds Today")
    #Pull in Refunds 2.0 Data from Chameleon Reports
    refunds_today = refunds_today.set_index('order_id')
    refunds_today

    st.header('Unpicked Items by Item (Last 7 Days)')
    #Pull in Unpicked Items by Items (Last 7 Days) from Chameleon Reports
    refunds_items_7days = refunds_items_7days.set_index('sku')
    refunds_items_7days
    st.bar_chart(refunds_items_7days.head(n=10))

    st.header('Unpicked Items by Location (Last 7 Days)')
    l = []
    for i in refunds_items_location_7days.iterrows():
        market = i[1]['orderNumber'].split("-")[0]
        tup = (market)
        l.append(market)
    refunds_items_location_7days['market']=l
    refunds_items_location_7days = refunds_items_location_7days.sort_values(by=['refund_qty'], ascending=False)
    refunds_items_location_7days
    st.text('Unpicked Items: Berkeley Warehouse')
    eb_sf_refunds = refunds_items_location_7days[(refunds_items_location_7days.market!='LA')].head(n=10)
    eb_sf_refunds = eb_sf_refunds.set_index('market')
    eb_sf_refunds
    c = alt.Chart(eb_sf_refunds).mark_circle().encode(
        x='sku', y='refund_qty', size='refund_qty', color='cost', tooltip=['sku', 'refund_qty', 'cost'])

    st.altair_chart(c, use_container_width=True)


    st.text('Unpicked Items: Los Angeles Warehouse')
    la_refunds = refunds_items_location_7days[(refunds_items_location_7days.market=='LA')].head(n=10)
    la_refunds = la_refunds.set_index('market')
    la_refunds
    c = alt.Chart(la_refunds).mark_circle().encode(
        x='sku', y='refund_qty', size='refund_qty', color='cost', tooltip=['sku', 'refund_qty', 'cost'])

    st.altair_chart(c, use_container_width=True)

elif choose == 'Order Distribution & Metrics':

    st.title('Order Data')
    #Pull in Lat Lon from Chameleon Reports
    lat_long = lat_long[['lat', 'lon']]
    st.write('Data starting from:')
    st.write(str(fromDate))

    st.header("Order Density By City")

    st.subheader("The Bay Area")
    st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
    latitude=37.76,
    longitude=-122.4,
    zoom=9,
    pitch=50,
    ),
    layers=[
    pdk.Layer(
    'HexagonLayer',
    data=lat_long,
    get_position='[lon, lat]',
    get_color='[200, 30, 0, 160]',
    radius=1000,
    elevation_scale=20,
    elevation_range=[0, 1000],
    pickable=True,
    #extruded=True,
    ),],))

    st.subheader("Los Angeles")
    st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
    latitude=34.1,
    longitude=-118.35,
    zoom=10,
    pitch=50,
    ),
    layers=[
    pdk.Layer(
    'HexagonLayer',
    data=lat_long,
    get_position='[lon, lat]',
    get_color='[200, 30, 0, 160]',
    radius=1000,
    elevation_scale=20,
    elevation_range=[0, 1000],
    pickable=True,
    #extruded=True,
    ),],))

    st.header('Orders Breakdown')

    st.subheader('Order Revenue by Date')
    inFlow_orders_revenue = orders.groupby(['Date', 'Market']).agg({'Revenue':['sum']})
    inFlow_orders_revenue = inFlow_orders_revenue.reset_index(drop=False, inplace=False)
    inFlow_orders_revenue.columns = inFlow_orders_revenue.columns.map(lambda x:x[0])
    inFlow_orders_revenue = inFlow_orders_revenue.reset_index(drop=True, inplace=False)
    inFlow_orders_revenue = inFlow_orders_revenue.set_index('Date')
    inFlow_agg_revenue = orders.agg({'Revenue':['sum']})
    st.write("Total Revenue: ", '$',str(round(inFlow_agg_revenue['Revenue'][0])))
    inFlow_orders_revenue
    st.bar_chart(inFlow_orders_revenue)

    st.text('Order Revenue: EB')
    st.line_chart(inFlow_orders_revenue[(inFlow_orders_revenue.Market=="EB")])

    st.text('Order Revenue: SF')
    st.line_chart(inFlow_orders_revenue[(inFlow_orders_revenue.Market=="SF")])

    st.text('Order Revenue: LA')
    st.line_chart(inFlow_orders_revenue[(inFlow_orders_revenue.Market=="LA")])

    st.subheader('Order Volume by Date')
    inFlow_orders_volume = orders.groupby(['Date', 'Market']).agg({'Order Count':['sum']})
    inFlow_orders_volume = inFlow_orders_volume.reset_index(drop=False, inplace=False)
    inFlow_orders_volume.columns = inFlow_orders_volume.columns.map(lambda x:x[0])
    inFlow_orders_volume = inFlow_orders_volume.set_index('Date')
    inFlow_agg_volume = orders.agg({'Order Count':['sum']})
    st.write("Total Orders: ", str(round(inFlow_agg_volume['Order Count'][0])))
    inFlow_orders_volume
    st.bar_chart(inFlow_orders_volume)

    st.text('Order Volume: EB')
    st.line_chart(inFlow_orders_volume[(inFlow_orders_volume.Market=="EB")])

    st.text('Order Volume: SF')
    st.line_chart(inFlow_orders_volume[(inFlow_orders_volume.Market=="SF")])

    st.text('Order Volume: LA')
    st.line_chart(inFlow_orders_volume[(inFlow_orders_volume.Market=="LA")])

    st.subheader('AOV by Date')
    inFlow_orders_aov = orders.groupby(['Date']).agg({'Order Value':['mean']})
    inFlow_orders_aov = inFlow_orders_aov.reset_index(drop=False, inplace=False)
    inFlow_orders_aov.columns = inFlow_orders_aov.columns.map(lambda x:x[0])
    inFlow_orders_aov = inFlow_orders_aov.set_index('Date')
    inFlow_agg_aov = orders.agg({'Order Value':['mean']})
    st.write("Aggregate AOV: ", "$",str(round(inFlow_agg_aov['Order Value'][0])))
    inFlow_orders_aov
    st.bar_chart(inFlow_orders_aov)
