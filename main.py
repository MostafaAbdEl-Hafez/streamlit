import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
import spacy

st.markdown("# DCA using ARP's model ")
st.sidebar.write("## Inputs ")

file = st.sidebar.file_uploader("Upload your Excel file for a single well : ")



if file:
   try:
        df = pd.read_excel(file)
        cols = list(df.columns)
        production_col = st.sidebar.selectbox("production column",cols)
        date_col = st.sidebar.selectbox("date column",cols)
        freq = st.sidebar.selectbox("date frequency",[ "Daily" , "Monthly" , "Yearly" ])

        # smoothing the data
        st.write("#### Smooting the data using `Moving Average` ")
        def remove_outliers(df, col_name="production"):
            df = df[ df["BORE_OIL_VOL"] != 0]
            return df



        def normalize(df, x,y):
            df[x + " smoothed"] = df[x].rolling(window=y, center=True).mean()
            return df


        def days(df, x):
            df["days"] = (df[x] - df[x].min()).dt.days
            return df

        window_size = st.slider("Window size for MA",value=400,min_value=0,max_value=500,step=10)
        st.write(window_size)
        remove_outliers(df,production_col)
        normalize(df,production_col,window_size)
        days(df,date_col)

        plt.style.use('classic')
        plt.figure(figsize=(8, 6))
        fig, ax = plt.subplots()
        ax.plot(df["DATEPRD"],df["BORE_OIL_VOL"],label="original",color="#89cff0")
        ax.plot(df["DATEPRD"],df["BORE_OIL_VOL smoothed"],label="smoothed",color="#007fff")
        ax.grid(which="major",color="#6666",linestyle="-",alpha=.5)
        ax.minorticks_on()
        ax.grid(which="minor", linestyle="--", alpha=.2, color="#3d85c6")
        plt.legend()
        ax.set_facecolor("#f0fff0")
        ax.set_xlabel("Time", fontsize= 16, labelpad = 20,color="#073763")
        ax.set_ylabel("Oil Production (Q)", fontsize= 16, labelpad = 20,color="#073763")
        ax.set_title("smoothed and before smooth production", fontsize= 20, pad = 20,color="#073763",loc="Center")
        st.pyplot(fig)



        # fitting the data
        st.write("### Fitting data `All ARP's models`")

        df3 = df[["BORE_OIL_VOL smoothed", "days"]].dropna()
        T = df3["days"]
        Q = df3["BORE_OIL_VOL smoothed"]
        T_normalized = T / max(T)
        Q_normalized = Q / max(Q)

        #exponential model
        def exponential(t, qi, di):
            return qi * np.exp(-di * t)
        params, pcov = curve_fit(exponential, T_normalized, Q_normalized)
        qi, di = params
        qi = qi * max(Q)
        di = di / max(T)
        qie=qi
        die=di
        q_E = exponential(T, qi, di)
        plt.style.use('classic')
        plt.figure(figsize=(8, 6))
        fig, ax = plt.subplots()
        ax.plot(T,Q,label="smoothed",color="#89cff0")
        ax.plot(T,q_E,label="Exponential",color="#00cccc")
        ax.grid(which="major", color="#6666", linestyle="-", alpha=.5)
        ax.minorticks_on()
        ax.grid(which="minor", linestyle="--", alpha=.2, color="#3d85c6")
        ax.set_facecolor("#f0fff0")
        ax.set_xlabel("Time", fontsize=16, labelpad=20, color="#073763")
        ax.set_ylabel("Oil Production (Q)", fontsize=16, labelpad=20, color="#073763")
        ax.set_title("All ARP's models", fontsize=20, pad=20, color="#073763", loc="Center")


        #Harmonic model
        def Harmonic(t, qi, di):
            return qi / (1 + di * t)
        params, pcov = curve_fit(Harmonic, T_normalized, Q_normalized)
        qi, di = params
        qi = qi * max(Q)
        di = di / max(T)
        qih=qi
        dih=di
        q_H= Harmonic(T, qi, di)
        ax.plot(T,q_H,label="Harmonic",color="#007fff")

        # Hyperbolic model
        def hyperbolic(t, qi, di, b):
            return qi / (np.abs((1 + b * di * t)) ** (1 / b))


        popt, pcov = curve_fit(hyperbolic, T_normalized, Q_normalized)
        qi, di, b = popt
        qi = qi * max(Q)
        di = di / max(T)
        qihy=qi
        dihy=di
        q_HP= hyperbolic(T, qi, di, b)
        ax.plot(T, q_HP, label = "Hyperbolic", color = "#008080")
        plt.legend()
        st.pyplot(fig)

        # root mean squared error
        def RMSE(q, q_fit):
            """Get the root mean squared error between the fit line model and the normalizea data"""
            N = len(q)
            return np.sqrt(np.sum(q - q_fit) ** 2 / N)

        re=RMSE(Q,q_E)
        rh=RMSE(Q,q_H)
        rhy=RMSE(Q,q_HP)

        d={"ARPS FIT TYPE":["Exponential","Harmonic","Hyperbolic"],"qi":[qie,qih,qihy],"di":[die,dih,dihy],"b":[0,1,np.abs(b)],"RMSE":[re,rh,rhy]}
        df = pd.DataFrame(data=d)
        st.dataframe(df,use_container_width=True)
        filter_criteria = df["RMSE"] == min(df["RMSE"])
        best_fit = df[filter_criteria]
        st.table("The best fit model is :"+ best_fit["ARPS FIT TYPE"].astype(str))
        st.write("#### Made by Mostafa Abd El-Hafez")
   except:
        st.error("make sure you entered the data right")

