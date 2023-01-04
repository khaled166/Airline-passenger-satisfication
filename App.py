# Liberary and modules
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import bz2
        
# All features user input.
st.title("Passenger Satisfication prediction")
Gender = st.selectbox("Gender",["Male","Female"])
CustomerType = st.selectbox("Customer Type",['Loyal Customer','disloyal Customer'])
Age = st.slider("Age",0,90)
TypeofTravel = st.selectbox("Type of Travel",['Personal Travel','Business travel'])
Class = st.selectbox("Class",['Eco','Business','Eco Plus'])
Seatcomfort = st.slider("Seat comfort",0,5)
DepartureArrivaltimeconvenient = st.selectbox("Departure/Arrival time convenient",(0,1,2,3,4,5))
Foodanddrink = st.slider("Food and drink",0,5)
Inflightwifiservice = st.slider("Inflight wifi service",0,5)
Inflightentertainment = st.slider("Inflight entertainment",0,5)
Onlinesupport = st.slider("Online support",0,5)
EaseofOnlinebooking = st.selectbox("Ease of Online booking",(0,1,2,3,4,5))
Onboardservice = st.selectbox("On-board service",(0,1,2,3,4,5))
Legroomservice = st.selectbox("Leg room service",(0,1,2,3,4,5))
Baggagehandling = st.selectbox("Baggage handling",(1,2,3,4,5))
Checkinservice = st.selectbox("Checkin service",(0,1,2,3,4,5))
Cleanliness = st.selectbox("Cleanliness",(0,1,2,3,4,5))
Onlineboarding = st.selectbox("Online boarding",(0,1,2,3,4,5))
Delay = st.slider("Delay",0,62) 




    

    
    
 # pandas dataframe format   
df = pd.DataFrame({"Gender":[Gender],"CustomerType":[CustomerType],"Age":[Age],"TypeofTravel":[TypeofTravel],"Class":[Class],
                       "Seatcomfort":[Seatcomfort],"DepartureArrivaltimeconvenient":[DepartureArrivaltimeconvenient],
                       "Foodanddrink":[Foodanddrink],"Inflightwifiservice":[Inflightwifiservice],
                       "Inflightentertainment":[Inflightentertainment],"Onlinesupport":[Onlinesupport],
                       "EaseofOnlinebooking":[EaseofOnlinebooking],"Onboardservice":[Onboardservice],"Legroomservice":[Legroomservice],
                       "Baggagehandling":[Baggagehandling],
                       "Checkinservice":[Checkinservice],"Cleanliness":[Cleanliness],"Onlineboarding":[Onlineboarding],"Delay":[Delay]})   
    


#df = pd.get_dummies(df,columns = ["Gender","CustomerType","TypeofTravel","Class"])

    
transformer = pickle.load(open("C:/Users/khaled_seifaldin/Downloads/FireShot/Airline_passenger/Transformer.pkl","rb"))
    
    
    
for i in range(0,df.shape[1]):
    if df.dtypes[i]=='object':
        df[df.columns[i]] = transformer.fit_transform(df[df.columns[i]])
        


#Load model 
load_model = pickle.load(open("model.pkl","rb"))
   


#Prediction results based on input data from user.
prediction_1 = load_model.predict(df)
Prediction = load_model.predict_proba(df)[0][1]*100
if st.button("Input Result"):
    if(prediction_1==1):
        st.success("Passenger is satisfied")
        st.markdown(f"#### approximately satisfication percentage is: {np.round(Prediction,2)}%")
    else:
        st.error("Passenger is not satisfied")
        st.markdown(f"#### approximately satisfication percentage is: {np.round(Prediction,2)}%")

        
        

