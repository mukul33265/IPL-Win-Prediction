import streamlit as st
import pickle
import pandas as pd
import sklearn
st.title('IPL Win Predictor')
teams=[
   'Royal Challengers Bengaluru',
    'Gujarat Titans',
    'Lucknow Super Giants',
    'Punjab Kings',
    'Delhi Capitals',
    'Sunrisers Hyderabad',
    'Rajasthan Royals',
    'Kolkata Knight Riders',
    'Mumbai Indians',
    'Chennai Super Kings'
]
pipe=pickle.load(open('pipe.pkl','rb'))
col1,col2=st.columns(2)

with col1:
    batting_team=st.selectbox('Select the batting team ',sorted(teams))

with col2:
    bowling_team = st.selectbox(
        'Select the bowling team',
        sorted([team for team in teams if team != batting_team])
    )


target=st.number_input('Target',min_value=0,step=1)
col3,col4,col5=st.columns(3)
with col3:
    score=st.number_input('Score',min_value=0, step=1)
with col4:
    overs=st.number_input('Overs completed',min_value=0.0, step=0.1)
with col5:
    wickets=st.number_input('Wickets out',min_value=0, step=1)

if(st.button('Predict Probability')):
    runs_left=target-score
    over_full = int(overs)
    ball_uncompleted = int((overs - over_full) * 10)
    balls_bowled = over_full * 6 + ball_uncompleted
    balls_left = 120 - balls_bowled
    wickets=10-wickets
    crr = score / overs if overs > 0 else 0
    rrr=runs_left*6/balls_left

    input=pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})
    result=pipe.predict_proba(input)
    loss=result[0][0]
    win=result[0][1]
    if wickets == 0 and runs_left > 0:
        st.text(batting_team + " - 0%")
        st.text(bowling_team + " - 100%")
    elif rrr > 36:
        st.text(batting_team + " - 0%")
        st.text(bowling_team + " - 100%")
    else:
        st.text(batting_team + "- " + str(round(win*100)) + "%" )
        st.text(bowling_team + "- " + str(round(loss*100)) + "%")
