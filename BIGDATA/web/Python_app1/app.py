# Web 구동 
import dash
# Web구성 Layer 설정 (View) 
import dash_bootstrap_components as dbc
import dash_html_components as html
# 동적인 자료를 다룰때 사용하는 함수
import dash_core_components as dcc

# Dash 함수를 가져와 웹을 구성
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
#########################################################################################
# 데이터를 불러와 시각화 하는 부분 (Model)
import pandas as pd 
import plotly.express as px 
df1 = pd.read_csv('12_Data.csv')

# 데이터를 시각화
figure1 = px.scatter(df1, x='thickness', y='Target')
# 관리도 시각화 thickness 항목에 대해 
USL = df1['thickness'].mean() + df1['thickness'].std() * 3
LSL = df1['thickness'].mean() - df1['thickness'].std() * 3
df1_xbar = df1[['thickness']].reset_index().head(50)

figure2 = px.line(df1_xbar, x= df1_xbar.index, y='thickness')
figure2.add_hline(y= USL, line_color='Red')
figure2.add_hline(y= LSL, line_color='Red')
figure2.update_layout(yaxis_range = [USL+10 , LSL-10])
figure2.update_traces(mode= 'markers+lines')
########################################################################################
# Web에서 입력받은 값을 Model로 처리하여 Output 계산 Part 
from dash.dependencies import Input, Output
import pickle
model = pickle.load(open('model_LGBR.sav','rb'))

# Callback (Dynamic)
@app.callback(
    Output('result', 'children'),
    Input('thickness','value'),
    Input('resist_target','value'),
    Input('Line_CD','value'),
    Input('Etching_rate','value')    
)
def predict_func(x1,x2,x3,x4):
    input_data = pd.DataFrame([[x1,x2,x3,x4]])
    result = model.predict(input_data)
    return result

#########################################################################################
input_layout = [
    html.Div([
        html.Label('thickness'), 
        dcc.Input(id='thickness', placeholder='값을 입력하시오...', type='text')
        ]),
    html.Div([
        html.Label('resist_target'), 
        dcc.Input(id='resist_target', placeholder='값을 입력하시오...', type='text')
        ]),
    html.Div([
        html.Label('Line_CD'), 
        dcc.Input(id='Line_CD', placeholder='값을 입력하시오...', type='text')
        ]),
    html.Div([
        html.Label('Etching_rate'), 
        dcc.Input(id='Etching_rate', placeholder='값을 입력하시오...', type='text')
        ])    
]
#########################################################################################
# Web에 나오는화면을 구성하는 부분 (app.layout) 
app.layout = dbc.Container(
    # Layer 틀을 구성 
    [
        html.H3(children='Web Application'),
        dbc.Col(html.Div(input_layout)),
        html.Hr(), # 밑줄 
        dbc.Card([
            html.H2(id='result', className='card_title'),
            html.P('불량 개수', className='card_text')
        ], body=True ),
        dcc.Graph(figure=figure1),
        dcc.Graph(figure=figure2)
    ] 
)
# Web Frame 구동
if __name__ == '__main__' : 
    app.run_server(debug=True)