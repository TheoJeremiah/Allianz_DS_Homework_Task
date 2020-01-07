import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
import dash_table
from dash.dependencies import Input, Output, State
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
df = pd.read_excel('Part 1. Modeling - Credit_Data.xls')

f = open('loan_prediction_model_2.pkl', 'rb')
logmodel = pickle.load(f)

standard_scaler = StandardScaler()


def hasil_prediksi(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20):
    
    temp=[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20]
    row_test = df.head(1).drop('default',axis=1)
    j=0
    for i in row_test.columns:
        row_test[i] = temp[j]
        j+=1
    
    df_row_test = df.copy()
    df_row_test.drop('default',axis=1,inplace=True)
    df_row_test = df_row_test.append(row_test)

    for i in ['duration_in_month','credit_amount','installment_as_income_perc','age','present_res_since','credits_this_bank','people_under_maintenance']:
        df_row_test[i] = pd.DataFrame(standard_scaler.fit_transform(df_row_test[[i]]))[0]

    df_row_test = pd.get_dummies(data=df_row_test, drop_first=True, columns=['account_check_status', 'credit_history', 'purpose', 'savings','present_emp_since', 'personal_status_sex','other_debtors','property','other_installment_plans','housing','job','telephone','foreign_worker'])

    row_test = df_row_test.tail(1)

    hasil=''
    hasil='Loan Status: \n'
    if logmodel.predict(row_test)[0] == 0:
        hasil+='Rejected'
    else:
        hasil+='Accepted'
    hasil+='\n\n'
    hasil+='Credit Score: \n'
    hasil+=str(logmodel.predict_proba(row_test)[:,1][0]*100)

    return hasil


app.layout = html.Div(children = [
    html.H1('Loan Application Model'),
    html.P('Created by: Theo Jeremiah'),
    dcc.Tabs(value = 'tabs', id = 'tabs-1', children = [

        dcc.Tab(label = 'Application Profile', id = 'tab-dua', children = [
        
        html.Div(children = [
            html.P('Account Check Status:'),
            dcc.Dropdown(id = 'input_account_check_status',
            options = [{'label' : i, 'value' : i} for i in df['account_check_status'].unique()],
            value = 'kosong')                
            ],className = 'col-3'),
        
        html.Br(),

        html.Div(children = [
            html.P('Loan Duration (in Month):'),
            dcc.Input(
            id='input_duration_in_month',
            type='number',
            placeholder='input month',)                
            ],className = 'col-3'),

        html.Br(),

        html.Div(children = [
            html.P('Credit History:'),
            dcc.Dropdown(id = 'input_credit_history',
            options = [{'label' : i, 'value' : i} for i in df['credit_history'].unique()],
            value = 'kosong')                
            ],className = 'col-3'),
        
        html.Br(),

        html.Div(children = [
            html.P('Purpose:'),
            dcc.Dropdown(id = 'input_purpose',
            options = [{'label' : i, 'value' : i} for i in df['purpose'].unique()],
            value = 'kosong')                
            ],className = 'col-3'),
        
        html.Br(),
        
        html.Div(children = [
            html.P('Credit Amount: '),
            dcc.Input(
            id='input_credit_amount',
            type='number',
            placeholder='input credit amount',)                
            ],className = 'col-3'),

        html.Br(),

        html.Div(children = [
            html.P('Saving Account Status:'),
            dcc.Dropdown(id = 'input_savings',
            options = [{'label' : i, 'value' : i} for i in df['savings'].unique()],
            value = 'kosong')                
            ],className = 'col-3'),
        
        html.Br(),

        html.Div(children = [
            html.P('Present Employee Since:'),
            dcc.Dropdown(id = 'input_present_emp_since',
            options = [{'label' : i, 'value' : i} for i in df['present_emp_since'].unique()],
            value = 'kosong')                
            ],className = 'col-3'),
        
        html.Br(),

        html.Div(children = [
            html.P('Installment as Income Percentage: '),
            dcc.Input(
            id='input_installment_as_income_perc',
            type='number',
            placeholder='input installment as income perc',)                
            ],className = 'col-3'),

        html.Br(),

        html.Div(children = [
            html.P('Personal Status Sex:'),
            dcc.Dropdown(id = 'input_personal_status_sex',
            options = [{'label' : i, 'value' : i} for i in df['personal_status_sex'].unique()],
            value = 'kosong')                
            ],className = 'col-3'),
        
        html.Br(),

        html.Div(children = [
            html.P('Other Debtors:'),
            dcc.Dropdown(id = 'input_other_debtors',
            options = [{'label' : i, 'value' : i} for i in df['other_debtors'].unique()],
            value = 'kosong')                
            ],className = 'col-3'),
        
        html.Br(),

        html.Div(children = [
            html.P('Present Residence Since: '),
            dcc.Input(
            id='input_present_res_since',
            type='number',
            placeholder='input present res since',)                
            ],className = 'col-3'),

        html.Br(),

        html.Div(children = [
            html.P('Property:'),
            dcc.Dropdown(id = 'input_property',
            options = [{'label' : i, 'value' : i} for i in df['property'].unique()],
            value = 'kosong')                
            ],className = 'col-3'),
        
        html.Br(),

        html.Div(children = [
            html.P('Age: '),
            dcc.Input(
            id='input_age',
            type='number',
            placeholder='input age',)                
            ],className = 'col-3'),

        html.Br(),

        html.Div(children = [
            html.P('Other Installment Plans:'),
            dcc.Dropdown(id = 'input_other_installment_plans',
            options = [{'label' : i, 'value' : i} for i in df['other_installment_plans'].unique()],
            value = 'kosong')                
            ],className = 'col-3'),
        
        html.Br(),

        html.Div(children = [
            html.P('Housing:'),
            dcc.Dropdown(id = 'input_housing',
            options = [{'label' : i, 'value' : i} for i in df['housing'].unique()],
            value = 'kosong')                
            ],className = 'col-3'),
        
        html.Br(),

        html.Div(children = [
            html.P('Credit in this bank: '),
            dcc.Input(
            id='input_credits_this_bank',
            type='number',
            placeholder='input credit this bank',)                
            ],className = 'col-3'),

        html.Br(),

        html.Div(children = [
            html.P('Job:'),
            dcc.Dropdown(id = 'input_job',
            options = [{'label' : i, 'value' : i} for i in df['job'].unique()],
            value = 'kosong')                
            ],className = 'col-3'),
        
        html.Br(),

        html.Div(children = [
            html.P('People under maintenance: '),
            dcc.Input(
            id='input_people_under_maintenance',
            type='number',
            placeholder='input people under maintenance',)                
            ],className = 'col-3'),

        html.Br(),

        html.Div(children = [
            html.P('Telephone:'),
            dcc.Dropdown(id = 'input_telephone',
            options = [{'label' : i, 'value' : i} for i in df['telephone'].unique()],
            value = 'kosong')                
            ],className = 'col-3'),
        
        html.Br(),

        html.Div(children = [
            html.P('Foreign Worker:'),
            dcc.Dropdown(id = 'input_foreign_worker',
            options = [{'label' : i, 'value' : i} for i in df['foreign_worker'].unique()],
            value = 'kosong')                
            ],className = 'col-3'),
        
        html.Br(),

        html.Div(html.Button('Search'), id = 'search3', className = 'col-3'),

        html.Br(),

        html.Div(id = 'predict', children= [
            html.H1('Fill in the Parameters')
            ],style={'text-align':'center'})


        ]),

    ],
    content_style = {
        'fontFamily' : 'Arial',
        'borderBottom' : '1px solid #d6d6d6',
        'borderLeft' : '1px solid #d6d6d6',
        'borderRight' : '1px solid #d6d6d6',
        'padding' : '44px'
    })
],style={'maxWidth': '1200px', 'margin': '0 auto'})

@app.callback(
    [Output(component_id = 'predict', component_property = 'children')],
    [Input(component_id = 'search3', component_property = 'n_clicks')],
    [State(component_id = 'input_account_check_status', component_property = 'value'),
    State(component_id = 'input_duration_in_month', component_property = 'value'),
    State(component_id = 'input_credit_history', component_property = 'value'),
    State(component_id = 'input_purpose', component_property = 'value'),
    State(component_id = 'input_credit_amount', component_property = 'value'),
    State(component_id = 'input_savings', component_property = 'value'),
    State(component_id = 'input_present_emp_since', component_property = 'value'),
    State(component_id = 'input_installment_as_income_perc', component_property = 'value'),
    State(component_id = 'input_personal_status_sex', component_property = 'value'),
    State(component_id = 'input_other_debtors', component_property = 'value'),
    State(component_id = 'input_present_res_since', component_property = 'value'),
    State(component_id = 'input_property', component_property = 'value'),
    State(component_id = 'input_age', component_property = 'value'),
    State(component_id = 'input_other_installment_plans', component_property = 'value'),
    State(component_id = 'input_housing', component_property = 'value'),
    State(component_id = 'input_credits_this_bank', component_property = 'value'),
    State(component_id = 'input_job', component_property = 'value'),
    State(component_id = 'input_people_under_maintenance', component_property = 'value'),
    State(component_id = 'input_telephone', component_property = 'value'),
    State(component_id = 'input_foreign_worker', component_property = 'value')]
)

def create_predictions(n_clicks, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20):
    if x1=='kosong':
        children = [
            html.H1('Fill in the Parameters')
        ]
    else:
        children = [
            dcc.Markdown(
            hasil_prediksi(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20),
            style={"white-space": "pre"}),
                ]
    return children

if __name__ == '__main__':
    app.run_server(debug=True)