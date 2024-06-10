import pandas as pd
from dash import Dash, dcc, html, callback, Input, Output
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


ages = []
for age in range(1, 101):
    d = {}
    d['label'] = age
    d['value'] = age
    ages.append(d)
style={'font-family': 'Calibri', 'font-size': 30}


features = pd.read_csv('Features.csv')
survived = pd.read_csv('Survived.csv')
print(features.shape, survived.shape)

forest = RandomForestClassifier()
param_grid = [{"n_estimators":[10,100,200,500], "max_depth":[None,5,10],"min_samples_split":[2,3,4]}]
forest_grid = GridSearchCV(forest, param_grid)
forest_grid.fit(features, survived)


app = Dash(__name__)

app.layout = html.Div([
    html.H1('Would you survive on the Titanik?', style={'font-family': 'Calibri', 'font-size': 60}),
    html.Br(),

    dcc.RadioItems(id='sex',
                   options=[{'label': 'Male', 'value': 0},
                            {'label': 'Female', 'value': 1}],
                   value=0, inline=True, style=style),
    html.Br(),

    dcc.RadioItems(id='class',
                   options=[{'label': '1st Class', 'value': 1},
                            {'label': '2nd Class', 'value': 2},
                            {'label': '3rd Class', 'value': 3}],
                   value=1, inline=True, style=style),
    html.Br(),

    dcc.RadioItems(id='spouse',
                   options=[{'label': 'No sibling / spouse aboard the Titanik', 'value': 0},
                            {'label': 'With sibling / spouse aboard the Titanik', 'value': 1}],
                   value=0, inline=True, style=style),
    html.Br(),

    dcc.RadioItems(id='parents', options=[{'label': 'No parents aboard the Titanik', 'value': 0},
                            {'label': 'With parents the Titanik', 'value': 1}],
                   value=0, inline=True, style=style),
    html.Br(),

    dcc.RadioItems(id='port', options=[{'label': 'Cherbourg', 'value': '100'},
                            {'label': 'Queenstown', 'value': '010'},
                            {'label': 'Southampton', 'value': '001'}],
                   value='100', inline=True, style=style),
    html.Br(),

    dcc.Dropdown(id='age', options=ages, value = 33, placeholder='Your age', style=style),
    html.Br(),

    html.H1(id='survived', style={'font-family': 'Calibri', 'color': 'red', 'font-size': 160})
])

@callback(
        Output(component_id='survived', component_property='children'),
        [Input(component_id='sex', component_property='value'),
         Input(component_id='class', component_property='value'),
         Input(component_id='spouse', component_property='value'),
         Input(component_id='parents', component_property='value'),
         Input(component_id='port', component_property='value'),
         Input(component_id='age', component_property='value'),]
          )
def viz(sex, ticket_class, spouse, parents, port, age):
    x = pd.DataFrame([sex, ticket_class, spouse, parents, port[0], port[1], port[2], age]).T
    yhat = forest_grid.predict(x)
    return 'YES' if yhat == 1 else 'NO'



if __name__ == '__main__':
    app.run_server()
