#!/usr/bin/env python
# coding: utf-8

# In[1]:


import turicreate as tc


data = tc.SFrame(
    'http://app.compu-tron.net/CDSinstalador/pruebaSolicitudCredito.csv')

data = data.dropna()

data.materialize()

data

train, test = data.random_split(0.8)

model = tc.boosted_trees_regression.create(train, target='atraso',

                                           features=[
                                               'Sexo', 'EstadoCivil', 'Profesion', 'NivelEstudio', 'AgeIntYears', 'Score', 'Salario'],
                                           max_iterations=100,
                                           Step_size=0.2,
                                           row_subsample=0.99,
                                           max_depth=7,
                                           #metric=  { 'max_error','rmse'},
                                           metric={'rmse'})

# Make predictions and evaluate results.
predictions = model.predict(test)
print(predictions)


results = model.evaluate(test)
print(results)
