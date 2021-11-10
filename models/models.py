# -*- coding: utf-8 -*-
from odoo import models, fields, api
from odoo.exceptions import ValidationError
import turicreate as tc
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import time
import datetime as dt
import pyodbc



class cartimex(models.Model):
    _name = 'cartimex.cartimex'
    _description = 'cartimex.cartimex'

    name = fields.Text()
    value = fields.Integer()
    value2 = fields.Float(compute="_value_pc", store=True)
    description = fields.Text()
    categorias = fields.Many2one('cartimex.cartimex', 'categorias')

    @api.depends('value')
    def _value_pc(self):
        for record in self:
            record.value2 = float(record.value) / 100

    def make_request(self):
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
        raise ValidationError(str(data))

    def prediccion(self):
        tmp = time.time()
        server = '10.5.1.247'
        database = 'CARTIMEX'
        username = 'esanchezf'
        password = 'Dosmillones354879'
        cnxn = pyodbc.connect('DRIVER={ODBC Driver 17 for SQL Server};SERVER=' +
                            server+';DATABASE='+database+';UID='+username+';PWD=' + password)
        #cursor = cnxn.cursor()
        codigo = ""
        for record in self:
            codigo = record.name
        #codigo = 'L3150'
        query = "EXEC [dbo].[VEN_DIARIAS_CODIGO] @Codigo ='"+codigo+"' "
        # print(query)
        # guarda el resultado del query en main_df
        main_df = pd.read_sql(query, cnxn)

        print("Data sourcing Time: %s seconds" % (str(time.time() - tmp)))
        print(query)


        # sobreescribe el indice defaul que viene con el dataframe, un indice numerico que genera sql, con la fecha
        main_df.index = pd.to_datetime(main_df['fecha'], format='%d%m%Y')
        main_df.index = pd.to_datetime(main_df.index, infer_datetime_format=True)
        main_df = main_df.drop(['fecha'], axis=1)


        print(main_df)


        comienzo = main_df.ne(0).idxmax().cantidad
        print(comienzo)


        training_df = main_df


        def cap_outliers(series, zscore_threshold=6, verbose=False):
            '''Caps outliers to closest existing value within threshold (Z-score).'''
            mean_val = series.mean()
            std_val = series.std()

            z_score = (series - mean_val) / std_val
            outliers = abs(z_score) > zscore_threshold

            series = series.copy()
            series.loc[z_score > zscore_threshold] = series.loc[~outliers].max()
            series.loc[z_score < -zscore_threshold] = series.loc[~outliers].min()

            # For comparison purposes.
            if verbose:
                lbound = mean_val - zscore_threshold * std_val
                ubound = mean_val + zscore_threshold * std_val

            return series


        # PODA LAS VENTAS OUTLIERS,EN 6 DESVIACIONES ESTANDAR DE LA MEDIA
        # verbose = TRUE PARA VISUALIZAR LOS LIMITES
        main_df['cantidad'] = cap_outliers(
            main_df['cantidad'], zscore_threshold=6, verbose=True)


        training_df = main_df


        def mape_vectorized(a, b):
            mask = a != 0
            return (np.fabs(a[mask] - b[mask])/a[mask]).mean()


        start = comienzo
        training_df = training_df.loc[training_df.index >= start].copy()
        split_date = "09-01-2020"  # ENTRENAMIENTO HASTA EL 2020, septiembre, LUEGO TEST

        fecha = pd.to_datetime(split_date, format="%m-%d-%Y")

        df_train = training_df.loc[training_df.index <= fecha].copy()
        df_test = training_df.loc[training_df.index > fecha].copy()


        def create_features(df, label=None):
            df['date'] = df.index
            df['quarter'] = df['date'].dt.quarter
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            df['weekofyear'] = df['date'].dt.weekofyear  # .dt.isocalendar().week
            df['promo'] = df['promo'].astype("int64")
            df['dia'] = df['date'].dt.day
            df['dayofweek'] = df['date'].dt.dayofweek
            df['pandemia'] = df['pandemia'].astype("int64")
            df['feriado'] = df['feriado'].astype("int64")

            X = df[[
                'quarter', 'month', 'year', 'pandemia', 'weekofyear', 'sucursales', 'promo', 'feriado', 'dia', 'dayofweek'
            ]]
            if label:
                y = df[label]
                return X, y
            return X

        # dataset para training
        X_train, y_train = create_features(df_train, label='cantidad')
        # dataset para testing
        X_test, y_test = create_features(df_test, label='cantidad')
        # dataset para fit final una vez lavidado el modelo
        X_final, y_final = create_features(training_df, label='cantidad')


        params = {
            'min_child_weight': [1],
            'gamma': [0],
            'subsample': [1],
            'colsample_bytree': [1],
            'max_depth': [6],
            'verbosity': [1],
            'eta': [0.3],
            'n_estimators': [10, 20, 30, 50, 100, 200]
        }
        # Initialize XGB and GridSearch
        xgb_reg = xgb.XGBRegressor(objective='reg:linear', eval_metric='rmse')
        grid = GridSearchCV(xgb_reg, params)
        grid2 = grid  # se inicializa grid 2 para hacer fit al dataset final
        grid.fit(X_train, y_train)
        gridcv_xgb = grid.best_estimator_
        grid2.fit(X_final, y_final)
        gridcv_xgb2 = grid2.best_estimator_
        predicciones = [0 if i < 0 else i for i in gridcv_xgb.predict(X_test)]

        print(predicciones)

        #df_test['Prediction'] = gridcv_xgb.predict(X_test)
        df_test['Prediction'] = predicciones
        df_all = pd.concat([df_train, df_test], sort=False)

        print('FIN')
        raise ValidationError(str(predicciones))
        

