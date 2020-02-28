from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score, classification_report
from sklearn.metrics import plot_confusion_matrix

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class predict_purchases():
    """docstring for predict_purchases"""

    
    def __init__(self):

        self.input_path = 'input/'
        self.output_path = 'output/'

        self.input_file_name = 'online_shoppers_intention.csv'

        self.ordinal_columns = ['Administrative', 'Informational', 'ProductRelated']
        self.numerical_columns = ['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']
        self.categorical_columns = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend', 'Revenue']
        self.onehot_colms = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType']
        self.std_colms_encoded = self.ordinal_columns + self.numerical_columns
        self.std_colms_generated_1 = ['ProductRelated', 'BounceRates', 'PageValues', 'SpecialDay', 'Admin+Admin_duration', 'Inform+Inform_duration']
        self.std_colms_generated_2 = ['BounceRates', 'PageValues', 'SpecialDay', 'A+I+PR', 'Ad+Id+PRd']

        self.month_dict_mapping = {'Feb':2, 'Mar':3, 'May':5, 'June':6, 'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
        self.visitor_dict_mapping = {'New_Visitor':1, 'Returning_Visitor':2, 'Other':3}
        self.weekend_dict_mapping = {False:0, True:1}
        self.revenue_dict_mapping = {False:0, True:1}

        self.random_state = 2

        self.train_ratio = 0.6

        self.n_jobs = -1


    def read_data(self):
        
        self.original_data = pd.read_csv(self.input_path + self.input_file_name)

        self.original_features = self.original_data.copy(deep = True)
        self.original_features.Weekend = self.original_features.Weekend.map(self.weekend_dict_mapping)
        self.original_features.Revenue = self.original_features.Revenue.map(self.revenue_dict_mapping)
        self.original_labels = self.original_features.pop('Revenue')
        
        # Level Encoded features        
        self.encoded_features = self.original_features.copy(deep = True)
        self.encoded_labels = self.original_labels.copy(deep = True)
        
        self.encoded_features.Month = self.encoded_features.Month.map(self.month_dict_mapping)
        self.encoded_features.VisitorType = self.encoded_features.VisitorType.map(self.visitor_dict_mapping)

        # combine A+Ad, I+Id, drop Exit, PRd
        self.generated_features_1 = self.encoded_features.copy(deep = True)
        self.gen_labels_1 = self.encoded_labels.copy(deep = True)        
        
        self.generated_features_1['Admin+Admin_duration'] = self.generated_features_1['Administrative'] + self.generated_features_1['Administrative_Duration']
        self.generated_features_1['Inform+Inform_duration'] = self.generated_features_1['Informational'] + self.generated_features_1['Informational_Duration']
        # self.generated_features_1['ProdR+ProdR_duration'] = self.generated_features_1['ProductRelated'] + self.generated_features_1['ProductRelated_Duration']
        drop_columns = ['ExitRates', 'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated_Duration']
        self.generated_features_1 = self.generated_features_1.drop(columns = drop_columns)


        # combine A+I+PR, Ad+Id+PRd, drop Exit 
        self.generated_features_2 = self.encoded_features.copy(deep = True)
        self.generated_labels_2 = self.encoded_labels.copy(deep = True)
        self.generated_features_2['A+I+PR'] = self.generated_features_2['Administrative'] + self.generated_features_2['Informational'] + self.generated_features_2['ProductRelated']
        self.generated_features_2['Ad+Id+PRd'] = self.generated_features_2['Administrative_Duration'] + self.generated_features_2['Informational_Duration'] + self.generated_features_2['ProductRelated_Duration']
        drop_columns = ['ExitRates', 'Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration']
        self.generated_features_2 = self.generated_features_2.drop(columns = drop_columns)


    def conduct_eda(self):
        
        # print(self.original_data.describe())
        # print(self.original_data.info())

        print('Plot(s) for categorical distributions')
        for colm in self.categorical_columns:
            cat = sns.catplot(x=colm, kind="count", data=self.original_data)
            cat.ax.set_axisbelow(True)
            plt.grid(axis = 'y')
            filename = 'categorical' + '_' + colm + '_catplot' + '.png'
            plt.savefig(self.output_path + filename, bbox_inches = 'tight', pad_inches = 0)

        print('Plot(s) for numerical distributions')
        # for colm in self.std_colms_encoded:
        #     fig, ax = plt.subplots()
        #     sns.distplot(self.original_data[colm], kde = False, bins = 10, ax = ax)
        #     filename = 'numerical' + '_' + colm + '_distplot' + '.png'
        #     plt.savefig(self.output_path + filename, bbox_inches = 'tight', pad_inches = 0)

        for colm in self.std_colms_encoded:
            fig, ax = plt.subplots()
            plt.hist(self.original_data[colm], bins = 10, color = 'skyblue')
            plt.grid(axis = 'x')
            plt.xlabel(colm)
            ax.set_axisbelow(True)
            filename = 'numerical' + '_' + colm + '_distplot' + '.png'
            plt.savefig(self.output_path + filename, bbox_inches = 'tight', pad_inches = 0)

        print('Plot(s) for numerical pairplots')
        pairplot_kind = ['scatter', 'reg']
        for kind in pairplot_kind:
            sns.pairplot(vars = self.std_colms_encoded, kind = kind, data = self.original_features)
            filename = 'numerical' + '_' + kind + '_pairplot' + '.png'
            plt.savefig(self.output_path + filename, bbox_inches = 'tight', pad_inches = 0)

        print('Plot(s) for numerical pairplots over Revenue')
        pairplot_kind = ['scatter', 'reg']
        for kind in pairplot_kind:
            sns.pairplot(vars = self.std_colms_encoded, hue = 'Revenue', kind = kind, data = self.original_data)
            filename = 'numerical' + '_' + kind + '_pairplot_over_Revenue' + '.png'
            plt.savefig(self.output_path + filename, bbox_inches = 'tight', pad_inches = 0)

        print('Plot(s) for categorical distributions over Revenue')
        for colm in self.categorical_columns:
            cat = sns.catplot(x=colm, y = 'Revenue', kind="point", data=self.original_data)
            cat.ax.set_axisbelow(True)
            plt.grid(axis = 'y')
            filename = 'categorical' + '_' + colm + '_catplot_over_Revenue' + '.png'
            plt.savefig(self.output_path + filename, bbox_inches = 'tight', pad_inches = 0)

        print('Plot(s) for correlations')
        encoded_data = self.original_data.copy(deep = True)
        encoded_data.Weekend = encoded_data.Weekend.map(self.weekend_dict_mapping)
        encoded_data.Revenue = encoded_data.Revenue.map(self.revenue_dict_mapping)
        encoded_data.Month = encoded_data.Month.map(self.month_dict_mapping)
        encoded_data.VisitorType = encoded_data.VisitorType.map(self.visitor_dict_mapping)

        fig, ax = plt.subplots(figsize=(12,12))
        sns.heatmap(data = encoded_data.corr(), cmap = 'RdYlGn', linewidths = 0.5,  vmin = -1, vmax = 1, annot = True, square = True, fmt=".2f")
        filename = 'correlations' + '_heatmap' + '.png'
        plt.savefig(self.output_path + filename, bbox_inches = 'tight', pad_inches = 0)

        print('Plots for PageValues, BounceRates over Revenue')
        num_vars = ['PageValues', 'BounceRates']
        for var in num_vars:
            plt.figure()
            sns.distplot(a=self.original_data[var][self.original_data.Revenue == False], label="Revenue False", kde=True, hist = False)
            sns.distplot(a=self.original_data[var][self.original_data.Revenue == True], label="Revenue True", kde=True, hist = False)
            plt.legend()
            filename = 'numerical' + '_' + var + '_over_Revenue' + '.png'
            plt.savefig(self.output_path + filename, bbox_inches = 'tight', pad_inches = 0)


    def check_different_test_train_splits(self):
        
        features = self.encoded_features
        labels = self.encoded_labels
        
        for train_ratio in np.arange(0.5, 1, 0.1):
            
            sss = StratifiedShuffleSplit(n_splits = 1, train_size = train_ratio)
            
            for train_indx, test_indx in sss.split(features, labels):
                # print(len(train_indx)/len(features), len(test_indx)/len(features))
                # print('% Survived:', labels[test_indx].mean())
                # print(train_indx, test_indx)
                clf = RandomForestClassifier(random_state = self.random_state, n_jobs = self.n_jobs)
                clf.fit(features.iloc[train_indx], labels.iloc[train_indx])
                # print('____Train____')
                # pred_proba = clf.predict_proba(features.iloc[train_indx])
                # print('AUC for {0} train_ratio:'.format(train_ratio), roc_auc_score(labels.iloc[train_indx], pred_proba[:,-1]))
                # pred_labels = clf.predict(features.iloc[train_indx])
                # print('Classification report score for {0} train_ratio:\n'.format(train_ratio), classification_report(labels.iloc[train_indx], pred_labels))
                # print('____Test____')
                pred_proba = clf.predict_proba(features.iloc[test_indx])
                print('AUC for {0} train_ratio:'.format(train_ratio), roc_auc_score(labels.iloc[test_indx], pred_proba[:,-1]))
                # pred_labels = clf.predict(features.iloc[test_indx])
                # print('Classification report score for {0} train_ratio:\n'.format(train_ratio), classification_report(labels.iloc[test_indx], pred_labels))


    def check_level_v_onehot_encoding(self):
        
        level_features = self.encoded_features
        level_labels = self.encoded_labels
        
        onehot_features = self.original_features
        onehot_labels = self.original_labels

        onehot_encoder = OneHotEncoder(handle_unknown = 'error', sparse = False)
        onehot_encoder.fit(onehot_features[self.onehot_colms])
        onehot_transformed_colms = onehot_encoder.get_feature_names(self.onehot_colms)
        onehot_transformed_features = onehot_encoder.transform(onehot_features[self.onehot_colms])
        onehot_features = onehot_features.join( pd.DataFrame(onehot_transformed_features, index = onehot_features.index, columns = onehot_transformed_colms), how = 'inner')
        # print(onehot_features.info())
        # print(onehot_transformed_colms)
        onehot_features = onehot_features.drop(columns = self.onehot_colms)
        # print(onehot_features.info())
        # print(self.original_features.loc[0:5,'Region'])
        # print(onehot_features.loc[0:5, ['Region_1', 'Region_2', 'Region_3', 'Region_4', 'Region_5', 'Region_6', 'Region_7', 'Region_8', 'Region_9'] ] )

        train_ratio = 0.6
        sss = StratifiedShuffleSplit(n_splits = 1, train_size = train_ratio, random_state = self.random_state)
        for train_indx, test_indx in sss.split(level_features, level_labels):
            # print(len(train_indx)/len(features), len(test_indx)/len(features))
            # print('% Survived:', labels[test_indx].mean())
            # print(train_indx, test_indx)
            
            # performance for level encoding
            clf = RandomForestClassifier(random_state = self.random_state, n_jobs = self.n_jobs)
            # clf = LogisticRegression()
            clf.fit(level_features.iloc[train_indx], level_labels.iloc[train_indx])
            pred_proba = clf.predict_proba(level_features.iloc[test_indx])
            print('AUC for {0} train_ratio:'.format(train_ratio), roc_auc_score(level_labels.iloc[test_indx], pred_proba[:,-1]))
            pred_labels = clf.predict(level_features.iloc[test_indx])
            print('Classification report score for {0} train_ratio:\n'.format(train_ratio), classification_report(level_labels.iloc[test_indx], pred_labels))

            # performance for onehot encoding
            clf = RandomForestClassifier(random_state = self.random_state, n_jobs = self.n_jobs)
            # clf = LogisticRegression()
            clf.fit(onehot_features.iloc[train_indx], onehot_labels.iloc[train_indx])
            pred_proba = clf.predict_proba(onehot_features.iloc[test_indx])
            print('AUC for {0} train_ratio:'.format(train_ratio), roc_auc_score(onehot_labels.iloc[test_indx], pred_proba[:,-1]))
            pred_labels = clf.predict(onehot_features.iloc[test_indx])
            print('Classification report score for {0} train_ratio:\n'.format(train_ratio), classification_report(onehot_labels.iloc[test_indx], pred_labels))


    def feature_selection(self):
        
        onehot_features = self.original_features
        onehot_labels = self.original_labels

        onehot_encoder = OneHotEncoder(handle_unknown = 'error', sparse = False)
        onehot_encoder.fit(onehot_features[self.onehot_colms])
        onehot_transformed_colms = onehot_encoder.get_feature_names(self.onehot_colms)
        onehot_transformed_features = onehot_encoder.transform(onehot_features[self.onehot_colms])
        onehot_features = onehot_features.join( pd.DataFrame(onehot_transformed_features, index = onehot_features.index, columns = onehot_transformed_colms), how = 'inner')
        # print(onehot_features.info())
        # print(onehot_transformed_colms)
        onehot_features = onehot_features.drop(columns = self.onehot_colms)
        # print(onehot_features.info())
        # print(self.original_features.loc[0:5,'Region'])
        # print(onehot_features.loc[0:5, ['Region_1', 'Region_2', 'Region_3', 'Region_4', 'Region_5', 'Region_6', 'Region_7', 'Region_8', 'Region_9'] ] )

        sss = StratifiedShuffleSplit(n_splits = 1, train_size = self.train_ratio, random_state = self.random_state)
        for train_indx, test_indx in sss.split(onehot_features, onehot_labels):
            # print(len(train_indx)/len(features), len(test_indx)/len(features))
            # print('% Survived:', labels[test_indx].mean())
            
            # Using RandomForestClassifier gives non-linear decision boundary
            clf = RandomForestClassifier(random_state = self.random_state, n_jobs = self.n_jobs)
            
            # Using LogisticRegression (default L1) gives linear decision boundary
            # clf = LogisticRegression()
            
            clf.fit(onehot_features.iloc[train_indx], onehot_labels.iloc[train_indx])
            
            # Using mean threshold in SelectFromModel
            feature_selection_model = SelectFromModel(clf, prefit = True, threshold = 'mean')
            selected_features = feature_selection_model.transform(onehot_features.iloc[train_indx])
            selected_features = pd.DataFrame(feature_selection_model.inverse_transform(selected_features), index = onehot_features.iloc[train_indx].index, columns = onehot_features.iloc[train_indx].columns)
            self.selected_columns_mean = selected_features.columns[selected_features.var() != 0]
            print('Mean threshold:', self.selected_columns_mean)

            # Using Median threshold for SelectFromModel
            feature_selection_model = SelectFromModel(clf, prefit = True, threshold = 'median')
            selected_features = feature_selection_model.transform(onehot_features.iloc[train_indx])
            selected_features = pd.DataFrame(feature_selection_model.inverse_transform(selected_features), index = onehot_features.iloc[train_indx].index, columns = onehot_features.iloc[train_indx].columns)
            self.selected_columns_median = selected_features.columns[selected_features.var() != 0]
            print('Median threshold', self.selected_columns_median)


    def return_train_val_test_sets(self, features, labels):
        
        sss_train_val_test = StratifiedShuffleSplit(n_splits = 1, train_size = self.train_ratio, random_state = self.random_state)
        for train_indx, val_test_indx in sss_train_val_test.split(features, labels):

            train_features = features.iloc[train_indx]
            train_labels = labels.iloc[train_indx]
            val_test_features = features.iloc[val_test_indx]
            val_test_labels = labels.iloc[val_test_indx]

            # print(len(train_features)/len(features), len(val_test_features)/len(features))
            # print('% Survived:', train_labels.mean(), val_test_labels.mean())

            sss_val_test = StratifiedShuffleSplit(n_splits = 1, test_size = 0.5, random_state = self.random_state)
            for val_indx, test_indx in sss_val_test.split(val_test_features, val_test_labels):

                val_features = val_test_features.iloc[val_indx]
                val_labels = val_test_labels.iloc[val_indx]
                test_features = val_test_features.iloc[test_indx]
                test_labels = val_test_labels.iloc[test_indx]

                # print(len(val_features)/len(features), len(test_features)/len(features))
                # print('% Survived:', val_labels.mean(), test_labels.mean())

        return train_features, train_labels, val_features, val_labels, test_features, test_labels


    def save_confusion_matrix(self, estimator_name, fitted_estimator, features, labels):
        normalizations = ['all', 'true', 'pred']
        for normalization in normalizations:
            disp = plot_confusion_matrix(fitted_estimator, features, labels, cmap = plt.cm.Blues, normalize = normalization)
            filename = estimator_name + '_confusion_matrix_norm_' + normalization + '.png'
            plt.savefig(self.output_path + filename, bbox_inches = 'tight', pad_inches = 0)
    

    def plot_save_feature_importance(self, estimator_name, best_estimator):
        feature_importance = pd.DataFrame(data = {'feature_name':self.selected_columns, 'feature_importance':best_estimator.feature_importances_})
        feature_importance = feature_importance.sort_values(by = 'feature_importance', ascending=False)
        fig, ax = plt.subplots()
        ax.grid(axis = 'x')
        ax.set_axisbelow(True)
        sns.barplot(y = 'feature_name', x = 'feature_importance', data = feature_importance)
        filename = estimator_name + '_feature_importance_' + '.png'
        plt.savefig(self.output_path + filename, bbox_inches = 'tight', pad_inches = 0)


    def plot_save_feature_weights(self, estimator_name, best_estimator):

        feature_name = self.selected_columns.to_list()
        feature_name.append('Intercept')
        feature_weight = best_estimator.coef_[0].tolist()
        feature_weight.append(best_estimator.intercept_[0])
        feature_weight = pd.DataFrame(data = {'feature_name': feature_name, 'feature_weight': feature_weight})
        feature_weight = feature_weight.sort_values(by = 'feature_weight', ascending=False)
        fig, ax = plt.subplots()
        ax.grid(axis = 'x')
        ax.set_axisbelow(True)
        # plt.grid(axis = 'x')
        sns.barplot(y = 'feature_name', x = 'feature_weight', data = feature_weight)
        filename = estimator_name + '_feature_weight_' + '.png'
        plt.savefig(self.output_path + filename, bbox_inches = 'tight', pad_inches = 0)


    def run_grid_pipeline(self, features, labels, standardization_colms, parameters, estimator, feature_selection_threshold_type):
        
        # Preprocessing for numerical data
        numerical_transformer = StandardScaler()

        # Preprocessing for categorical data
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Bundle preprocessing for numerical and categorical data
        preprocessor = ColumnTransformer(
            transformers = [
                ('num', numerical_transformer, standardization_colms),
                # ('cat', categorical_transformer, self.onehot_colms)
            # ], n_jobs = self.n_jobs)
            ], n_jobs = self.n_jobs, remainder = 'passthrough')

        feature_selection_clf = RandomForestClassifier(random_state = self.random_state, n_jobs = self.n_jobs)
        feature_selection_model = SelectFromModel(feature_selection_clf, threshold = feature_selection_threshold_type)

        grid = GridSearchCV(estimator = estimator, param_grid = parameters, cv = 5, scoring = 'accuracy', refit = True, n_jobs = -1)

        pipeline = Pipeline(
            steps = [
                ('preprocessor', preprocessor),
                ('feature_selection', feature_selection_model),
                ('grid_search', grid)
            ])
        
        pipeline.fit(features, labels)

        def print_results(results):
            print('BEST PARAMS: {}\n'.format(results.best_params_))

            means = results.cv_results_['mean_test_score']
            stds = results.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, results.cv_results_['params']):
                print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), params))

        print_results(pipeline['grid_search'])

        # print(features.columns)
        feature_selection_model = pipeline['feature_selection']
        selected_features = feature_selection_model.transform(features)
        selected_features = pd.DataFrame(feature_selection_model.inverse_transform(selected_features), index = features.index, columns = features.columns)
        self.selected_columns = selected_features.columns[selected_features.var() != 0]
        print('\nColumns selected for {0} threshold'.format(feature_selection_threshold_type), self.selected_columns)

        # print('\nBest estimator:\n')
        # print(pipeline['grid_search'].best_estimator_)
        # print(pipeline['grid_search'].best_score_)
        # print(pipeline['grid_search'].best_params_)
        # print(pipeline['grid_search'].scorer_)

        return pipeline


    def run_train_val_test(self):
        
        # features = self.original_features
        # labels = self.original_labels
        # features = self.encoded_features
        # labels = self.encoded_labels
        # features = self.generated_features_1
        # labels = self.gen_labels_1
        features = self.generated_features_2
        labels = self.generated_labels_2

        if features.shape[1] == self.encoded_features.shape[1]:
            standardization_colms = self.std_colms_encoded
        elif features.shape[1] == self.generated_features_1.shape[1]:
            standardization_colms = self.std_colms_generated_1
        elif features.shape[1] == self.generated_features_2.shape[1]:
            standardization_colms = self.std_colms_generated_2

        train_features, train_labels, val_features, val_labels, test_features, test_labels = self.return_train_val_test_sets(features, labels)

        # print(len(train_features)/len(features), len(val_features)/len(features), len(test_features)/len(features))
        # print('% Survived:', train_labels.mean(), val_labels.mean(), test_labels.mean(), '\n')
        # print(features.info())


        print('___RandomForestClassifier___\n')
        feature_selection_threshold_type = 'median'
        parameters = {
            'class_weight': [None, 'balanced', 'balanced_subsample'],
            'n_estimators': [5, 50, 75, 100],
            'max_depth': [2, 10, 20, None]
        }
        estimator = RandomForestClassifier(random_state = self.random_state, n_jobs = self.n_jobs)
        rfc_pipe = self.run_grid_pipeline(train_features, train_labels, standardization_colms, parameters, estimator, feature_selection_threshold_type)
        
        print('\nFeature Selection threshold type:', feature_selection_threshold_type)
        pred_proba = rfc_pipe.predict_proba(val_features)
        print('\nROC AUC - Validation Set:', roc_auc_score(val_labels, pred_proba[:,-1]))
        pred_labels = rfc_pipe.predict(val_features)
        print('\nClassification report - Validation Set:\n', classification_report(val_labels, pred_labels))

        print('\nBest estimator:\n')
        rfc_best_estimator = rfc_pipe['grid_search'].best_estimator_
        print('# of features:', rfc_best_estimator.n_features_)
        print('Features Importance:', rfc_best_estimator.feature_importances_)

        self.plot_save_feature_importance('rfc', rfc_best_estimator)
        self.save_confusion_matrix('rfc', rfc_pipe, val_features, val_labels)

        pred_proba = rfc_pipe.predict_proba(test_features)
        print('\nROC AUC - Test Set:', roc_auc_score(test_labels, pred_proba[:,-1]))
        pred_labels = rfc_pipe.predict(test_features)
        print('\nClassification report - Test Set:\n', classification_report(test_labels, pred_labels))

        self.save_confusion_matrix('rfc_test', rfc_pipe, test_features, test_labels)


        print('\n___LogisticRegression___\n')
        feature_selection_threshold_type = 'median'
        parameters = {
            'class_weight': [None, 'balanced'],
            'C': np.arange(0.1,1.05,0.05),
        }
        estimator = LogisticRegression(random_state = self.random_state, n_jobs = self.n_jobs)
        lr_pipe = self.run_grid_pipeline(train_features, train_labels, standardization_colms, parameters, estimator, feature_selection_threshold_type)
        
        print('\nFeature Selection threshold type:', feature_selection_threshold_type)
        pred_proba = lr_pipe.predict_proba(val_features)
        print('\nROC AUC - Validation Set:', roc_auc_score(val_labels, pred_proba[:,-1]))
        pred_labels = lr_pipe.predict(val_features)
        print('\nClassification report - Validation Set:\n', classification_report(val_labels, pred_labels))

        print('\nBest estimator:\n')
        lr_best_estimator = lr_pipe['grid_search'].best_estimator_
        print('Coefficients:', lr_best_estimator.coef_)
        print('Intercept:', lr_best_estimator.intercept_)

        self.plot_save_feature_weights('lr', lr_best_estimator)
        self.save_confusion_matrix('lr', lr_pipe, val_features, val_labels)


        print('\n___KNeighborsClassifier___\n')
        feature_selection_threshold_type = 'mean'
        parameters = {
            'weights': ['uniform', 'distance'],
            'p': np.arange(1,3,1),
            'n_neighbors': np.arange(1,20,1)
        }
        estimator = KNeighborsClassifier(n_jobs = self.n_jobs)
        knn_pipe = self.run_grid_pipeline(train_features, train_labels, standardization_colms, parameters, estimator, feature_selection_threshold_type)
        
        
        print('\nFeature Selection threshold type:', feature_selection_threshold_type)
        pred_proba = knn_pipe.predict_proba(val_features)
        print('\nROC AUC - Validation Set:', roc_auc_score(val_labels, pred_proba[:,-1]))
        pred_labels = knn_pipe.predict(val_features)
        print('\nClassification report - Validation Set:\n', classification_report(val_labels, pred_labels))

        self.save_confusion_matrix('knn', knn_pipe, val_features, val_labels)


    def start_execution(self):

        self.read_data()
        self.conduct_eda()
        # self.check_different_test_train_splits()
        # self.check_level_v_onehot_encoding()
        # self.feature_selection()
        # self.return_train_val_test_sets(self.encoded_features, self.encoded_labels)
        self.run_train_val_test()

        return




if __name__ == "__main__":

    print('ERROR: Execute py main.py to start the execution of the program! ')