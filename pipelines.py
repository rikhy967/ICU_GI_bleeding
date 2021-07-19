from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV,calibration_curve
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.metrics import plot_roc_curve,plot_confusion_matrix
from sklearn.inspection import plot_partial_dependence
from sklearn.inspection import permutation_importance
import sklearn.metrics as metrics

import scipy.stats as stats

import numpy as np

import warnings

from hyperopt import  Trials,fmin,tpe,hp

from costum_utils import performance_visualizer

import matplotlib.pyplot as plt

import pickle


##########################################################################################################################
############################################ Optimization pipeline #######################################################
##########################################################################################################################

def optim_pipeline(X_train, y_train, space,
            calibration_method='sigmoid',
            kfold=6,max_evals=100,
            random_state=None,accuracy_weight=0.3):


    def objective(params):
        ######################## hyper params ##############################

        params_RF = {}
        params_LR = {}

        params_RF['max_samples'] = params['max_samples']
        params_RF['max_features'] = params['max_features']
        params_RF['criterion'] = params['criterion']
        params_RF['max_depth'] = params['max_depth']
        params_RF['ccp_alpha'] = params['ccp_alpha']
        params_RF['bootstrap'] = params['bootstrap']
        params_RF['min_samples_leaf'] = params['min_samples_leaf']
        params_RF['min_samples_split'] = params['min_samples_split']

        params_LR['C'] = params['C']
        params_LR['penalty'] = params['penalty']
        params_LR['l1_ratio'] = params['l1_ratio']

        treshold = params['treshold']

        ########################## Classifier definition ###################

        classifier1 = RandomForestClassifier(class_weight='balanced',n_estimators=150, n_jobs=-1)
        classifier1.set_params(**params_RF)

        classifier2 = LogisticRegression(class_weight='balanced', max_iter=10000, solver='saga',
                                         random_state=random_state, n_jobs=-1)
        classifier2.set_params(**params_LR)

        estimators = [('rf', classifier1),
                      ('lr', classifier2)]

        classifier = CalibratedClassifierCV(VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1),
                                            method=calibration_method)

        ######################### Model testing ################################

        shuffle = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=random_state)

        def treshold_scorer(estimator, X, y, ):
            pred_proba = estimator.predict_proba(X)[:, 1]
            pred = np.array([pred_proba > treshold]).astype(np.int).ravel()
            
            f1_score = metrics.f1_score(y, pred)
            accuracy_score = metrics.accuracy_score(y,pred)
            
            
            score = f1_score*(1-accuracy_weight) + accuracy_score*accuracy_weight
            
            return score

        score = cross_val_score(classifier, X_train, y_train, cv=shuffle, scoring=treshold_scorer, n_jobs=-1)
        score = np.mean(score)
        
        

        return 1 - score

    ######################## Start optim ####################################

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

    # The Trials object will store details of each iteration
    trials = Trials()

    # Run the hyperparameter search using the tpe algorithm
    best = fmin(objective,
                space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    return best, trials

##########################################################################################################################
############################################ Costum CalibratedCV ########################################################
##########################################################################################################################


class modified_CalibratedClassifierCV(CalibratedClassifierCV):
    
        def __init__(self,base_estimator=None, method='sigmoid', cv=None,treshold=0.5):

        #inhertis from SparseGraph_wutils
            super().__init__(base_estimator, method, cv)
            
            self.treshold = treshold
        
        
        def predict(self,X):
        
            
            return np.array([self.predict_proba(X)[:,1] > self.treshold]).astype(np.int).ravel()

        
##########################################################################################################################
#################################################### Costum Voting #######################################################
##########################################################################################################################
        
        
class modified_VotingClassifier(VotingClassifier):
    
    def __init__(self,estimators, voting='hard', weights=None,
                 n_jobs=None):
        
        super().__init__(estimators, voting, weights,n_jobs)
        
        
    def _predict(self, X):
        return np.asarray([est.predict(X) for est in self.estimators_]).T
        
    
    @property
    def predict_proba(self):
        
        return self._predict_proba


##########################################################################################################################
#################################################### Best model #########################################################
##########################################################################################################################
        

class BestModel_with_bagging(object):

    def __init__(self, trials,names, train_label,
                 bag_k_best=2, calibration_method = 'sigmoid',
                 dump_param = False,dump_path=r"05-24_eICU",
                 load_dump = False, load_path=""):

        self.names = names
        self.importances=None
        self.train_label = train_label
        
        self.classifier_list = []
      

        cat = {}
        cat['criterion'] = ['gini', 'entropy']
        cat['bootstrap'] = [False, True]
        cat['penalty'] = ['l2', 'l1', 'elasticnet']
        
        best_models = performance_visualizer(trials, bag_k_best, choice=True, **cat)

        for i in range(bag_k_best):
           
            best_dict = best_models.to_dict(orient='records')[-i]

            params_RF = {}
            params_LR = {}

            params = best_dict.copy()

       
            params_RF['max_samples'] = params['max_samples']
            params_RF['max_features'] = params['max_features']
            params_RF['criterion'] = params['criterion']
            params_RF['max_depth'] = int(params['max_depth'])
            params_RF['ccp_alpha'] = params['ccp_alpha']
            params_RF['bootstrap'] = params['bootstrap']
            params_RF['min_samples_leaf'] = int(params['min_samples_leaf'])
            params_RF['min_samples_split'] = int(params['min_samples_split'])

            params_LR['C'] = params['C']
            params_LR['penalty'] = params['penalty']
            params_LR['l1_ratio'] =params['l1_ratio']

            classifier1 = RandomForestClassifier(class_weight='balanced',n_estimators=150,n_jobs=-1)
            classifier1.set_params(**params_RF)

            classifier2 = LogisticRegression(class_weight='balanced',solver='saga')
            classifier2.set_params(**params_LR)

            estimators = [('rf', classifier1),
                          ('bg_lr', classifier2)]


            cv = StratifiedKFold()
            
            clf = modified_CalibratedClassifierCV(VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1), cv=cv,method=calibration_method,treshold = params['treshold'])
            
            name = 'clf'+str(i)
            
            self.classifier_list.append((name,clf))
            
            
        self.classifier = modified_VotingClassifier(estimators = self.classifier_list,voting='hard',n_jobs=-1)


    def fit(self,X,y):

        self.classifier.fit(X,y)

    def predict(self,X):
        return self.classifier.predict(X)

    def predict_proba(self,X):
        return self.classifier.predict_proba(X)

    def print_metrics(self,X_test_mimic,X_test_eICU,
                      y_test_mimic,y_test_eICU):

        y_pred_proba_test_mimic = self.predict_proba(X_test_mimic)
        y_pred_proba_test_eicu = self.predict_proba(X_test_eICU)
        y_pred_test_mimic = self.predict(X_test_mimic)
        y_pred_test_eicu = self.predict(X_test_eICU)

        ######## PRINT ACCURACY ###########

        print('Accuracy Test on MIMIC: ', metrics.accuracy_score(y_test_mimic, y_pred_test_mimic))
        print('Accuracy Test on eICU: ', metrics.accuracy_score(y_test_eICU, y_pred_test_eicu))
        print('')

        # print ('Accuracy Test on MIMIC: ',classifier.score(X_test_mimic,y_test_mimic))
        # print ('Accuracy Test on eICU: ',classifier.score(X_test_eICU,y_test_eICU))
        # print('')

        ####### PRINT RECALL (SENSITIVITY) ##########

        print('Recall Test on MIMIC: ', metrics.recall_score(y_test_mimic, y_pred_test_mimic))
        print('Recall Test on eICU: ', metrics.recall_score(y_test_eICU, y_pred_test_eicu))
        print('')

        ######## SPECIFICITY ###########

        print('Specificity Test on MIMIC: ', metrics.recall_score(y_test_mimic, y_pred_test_mimic, pos_label=0))
        print('Specificity Test on eICU: ', metrics.recall_score(y_test_eICU, y_pred_test_eicu, pos_label=0))
        print('')

        ######## PRINT ROC_AUC ###########

        fpr, tpr, thresholds = metrics.roc_curve(y_test_mimic, y_pred_proba_test_mimic[:, 1])
        roc_auc = metrics.auc(fpr, tpr)
        print('ROC AUC Test on MIMIC: ', roc_auc)

        fpr, tpr, thresholds = metrics.roc_curve(y_test_eICU, y_pred_proba_test_eicu[:, 1])
        roc_auc = metrics.auc(fpr, tpr)
        print('ROC AUC Test on eICU: ', roc_auc)

        print('')
        
        name = self.train_label+ ' (train)'

        eICU_roc = plot_roc_curve(self.classifier, X_test_eICU, y_test_eICU, name=name)
        ax = plt.gca()
        MIMIC_roc = plot_roc_curve(self.classifier, X_test_mimic, y_test_mimic, ax=ax, alpha=0.8, name='MIMIC')
        fig = plt.gcf()
        fig.set_figheight(5)
        fig.set_figwidth(5)
        fig.set_dpi(150)
        
        name = 'ROC curves (trained on '+self.train_label+')'
        
        fig.suptitle(name, fontsize=10)


    def print_feature_importance(self,X_train,y_train,
                                 n_features=5,figsize=(6,6)):

        self.importances = permutation_importance(self.classifier, X_train, y_train, n_jobs=-1, n_repeats=100)
        self.importances = self.importances['importances_mean']
        self.indices = np.argsort(self.importances)
        
        name = 'Feature Importance (' + self.train_label + ')'
        
        plt.figure(figsize=figsize, dpi=150)
        plt.title(name)
        plt.barh(range(n_features), self.importances[self.indices][-n_features:], color='b', align='center')
        plt.yticks(range(n_features), np.array(self.names)[self.indices][-n_features:])
        plt.xlabel('Relative Importance')
        


    def print_partial_dependence(self,X_train,
                                 fig_height=6,fig_width=6,
                                 xlim=(10,40),ylim=(0,1)):


        plot_partial_dependence(self.classifier, X_train, features=[self.indices[-1]], feature_names=self.names)
        fig = plt.gcf()
        fig.set_figheight(fig_height)
        fig.set_figwidth(fig_width)
        fig.set_dpi(150)
        ax = fig.get_axes()[-1]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
        name = 'Partial dependence (' + self.train_label + ')'
        
        fig.suptitle(name, fontsize=20)
        fig.subplots_adjust(hspace=0.3)


    def print_calibration_plot(self,X_test_mimic,y_test_mimic,X_test_eICU,y_test_eICU):

        y_pred_proba_test_mimic=self.predict_proba(X_test_mimic)[:, 1]
        y_pred_proba_test_eICU = self.predict_proba(X_test_eICU)[:, 1]
        
        fraction_of__positives_mimic, mean_predicted_values_mimic = calibration_curve(y_test_mimic.values.ravel(),
                                                                                      y_pred_proba_test_mimic)
        fraction_of__positives_eICU, mean_predicted_values_eICU = calibration_curve(y_test_eICU.values.ravel(),
                                                                                    y_pred_proba_test_eICU)

        fig, (ax1, ax2) = plt.subplots(2, figsize=(5, 10), dpi=150)
        ax1.plot(mean_predicted_values_mimic, fraction_of__positives_mimic, label='MIMIC')
        ax1.plot(mean_predicted_values_eICU, fraction_of__positives_eICU, label='eICU')
        ax1.plot([0, 1], [0, 1], lw=1, color='black', linestyle='dashed')
        ax1.set_xlim((0, 1))
        
        name = 'Calibration plots (Train on '+self.train_label+')'
        
        ax1.title.set_text(name)
        ax1.legend(loc='lower right')
        ax1.set(ylabel='fraction of positives')

        ax2.hist(y_pred_proba_test_mimic, bins=10, histtype='step', lw=2, label='MIMIC')
        ax2.hist(y_pred_proba_test_eICU, bins=10, histtype='step', lw=2, label='eICU')
        ax2.set_xlim((0, 1))
        ax2.legend(loc="upper center")
        ax2.set(xlabel='Mean predicted value', ylabel='Count')
        
    
    def print_confusion_matrix(self,X,y,width=8,height=8):
        
        
        plot_confusion_matrix(self.classifier,
                              X,y,
                              values_format='d',
                              cmap='Blues')
        
        name = " Confusion matrix (Train on "+self.train_label+")" 
        
        fig = plt.gcf()
        fig.suptitle(name, fontsize=20)
        fig.set_figheight(height)
        fig.set_figwidth(width)
        fig.set_dpi(150)