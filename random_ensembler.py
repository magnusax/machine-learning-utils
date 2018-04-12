import numpy as np
from sklearn.exceptions import NotFittedError


class RandomEnsembler():
    
    def __init__(self, estimator_class, fitted_params, n_ensembles=5):
        """
        Convenient way to 'average' predictions from multiple instances of a single classifier, using the
        best parameters found from some type of hyperparameter search.
        
        Parameters:
        -----------
            estimator_class : 
                a class of scikit-learn estimator Note: it should not be instanced; 
                e.g. 'RandomForestClassifier' NOT 'RandomForestClassifier()'
            fitted_params : 
                dictionary, a set of (preferably optimized) parameters to be fed to `estimator`
            n_ensembles : 
                integer, controls number of instances to ensemble over (default: 5)
        """
        # Check that algorithm implements a random seed
        assert hasattr(estimator_class(), 'random_state')
        self.estimator_class = estimator_class
        # Sanity check on input parameters
        assert isinstance(fitted_params, dict)
        self.fitted_params = fitted_params
        self.n_ensembles = int(n_ensembles)
        # Variable checking the status of fitting
        self.is_fitted = False
        # The individual models are stored here
        self.models = list()
    
    def get_models(self):
        """
        Returns the individual fitted models if `fit` has 
        been called. Otherwise return `None`.
        """
        if self.is_fitted:
            return self.models
        else:
            return None
    
    def fit(self, X_train, y_train, **kwargs):
        """
        Parameters:
        -----------
        X_train : 
            array-like, shape (n_samples_train, n_features). Training data
        y_train : 
            array-like, shape (n_samples_train,). Training labels
        **kwargs : 
            other parameters to input to `fit` method if applicable
        """
        # Generate and train new model for each seed
        for _ in range(self.n_ensembles):
            model = self.estimator_class(
                random_state=np.random.randint(low=0, high=1e9))
            model.set_params(**self.fitted_params)
            model.fit(X_train, y_train, **kwargs)
            self.models.append(model)
            del model    
        self.is_fitted = True
        return self
    
    def predict(self, X_test):
        """ 
        Parameters:
        -----------
        X_test : 
            matrix-like, shape (n_samples_test, n_features). Test data to predict on.
            
        Returns:
        -----------
        ensembled_preds : 
            numpy-array, shape (n_samples_test,). Majority ensembled predictions 
            which usually beat the single best classifier predictions trained on 
            the same data.
        """    
        if not self.is_fitted:
            raise NotFittedError('Call `fit` first.')
            
        preds = [m.predict(X_test) for m in self.models]
        ensembled_preds = []
        for jj in range(X_test.shape[0]):
            ensembled_preds.append(
                self._majority_vote([preds[ii][jj] 
                for ii in range(self.n_ensembles)]))
        return np.array(ensembled_preds)
    
    def predict_proba(self, X_test):
        """ 
        Parameters:
        -----------
        X_test : 
            matrix-like, shape (n_samples_test, n_features). Test data to predict on.
            
        Returns:
        -----------
        ensembled_preds : 
            numpy-array, shape (n_samples_test,). Numerically averaged probability 
            predictions which usually beat the single best classifier predictions 
            trained on the same data.
        """  
        if not self.is_fitted:
            raise NotFittedError('Call `fit` first.')
            
        if not hasattr(self.estimator_class(), 'predict_proba'):
            raise TypeError('Estimator should implement `predict_proba`.')
        
        preds = [m.predict_proba(X_test) for m in self.models]
        ensembled_preds = []
        for jj in range(X_test.shape[0]):
            ensembled_preds.append(
                self._average_prob([preds[ii][jj] 
                for ii in range(self.n_ensembles)]))
        return np.array(ensembled_preds)
    
    def _majority_vote(self, lst):
        return max(set(lst), key=lst.count)
        
    def _average_prob(self, lst):
        return np.array(lst).mean()