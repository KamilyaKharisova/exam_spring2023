import numpy as np
class LinearRegression():

    def __init__(self, base_functions: list, learning_rate: float, reg_coefficient: float):
        self.weights = np.random.randn(len(base_functions)+1)
        self.base_functions = base_functions
        self.learning_rate = learning_rate
        self.reg_coefficient = reg_coefficient

    # Methods related to the Normal Equation

    def _pseudoinverse_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Compute the pseudoinverse of a matrix  using SVD with regularization.
        """
        pass

    def _calculate_weights(self, pseudoinverse_plan_matrix: np.ndarray, targets: np.ndarray) -> None:
        """Calculate the optimal weights using the normal equation.
        Calculate  Φ^+ using _pseudoinverse_matrix function
        """
        pass

    # General methods
    def _plan_matrix(self, inputs: np.ndarray) -> np.ndarray:
        """Construct the design matrix (Φ) using base functions.
         Implement this method using one loop over the base functions.

        """
        pass

    def calculate_model_prediction(self, plan_matrix: np.ndarray) -> np.ndarray:
        """Calculate the predictions of the model.
        Implement this method without using loop

        """
        pass

    def train(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        """
        """
        plan_matrix = self._plan_matrix(inputs)
        pseudoinverse_plan_matrix = self._pseudoinverse_matrix(plan_matrix)
        # train process
        self._calculate_weights(pseudoinverse_plan_matrix, targets)


    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """return prediction of the model"""
        plan_matrix = self._plan_matrix(inputs)
        predictions = self.calculate_model_prediction(plan_matrix)

        return predictions
