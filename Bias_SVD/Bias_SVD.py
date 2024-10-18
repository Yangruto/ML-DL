import pandas as pd
import tensorflow as tf

# parameters
EMBEDDING_SIZE = 100
LEARNING_RATE = 0.001 # must be float
TRAINING_STEP = 3500
REGULARIZATION_RATE = 0.1
WEIGHTS_FOLDER = './weights/'
optimizer = tf.optimizers.Adam(LEARNING_RATE)

def combine_purchase_register(purchase_path:str, register_path:str) -> pd.DataFrame:
    """
    Combine the purchased data and registered data. If customers have both purchased data and registered data, 
    product score take the maximum; recency take the minimum.
        purchase_path: the customer purchase data path 
        register_path: the customer register data path
    """
    purchase_data = pd.read_csv(purchase_path) 
    register_data = pd.read_csv(register_path)
    all_data = pd.concat([purchase_data, register_data])
    maximum_score = all_data.groupby(['email']).max().reset_index()
    minimum_recency = all_data.groupby(['email']).recency.min().reset_index()
    final_data = maximum_score.drop(columns=['recency'])
    final_data = final_data.merge(minimum_recency, how='left', on=['email'])
    return final_data

class Bias_SVD_model:
    """
    Build bias SVD model.
    Data must be user product matrix with 'member_id' and 'email' columns.
    """
    def __init__(self, data_path:str):
        print(f'embedding_size : {EMBEDDING_SIZE}, learning_rate: {LEARNING_RATE}, TRAINING_STEP: {TRAINING_STEP}')
        self.user_product_matrix = pd.read_csv(data_path, index_col='email').drop(columns=['recency'])
        self.user_product_index = self.user_product_matrix.index
        self.user_product_column = self.user_product_matrix.columns
        self.user_number, self.product_number = self.user_product_matrix.shape

    def tranform2tensor(self) -> None:
        """
        Transform user_product_matrix from pd.DataFrame to tf.tensor
        """
        self.user_product_matrix = tf.convert_to_tensor(self.user_product_matrix)
        self.user_product_matrix = tf.cast(self.user_product_matrix, tf.float32)
    
    def create_random_variable(self, dimension:list) -> tf.Variable:
        """
        Create tensorflow random variable for model training
            dimension: the dimension of tensor, e.g.[20, 20]
        """
        random_variable = tf.Variable(tf.random.normal(dimension, mean=0, stddev=1))
        return random_variable 

    def create_constant_variable(self, dimension:list) -> tf.Variable:
        """
        Create tensorflow constant variable for model training
        * It is not good to set weights randomly because the same users would convergence to different results
            dimension: the dimension of tensor, e.g.[2, 2]
        """
        constant_variable = tf.Variable(tf.ones(dimension, tf.float32))
        return constant_variable

    def initialize_variable(self, embedding_size) -> None:
        """
        Initialize and embed variables according to the user and product dimensions.
            embedding_size: embedding size of product and user weights
        """
        self.user_matrix = self.create_constant_variable([self.user_number, embedding_size])
        self.product_matrix = self.create_random_variable([embedding_size, self.product_number])
        self.user_bias = self.create_constant_variable([self.user_number, 1])
        self.product_bias = self.create_random_variable([1, model.product_number])
        self.total_bias = self.create_random_variable([1])

    def funk_SVD(self) -> tf.Tensor:
        """
        Funk SVD
        """
        return tf.matmul(self.user_matrix, self.product_matrix)

    def bias_SVD(self) -> tf.Tensor:
        """
        Bias SVD (Funk SVD + Bias)
        """
        pred_user_product_matrix = self.funk_SVD()
        pred_user_product_matrix = pred_user_product_matrix + self.user_bias + self.product_bias + self.total_bias
        # pred_user_product_matrix = tf.keras.activations.relu(pred_user_product_matrix)
        return pred_user_product_matrix

    def loss_function(self, pred:tf.Tensor, ground_truth:pd.DataFrame) -> tf.Tensor:
        """
        The loss function of Bias SVD
            pred: the prediction of model  
            ground_truth: the ground truth of predicted data
        """
        # pred_user_product_matrix = bias_SVD()
        # threshold = np.where(pred_user_product_matrix > 5, 5, pred_user_product_matrix)
        # threshold = np.where(threshold < 1, 1, threshold)
        # threshold_error = tf.pow(threshold - pred_user_product_matrix, 2)
        pred_error = tf.pow(ground_truth - pred, 2)
        # return tf.reduce_sum(pred_error) + tf.reduce_sum(threshold_error)
        return tf.reduce_sum(pred_error)

    def regularization(self) -> tf.Tensor:
        """
        The regularizaion function of Bias SVD
        """
        return REGULARIZATION_RATE * (tf.reduce_sum(tf.pow(self.user_matrix, 2)) + tf.reduce_sum(tf.pow(self.product_matrix, 2)) + 
        tf.reduce_sum(tf.pow(self.user_bias, 2)) + tf.reduce_sum(tf.pow(self.product_bias, 2)))

    def calculation_loss(self) -> tf.Tensor:
        """
        Calculate the loss of model
        """
        pred = self.bias_SVD()
        pred = pred[~tf.math.is_nan(self.user_product_matrix)]
        no_null_user_product_matrix = self.user_product_matrix[~tf.math.is_nan(self.user_product_matrix)]
        loss = self.loss_function(pred, no_null_user_product_matrix) + self.regularization()   
        return loss
    
    @tf.function
    def gradient_descent(self) -> None:
        """
        Calculate gradient descent and update algorithm weights
        """
        with tf.GradientTape() as g:  
            loss = self.calculation_loss()
        gradients = g.gradient(loss, [self.user_matrix, self.product_matrix, self.total_bias, self.user_bias, self.product_bias])
        optimizer.apply_gradients(zip(gradients, [self.user_matrix, self.product_matrix, self.total_bias, self.user_bias, self.product_bias]))
    
    def training(self, training_step:int, learning_rate:int) -> None:
        """
        Training bias SVD model
            TRAINING_STEP: the number of training steps
            learning_rate: the model learning rate
        """
        for i in range(1, training_step + 1):
            self.gradient_descent()
            # every 10 epochs print training situation
            if i % 10 == 0:
                loss = self.calculation_loss() - self.regularization()
                print(f'Step: {i}, Loss: {loss}')
                # every 100 epochs check out the loss and update the learning rate
                if (i % 100 == 0) & (loss < 1) & (learning_rate >= 0.01):
                    learning_rate = learning_rate / 10
                    optimizer.lr.assign(learning_rate)
                    print(f'Update learning rate to: {learning_rate}')
    
    def predict(self):
        pred = self.bias_SVD()
        pred = pd.DataFrame(pred.numpy())
        pred.index = self.user_product_index
        pred.columns = self.user_product_column
        return pred
    
    def save_weights(self, folder_path:str) -> None:
        """
        Save model weights to the specific folder.
        There are five weights: user, product, user bias, product bias, and total bias
            folder_path: the path of the folder you want to save to
        """
        pd.DataFrame(self.user_matrix.numpy(), index=self.user_product_index).to_csv(folder_path + 'user_matrix.csv', float_format='%.15f')
        pd.DataFrame(self.product_matrix.numpy(), columns=self.user_product_column).to_csv(folder_path + 'product_matrix.csv', float_format='%.15f')
        pd.DataFrame(self.user_bias.numpy(), index=self.user_product_index).to_csv(folder_path + 'user_bias.csv', float_format='%.15f')
        pd.DataFrame(self.product_bias.numpy(), columns=self.user_product_column).to_csv(folder_path + 'product_bias.csv', float_format='%.15f')
        pd.DataFrame(self.total_bias.numpy()).to_csv(folder_path + 'total_bias.csv', float_format='%.15f')

if __name__ == "__main__":
    combine_data = combine_purchase_register('./data/purchase_recommendation_data.csv', './data/register_recommendation_data.csv')
    combine_data.to_csv('./data/purchase_register_recommendation_data.csv', index=False)

    model = Bias_SVD_model('./data/purchase_register_recommendation_data.csv')
    model.tranform2tensor()
    model.initialize_variable(EMBEDDING_SIZE)
    model.training(TRAINING_STEP, LEARNING_RATE)
    model.save_weights(WEIGHTS_FOLDER)

    pred = model.predict()
    recency = pd.read_csv('./data/purchase_register_recommendation_data.csv', usecols=['email', 'recency'])
    pred = pred.merge(recency, how='left', on='email')
    pred.to_csv('./data/bias_svd_result.csv', index=False, float_format='%.15f')