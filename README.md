# deep-learning-churn-ANN
**Customer Churn Prediction with Artificial Neural Networks**
## **Overview**
This project aims to predict customer churn in a banking context using an Artificial Neural Network (ANN). The goal is to identify customers who are more likely to leave the bank, allowing proactive measures to be taken for customer retention. By leveraging deep learning techniques, the ANN model analyzes various customer features and generates probabilities indicating the likelihood of churn.

## **Dataset**
The dataset used for this project contains a diverse range of customer attributes, including age, gender, account balance, credit score, and more. It was carefully collected and curated to ensure its relevance to the problem of customer churn prediction.

## **Methodology**
The project follows the following methodology:

Data Preprocessing: The dataset underwent preprocessing steps, including handling missing values, encoding categorical variables, and normalizing numerical features.
Model Architecture: An ANN model was constructed using TensorFlow, a popular deep learning library, and the Pandas library for data manipulation. The model consists of an input layer, two hidden layers with ReLU activation, and an output layer with sigmoid activation.
Training and Evaluation: The model was trained using the training dataset and evaluated on a separate validation dataset. The binary cross-entropy loss function was used, and the model's performance was assessed based on accuracy and other relevant metrics.
Prediction and Interpretation: Once trained, the model was utilized to predict customer churn probabilities for new, unseen data. These probabilities provide insights into the customers at higher risk of leaving, enabling the bank to take proactive actions for customer retention.

## **Results**
The ANN model achieved promising results in predicting customer churn. With an accuracy of 86.67% on the validation dataset, the model effectively identifies customers who are more likely to churn. The probabilities generated by the model provide a quantitative measure of the likelihood of churn, aiding the bank in decision-making processes and customer retention strategies.

## **Usage and Reproducibility**
To reproduce the results and explore the project further, follow the instructions below:

Clone this GitHub repository to your local machine.
Install the required dependencies, including TensorFlow and Pandas.
Run the churn_prediction.ipynb notebook, which contains the complete code for data preprocessing, model construction, training, evaluation, and prediction.
Feel free to modify the code or experiment with different hyperparameters to further enhance the model's performance.

## **Contributions and Feedback**

Contributions to this project are more than welcome! If you have any ideas for improvement or would like to collaborate, please feel free to submit a pull request or open an issue. Your feedback and suggestions are valuable in refining and advancing the customer churn prediction model.
