from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import abs
from pyspark.sql.functions import row_number, lit
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
import numpy as np


def create_ma(df, feature, period):
    time_window = Window.orderBy(F.col("Date")).rowsBetween(-period + 1, 0)
    new_df = df.withColumn('ma' + str(period) + '_' + feature, F.avg(feature).over(time_window) / df[feature])
    return new_df


def create_rsi(df, feature, period):
    new_df = df.withColumn("1d_past_" + feature, F.lag(feature, 1).over(Window().orderBy('Date')))
    new_df = new_df.withColumn('diff', new_df[feature] - new_df["1d_past_" + feature])
    #     new_df = new_df.na.drop()
    mask_up = F.udf(lambda x: None if x is None else (1 if x > 0 else 0))
    mask_down = F.udf(lambda x: None if x is None else (1 if x < 0 else 0))
    new_df = new_df.withColumn('mask_up', mask_up(new_df.diff)).withColumn('mask_down', mask_down(new_df.diff))
    new_df = new_df.withColumn('up_change', new_df.diff * new_df.mask_up).withColumn('down_change',
                                                                                     new_df.diff * new_df.mask_down)
    time_window = Window.orderBy(F.col("Date")).rowsBetween(-period + 1, 0)
    new_df = new_df.withColumn("up_chg_avg", F.avg("up_change").over(time_window)).withColumn("down_chg_avg",
                                                                                              F.avg("down_change").over(
                                                                                                  time_window))
    new_df = new_df.withColumn("rsi" + str(period) + '_' + feature,
                               (100 - 100 / (1 + abs(new_df.up_chg_avg / new_df.down_chg_avg))))
    return new_df


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_params(dim, value):
    w = np.array([value] * dim)
    b = value
    return w, b


def propagate(RDD, w, b, size, class_weights=None):
    '''
    calculate cost and derivative using map and treeAggregate
    gradientCost = RDD.map(y,x).map(y,x,a).map(cost,dw,db)
    '''
    seqOp = (lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2]))
    combOp = (lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2]))
    if class_weights == None:
        gradientCost = RDD.map(lambda x: (x[0], x[1], sigmoid(np.dot(x[1], w) + b))) \
            .map(lambda x: ((x[0] * np.log(x[2]) + (1 - x[0]) * np.log(1 - x[2])),
                            (x[1] * (x[2] - x[0])),
                            (x[2] - x[0]))) \
            .treeAggregate((0, 0, 0), seqOp, combOp)
    else:
        gradientCost = RDD.map(lambda x: (x[0], x[1], sigmoid(np.dot(x[1], w) + b))) \
            .map(lambda x: (class_weights[0] * (x[0] * np.log(x[2]) + class_weights[1] * (1 - x[0]) * np.log(1 - x[2])),
                            (x[1] * (class_weights[0] * x[0] * (x[2] - 1) + class_weights[1] * x[2] * (1 - x[0]))),
                            (class_weights[0] * x[0] * (x[2] - 1) + class_weights[1] * x[2] * (1 - x[0])))) \
            .treeAggregate((0, 0, 0), seqOp, combOp)

    cost = (-1 / float(size)) * gradientCost[0]
    dw = (1 / float(size)) * gradientCost[1]
    db = (1 / float(size)) * gradientCost[2]
    grads = {"dw": dw, "db": db}

    return grads, cost


# ## Optimization
def optimize(RDD, w, b, num_iterations, learningRate, size, lambd=0, weights=None, bold_drive=False):
    temp_w = np.zeros(dim)
    temp_b = 0
    temp_cost = 0
    print(
        f'Model Parmaeters: initialization_value={b}; num_terations={num_iterations}; initial_learningRate={learningRate}; '
        f'lambda={lambd}; weights={weights}; bold_drive={bold_drive}')
    n = 0
    for i in range(num_iterations):
        grads, cost = propagate(RDD, w, b, size, class_weights=weights)
        # if lambd != 0, use the regularization
        if lambd != 0:
            dw = grads['dw'] + (lambd / size) * w
            cost = cost + (1. / size) * (lambd / 2) * (np.sum(w ** 2))
        else:
            dw = grads["dw"]
        db = grads["db"]
        # update parameters
        w = w - learningRate * dw
        b = b - learningRate * db
        print(f'Epoch {i}: w={w}, b={b}, Cost={cost}, LearningRate={learningRate}')
        # Adapt learning rate by "Bold Drive"
        # If cost increase, decrease learning rate by 50% and reset the m and b to the values of previous iteration
        if bold_drive:
            if cost > temp_cost:
                learningRate = learningRate * 0.5
                w = temp_w
                b = temp_b
                n = 0
            # If cost decrease, increase learning rate by 5%
            elif cost < temp_cost:
                learningRate = learningRate * 1.05
                n = 0
            # if cost is not changing count the number of times in a row
            else:
                n += 1
            # if cost is not changing for more than three times in a row, exit the training
            if n > 3:
                print('Cost stabilized, exit training.')
                break
            temp_w = w
            temp_b = b
            temp_cost = cost
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, cost


def predict(features, w, b, threshhold=0.5):
    predictions = features.map(lambda x: sigmoid(np.dot(x, w) + b)).map(lambda p: 1 if p > threshhold else 0)

    return predictions


if __name__ == '__main__':

    sc = SparkContext.getOrCreate()
    ss = SparkSession.builder.getOrCreate()

    # ================================== 1. Read raw time series data ====================================================
    # NOTE: Read Local CSV using com.databricks.spark.csv Format, so that we can specify timestamp datatype and format.
    # this method is dependent on the “com.databricks:spark-csv_2.10:1.2.0” package. if not installed, run the commandline
    # to install `pyspark --packages com.databricks:spark-csv_2.10:1.2.0`
    stock_data = ss.read.format("com.databricks.spark.csv") \
        .option("header", "true") \
        .option("treatEmptyValuesAsNulls", "true") \
        .option("inferSchema", "true") \
        .option("mode", "DROPMALFORMED") \
        .option("timestampFormat", "yyyy-M-d") \
        .load('data/stock_data.csv')

    stock_data.printSchema()
    stock_data.show()

    # ======================================== 2. Data transformation ================================================
    # using windows function in pySpark to calculate percentage change, similar to pandas shift function
    # Calculate Percentage of Price feature: `5d_close_pct`
    window_spec = Window().orderBy('Date')
    stock_data = stock_data.withColumn("5d_past_close", F.lag("Adj_Close", 5).over(window_spec))
    stock_data = stock_data.withColumn('5d_close_pct',
                                       (stock_data['Adj_Close'] - stock_data['5d_past_close']) / stock_data[
                                           '5d_past_close'])
    # stock_data.show()

    # Calculate percentage of Volume feature: `1d_volume_pct`
    stock_data = stock_data.withColumn("1d_past_volume", F.lag("Volume", 1).over(Window().orderBy('Date')))
    stock_data = stock_data.withColumn('1d_volume_pct',
                                       (stock_data['Volume'] - stock_data['1d_past_volume']) / stock_data[
                                           '1d_past_volume'])
    primary_features = ['5d_close_pct', '1d_volume_pct']
    # stock_data.show()

    # Use 5 day future price percentage change as the response variable: `5d_future_pct`
    stock_data = stock_data.withColumn("5d_future_close", F.lead("Adj_Close", 5).over(window_spec))
    stock_data = stock_data.withColumn('5d_future_pct',
                                       (stock_data['5d_future_close'] - stock_data['Adj_Close']) / stock_data[
                                           'Adj_Close'])
    # stock_data.show()
    # change response variable to binary results: 1 for buy and 0 for sell
    buyorsell = F.udf(lambda x: None if x is None else (1.0 if x > 0 else 0))
    stock_data = stock_data.withColumn('buyorsell', buyorsell(stock_data['5d_future_pct']))
    response = ['buyorsell']

    # ========================================= 3. Feature engineering ===============================================
    # Create moving average(ma) and rsi feature.
    # Another common technical indicator is the relative strength index (RSI). This is defined by:
    #
    # $$RSI = 100 - \frac{100} {1 + RS}$$
    # $$RS = \frac{\text{average gain over } n \text{ periods}} {\text{average loss over } n \text{ periods}}$$
    #
    # A common period for RSI is 14, so we'll use that as one setting in our calculations.

    # Create moving averages and rsi for timeperiods of 14, 30, 50, and 200
    for n in [14, 30, 50, 200]:
        stock_data = create_ma(stock_data, feature="Adj_Close", period=n)
        stock_data = create_ma(stock_data, feature="1d_volume_pct", period=n)
        stock_data = create_rsi(stock_data, feature="Adj_Close", period=n)
        # Add rsi and moving average to the feature list
        features = primary_features + ['ma' + str(n) + '_Adj_Close'] + ['rsi' + str(n) + '_Adj_Close'] + [
            'ma' + str(n) + '_1d_volume_pct']

    # =================== 4. Implementation of Logistic Regression from scratch ========================================
    # Drop missing value
    # Our indicators also cause us to have missing values at the beginning of the DataFrame due to the calculations.
    w = Window().partitionBy(lit('a')).orderBy(F.col("Date"))
    stock_data = stock_data.withColumn('row_number', row_number().over(w))
    stock_data = stock_data.filter(F.col('row_number') > 200)

    stock_data = stock_data.na.drop()

    # convert the response field to integer field, not string!
    stock_data = stock_data.withColumn('buyorsell', stock_data.buyorsell.cast('int'))
    stock_data.printSchema()
    print(features)
    print(response)

    data = stock_data.select(['Date'] + features + response)
    # data.show()

    # Split data into train and test data
    size = data.count()
    train_size = data.count() * 0.8

    w = Window().orderBy("Date")
    data = data.withColumn('row_number', row_number().over(w))
    data.show(5)
    data_train = data.where(F.col('row_number') < train_size)
    data_test = data.where(F.col('row_number') >= train_size)

    # Reformat data ready for training
    RDD_train = data_train.rdd.map(lambda x: x[1:-1]) \
        .map(lambda x: (x[-1], np.array((x[0:-1])))) \
        .map(lambda x: (x[0], (x[1] - np.mean(x[1])) / np.std(x[1])))
    # RDD_train.take(2)

    RDD_test = data_test.rdd.map(lambda x: x[1:-1]).map(lambda x: (x[-1], np.array((x[0:-1])))).map(
        lambda x: (x[0], (x[1] - np.mean(x[1])) / np.std(x[1])))
    # RDD_test.take(2)

    # NOTE: if RDD_train has null value, count() will get error!
    # RDD_train.count()
    # RDD_test.count()

    # Train Logistic Regression Model
    RDD_train.cache()
    dim = len(features)
    w, b = initialize_params(dim, 0)
    learningRate = 1E-4
    train_size = RDD_train.count()
    num_iterations = 200
    print('==>Training Model...')
    # lambd is the lambda value for regularization, set to 0 if you don't use regularization
    # weights is for the imbalance classes; default to native without balancing weights
    parameters, grads, cost = optimize(RDD_train, w, b, num_iterations, learningRate, train_size,
                                       lambd=0.1, weights=None, bold_drive=True)
    print('Training model is done!')

    # ============================================ 5. Prediction ======================================================
    print('==>Making predictions... ')
    w, b = parameters['w'], parameters['b']
    test_features = RDD_test.map(lambda x: x[1])
    test_label = RDD_test.map(lambda x: x[0])
    predictions_test = predict(test_features, w, b, threshhold=0.5)

    # ============================================= 6. Calculate Metrics ==============================================
    # - True Positive (TP) - label is positive and prediction is also positive
    # - True Negative (TN) - label is negative and prediction is also negative
    # - False Positive (FP) - label is negative but prediction is positive
    # - False Negative (FN) - label is positive but prediction is negative
    # - Precision = TruePositives / (TruePositives + FalsePositives)
    # - Recall = TruePositives / (TruePositives + FalseNegatives)
    # - F1 = 2 * (precision * recall) / (precision + recall)

    lp_RDD = test_label.zip(predictions_test).zipWithIndex()

    TP = lp_RDD.filter(lambda x: (x[0][0] == 1 and x[0][1] == 1)).count()
    FP = lp_RDD.filter(lambda x: (x[0][0] == 0 and x[0][1] == 1)).count()
    FN = lp_RDD.filter(lambda x: (x[0][0] == 1 and x[0][1] == 0)).count()

    if TP == 0 and FP == 0:
        precision = 0
        recall = 0
        f1_score = 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * precision * recall / (precision + recall)
    print(f"Precision = {precision}")
    print(f"Recall = {recall}")
    print(f'F1 score: {f1_score}')

    # ====================================== 7.Implementation using MLlib library ======================================
    # Create labeled data for both training dataset and test dataset
    labeled_train_data = RDD_train.map(lambda x: LabeledPoint(x[0], x[1]))
    labeled_test_data = RDD_test.map(lambda x: LabeledPoint(x[0], x[1]))

    # Build and train the model
    print('==> Train model using LogisticRegressionWithLBFGS ...')
    # default iterations 100
    model = LogisticRegressionWithLBFGS.train(labeled_train_data, iterations=100)
    # Predict and Compute raw scores on the test set
    print('==> Precition...')
    predictionAndLabels = labeled_test_data.map(lambda lp: (float(model.predict(lp.features)), lp.label))
    metrics_multi = MulticlassMetrics(predictionAndLabels)
    # Overall statistics
    precision_lib = metrics_multi.precision(1.0)
    recall_lib = metrics_multi.recall(1.0)
    f1Score_lib = metrics_multi.fMeasure(1.0)
    print("==> Summary Stats for LogisticRegressionWithLBFGS:")
    print(f"Precision = {precision_lib}")
    print(f"Recall = {recall_lib}")
    print(f"F1 Score = {f1Score_lib}")
