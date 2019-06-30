# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:24:36 2019

@author: Shivaranjani
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import time
import surprise
from surprise import Reader, Dataset
from surprise import SVD, CoClustering, NMF
from surprise import KNNBasic, KNNWithMeans
from numpy import tensordot
from numpy.linalg import norm
from itertools import product
from PIL import Image

sns.set()
pd.set_option('display.expand_frame_repr', False)
labelencoder_PID = LabelEncoder()
labelencoder_UID = LabelEncoder()

svdModel = SVD(n_factors = 20, n_epochs = 10, biased=True)
sim_options = {'name': 'msd', 'user_based': False}
knnBasicModel = KNNBasic(k =10, sim_options=sim_options)
coCluster = CoClustering(n_cltr_u = 10, n_cltr_i = 10)
nmfModel = NMF(n_factors = 10, n_epochs = 40, biased=True)
predictionsEn = []
listOfBooksReadByTop10 = []
recBooksEn = []

def getDataFromFile():
    df = pd.read_csv("goodbooks-10k/ratings.csv")
    threeabvrating = df[df["rating"]>=3]
    books = threeabvrating.groupby("book_id").agg({"user_id":"count", "rating" : "mean"}).reset_index().rename(columns = {"user_id" : "count_users","rating": "avg_rating"})
    sorted_bks = books.sort_values(by=['count_users', 'avg_rating'], ascending=False).reset_index()
    top500famousBookIds = sorted_bks.book_id.unique()[:500]
    famousBooks = df[df['book_id'].isin(top500famousBookIds)]
    countUsers = famousBooks.groupby("user_id").agg({"rating" : "count"}).reset_index().rename(columns = {"rating":"count"})
    top500users =  countUsers.sort_values(by = "count", ascending=False).reset_index().user_id.unique()[:500]
    df = famousBooks[famousBooks['user_id'].isin(top500users)].reset_index()
    df.to_csv("TopBookRatings.csv", index=False)
    ratings_dict = {'user_id': list(df["user_id"]),
                'book_id': list(df["book_id"]),
                'rating':list(df["rating"])}
    df = pd.DataFrame(ratings_dict)
    train, test = train_test_split(df, test_size=0.25, random_state=43)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train[["user_id","book_id","rating"]], reader)
    test = Dataset.load_from_df(test[["user_id","book_id","rating"]], reader)
    return data, test

def trainModel(data):
    kFold = surprise.model_selection.split.KFold(n_splits=5, random_state=43, shuffle=True)
    knnResults = []
    svdResults = []
    #coClusterResults = []
    nmfResults = []
    for train, test in kFold.split(data):
        print("Starting a new fold...")
        print("========================")
        print("Running SVD..")
        start = time.time()
        svdModel.fit(train)
        svdResult = svdModel.test(test)
        svdResults.append({"rmse" : surprise.accuracy.rmse(svdResult, verbose=True),
                           "mae" : surprise.accuracy.mae(svdResult, verbose=True),
                           "fcp" :surprise.accuracy.fcp(svdResult, verbose=True)})
        end = time.time()
        time_taken = end - start
        print("Time taken for SVD: ", time_taken)
        print("Running KNN..")
        start = time.time()
        knnBasicModel.fit(train)
        knnResult = knnBasicModel.test(test)
        knnResults.append({"rmse" : surprise.accuracy.rmse(knnResult, verbose=True),
                           "mae" : surprise.accuracy.mae(knnResult, verbose=True),
                           "fcp" :surprise.accuracy.fcp(knnResult, verbose=True)})
        end = time.time()
        time_taken = end - start
        print("Time taken for KNN: ", time_taken)
    
        '''
        print("Running CoClustering..")
        start = time.time()
        coCluster.fit(train)
        coClusterResult = coCluster.test(test)
        coClusterResults.append({"rmse" : surprise.accuracy.rmse(coClusterResult, verbose=True),
                           "mae" : surprise.accuracy.mae(coClusterResult, verbose=True),
                           "fcp" :surprise.accuracy.fcp(coClusterResult, verbose=True)})
        end = time.time()
        time_taken = end - start
        print("Time taken for coClustering: ", time_taken)'''
        print("Running NMF..")
        start = time.time()
        nmfModel.fit(train)
        nmfResult = nmfModel.test(test)
        nmfResults.append({"rmse" : surprise.accuracy.rmse(nmfResult, verbose=True),
                           "mae" : surprise.accuracy.mae(nmfResult, verbose=True),
                           "fcp" :surprise.accuracy.fcp(nmfResult, verbose=True)})
        end = time.time()
        time_taken = end - start
        print("Time taken for NMF: ", time_taken)
    plt.plot([o['rmse'] for o in knnResults],label='knn')
    plt.plot([o['rmse'] for o in svdResults],label='svd')
    plt.plot([o['rmse'] for o in nmfResults],label='nmf')
    #plt.plot([o['rmse'] for o in coClusterResults],label='cc')
    plt.xlabel('folds')
    plt.ylabel('rmse')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    plt.plot([o['mae'] for o in knnResults],label='knn')
    plt.plot([o['mae'] for o in svdResults],label='svd')
    plt.plot([o['mae'] for o in nmfResults],label='nmf')
    #plt.plot([o['mae'] for o in coClusterResults],label='cc')
    plt.xlabel('folds')
    plt.ylabel('mae')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    plt.plot([o['fcp'] for o in knnResults],label='knn')
    plt.plot([o['fcp'] for o in svdResults],label='svd')
    plt.plot([o['fcp'] for o in nmfResults],label='nmf')
    #plt.plot([o['fcp'] for o in coClusterResults],label='cc')
    plt.xlabel('folds')
    plt.ylabel('fcp')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def getWeightedPrediction(weights, data):
    predSVD = svdModel.test(data)
    predKNN = knnBasicModel.test(data) 
    #predCC = coCluster.test(data)
    predNMF = nmfModel.test(data)
    estSVD = np.array([x.est for x in predSVD])
    estKNN = np.array([x.est for x in predKNN])
    #estCC = np.array([x.est for x in predCC])
    estNMF = np.array([x.est for x in predNMF])
    pred = np.vstack((estSVD, estKNN, estNMF)).T.dot(weights)
    trueRating = np.array([x.r_ui for x in predSVD])
    loss = mean_squared_error(trueRating, pred)
    return pred, loss

def optimiseEnsembleWeights(data, modelCount = 3):
    print("===================================================")
    print("Starting ensemble weights optimisation..: ")
    data = data.build_full_trainset().build_testset()
    w = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    best_loss, best_weights = 1.0, None
    for weights in product(w, repeat = modelCount):
        if len(set(weights)) == 1:
            continue
        result = norm(weights, 1)
        if result == 0.0:
            weights = weights
        else: 
            weights = weights / result
        pred, loss = getWeightedPrediction(weights, data)
        if loss < best_loss:
            best_loss, best_weights = loss, weights
            print('>%s %.3f' % (best_weights, best_loss))
    return best_weights, best_loss


def predict(weights):
    df = pd.read_csv("TopBookRatings.csv")
    df.drop(["index"],inplace=True, axis=1)
    top10UserIds = df.groupby("user_id").agg({"rating":"count"}).sort_values(by = "rating", ascending=False).reset_index().user_id[:10]
    listOfBooksReadByTop10 = []
    #usersBooks = df.groupby("book_id").agg({"rating":"count"}).reset_index().sort_values(by = "rating", ascending=False).reset_index().book_id[:10]
    for id in top10UserIds:
        usersBooks = list(df[df["user_id"] == id].reset_index().sort_values(by="rating", ascending=False).reset_index().book_id)[:10]
        for bk in usersBooks:
            listOfBooksReadByTop10.append([id, bk])
    listOfBooksReadByTop10 = pd.DataFrame(listOfBooksReadByTop10,  columns = ["user_id", "book_id"])
    predictions = {'user_id': list(),
                'book_id': list(),
                'rating':list()}
    print("Getting rating predictions for top 10 users....")
    users = top10UserIds
    items = df.book_id.unique()
    for u in users:
        for i in items:
            algoResults = np.array([x.est for x in [svdModel.predict(u,i), knnBasicModel.predict(u,i), nmfModel.predict(u,i)]]).reshape(1,len(weights))
            prediction = np.dot(algoResults, weights)[0,0]
            predictions["user_id"].append(u)
            predictions["book_id"].append(i)
            predictions["rating"].append(prediction)
    pred = pd.DataFrame(predictions)
    #pred = pred.sort_values(by = ["user_id","rating","book_id"], ascending=False).groupby(["user_id"])
    return pred, listOfBooksReadByTop10

def getRecommendedBooks():
    global recBooksEn
    tf = pd.read_csv("goodbooks-10k/books.csv").reset_index()
    predtop10 = predictionsEn.sort_values(by=["user_id","rating","book_id"], ascending=False).groupby(["user_id"]).head(20)
    recBooksEn = pd.merge(predtop10, tf[["id","title","authors","image_url","small_image_url"]], left_on = "book_id", right_on = "id")
    booksRead = pd.merge(listOfBooksReadByTop10, tf[["id","title","authors","image_url","small_image_url"]], left_on = "book_id", right_on = "id")
    print("Following are the recommendations..")
    for user in booksRead.user_id.unique():
        print("User ID: %d" %(user))
        print("=======================================================================")
        print("Books read:")
        rows = booksRead[booksRead["user_id"] == user].reset_index()
        for i, row in rows.iterrows():
            print("Title : " + row["title"])
            print("Author : " + row["authors"])
            #im = Image.open(requests.get(row["small_image_url"], stream=True).raw)
            #im.save(str(user)+"_"+row["title"],"JPEG")
        print("=======================================================================")
        print("Books recommended:")
        rows = recBooksEn[recBooksEn["user_id"] == user].reset_index()
        for i, row in rows.iterrows():
            print("Title : " + row["title"])
            print("Author : " + row["authors"])
            #im = Image.open(requests.get(row["small_image_url"], stream=True).raw)
            #im.save(str(user)+"_"+row["title"],"JPEG")
        print("=======================================================================")

def calculatePre_Re(pred, relevant, df, weights):
    book_ids = df.book_id.unique()
    predictionsForUser = []
    for u in relevant.user_id.unique():
        for b in book_ids:
            trueRating = df.loc[(df['user_id'] == u) & (df['book_id'] == b)].rating.item() if not df.loc[(df['user_id'] == u) & (df['book_id'] == b)].empty else -1
            algoResults = np.array([x.est for x in [svdModel.predict(u,b), knnBasicModel.predict(u,b), nmfModel.predict(u,b)]]).reshape(1,len(weights))
            prediction = np.dot(algoResults, weights)[0]
            predictionsForUser.append([u, b, prediction, trueRating])
    
    predictionsForUser = pd.DataFrame(predictionsForUser,  columns = ["user_id", "book_id", "predicted", "trueRating"])
    userMetric = {}
    for u in predictionsForUser.user_id.unique():
        bookSubset = predictionsForUser[predictionsForUser.user_id == u].reset_index()
        relevantSubset = bookSubset[bookSubset.trueRating>=2.5]
        recommSubSet = bookSubset.sort_values(by=['predicted'], ascending=False).reset_index()[:50]
        relSet = set(relevantSubset.book_id)
        recSet = set(recommSubSet.book_id)
        relAndRec = relSet.intersection(recSet)
        precision = len(relAndRec)/len(recSet)
        recall = len(relAndRec)/len(relSet)
        userMetric[u] = {"precision" : precision, "recall" : recall}
    return userMetric

def getPredictedMse(df, listOfBooksReadByTop10):
    predictionsForUser = []
    for row in listOfBooksReadByTop10.iterrows():
        u = row[1][0]
        b = row[1][1]
        prediction = predictionsEn.loc[(predictionsEn['user_id'] == u) & (predictionsEn['book_id'] == b)].rating.item() if not predictionsEn.loc[(predictionsEn['user_id'] == u) & (predictionsEn['book_id'] == b)].empty else -1
        trueRating = df.loc[(df['user_id'] == u) & (df['book_id'] == b)].rating.item() if not df.loc[(df['user_id'] == u) & (df['book_id'] == b)].empty else -1
        predictionsForUser.append([u, b, prediction, trueRating])

    predictionsForUser = pd.DataFrame(predictionsForUser,  columns = ["user_id", "book_id", "predicted", "trueRating"])
    predictionsForUser = pd.DataFrame(predictionsForUser,  columns = ["user_id", "book_id", "predicted", "trueRating"])
    print(mean_squared_error(predictionsForUser.trueRating, predictionsForUser.predicted))

def dcg(relevances, rank=10):
    """Discounted cumulative gain at rank (DCG)"""
    relevances = np.asarray(relevances)[:rank]
    n_relevances = len(relevances)
    if n_relevances == 0:
        return 0.

    discounts = np.log2(np.arange(n_relevances) + 2)
    return np.sum(relevances / discounts)

def ndcg(relevances, rank=10):
    """Normalized discounted cumulative gain (NDGC)"""
    best_dcg = dcg(sorted(relevances, reverse=True), rank)
    if best_dcg == 0:
        return 0.

    return dcg(relevances, rank) / best_dcg

def main():
    global predictionsEn, listOfBooksReadByTop10
    df = pd.read_csv("TopBookRatings.csv")
    train, test = getDataFromFile()
    trainModel(train)
    #Found from experiments loss = 0.356
    #weights = np.array([0.25, 0.3125, 0.4375])
    weights, mseEnsemble = optimiseEnsembleWeights(test)
    print("Best Weights for the ensemble model: ", weights)
    print("Ensemble mse for the best weights: ", mseEnsemble)
    test = test.build_full_trainset().build_testset()
    predSVD = svdModel.test(test)
    predKNN = knnBasicModel.test(test) 
    predNMF = nmfModel.test(test)
    estSVD = np.array([x.est for x in predSVD])
    estKNN = np.array([x.est for x in predKNN])
    estNMF = np.array([x.est for x in predNMF])
    trueRating = np.array([x.r_ui for x in predSVD])
    mseSVD = mean_squared_error(trueRating, estSVD)
    print("Error from SVD: ", mseSVD)
    mseKNN = mean_squared_error(trueRating, estKNN)
    print("Error from KNN: ", mseKNN)
    mseNMF = mean_squared_error(trueRating, estNMF)
    print("Error from NMF: ",mseNMF)
    predictionsEn, listOfBooksReadByTop10 = predict(np.array(weights).reshape(len(weights),1))
    #userMetric = calculatePre_Re(predictionsEn, listOfBooksReadByTop10, df, weights)
    #print(userMetric)
    getPredictedMse(df, listOfBooksReadByTop10)
    pre = predictionsEn.sort_values(by=["user_id","book_id"], ascending=False).reset_index().rating
    print("NDCG: " ,ndcg(pre, 10))
    getRecommendedBooks()

if __name__ == "__main__":
    main()

    

