# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 23:21:05 2019

@author: Shivaranjani
"""

import numpy as np
import pandas as pd
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Flatten
global predictions, listOfBooksReadByTop10, labelencoder_PID, labelencoder_UID
labelencoder_PID = LabelEncoder()
labelencoder_UID = LabelEncoder()
predictions = []
listOfBooksReadByTop10 = []
recBooks = []

def get_model(num_users, num_books, n_latent_factors=30, layers=[10], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers)
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    book_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    MF_Embedding_User = keras.layers.Embedding(num_users, n_latent_factors, name='MF-User-Embedding', 
                                               embeddings_initializer = keras.initializers.he_uniform(),
                                               embeddings_regularizer = l2(0.), input_length=1)(user_input)
    MF_Embedding_Book = keras.layers.Embedding(num_books, n_latent_factors, name='MF-Book-Embedding', 
                                               embeddings_initializer = keras.initializers.he_uniform(),
                                               embeddings_regularizer = l2(0.), input_length=1)(book_input)
    MLP_Embedding_User = keras.layers.Embedding(num_users, int(layers[0]/2), name='MLP-User-Embedding', 
                                               embeddings_initializer = keras.initializers.he_uniform(),
                                               embeddings_regularizer = l2(0.), input_length=1)(user_input)
    MLP_Embedding_Book = keras.layers.Embedding(num_books, int(layers[0]/2), name='MLP-Book-Embedding', 
                                               embeddings_initializer = keras.initializers.he_uniform(),
                                               embeddings_regularizer = l2(0.), input_length=1)(book_input)
    mf_user_latent = Flatten()(MF_Embedding_User)
    mf_item_latent = Flatten()(MF_Embedding_Book)
    mf_vector = keras.layers.Multiply()([mf_user_latent, mf_item_latent])
    mlp_user_latent = Flatten()(MLP_Embedding_User)
    mlp_item_latent = Flatten()(MLP_Embedding_Book)
    mlp_vector = keras.layers.concatenate([mlp_user_latent, mlp_item_latent], axis = 1, name='Concat-MLP')
    for idx in range(1, num_layer):
        layer = keras.layers.Dense(layers[idx], W_regularizer= l2(reg_layers[idx]), activation='relu', name="Layer-%d" %idx)
        mlp_vector = layer(mlp_vector)
    predict_vector = keras.layers.concatenate([mf_vector, mlp_vector], axis = 1, name='Concat-Final')
    #dense_4 = keras.layers.Dense(8,name='Layer-5', activation='relu')(predict_vector)
    prediction = keras.layers.Dense(1, activation="relu", name='Prediction')(predict_vector)
    model = Model(input=[user_input, book_input], output=prediction)
    model.summary()
    return model

def trainModel(model, df, num_users, num_books, num_epochs = 12):
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
    cvscores = []
    trainscores = []
    
    earlyStopping = keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, restore_best_weights=True, patience = 1)
    for trainIndex, testIndex in kfold.split(df, df.rating):
        print("Starting a new fold...")
        train = df.iloc[trainIndex]
        test = df.iloc[testIndex]
        history = model.fit([np.array(train.user_id), np.array(train.book_id)], np.array(train.rating), epochs=num_epochs, verbose=1, callbacks=[earlyStopping])
        results = model.evaluate([test.user_id,test.book_id], test.rating)
        #plt.plot(history.history['mean_squared_error'])
        trainscores.append(history)
        cvscores.append(results)
    plt.plot(list(x[0] for x in cvscores))
    #plt.plot(list(x[1] for x in cvscores))
    plt.xlabel('folds')
    plt.ylabel('mse')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
    model.save('recommenderNeuMF.h5')
    return model

def predict(model, df):
    top10UserIds = df.groupby("user_id").agg({"rating":"count"}).sort_values(by = "rating", ascending=False).reset_index().user_id[:10]
    listOfBooksReadByTop10 = []
    #usersBooks = df.groupby("book_id").agg({"rating":"count"}).reset_index().sort_values(by = "rating", ascending=False).reset_index().book_id[:10]
    for id in top10UserIds:
        usersBooks = list(df[df["user_id"] == id].reset_index().sort_values(by="rating", ascending=False).reset_index().book_id)[:10]
        for bk in usersBooks:
            listOfBooksReadByTop10.append([id, bk])
    listOfBooksReadByTop10 = pd.DataFrame(listOfBooksReadByTop10,  columns = ["user_id", "book_id"])
    users = top10UserIds
    items = df.book_id.unique()
    predictions = {'user_id': list(),
                'book_id': list(),
                'rating':list()}
    for u in users:
        for i in items:
            predictions["user_id"].append(u)
            predictions["book_id"].append(i)
            predictions["rating"].append(model.predict([[[u]], [[i]]])[0][0])
    pred = pd.DataFrame(predictions)
    pred.rating = pred.rating/max(pred.rating)*5
    #pred = pred.sort_values(by = ["user_id", "rating", "book_id"], ascending=False).groupby(["user_id"]).head(20)
    return pred, listOfBooksReadByTop10

def getRecommendedBooks():
    global recBooks, labelencoder_PID, labelencoder_UID
    predtop10 = predictions.sort_values(by=["user_id","rating","book_id"], ascending=False).groupby(["user_id"]).head(20)
    tf = pd.read_csv("goodbooks-10k/books.csv").reset_index()
    recBooks = pd.merge(predtop10, tf[["id","title","authors","image_url","small_image_url"]], left_on = "book_id", right_on = "id")
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
        rows = recBooks[recBooks["user_id"] == user].reset_index()
        for i, row in rows.iterrows():
            print("Title : " + row["title"])
            print("Author : " + row["authors"])
            #im = Image.open(requests.get(row["small_image_url"], stream=True).raw)
            #im.save(str(user)+"_"+row["title"],"JPEG")
        print("=======================================================================")


def dcg_at_k(ratings, k, method=0):
    ratings = np.asfarray(ratings)[:k]
    if ratings.size:
        if method == 0:
            return ratings[0] + np.sum(ratings[1:] / np.log2(np.arange(2, ratings.size + 1)))
        elif method == 1:
            return np.sum(ratings / np.log2(np.arange(2, ratings.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def calculatePre_Re(model, pred, relevant, df):
    book_ids = df.book_id.unique()
    predictionsForUser = []
    for u in relevant.user_id.unique():
        for b in book_ids:
            trueRating = df.loc[(df['user_id'] == u) & (df['book_id'] == b)].rating.item() if not df.loc[(df['user_id'] == u) & (df['book_id'] == b)].empty else -1
            predictionsForUser.append([u, b, model.predict([[[u]], [b]])[0][0], trueRating])
    
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
        prediction = predictions.loc[(predictions['user_id'] == u) & (predictions['book_id'] == b)].rating.item() if not predictions.loc[(predictions['user_id'] == u) & (predictions['book_id'] == b)].empty else -1
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

if __name__ == '__main__':
    num_epochs = 12
    mf_dim = 30
    layers = [64,32,16,8]
    reg_mf = 0
    reg_layers = [0,0,0,0]
    learning_rate = 0.005
    df = pd.read_csv("TopBookRatings.csv")
    df.drop(["index"],inplace=True, axis=1)
    df["book_id"] = labelencoder_PID.fit_transform(df["book_id"])
    df["user_id"] = labelencoder_UID.fit_transform(df["user_id"])
    num_users = df.user_id.unique().shape[0]
    num_books = df.book_id.unique().shape[0]
    model = get_model(num_users, num_books, mf_dim, layers, reg_layers, reg_mf)
    model.compile(optimizer=Adam(lr=learning_rate), loss= 'mean_squared_error', metrics=['mse'])
    model = load_model('recommenderNeuMF.h5')
    #model = trainModel(model, df, num_users, num_books, num_epochs = 12)
    predictions, listOfBooksReadByTop10 = predict(model, df)
    df["book_id"] = labelencoder_PID.inverse_transform(df["book_id"])
    df["user_id"] = labelencoder_UID.inverse_transform(df["user_id"])
    predictions["user_id"] = labelencoder_UID.inverse_transform(predictions["user_id"])
    predictions["book_id"] = labelencoder_PID.inverse_transform(predictions["book_id"])
    listOfBooksReadByTop10["user_id"] = labelencoder_UID.inverse_transform(listOfBooksReadByTop10["user_id"])
    listOfBooksReadByTop10["book_id"] = labelencoder_PID.inverse_transform(listOfBooksReadByTop10["book_id"])
    getPredictedMse(df, listOfBooksReadByTop10)
    pre = predictions.sort_values(by=["user_id","book_id"], ascending=False).reset_index().rating
    print("NDCG: " ,ndcg(pre, 10))
    #userMetric = calculatePre_Re(model, predictions, listOfBooksReadByTop10, df)
    #print(userMetric)
    getRecommendedBooks()

