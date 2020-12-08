import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn import preprocessing
from collections import defaultdict, Counter
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
import sklearn.metrics as metrics
import surprise
from surprise import Reader
from surprise import Dataset
from collections import defaultdict
import pymongo
import warnings
warnings.filterwarnings("ignore")

data_path = Path('../EDA/outputs')

def get_db(port=27017,host='localhost'):
    db_client = pymongo.MongoClient(host=host,port=port)
    db = db_client['beeptunes']
    return db

def getFeature(col,track_tag):
    groupby_typekey = track_tag.groupby(['TYPE_KEY'])
    feature = groupby_typekey.get_group(col)
    le = LabelEncoder()
    feature['TAG_ID'] = le.fit_transform(feature['TAG_ID'])
    feature.drop(['TYPE_KEY'], axis = 1, inplace = True)
    feature = pd.get_dummies(feature, columns=['TAG_ID'])
    feature_cols= feature.columns.drop(['TRACK_ID'])
    feature=feature.groupby(['TRACK_ID'])[feature_cols].sum().reset_index()
    feature_dict=defaultdict(list)
    a=feature.drop(['TRACK_ID'],axis=1).values.tolist()
    for i in range(len(a)):
        feature_dict[feature['TRACK_ID'][i]].extend(a[i])
    return feature_dict

class PopularRecommender:
    def fit(self, scores):
        self.popular = scores.groupby('track').score.sum().reset_index().sort_values('score', ascending=False)[
            'track'].values

    def recommend(self, user_id=None, count=None):
        result = self.popular
        if count is not None:
            result = result[:count]
        return result

    def recommend_all(self, user_ids=None, count=None):
        recomms = dict()
        for user_id in user_ids:
            recomms[user_id] = self.recommend(user_id, count)

        return recomms



class Skl_CollaborativeRecommender:
    def __init__(self, rank=20, alpha=0.0, max_iter=1000):
        self.rank = rank
        self.alpha = alpha
        self.iter = max_iter
        self.new_user_recommender = PopularRecommender()

    def fit(self, scores):
        scores = scores.assign(user=scores.user.astype('category'), track=scores.track.astype('category'))
        mat = sparse.coo_matrix((
            scores.score.values.astype('float32'),
            (scores.user.cat.codes.values,
             scores.track.cat.codes.values))).tocsr()
        model = NMF(n_components=self.rank, alpha=self.alpha, max_iter=self.iter)
        model.fit(mat)
        self.new_user_recommender.fit(scores)
        self.mat = mat
        self.model = model
        self.track_categories = scores.track.cat.categories
        self.user_index = {Id:index for index, Id in enumerate(scores.user.cat.categories)}
        self.track_index = {Id:index for index, Id in enumerate(scores.track.cat.categories)}

    def recommend(self, user_id, count=None):
        if user_id not in self.user_index:
            return self.new_user_recommender.recommend(user_id, count)
        index = self.user_index[user_id]
        user_history = self.mat[index]
        score_pred = self.model.inverse_transform(self.model.transform(user_history)).ravel()
        result = self.track_categories[np.argsort(score_pred)[::-1]]

        previously_downloaded = set(self.track_categories[self.mat[index].nonzero()[1]])
        result = [x for x in result if x not in previously_downloaded]

        if count is not None:
            result = result[:count]
        return np.array(result)

    def recommend_all(self, user_ids=None, count=None):
        recomms = dict()
        for user_id in user_ids:
            recomms[user_id] = self.recommend(user_id, count)

        return recomms

    def predict(self,test_df):
        preds = pd.DataFrame(columns=['user','track','actual','est'])
        preds[['user','track','actual']] = test_df[['user','track','score']].copy()
        for i,row in preds.iterrows():
            preds.loc[i,'est'] =  (self.model.inverse_transform(self.model.transform(self.mat[self.user_index[row['user']]]))).squeeze()[self.track_index[row['track']]]
        return preds


class Surp_CollaborativeRecommender:

    def __init__(self):
        self.top_n = defaultdict(list)
        return

    def fit(self, scores, n_factors=30, n_epochs=100, biased=True, random_state=None, verbose=False):
        self.reader = Reader(rating_scale=(scores['score'].min(), scores['score'].max()))
        self.scores = scores
        self.scores_ds = Dataset.load_from_df(self.scores, reader=self.reader)
        self.trainset = self.scores_ds.build_full_trainset()
        self.algo = surprise.NMF(n_factors=n_factors, n_epochs=n_epochs, biased=biased, random_state=random_state,
                                 verbose=verbose)
        self.algo.fit(trainset=self.trainset)

    def recommend(self, user_id, count=10):
        user_id = str(user_id)
        if len(self.top_n[user_id]) != 0:
            return np.array(self.top_n[user_id])[:, 0]
        predictions = []
        for track_id in self.scores.track.unique():
            predictions.append(self.algo.predict(user_id, track_id))
        for uid, iid, true_r, est, _ in predictions:
            self.top_n[uid].append((iid, est))

        for uid, user_ratings in self.top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            self.top_n[uid] = user_ratings[:count]

        return np.array(self.top_n[user_id])[:, 0]

    def recommend_all(self, user_ids=None, count=10):
        recomms = dict()
        for user_id in user_ids:
            recomms[user_id] = self.recommend(user_id, count)

        return recomms

    def predict(self, test_df):
        testset = [(row['user'], row['track'], row['score']) for i, row in test_df.iterrows()]
        predictions = self.algo.test(testset=testset)
        predictions_df = pd.DataFrame(predictions)
        return predictions_df


class ContentBasedRecommender:

    def __init__(self, track_tag,k=40):
        self.k = k
        self.track_tag = track_tag


    def fit(self, scores):
        #TODO track C_DURATION_MIN,C_DURATION_SEC,TIME_CREATED,PRICE can also be used for featrue build.

        # Compute item similarity matrix based on content attributes

        # Load up feature vectors for every track
        genre = getFeature('GENRE',self.track_tag)
        orchestration = getFeature('ORCHESTRATION',self.track_tag)
        #curation = bp.getFeature('CURATION')
        #mood = bp.getFeature('MOOD')
        #form = bp.getFeature('FORM')

        print("Computing content-based similarity matrix...")

        # Compute feature distance for every track combination as a 2x2 matrix


        self.total_unique_tracks = sorted(scores['track'].unique())
        self.total_user = scores.groupby(['user'])
        num_tracks = len(self.total_unique_tracks)
        self.similarities = np.zeros((num_tracks,num_tracks))
        for i in range(num_tracks):
            for j in range(i+1,num_tracks):
                sim = 0
                try:
                    genreSim = self.computeSimilarity(self.total_unique_tracks[i],self.total_unique_tracks[j],genre)
                    sim += sim + 0.2*genreSim
                except IndexError:
                    pass
                except ZeroDivisionError:
                    pass
                try:
                    orchSim = self.computeSimilarity(self.total_unique_tracks[i],self.total_unique_tracks[j],orchestration)
                    sim += sim + 0.8*orchSim
                except IndexError:
                    pass
                except ZeroDivisionError:
                    pass

                self.similarities[i,j] = sim
                self.similarities[j,i] = self.similarities[i,j]

        self.similarities = self.similarities /np.amax(self.similarities)

        return self

    def computeSimilarity(self, track1, track2, feature_dict):
        feature1 = feature_dict[track1]
        feature2 = feature_dict[track2]
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(feature1)):
            x = feature1[i]
            y = feature2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y

        return sumxy/math.sqrt(sumxx*sumyy)

    def recommend(self, user_id, count=None):
        track_ratings={}
        num_tracks = len(self.total_unique_tracks)
        for i in range(num_tracks):
            pred = self.predict(user_id,self.total_unique_tracks[i])
            track_ratings[self.total_unique_tracks[i]]=pred
        recom=Counter(track_ratings)

        result = [track for track, cnt in recom.most_common()]
        if count is not None:
            result = result[:count]
        return np.array(result)

    def recommend_all(self, user_ids=None, count=None):
        recomms=dict()
        for user_id in user_ids:
            recomms[user_id] = self.recommend(user_id, count)
        return recomms

    def predict(self, user_id, item_id):
        # Build up similarity scores between this item and everything the user rated
        neighbors = []

        user = self.total_user.get_group(user_id)
        user = user.reset_index().drop(['index'],axis=1)
        p=0
        item_i = self.total_unique_tracks.index(item_id)
        item_index=np.where(np.isin(self.total_unique_tracks,user['track']))[0]
        for rating in user['score']:
            try:
                featureSimilarity = self.similarities[item_i, item_index[p]]
                neighbors.append((featureSimilarity, rating))

            except ValueError:
                pass
            p+=1

        # Extract the top-K most-similar ratings
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # Compute average sim score of K neighbors weighted by user ratings
        simTotal = weightedSum = 0
        for (simScore, rating) in k_neighbors:
            if (simScore > 0):
                simTotal += simScore
                weightedSum += simScore * rating
        if (simTotal == 0):
            return -1

        predictedRating = round(weightedSum/simTotal,3)

        return predictedRating


class HybridRecommender:
    def __init__(self, recommenders):
        self.recommenders = recommenders

    def fit(self, scores):
        self.recommenders[0].fit(scores)
        self.recommenders[1].fit(scores)

    def recommend(self, user_id, count=None):
        scores = defaultdict(lambda: 0)
        for recommender in self.recommenders:
            recommendations = recommender.recommend(user_id)
            for i, track in enumerate(recommendations):
                scores[track] += 1 / (i + 1)
        result = [track for track, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        if count is not None:
            result = result[:count]
        return np.array(result)

    def recommend_all(self, user_ids=None, count=None):
        recomms = dict()
        for user_id in user_ids:
            recomms[user_id] = self.recommend(user_id, count)

        return recomms


class RandomRecommender:
    def fit(self, scores):
        self.tracks = np.unique(scores.track)

    def recommend(self, user_id, count=None):
        result = self.tracks.copy()
        np.random.shuffle(result)
        if count is not None:
            result = result[:count]
        return result

    def recommend_all(self, user_ids=None, count=None):
        recomms = dict()
        for user_id in user_ids:
            recomms[user_id] = self.recommend(user_id, count)

        return recomms

print('Loading data...')

track_like_df = pd.read_csv(data_path / 'track_like_expanded.csv')
track_like_df = track_like_df[['USER_ID', 'TRACK_ID']].assign(r=3)
track_like_df.columns = ['user', 'track', 'score']

track_download_df = pd.read_csv(data_path / 'track_download_expanded.csv')
track_download_df = track_download_df[['USER_ID', 'TRACK_ID']].assign(r=1)
track_download_df.columns = ['user', 'track', 'score']

track_purchase_df = pd.read_csv(data_path / 'track_purchase_expanded.csv')
track_purchase_df = track_purchase_df[['USER_ID', 'TRACK_ID']].assign(r=2)
track_purchase_df.columns = ['user', 'track', 'score']

track_info = pd.read_csv(data_path / 'track_infov2_df.csv', index_col=0)
track_tag = pd.read_csv(data_path / 'track_tag_df.csv', index_col=0)
user_info_df = pd.read_csv(data_path / 'user_info.csv')
user_info_df = user_info_df.sort_values('total_interactions', ascending=False)
track_interaction_info = pd.read_csv(data_path / 'track_info.csv', index_col=0)

print('Processing data...')

print('computing total interactions...')
track_interaction_info[
    'total_interactions'] = track_interaction_info.noDownloads + track_interaction_info.noPurchases + track_interaction_info.noLikes
user_info_df.fillna(0, inplace=True)
user_info_df['total_interactions'] = user_info_df['noAlbum_downloads'] + user_info_df['noAlbum_likes'] + user_info_df[
    'noAlbum_purchases'] + user_info_df['noTrack_downloads'] + user_info_df['noTrack_likes'] + user_info_df[
                                         'noTrack_purchases']

print('removing unnecessary track_tag records...')
target_orch = track_tag.groupby('TYPE_KEY').get_group("ORCHESTRATION").TAG_ID.value_counts().index[:50].values
target_tag = track_tag.groupby('TYPE_KEY').get_group("GENRE").TAG_ID.value_counts().index[:35].values
track_tag = track_tag.loc[track_tag.TAG_ID.isin(np.concatenate([target_tag, target_orch]))]

print('Preparing data...')
total = pd.concat([track_download_df, track_like_df, track_purchase_df])
total = total.groupby(['user', 'track']).score.sum().reset_index()
print('raw data records:', total.shape[0])

pop_recommender = PopularRecommender()
print('Fitting the model {}...'.format(pop_recommender.__class__.__name__))
print('number of target users:', total.user.nunique())
print('number of target tracks:', total.track.nunique())
pop_recommender.fit(total)

print('removing records containing tracks without info or tag...')
total = total.loc[total['track'].isin(track_tag['TRACK_ID'].unique())]
total = total.loc[total['track'].isin(track_info['TRACK_ID'].unique())]

min_interactions = 50
print('removing records containing users and tracks less than minimum number of interactions({})'.format(min_interactions))
target_users = user_info_df.loc[user_info_df['total_interactions'] >= min_interactions]['USER_ID'].values
target_tracks = track_interaction_info.loc[track_interaction_info['total_interactions'] >= min_interactions][
    'TRACK_ID'].values
total = total.loc[total['user'].isin(target_users)]
total = total.loc[total['track'].isin(target_tracks)]
total = total.sort_values("track")

track_tag_total = track_tag.loc[track_tag['TRACK_ID'].isin(total['track'].unique())]
track_tag_total = track_tag_total.sort_values(by=['TRACK_ID'])
track_info_total = track_info.loc[track_info['TRACK_ID'].isin(total['track'].unique())]
track_info_total = track_info_total.sort_values(by=['TRACK_ID'])

print('final data records:', total.shape[0])

recommender = Skl_CollaborativeRecommender(rank=20)
print('Fitting the model {}...'.format(recommender.__class__.__name__))
print('number of target users:', total.user.nunique())
print('number of target tracks:', total.track.nunique())
recommender = Skl_CollaborativeRecommender(rank=20,max_iter=100)
recommender.fit(total)

print('Making playlists...')
recc_user_ids = total.user.unique()
pop_user_ids = pd.concat([total.user,user_info_df.USER_ID]).drop_duplicates(keep=False).values
start = time.time()
sample_ids = recc_user_ids[:100]
recomms = recommender.recommend_all(sample_ids,10)
sample_ids = pop_user_ids[:100]
recomms.update(pop_recommender.recommend_all(sample_ids,10))
print('Recommendation time for {} users:'.format(len(recomms)),time.time()-start,'seconds')

print('Saving playlists to DB...')
start = time.time()
db = get_db()
coll = db['sample_mix_recomms']
docs = [{"USER_ID":int(x),"TRACK_ID":int(v)} for x in recomms for v in recomms[x]]
coll.insert_many(docs)
print('saving time:',time.time()-start,'seconds')

#      surprise prediction speedup by using algo.test and trainset.build_anti_testset()
