from flask import (
    Blueprint, g, request, url_for, jsonify, Response
)
from werkzeug.exceptions import abort
import numpy as np
from flaskr.db import get_db

bp = Blueprint('recommendation', __name__, url_prefix='/api/v1/recommendation')

@bp.route('/playlists/discovery', methods=['GET'])
def discovery_recomm():
    '''parameters:
        USER_ID, Playlist_size
        '''
    db_client = get_db()
    db = db_client['beeptunes']
    coll = db['sample_mix_recomms']
    status = 200
    message = None
    if 'USER_ID' not in request.form:
        message = 'USER_ID required.'
        status = 400
    elif 'Playlist_size' not in request.form:
        message = 'Playlist_size required.'
        status = 400

    if status == 200:
        cur = coll.find({"USER_ID":int(request.form['USER_ID'])})
        count = cur.count()
        pl_size = int(request.form['Playlist_size'])
        recomms = []
        for record in cur:
            recomms.append(record['TRACK_ID'])
        if pl_size < count:
            recomms = recomms[:pl_size]
        np.random.shuffle(recomms)
        response = dict()
        response['TRACK_ID'] = recomms
        response['size'] = len(recomms)
        response['available'] = count
        response['message'] = 'discovery playlist retrieved successfully.'
        return jsonify(response)

    return Response(message,status=status,mimetype='application/json')

@bp.route('/playlists/mix', methods=['GET'])
def mix_recomm():
    db_client = get_db()
    # TODO
