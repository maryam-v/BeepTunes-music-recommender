from flask import (
    Blueprint, g, request, url_for, Response, flash
)
import ast
from werkzeug.exceptions import abort

from flaskr.db import get_db

bp = Blueprint('entities', __name__,url_prefix='/api/v1/entities')

@bp.route('/tracks', methods=['POST'])
def new_track():
    '''parameters:
        TRACK_ID,TIME_CREATED,PRICE,PUBLISH_DATE,C_DURATION_MIN,C_DURATION_SEC,ALBUM_ID,ARTIST_IDS,TAG_IDS,TYPE_KEYS
        '''
    parameters = ["TRACK_ID","TIME_CREATED","PRICE","PUBLISH_DATE","C_DURATION_MIN","C_DURATION_SEC","ALBUM_ID","ARTIST_IDS","TAG_IDS","TYPE_KEYS"]
    db_client = get_db()
    db = db_client['beeptunes']
    status = 200
    message = 'track saved successfully.'

    for par in parameters:
        if par not in request.form:
            status = 400
            message = par + " required."
            break


    if status == 200:
        track_id = request.form['TRACK_ID']
        album_id = request.form['ALBUM_ID']
        artist_ids = ast.literal_eval(request.form['ARTIST_IDS'])
        tag_ids = ast.literal_eval(request.form['TAG_IDS'])
        tag_keys = ast.literal_eval(request.form['TYPE_KEYS'])

        if len(tag_ids) == len(tag_keys):

            coll = db['track_artist']
            for artist_id in artist_ids:
                coll.insert_one({"TRACK_ID":track_id,"ARTIST_ID":artist_id})

            coll = db['track_tag']
            for i,_ in enumerate(tag_ids):
                coll.insert_one({"TRACK_ID":track_id,"TAG_ID":tag_ids[i],"TYPE_KEY":tag_keys[i]})

            coll = db['track_info_v2']
            coll.insert_one({"TRACK_ID":track_id,"TIME_CREATED":request.form['TIME_CREATED'],"PRICE":request.form['PRICE'],
                            "PUBLISH_DATE":request.form['PUBLISH_DATE'],"C_DURATION_MIN":request.form["C_DURATION_MIN"],
                            "C_DURATION_SEC":request.form["C_DURATION_SEC"],"ALBUM_ID":request.form["ALBUM_ID"]})

            coll = db['album_artist']
            for artist_id in artist_ids:
                cur = coll.find({"ALBUM_ID":int(album_id),"ARTIST_ID":int(artist_id)})
                if cur.count() == 0:
                    coll.insert_one({"ALBUM_ID":int(album_id),"ARTIST_ID":int(artist_id)})

        else :
            status = 400
            message = 'tag_ids and tag_keys length must match.'

    flash(message)
    return Response(message,status=status,mimetype='application/json')

@bp.route('/tracks/likes', methods=['POST'])
def new_track_like(track_id=None,user_id=None):
    '''parameters:
        USER_ID, TRACK_ID, C_DATE
        '''
    db_client = get_db()
    db = db_client['beeptunes']
    coll = db['track_like']
    status = 200
    message = 'record save successfully.'
    if 'USER_ID' not in request.form and user_id is None:
        message = 'USER_ID required.'
        status = 400
    elif 'TRACK_ID' not in request.form and track_id is None:
        message = 'TRACK_ID required.'
        status = 400

    if status==200:
        if user_id is None:
            user_id = int(request.form['USER_ID'])
        if track_id is None:
            track_id = int(request.form['TRACK_ID'])
        c_date = None
        if 'C_DATE' in request.form:
            c_date = request.form['C_DATE']
        record = {"USER_ID":user_id,"TRACK_ID":track_id,"C_DATE":c_date}
        coll.insert_one(record)
    flash(message)
    return Response(message,status=status,mimetype='application/json')

@bp.route('/tracks/downloads', methods=['POST'])
def new_track_download():
    '''parameters:
        USER_ID, TRACK_ID
        '''
    db_client = get_db()
    db = db_client['beeptunes']
    coll = db['track_download']
    status = 200
    message = 'record save successfully.'
    if 'USER_ID' not in request.form:
        message = 'USER_ID required.'
        status = 400
    elif 'TRACK_ID' not in request.form:
        message = 'TRACK_ID required.'
        status = 400

    if status==200:
        user_id = int(request.form['USER_ID'])
        track_id = int(request.form['TRACK_ID'])
        c_date = None
        if 'C_DATE' in request.form:
            c_date = request.form['C_DATE']
        record = {"USER_ID":user_id,"TRACK_ID":track_id,"C_DATE":c_date}
        coll.insert_one(record)
    flash(message)
    return Response(message,status=status,mimetype='application/json')

@bp.route('/tracks/purchases', methods=['POST'])
def new_track_purchase():
    '''parameters:
        USER_ID, TRACK_ID
        '''
    db_client = get_db()
    db = db_client['beeptunes']
    coll = db['track_purchase']
    status = 200
    message = 'record save successfully.'
    if 'USER_ID' not in request.form:
        message = 'USER_ID required.'
        status = 400
    elif 'TRACK_ID' not in request.form:
        message = 'TRACK_ID required.'
        status = 400

    if status==200:
        user_id = int(request.form['USER_ID'])
        track_id = int(request.form['TRACK_ID'])
        c_date = None
        if 'C_DATE' in request.form:
            c_date = request.form['C_DATE']
        record = {"USER_ID":user_id,"TRACK_ID":track_id,"C_DATE":c_date}
        coll.insert_one(record)
    flash(message)
    return Response(message,status=status,mimetype='application/json')

@bp.route('/albums/likes', methods=['POST'])
def new_album_like():
    '''parameters:
        USER_ID, ALBUM_ID
        '''
    db_client = get_db()
    db = db_client['beeptunes']
    coll = db['album_like']
    status = 200
    message = 'record save successfully.'
    if 'USER_ID' not in request.form:
        message = 'USER_ID required.'
        status = 400
    elif 'ALBUM_ID' not in request.form:
        message = 'ALBUM_ID required.'
        status = 400

    if status==200:
        user_id = int(request.form['USER_ID'])
        album_id = int(request.form['ALBUM_ID'])
        c_date = None
        if 'C_DATE' in request.form:
            c_date = request.form['C_DATE']
        record = {"USER_ID":user_id,"ALBUM_ID":album_id,"C_DATE":c_date}
        coll.insert_one(record)
        coll = db['track_info_v2']
        cur = coll.find({'ALBUM_ID':album_id})
        for record in cur:
            new_track_like(record['TRACK_ID'],user_id)

    flash(message)
    return Response(message,status=status,mimetype='application/json')

@bp.route('/albums/downloads', methods=['POST'])
def new_album_download():
    '''parameters:
        USER_ID, ALBUM_ID
        '''
    db_client = get_db()
    db = db_client['beeptunes']
    coll = db['album_download']
    status = 200
    message = 'record save successfully.'
    if 'USER_ID' not in request.form:
        message = 'USER_ID required.'
        status = 400
    elif 'ALBUM_ID' not in request.form:
        message = 'ALBUM_ID required.'
        status = 400

    if status==200:
        user_id = int(request.form['USER_ID'])
        album_id = int(request.form['ALBUM_ID'])
        c_date = None
        if 'C_DATE' in request.form:
            c_date = request.form['C_DATE']
        record = {"USER_ID":user_id,"ALBUM_ID":album_id,"C_DATE":c_date}
        coll.insert_one(record)
    flash(message)
    return Response(message,status=status,mimetype='application/json')

@bp.route('/albums/purchases', methods=['POST'])
def new_album_purchase():
    '''parameters:
        USER_ID, ALBUM_ID
        '''
    db_client = get_db()
    db = db_client['beeptunes']
    coll = db['album_purchase']
    status = 200
    message = 'record save successfully.'
    if 'USER_ID' not in request.form:
        message = 'USER_ID required.'
        status = 400
    elif 'ALBUM_ID' not in request.form:
        message = 'ALBUM_ID required.'
        status = 400

    if status==200:
        user_id = int(request.form['USER_ID'])
        album_id = int(request.form['ALBUM_ID'])
        c_date = None
        if 'C_DATE' in request.form:
            c_date = request.form['C_DATE']
        record = {"USER_ID":user_id,"ALBUM_ID":album_id,"C_DATE":c_date}
        coll.insert_one(record)
    flash(message)
    return Response(message,status=status,mimetype='application/json')


@bp.route('/artists/likes', methods=['POST'])
def new_artist_like():
    '''parameters:
        USER_ID, ARTIST_ID
        '''
    db_client = get_db()
    db = db_client['beeptunes']
    coll = db['artist_like']
    status = 200
    message = 'record save successfully.'
    if 'USER_ID' not in request.form:
        message = 'USER_ID required.'
        status = 400
    elif 'ARTIST_ID' not in request.form:
        message = 'ARTIST_ID required.'
        status = 400

    if status==200:
        user_id = int(request.form['USER_ID'])
        artist_id = int(request.form['ARTIST_ID'])
        c_date = None
        if 'C_DATE' in request.form:
            c_date = request.form['C_DATE']
        record = {"USER_ID":user_id,"ARTIST_ID":track_id,"C_DATE":c_date}
        coll.insert_one(record)
    flash(message)
    return Response(message,status=status,mimetype='application/json')
