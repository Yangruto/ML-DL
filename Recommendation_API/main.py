from flask import Flask, request, Response, send_from_directory
from herp.derp.xgboost_for_recommendation import Recommendation, FEATUE_TEMPLATE

app = Flask(__name__)

# def ip_check():
#     ip = request.remote_addr
#     if ip not in white_list:
#         print(ip)
#         flask.abort(403)
    
@app.route('/recommendation/threshold/<string:product_category>/<int:max_recency>/<int:min_open_prob>', methods=["GET", "POST"])
def threshold_recommendation(product_category:str, max_recency:int, min_open_prob:int):
    # ip_check()
    recommendation = Recommendation()
    recommendation.MODEL_FEATURES = [i for i in filter(lambda x: (product_category in x or x in ('recency')), FEATUE_TEMPLATE)]
    recommendation.load_model(f'./data/model/{product_category}_model.pkl')
    recommend_list = recommendation.edm_recommend('threshold', max_recency=max_recency, min_open_prob=min_open_prob)
    return recommend_list.to_dict()

@app.route('/recommendation/headcount/<string:product_category>/<int:max_recency>/<int:num_people>', methods=["GET", "POST"])
def headcount_recommendation(product_category:str, max_recency:int, num_people:int):
    # ip_check()
    recommendation = Recommendation()
    recommendation.MODEL_FEATURES = [i for i in filter(lambda x: (product_category in x or x in ('recency')), FEATUE_TEMPLATE)]
    recommendation.load_model(f'./data/model/{product_category}_model.pkl')
    recommend_list = recommendation.edm_recommend('headcount', max_recency=max_recency, headcount=num_people)
    recommend_list.to_excel('./result/result.xlsx', index=False)
    return send_from_directory('./result/', 'result.xlsx')

@app.route('/ip-check', methods=["GET"])
def my_endpoint():
    ip = request.remote_addr
    return ip

@app.route('/', methods=['GET'])
def main_page():
    return 'Welcome, this is My ML API.'

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9850)

"""
gunicorn -w 1 -b 0.0.0.0:9850 wsgi:app

/etc/nginx/sites-available
172.17.168.206 host
172.17.168.193 my ip

// stop Nginx
$ service nginx stop

// restart Nginx
$ service nginx restart

// check out Nginx access log
$ tail -f /var/log/nginx/access.log

// check out Nginx error log
$ tail -f /var/log/nginx/error.log

// get Port:80 status 
$ netstat -lpn |grep 80

// get Nginx status
$ ps -ef | grep nginx

// delete Port 
$ kill PID 
"""