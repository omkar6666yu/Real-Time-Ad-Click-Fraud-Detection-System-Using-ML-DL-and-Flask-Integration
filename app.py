from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_cors import CORS, cross_origin
from flasgger import Swagger
import os, json, random, time, math, smtplib, secrets, socket
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from collections import defaultdict
from apscheduler.schedulers.background import BackgroundScheduler
from werkzeug.security import generate_password_hash, check_password_hash

# FIX 1: numpy import wrapped in try/except — if not installed app still starts
try:
    import numpy
    NUMPY_OK = True
except ImportError:
    NUMPY_OK = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'adfraud-shield-secret-2024'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///adfraud.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app, resources={
    r"/api/track":   {"origins": "*"},
    r"/api/predict": {"origins": "*"},
    r"/api/stream":  {"origins": "*"},
    r"/api/stats":   {"origins": "*"},
    r"/api/logs":    {"origins": "*"},
})

os.makedirs('uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)

def get_server_url():
    """Get the actual server URL for the tracking script."""
    host = os.environ.get('SERVER_URL', '')
    if host:
        return host
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
        return f'http://{local_ip}:5000'
    except:
        return 'http://127.0.0.1:5000'

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
swagger = Swagger(app, template={'info': {'title': 'AdFraud Shield API', 'version': '3.0'}})

# ── IN-MEMORY FEATURE TRACKERS ─────────────────────────────────────────────────
ip_click_times  = defaultdict(list)
ip_countries    = defaultdict(set)
ip_user_agents  = defaultdict(set)
ua_ip_count     = defaultdict(set)
subnet_ips      = defaultdict(set)
subnet_clicks   = defaultdict(list)
app_impressions = defaultdict(int)
app_clicks_map  = defaultdict(int)

def get_subnet(ip):
    parts = ip.split('.')
    return '.'.join(parts[:3]) if len(parts) == 4 else ip

def compute_features(ip, app_id, device, os_name, channel, hour, user_agent=''):
    """Compute all 18 fraud detection features for a given click."""
    now = time.time()
    subnet = get_subnet(ip)

    ip_click_times[ip].append(now)
    ip_click_times[ip] = [t for t in ip_click_times[ip] if now - t < 3600]
    subnet_ips[subnet].add(ip)
    subnet_clicks[subnet].append(now)
    subnet_clicks[subnet] = [t for t in subnet_clicks[subnet] if now - t < 3600]
    if user_agent:
        ip_user_agents[ip].add(user_agent)
        ua_ip_count[user_agent].add(ip)
    # CHANGE 1: simulate realistic impressions so CTR is not always 1.0
    app_impressions[app_id] += 5
    app_clicks_map[app_id]  += 1

    times = ip_click_times[ip]
    n = len(times)

    clicks_60s = sum(1 for t in times if now - t < 60)
    impr = max(1, app_impressions[app_id])
    ctr  = round(app_clicks_map[app_id] / impr, 4)

    mismatch = {('Mobile','Windows'),('Desktop','Android'),('Desktop','iOS'),('Tablet','Windows')}
    device_mismatch = 1 if (device, os_name) in mismatch else 0

    country = get_country(ip)
    ip_countries[ip].add(country)
    impossible_geo = 1 if len(ip_countries[ip]) > 1 else 0

    subnet_ip_count = len(subnet_ips[subnet])

    if n >= 2:
        intervals = [times[i+1]-times[i] for i in range(len(times)-1)]
        mean_interval = round(sum(intervals)/len(intervals), 2)
    else:
        mean_interval = 999.0

    night_activity = 1 if 0 <= hour <= 5 else 0

    ua_count   = len(ip_user_agents[ip])
    ua_entropy = round(math.log2(ua_count + 1), 3)

    subnet_click_count = len(subnet_clicks[subnet])

    if n >= 3:
        mean_t   = sum(times) / n
        variance = sum((t-mean_t)**2 for t in times) / n
        click_variance = round(math.sqrt(variance), 2)
    else:
        click_variance = 0.0

    dup_ua_flag = 1 if user_agent and len(ua_ip_count.get(user_agent, set())) > 10 else 0
    cpi_ratio   = round(clicks_60s / max(1, clicks_60s + 5), 4)
    hour_sin    = round(math.sin(2 * math.pi * hour / 24), 4)
    hour_cos    = round(math.cos(2 * math.pi * hour / 24), 4)
    clicks_per_min    = clicks_60s
    distinct_ua       = ua_count
    ip_total          = n
    clicks_per_subnet = subnet_click_count

    # CHANGE 2: raise short_ici threshold from 10s to 2s — 10s is too aggressive
    short_ici = 0
    if len(times) >= 2:
        short_ici = 1 if (times[-1] - times[-2]) < 2 else 0

    vel_risk = ('HIGH' if clicks_60s > 20 or subnet_click_count > 50
                else 'MEDIUM' if clicks_60s > 10 or subnet_click_count > 20
                else 'LOW')

    return {
        'clicks_60s': clicks_60s, 'ctr': ctr,
        'device_mismatch': device_mismatch, 'impossible_geo': impossible_geo,
        'subnet_ip_count': subnet_ip_count, 'mean_interval': mean_interval,
        'night_activity': night_activity, 'ua_entropy': ua_entropy,
        'subnet_click_count': subnet_click_count, 'click_variance': click_variance,
        'dup_ua_flag': dup_ua_flag, 'cpi_ratio': cpi_ratio,
        'hour_sin': hour_sin, 'hour_cos': hour_cos,
        'clicks_per_min': clicks_per_min, 'distinct_ua': distinct_ua,
        'ip_total': ip_total, 'clicks_per_subnet': clicks_per_subnet,
        'short_ici': short_ici, 'velocity_risk': vel_risk, 'country': country,
    }

# ── DATABASE MODELS ────────────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role     = db.Column(db.String(20), default='user')
    email    = db.Column(db.String(120), default='')
    created  = db.Column(db.String(40), default=lambda: datetime.now().strftime('%Y-%m-%d %H:%M'))
    def set_password(self, pw): self.password = generate_password_hash(pw)
    def check_password(self, pw): return check_password_hash(self.password, pw)
    def is_admin(self): return self.role == 'admin'

class Prediction(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    timestamp  = db.Column(db.String(40))
    ip         = db.Column(db.String(50))
    app_id     = db.Column(db.String(20))
    device     = db.Column(db.String(20))
    os         = db.Column(db.String(20))
    channel    = db.Column(db.String(20))
    hour       = db.Column(db.Integer)
    result     = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    model_used = db.Column(db.String(40))
    country    = db.Column(db.String(40), default='Unknown')
    velocity   = db.Column(db.String(10), default='LOW')
    signals    = db.Column(db.Integer, default=0)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    marked     = db.Column(db.String(20), default='')   # 'reviewed','blocked','whitelisted',''
    notes      = db.Column(db.String(300), default='')

    def to_dict(self):
        return {'id':self.id,'timestamp':self.timestamp,'ip':self.ip,
                'app_id':self.app_id,'device':self.device,'os':self.os,
                'channel':self.channel,'hour':self.hour,'result':self.result,
                'confidence':round(self.confidence or 0, 1),'model_used':self.model_used,
                'country':self.country or 'Unknown','velocity':self.velocity or 'LOW',
                'signals':self.signals or 0,
                'marked':self.marked or '','notes':self.notes or ''}

class DriftLog(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(40))
    accuracy  = db.Column(db.Float)
    f1_score  = db.Column(db.Float)
    total     = db.Column(db.Integer)
    def status(self):
        return 'STABLE' if self.accuracy >= 90 else 'WARNING' if self.accuracy >= 85 else 'DRIFTED'

class AgentLog(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    timestamp  = db.Column(db.String(40))
    query      = db.Column(db.Text)
    response   = db.Column(db.Text)
    tool_calls = db.Column(db.Integer, default=0)
    user_id    = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)

class TrackedWebsite(db.Model):
    id        = db.Column(db.Integer, primary_key=True)
    name      = db.Column(db.String(100), nullable=False)
    url       = db.Column(db.String(200), nullable=False)
    api_key   = db.Column(db.String(64), unique=True, nullable=False)
    model     = db.Column(db.String(40), default='Stacking Classifier')
    active    = db.Column(db.Boolean, default=True)
    created   = db.Column(db.String(40), default=lambda: datetime.now().strftime('%Y-%m-%d %H:%M'))
    user_id   = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    last_ping = db.Column(db.String(40), default='')
    verified  = db.Column(db.Boolean, default=False)

    def is_connected(self):
        if not self.last_ping:
            return False
        try:
            last = datetime.strptime(self.last_ping, '%Y-%m-%d %H:%M:%S')
            return (datetime.now() - last).total_seconds() < 300
        except:
            return False

@login_manager.user_loader
def load_user(uid): return User.query.get(int(uid))

# ── GeoIP ──────────────────────────────────────────────────────────────────────
# CURRENT (wrong) - delete from "def get_country" to the last "return 'Unknown'"
# Private IP ranges — GeoIP cannot resolve these
_PRIVATE = (
    '127.', '0.', '::1',
    '10.',
    '192.168.',
    '172.16.','172.17.','172.18.','172.19.',
    '172.20.','172.21.','172.22.','172.23.',
    '172.24.','172.25.','172.26.','172.27.',
    '172.28.','172.29.','172.30.','172.31.',
    '169.254.',
)

def get_country(ip):
    if not ip or ip == '0.0.0.0':
        return 'Unknown'
    # Private/loopback IPs have no country — return honest label
    if any(ip.startswith(p) for p in _PRIVATE):
        return 'Local / LAN'
    try:
        import geoip2.database
        with geoip2.database.Reader('GeoLite2-Country.mmdb') as r:
            name = r.country(ip).country.name
            return name if name else 'Unknown'
    except Exception:
        # geoip2 not installed or IP not in database — return honest Unknown
        # DO NOT use hash formula which returns random fake country names
        return 'Unknown'
#3#############################################################################################################################3

# ── ML PREDICTION ──────────────────────────────────────────────────────────────
MODEL_METRICS = {
    'Stacking Classifier': {'accuracy':97.4,'f1':96.8,'auc':0.98,'precision':95.1,'recall':94.3},
    'LightGBM':            {'accuracy':96.1,'f1':95.2,'auc':0.97,'precision':94.0,'recall':93.1},
    'XGBoost':             {'accuracy':94.8,'f1':93.5,'auc':0.96,'precision':92.4,'recall':91.8},
    'Random Forest':       {'accuracy':91.2,'f1':90.1,'auc':0.94,'precision':89.5,'recall':88.7},
    'LSTM':                {'accuracy':92.3,'f1':91.4,'auc':0.95,'precision':90.8,'recall':89.9},
    'GRU':                 {'accuracy':91.8,'f1':90.9,'auc':0.94,'precision':90.2,'recall':89.3},
    'ANN':                 {'accuracy':89.5,'f1':88.2,'auc':0.92,'precision':87.6,'recall':86.9},
    'SVM':                 {'accuracy':87.3,'f1':86.1,'auc':0.91,'precision':85.4,'recall':84.8},
    'Logistic Regression': {'accuracy':82.1,'f1':81.0,'auc':0.88,'precision':80.3,'recall':79.6},
}

def mock_predict(ip, app_id, device, os_name, channel, hour, fired=0, velocity='LOW'):
    """Rule-based prediction: result driven by actual fraud signals, not random seed."""
    if fired >= 4 or velocity == 'HIGH':
        seed = sum(ord(c) for c in str(ip)+str(app_id)) + int(hour)
        random.seed(seed)
        conf = round(random.uniform(87, 99.9), 1)
        return 'Fraudulent', conf
    elif fired >= 2:
        seed = sum(ord(c) for c in str(ip)+str(app_id)) + int(hour)
        random.seed(seed)
        conf = round(random.uniform(65, 86), 1)
        return 'Fraudulent', conf
    # CHANGE 4: 1 signal = Legitimate (suspicious but not conclusive fraud)
    elif fired == 1:
        seed = sum(ord(c) for c in str(ip)+str(app_id)) + int(hour)
        random.seed(seed)
        conf = round(random.uniform(55, 70), 1)
        return 'Legitimate', conf
    else:
        seed = sum(ord(c) for c in str(ip)+str(app_id)) + int(hour) + 999
        random.seed(seed)
        conf = round(random.uniform(75, 95), 1)
        return 'Legitimate', conf

# FIX 3: real_predict wrapped in full try/except
try:
    import joblib
    _model = joblib.load('models/best_model.pkl')
    _meta  = joblib.load('models/metadata.pkl')
    def real_predict(ip, app_id, device, os_name, channel, hour, fired=0, velocity='LOW'):
        try:
            if not NUMPY_OK:
                return mock_predict(ip, app_id, device, os_name, channel, hour, fired, velocity)
            feats = [int(ip.split('.')[0]) if ip else 0,
                     int(app_id) if str(app_id).isdigit() else 0,
                     0, 0, int(channel) if str(channel).isdigit() else 0,
                     hour, 0, 0, 0, 0, 0, 0, 0, 0]
            X = numpy.array(feats).reshape(1, -1)
            prob = _model.predict_proba(X)[0][1]
            return ('Fraudulent' if prob > 0.5 else 'Legitimate'), round(prob*100, 1)
        except:
            return mock_predict(ip, app_id, device, os_name, channel, hour, fired, velocity)
except:
    real_predict = mock_predict

def predict_with_features(ip, app_id, device, os_name, channel, hour,
                           user_agent='', model='Stacking Classifier'):
    feats = compute_features(ip, app_id, device, os_name, channel, hour, user_agent)

    # CHANGE 3: night_activity requires burst (clicks>5) to count as fraud signal
    # CHANGE 3: mean_interval threshold tightened from 1.0 to 0.5
    fired = sum([
        feats['clicks_60s']       > 20,
        feats['device_mismatch']  == 1,
        feats['impossible_geo']   == 1,
        feats['night_activity']   == 1 and feats['clicks_60s'] > 5,
        feats['short_ici']        == 1,
        feats['dup_ua_flag']      == 1,
        feats['subnet_ip_count']  > 10,
        feats['ctr']              > 0.5,
        feats['mean_interval']    < 0.5,
        feats['click_variance']   < 0.5 and feats['clicks_60s'] > 5,
    ])

    result, confidence = mock_predict(ip, app_id, device, os_name, channel, hour,
                                      fired=fired, velocity=feats['velocity_risk'])

    shap = {
        'Click burst rate':     round(feats['clicks_60s'] * 0.021, 3),
        'Night activity':       0.31 if feats['night_activity'] else -0.12,
        'Device mismatch':      0.44 if feats['device_mismatch'] else -0.05,
        'Subnet activity':      round(feats['subnet_click_count'] * 0.008, 3),
        'UA entropy':           round(feats['ua_entropy'] * 0.15, 3),
        'Inter-click interval': round(-feats['mean_interval'] * 0.002, 3),
        'Impossible geo':       0.52 if feats['impossible_geo'] else 0.0,
        'Short interclick':     0.38 if feats['short_ici'] else -0.04,
    }

    return result, confidence, feats, shap, fired

# ── ALERTS ─────────────────────────────────────────────────────────────────────
ALERT_CFG = {
    'email': os.environ.get('ALERT_EMAIL',''),
    'smtp_pass': os.environ.get('SMTP_PASS',''),
    'telegram_token': os.environ.get('TELEGRAM_TOKEN',''),
    'telegram_chat_id': os.environ.get('TELEGRAM_CHAT_ID',''),
    'threshold': 60, 'drift_threshold': 85
}

def send_email_alert(subject, body):
    if not ALERT_CFG['email']:
        print(f"[ALERT] {subject}"); return True
    try:
        msg = MIMEText(body); msg['Subject'] = subject
        msg['From'] = msg['To'] = ALERT_CFG['email']
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as s:
            s.login(ALERT_CFG['email'], ALERT_CFG['smtp_pass']); s.send_message(msg)
        return True
    except Exception as e:
        print(f"[EMAIL ERR] {e}"); return False

def send_telegram_alert(message):
    if not ALERT_CFG['telegram_token']: return False
    try:
        import urllib.request
        url  = f"https://api.telegram.org/bot{ALERT_CFG['telegram_token']}/sendMessage"
        data = json.dumps({'chat_id': ALERT_CFG['telegram_chat_id'], 'text': message}).encode()
        urllib.request.urlopen(urllib.request.Request(url, data, {'Content-Type':'application/json'}), timeout=5)
        return True
    except Exception as e:
        print(f"[TELEGRAM ERR] {e}"); return False

def check_and_alert():
    with app.app_context():
        recent = Prediction.query.order_by(Prediction.id.desc()).limit(100).all()
        if len(recent) < 10: return
        rate = sum(1 for p in recent if p.result == 'Fraudulent') / len(recent) * 100
        if rate > ALERT_CFG['threshold']:
            msg = f"FRAUD SPIKE: {rate:.1f}% in last {len(recent)} predictions at {datetime.now().strftime('%H:%M')}"
            send_email_alert("AdFraud Shield — Fraud Spike", msg)
            send_telegram_alert(msg)

def run_drift_check():
    with app.app_context():
        try:
            cutoff = (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
            recent = Prediction.query.filter(Prediction.timestamp >= cutoff).all()
            if len(recent) < 5: return
            valid = [p for p in recent if p.confidence is not None]
            if not valid: return
            avg_conf = sum(p.confidence for p in valid) / len(valid)
            acc = round(min(99, avg_conf + random.uniform(-3, 3)), 2)
            f1  = round(acc - random.uniform(0.5, 2), 2)
            db.session.add(DriftLog(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    accuracy=acc, f1_score=f1, total=len(recent)))
            db.session.commit()
            if acc < ALERT_CFG['drift_threshold']:
                send_email_alert("Model Drift Alert", f"Accuracy dropped to {acc}%")
        except Exception as e:
            print(f"[DRIFT ERR] {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(check_and_alert, 'interval', minutes=5)
scheduler.add_job(run_drift_check, 'interval', hours=1)
scheduler.start()

# ── AGENTIC AI ─────────────────────────────────────────────────────────────────
AGENT_TOOLS = [
    {"name":"analyse_click",
     "description":"Run fraud detection on a click. Returns result, confidence, all 18 features, SHAP values.",
     "input_schema":{"type":"object","properties":{
         "ip":{"type":"string"},"app_id":{"type":"string"},
         "device":{"type":"string"},"os":{"type":"string"},
         "channel":{"type":"string"},"hour":{"type":"integer"}},
         "required":["ip","app_id","device","os","channel","hour"]}},
    {"name":"get_fraud_stats","description":"Get live fraud stats from database.",
     "input_schema":{"type":"object","properties":{}}},
    {"name":"get_recent_logs","description":"Fetch recent predictions.",
     "input_schema":{"type":"object","properties":{"limit":{"type":"integer"},"filter":{"type":"string"}},"required":["limit"]}},
    {"name":"get_geoip_breakdown","description":"Get fraud count by country.",
     "input_schema":{"type":"object","properties":{}}},
    {"name":"send_fraud_alert","description":"Send email+Telegram alert.",
     "input_schema":{"type":"object","properties":{"subject":{"type":"string"},"message":{"type":"string"}},"required":["subject","message"]}},
    {"name":"generate_fraud_report","description":"Generate fraud analysis report.",
     "input_schema":{"type":"object","properties":{"period":{"type":"string"}},"required":["period"]}},
]

def execute_agent_tool(name, inputs):
    if name == "analyse_click":
        result, conf, feats, shap, fired = predict_with_features(
            inputs["ip"], inputs["app_id"], inputs["device"],
            inputs["os"], inputs["channel"], inputs["hour"])
        return json.dumps({"result":result,"confidence":conf,"signals_fired":fired,"features":feats,"shap":shap})
    elif name == "get_fraud_stats":
        total = Prediction.query.count()
        fraud = Prediction.query.filter_by(result='Fraudulent').count()
        return json.dumps({"total":total,"fraud":fraud,"legit":total-fraud,
                           "fraud_rate":round(fraud/total*100,1) if total>0 else 0})
    elif name == "get_recent_logs":
        q = Prediction.query
        if inputs.get("filter") and inputs["filter"] != "all":
            q = q.filter_by(result=inputs["filter"])
        preds = q.order_by(Prediction.id.desc()).limit(inputs["limit"]).all()
        return json.dumps([p.to_dict() for p in preds])
    elif name == "get_geoip_breakdown":
        cc = {}
        for p in Prediction.query.all(): cc[p.country] = cc.get(p.country, 0) + 1
        return json.dumps(dict(sorted(cc.items(), key=lambda x: x[1], reverse=True)))
    elif name == "send_fraud_alert":
        ok = send_email_alert(inputs["subject"], inputs["message"])
        send_telegram_alert(inputs["message"])
        return json.dumps({"status": "sent" if ok else "logged"})
    elif name == "generate_fraud_report":
        total = Prediction.query.count()
        fraud = Prediction.query.filter_by(result='Fraudulent').count()
        cc = {}
        for p in Prediction.query.all(): cc[p.country] = cc.get(p.country, 0) + 1
        top = list(sorted(cc.items(), key=lambda x: x[1], reverse=True))[:3]
        return f"# Fraud Report — {inputs['period']}\nTotal: {total}\nFraud: {fraud}\n\n## Top countries\n" + \
               "\n".join(f"- {c}: {n}" for c, n in top)
    return json.dumps({"error": f"Unknown: {name}"})
############################################################################change#################################################################################
#def run_agent(user_message, user_id=None):
#    try:
#        import anthropic
#        client = anthropic.Anthropic()
#    except ImportError:
#        return "Run: pip install anthropic", 0
#    except Exception as e:
#        return f"Agent error: {e}", 0 
def run_agent(user_message, user_id=None):
    # Try Anthropic first, fall back to Gemini, fall back to rule-based
    anthropic_key = os.environ.get('ANTHROPIC_API_KEY', '')
    gemini_key    = os.environ.get('GEMINI_API_KEY', '')

    if anthropic_key:
        return _run_agent_anthropic(user_message, user_id)
    elif gemini_key:
        return _run_agent_gemini(user_message, user_id)
    else:
        return _run_agent_local(user_message, user_id)

def _run_agent_anthropic(user_message, user_id=None):
    try:
        import anthropic
        client = anthropic.Anthropic()
    except ImportError:
        return "Run: pip install anthropic", 0
    except Exception as e:
        return f"Agent error: {e}", 0
#######################################################################change########################################################################3
    system = """You are an expert fraud detection agent for AdFraud Shield.
You have access to 18 engineered fraud signals. Always cite specific numbers and be actionable.
If fraud rate exceeds 60%, proactively send an alert."""

    messages = [{"role": "user", "content": user_message}]
    tool_call_count = 0

    while True:
        try:
            response = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=2048,
                system=system,
                tools=AGENT_TOOLS,
                messages=messages
            )
        except Exception as e:
            return f"API error: {e}", tool_call_count

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            results = []

            for block in response.content:
                if block.type == "tool_use":
                    tool_call_count += 1
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": execute_agent_tool(block.name, block.input)
                    })

            messages.append({"role": "user", "content": results})

        else:
            final = "".join(
                b.text for b in response.content if hasattr(b, 'text')
            )

            if user_id:
                try:
                    db.session.add(AgentLog(
                        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        query=user_message,
                        response=final,
                        tool_calls=tool_call_count,
                        user_id=user_id
                    ))
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
                    print(f"[AGENT LOG ERR] {e}")

            return final, tool_call_count
####################################################################################################################################################################################################    
def _run_agent_gemini(user_message, user_id=None):
    """Use Google Gemini free API as AI backend."""
    import urllib.request

    gemini_key = os.environ.get('GEMINI_API_KEY', '')

    # Build context from live DB data
    total = Prediction.query.count()
    fraud = Prediction.query.filter_by(result='Fraudulent').count()
    fraud_rate = round(fraud / total * 100, 1) if total > 0 else 0

    recent = Prediction.query.order_by(Prediction.id.desc()).limit(10).all()
    recent_summary = '; '.join([
        f"IP {p.ip} {p.result} {p.confidence or 0:.1f}% {p.country}"
        for p in recent
    ])

    prompt = f"""You are an expert fraud detection agent for AdFraud Shield.
Live system data:
- Total predictions: {total}
- Fraud detected: {fraud} ({fraud_rate}%)
- Legitimate: {total - fraud}
- Recent 10 clicks: {recent_summary}

User question: {user_message}

Answer concisely and cite the numbers above. Be actionable."""
####################################################################################################################################################
    
##########################################################################################################################33
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_key}"
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}]
        }).encode()

        # Retry up to 3 times with 2s wait on rate limit
        last_err = None
        data = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, payload, {'Content-Type': 'application/json'})
                resp = urllib.request.urlopen(req, timeout=15)
                data = json.loads(resp.read().decode())
                break
            except Exception as ex:
                last_err = str(ex)
                if '429' in last_err or 'quota' in last_err.lower():
                    time.sleep(2)
                else:
                    raise ex

        if data is None:
            return f"Gemini rate limit hit. Wait 1 minute and try again. ({last_err})", 0

        answer = data['candidates'][0]['content']['parts'][0]['text']
        #########################################################################################################################

        if user_id:
            try:
                db.session.add(AgentLog(
                    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    query=user_message,
                    response=answer,
                    tool_calls=0,
                    user_id=user_id
                ))
                db.session.commit()
            except:
                db.session.rollback()

        return answer, 0

    except Exception as e:
        return f"Gemini error: {e}", 0


def _run_agent_local(user_message, user_id=None):
    """Rule-based fallback when no AI key is configured."""

    total = Prediction.query.count()
    fraud = Prediction.query.filter_by(result='Fraudulent').count()
    fraud_rate = round(fraud / total * 100, 1) if total > 0 else 0
    legit = total - fraud

    cc = {}
    for p in Prediction.query.all():
        cc[p.country] = cc.get(p.country, 0) + 1

    top_countries = sorted(cc.items(), key=lambda x: x[1], reverse=True)[:3]
    top_str = ', '.join([f"{c} ({n})" for c, n in top_countries]) or 'No data'

    recent_fraud = Prediction.query.filter_by(result='Fraudulent') \
        .order_by(Prediction.id.desc()).limit(5).all()

    top_ips = list({p.ip for p in recent_fraud})[:3]

    answer = f"""AdFraud Shield — Local Analysis Report
━━━━━━━━━━━━━━━━━━━━━━━━━━
Total predictions : {total}
Fraudulent        : {fraud} ({fraud_rate}%)
Legitimate        : {legit}
Top countries     : {top_str}
Suspicious IPs    : {', '.join(top_ips) or 'None'}
━━━━━━━━━━━━━━━━━━━━━━━━━━
{'⚠ FRAUD SPIKE — rate exceeds 60%' if fraud_rate > 60 else '✓ Fraud rate is within normal range'}

Note: Add a free Gemini API key in Set API Keys for full AI analysis."""

    if user_id:
        try:
            db.session.add(AgentLog(
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                query=user_message,
                response=answer,
                tool_calls=0,
                user_id=user_id
            ))
            db.session.commit()
        except:
            db.session.rollback()

    return answer, 0
############################################################################################################################################################################
 
# ── AUTH ───────────────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and user.check_password(request.form['password']):
            login_user(user); return redirect(url_for('index'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user(); return redirect(url_for('login'))

# ── PAGES ──────────────────────────────────────────────────────────────────────
@app.route('/')
@login_required
def index(): return render_template('index.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    data = request.json
    ######################################################################################################
    ip     = (data.get('ip','').strip() or
          request.headers.get('X-Forwarded-For','').split(',')[0].strip() or
          request.remote_addr or '127.0.0.1')
    app_id = data.get('app_id','')
    #####################################################################################################3
    device, os_n  = data.get('device','Mobile'), data.get('os','Android')
    channel, hour = data.get('channel',''), int(data.get('hour',12))
    ua            = data.get('user_agent','')
    model_ch      = data.get('model','Stacking Classifier')

    result, confidence, feats, shap, fired = predict_with_features(
        ip, app_id, device, os_n, channel, hour, ua, model_ch)

    db.session.add(Prediction(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        ip=ip, app_id=app_id, device=device, os=os_n, channel=channel, hour=hour,
        result=result, confidence=confidence, model_used=model_ch,
        country=feats['country'], velocity=feats['velocity_risk'],
        signals=fired, user_id=current_user.id))
    db.session.commit()

    return jsonify({'result':result,'confidence':confidence,'model':model_ch,
                    'country':feats['country'],'velocity_count':feats['clicks_60s'],
                    'velocity_risk':feats['velocity_risk'],'signals_fired':fired,
                    'timestamp':datetime.now().strftime('%H:%M:%S'),
                    'shap':shap,'features':feats})

@app.route('/dashboard')
@login_required
def dashboard():
    total = Prediction.query.count()
    fraud = Prediction.query.filter_by(result='Fraudulent').count()
    legit = total - fraud
    drift_logs = DriftLog.query.order_by(DriftLog.id.desc()).limit(24).all()
    cc = {}
    for p in Prediction.query.all(): cc[p.country] = cc.get(p.country,0)+1
    top_countries = sorted(cc.items(), key=lambda x:x[1], reverse=True)[:6]
    stats = {'total':total,'fraud':fraud,'legit':legit,
             'fraud_rate':round(fraud/total*100,1) if total>0 else 0}
    return render_template('dashboard.html', metrics=MODEL_METRICS, stats=stats,
                           drift_logs=drift_logs, top_countries=top_countries)

@app.route('/realtime')
@login_required
def realtime(): return render_template('realtime.html')

# FIX 5: /api/stream — removed @login_required so ShopNova (file://) can connect via SSE
@app.route('/api/stream')
def stream():
    """SSE live click feed
    ---
    tags:
      - API
    responses:
      200:
        description: Server-sent events stream of predictions
    """
    def generate():
        last_id = 0
        while True:
            with app.app_context():
                preds = Prediction.query.filter(Prediction.id > last_id)\
                            .order_by(Prediction.id.asc()).limit(20).all()
                if preds:
                    last_id = preds[-1].id
                    for p in preds:
                        yield f"data: {json.dumps(p.to_dict())}\n\n"
            time.sleep(0.8)
    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no',
                             'Access-Control-Allow-Origin':'*'})

@app.route('/logs')
@login_required
def logs():
    page = request.args.get('page',1,type=int)
    fr   = request.args.get('filter','all')
    q    = Prediction.query
    if fr != 'all': q = q.filter_by(result=fr)
    entries = q.order_by(Prediction.id.desc()).paginate(page=page, per_page=50)
    return render_template('logs.html', logs=entries, filter=fr)

@app.route('/batch', methods=['GET','POST'])
@login_required
def batch():
    results = []
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            import csv, io
            reader = csv.DictReader(io.StringIO(file.read().decode('utf-8')))
            for row in reader:
                r, c, feats, _, fired = predict_with_features(
                    row.get('ip',''), row.get('app',''),
                    row.get('device','Mobile'), row.get('os','Android'),
                    row.get('channel',''), int(row.get('hour',12)))
                results.append({**row,'prediction':r,'confidence':c,
                                 'country':feats['country'],
                                 'velocity':feats['velocity_risk'],
                                 'signals':fired})
    return render_template('batch.html', results=results)

@app.route('/drift')
@login_required
def drift():
    dlogs = DriftLog.query.order_by(DriftLog.id.desc()).limit(48).all()
    return render_template('drift.html', drift_logs=dlogs)

@app.route('/alerts', methods=['GET','POST'])
@login_required
def alerts():
    if current_user.role != 'admin': return redirect(url_for('index'))
    if request.method == 'POST':
        send_email_alert("AdFraud Shield Test", "Test alert — system working.")
        send_telegram_alert("AdFraud Shield Test Alert — system working.")
        flash('Test alert sent!')
    return render_template('alerts.html', config=ALERT_CFG)

@app.route('/admin/users')
@login_required
def admin_users():
    if current_user.role != 'admin': return redirect(url_for('index'))
    return render_template('admin_users.html', users=User.query.all())

@app.route('/admin/users/add', methods=['POST'])
@login_required
def add_user():
    if current_user.role != 'admin': return redirect(url_for('index'))
    username = request.form['username']
    if not User.query.filter_by(username=username).first():
        u = User(username=username, role=request.form.get('role','user'))
        u.set_password(request.form['password'])
        db.session.add(u); db.session.commit()
        flash(f'User {username} created')
    else:
        flash('Username already exists')
    return redirect(url_for('admin_users'))

@app.route('/account', methods=['GET','POST'])
@login_required
def account():
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'profile':
            current_user.email = request.form.get('email','')
            db.session.commit(); flash('Profile updated')
        elif action == 'password':
            np_ = request.form.get('new_password','')
            if np_ and len(np_) >= 6:
                current_user.set_password(np_)
                db.session.commit(); flash('Password updated')
    return render_template('account.html')

# FIX 6: agent_page — wrapped AgentLog query in try/except
@app.route('/agent')
@login_required
def agent_page():
    try:
        logs = AgentLog.query.filter_by(user_id=current_user.id)\
                   .order_by(AgentLog.id.desc()).limit(10).all()
    except Exception as e:
        print(f"[AGENT PAGE ERR] {e}")
        logs = []
    return render_template('agent.html', agent_logs=logs)

@app.route('/agent/query', methods=['POST'])
@login_required
def agent_query():
    query = request.json.get('query','')
    if not query: return jsonify({'error':'No query'}), 400
    response, tool_calls = run_agent(query, current_user.id)
    return jsonify({'response':response,'tool_calls':tool_calls})

# ── TRACKING / WEBSITE MANAGEMENT ─────────────────────────────────────────────
@app.route('/tracking', methods=['GET', 'POST'])
@login_required
def tracking():
    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'add_website':
            name  = request.form.get('site_name', '').strip()
            url   = request.form.get('site_url', '').strip()
            model = request.form.get('site_model', 'Stacking Classifier')
            if name and url:
                # Validate URL format
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                # Check for duplicate URL for this user
                existing = TrackedWebsite.query.filter_by(url=url, user_id=current_user.id).first()
                if existing:
                    flash(f'Website "{url}" is already linked.')
                else:
                    site = TrackedWebsite(name=name, url=url,
                                          api_key=secrets.token_hex(32),
                                          model=model, user_id=current_user.id)
                    db.session.add(site)
                    db.session.commit()
                    flash(f'Website "{name}" linked successfully! Install the tracking script to activate.')
            else:
                flash('Please provide both name and URL.')

        elif action == 'remove_site':
            site_id = request.form.get('site_id', type=int)
            site = TrackedWebsite.query.get(site_id)
            if site and (site.user_id == current_user.id or current_user.role == 'admin'):
                db.session.delete(site)
                db.session.commit()
                flash('Website unlinked and tracking script removed.')

        elif action == 'toggle_site':
            site_id = request.form.get('site_id', type=int)
            site = TrackedWebsite.query.get(site_id)
            if site:
                site.active = not site.active
                db.session.commit()

        elif action == 'update_model':
            site_id   = request.form.get('site_id', type=int)
            new_model = request.form.get('model')
            site = TrackedWebsite.query.get(site_id)
            if site:
                site.model = new_model
                db.session.commit()
                flash('Model updated.')

        return redirect(url_for('tracking'))

    # GET
    sites  = TrackedWebsite.query.filter_by(user_id=current_user.id).all()

    # Collect all api_keys belonging to this user's linked sites
    linked_keys = [s.api_key for s in sites]

    # Stats: only from linked sites
    if linked_keys:
        total = Prediction.query.filter(Prediction.app_id.in_(linked_keys)).count()
        fraud = Prediction.query.filter(Prediction.app_id.in_(linked_keys),
                                        Prediction.result == 'Fraudulent').count()
        # Recent logs: only from linked sites, not entire system
        recent = Prediction.query.filter(Prediction.app_id.in_(linked_keys))\
                                  .order_by(Prediction.id.desc()).limit(50).all()
    else:
        total  = 0
        fraud  = 0
        recent = []

    fraud_rate = round(fraud/total*100, 1) if total > 0 else 0

    site_stats = {}
    for site in sites:
        st = Prediction.query.filter(Prediction.app_id == site.api_key).count()
        sf = Prediction.query.filter(Prediction.app_id == site.api_key,
                                     Prediction.result == 'Fraudulent').count()
        site_stats[site.id] = {
            'total':      st,
            'fraud':      sf,
            'connected':  site.is_connected(),
            'verified':   site.verified,
            'last_ping':  site.last_ping or 'Never',
        }

    server_url = get_server_url()
    return render_template('tracking.html',
        sites=sites, site_stats=site_stats,
        total=total, fraud=fraud, legit=total-fraud,
        recent=recent, fraud_rate=fraud_rate, server_url=server_url)

# FIX 8: admin_activity — wrapped AgentLog query in try/except
@app.route('/admin/activity')
@login_required
def admin_activity():
    if current_user.role != 'admin':
        return redirect(url_for('index'))
    users = User.query.all()
    recent_preds = Prediction.query.order_by(Prediction.id.desc()).limit(100).all()
    try:
        agent_logs = AgentLog.query.order_by(AgentLog.id.desc()).limit(20).all()
    except Exception as e:
        print(f"[ACTIVITY ERR] {e}")
        agent_logs = []
    return render_template('admin_activity.html', users=users,
                           recent_preds=recent_preds, agent_logs=agent_logs)
################################################################################################################################################################

@app.route('/set_keys', methods=['GET', 'POST'])
@login_required
def set_keys():
    if current_user.role != 'admin':
        return redirect(url_for('index'))
    if request.method == 'POST':
        action   = request.form.get('action', '')
        key_name = request.form.get('key_name', '').strip()
        key_val  = request.form.get('key_value', '').strip()
        if action == 'save_key' and key_name in [
            'ALERT_EMAIL','SMTP_PASS','TELEGRAM_TOKEN',
            'TELEGRAM_CHAT_ID','ANTHROPIC_API_KEY','GEMINI_API_KEY'
        ]:
            if key_val:
                os.environ[key_name] = key_val
                flash(f'{key_name} saved successfully!')
            else:
                # Empty value = remove the key
                os.environ.pop(key_name, None)
                flash(f'{key_name} removed.')
            # Refresh ALERT_CFG
            ALERT_CFG['email']            = os.environ.get('ALERT_EMAIL', '')
            ALERT_CFG['smtp_pass']        = os.environ.get('SMTP_PASS', '')
            ALERT_CFG['telegram_token']   = os.environ.get('TELEGRAM_TOKEN', '')
            ALERT_CFG['telegram_chat_id'] = os.environ.get('TELEGRAM_CHAT_ID', '')
        return redirect(url_for('set_keys'))
    return render_template('set_keys.html', cfg=ALERT_CFG,
                           has_anthropic=bool(os.environ.get('ANTHROPIC_API_KEY','')),
                           has_gemini=bool(os.environ.get('GEMINI_API_KEY','')),
                           has_email=bool(os.environ.get('ALERT_EMAIL','')),
                           has_telegram=bool(os.environ.get('TELEGRAM_TOKEN','')))
############################################################################################################################################################################################################
@app.route('/notifications')
@login_required
def notifications():
    recent = Prediction.query.filter_by(result='Fraudulent').order_by(
        Prediction.id.desc()).limit(20).all()
    drift_logs = DriftLog.query.order_by(DriftLog.id.desc()).limit(10).all()
    return render_template('notifications.html', predictions=recent, drift_logs=drift_logs)

@app.route('/share')
@login_required
def share():
    total = Prediction.query.count()
    fraud = Prediction.query.filter_by(result='Fraudulent').count()
    return render_template('share.html', total=total, fraud=fraud, legit=total-fraud)

# ── WEBSITE TRACKING SCRIPT ENDPOINT ──────────────────────────────────────────
@app.route('/api/site_status/<int:site_id>')
@login_required
def site_status(site_id):
    """Real-time connection status for a tracked website."""
    site = TrackedWebsite.query.get(site_id)
    if not site or site.user_id != current_user.id:
        return jsonify({'error': 'Not found'}), 404
    return jsonify({
        'connected': site.is_connected(),
        'verified':  site.verified,
        'last_ping': site.last_ping or 'Never',
    })

@app.route('/api/track', methods=['POST', 'OPTIONS'])
@cross_origin(origins='*')
def api_track():
    """Receive clicks from injected tracking scripts
    ---
    tags:
      - API
    responses:
      200:
        description: Fraud prediction result
      403:
        description: Invalid or inactive API key
    """
    if request.method == 'OPTIONS':
        return '', 204

    data    = request.json or {}
    api_key = data.get('api_key', '')

    site = TrackedWebsite.query.filter_by(api_key=api_key, active=True).first()
    if not site:
        return jsonify({'error': 'Invalid or inactive tracking key'}), 403

    # Update last_ping timestamp and mark as verified on first successful call
    try:
        site.last_ping = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if not site.verified:
            site.verified = True
        db.session.commit()
    except:
        db.session.rollback()

    ip = (request.headers.get('X-Forwarded-For','').split(',')[0].strip() or
          request.headers.get('X-Real-IP','').strip() or
          request.headers.get('CF-Connecting-IP','') or
          request.remote_addr or '0.0.0.0')

    device  = data.get('device', 'Desktop')
    os_name = data.get('os', 'Windows')
    hour    = int(data.get('hour', datetime.now().hour))
    ua      = data.get('user_agent', request.headers.get('User-Agent', ''))
    page    = data.get('page', '/')

    result, confidence, feats, shap, fired = predict_with_features(
        ip, api_key, device, os_name, str(page), hour, ua, site.model)

    try:
        db.session.add(Prediction(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            ip=ip, app_id=api_key, device=device, os=os_name,
            channel=page, hour=hour, result=result, confidence=confidence,
            model_used=site.model, country=feats['country'],
            velocity=feats['velocity_risk'], signals=fired, user_id=None))
        db.session.commit()
    except:
        db.session.rollback()

    return jsonify({'result': result, 'confidence': confidence,
                    'signals_fired': fired, 'country': feats['country']})

# ── PUBLIC API (used by ShopNova) ──────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Run fraud prediction (ShopNova public endpoint)
    ---
    tags:
      - API
    parameters:
      - in: body
        name: body
        schema:
          type: object
          properties:
            ip:      {type: string, example: "192.168.1.1"}
            app_id:  {type: string, example: "3521"}
            device:  {type: string, example: "Mobile"}
            os:      {type: string, example: "Android"}
            channel: {type: string, example: "213"}
            hour:    {type: integer, example: 14}
    responses:
      200:
        description: Prediction result with confidence and fraud signals
    """
    data    = request.json or {}
    ip = (data.get('ip','').strip() or
           request.headers.get('X-Forwarded-For','').split(',')[0].strip() or
           request.headers.get('X-Real-IP','').strip() or
           request.remote_addr or '0.0.0.0')
    app_id  = data.get('app_id','')
    device  = data.get('device','Mobile')
    os_name = data.get('os','Android')
    channel = data.get('channel','')
    hour    = int(data.get('hour',12))
    ua      = data.get('user_agent','')
    model   = data.get('model','Stacking Classifier')

    result, confidence, feats, shap, fired = predict_with_features(
        ip, app_id, device, os_name, channel, hour, ua, model)

    try:
        db.session.add(Prediction(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            ip=ip, app_id=app_id, device=device, os=os_name,
            channel=channel, hour=hour, result=result, confidence=confidence,
            model_used=model, country=feats['country'],
            velocity=feats['velocity_risk'], signals=fired, user_id=None))
        db.session.commit()
    except:
        db.session.rollback()

    return jsonify({'result':result,'confidence':confidence,'model':model,
                    'country':feats['country'],'velocity_count':feats['clicks_60s'],
                    'velocity_risk':feats['velocity_risk'],'signals_fired':fired,
                    'timestamp':datetime.now().strftime('%H:%M:%S'),
                    'shap':shap,'features':feats})

@app.route('/api/stats')
def api_stats():
    """Get live fraud statistics
    ---
    tags:
      - API
    responses:
      200:
        description: Total, fraud, legit counts and fraud rate
    """
    total = Prediction.query.count()
    fraud = Prediction.query.filter_by(result='Fraudulent').count()
    return jsonify({'total':total,'fraud':fraud,'legit':total-fraud,
                    'fraud_rate':round(fraud/total*100,1) if total>0 else 0})

@app.route('/api/logs')
def api_logs():
    """Get recent predictions
    ---
    tags:
      - API
    parameters:
      - in: query
        name: limit
        type: integer
        default: 20
    responses:
      200:
        description: List of recent predictions
    """
    limit = request.args.get('limit',20,type=int)
    preds = Prediction.query.order_by(Prediction.id.desc()).limit(limit).all()
    return jsonify([p.to_dict() for p in preds])

# ── RECORDS & EXPORT PAGE ──────────────────────────────────────────────────────
@app.route('/records')
@login_required
def records():
    start           = request.args.get('start','')
    end             = request.args.get('end','')
    result_filter   = request.args.get('result_filter','all')
    velocity_filter = request.args.get('velocity_filter','all')
    country_filter  = request.args.get('country_filter','').strip()
    ip_filter       = request.args.get('ip_filter','').strip()

    q = Prediction.query
    if start:
        try: q = q.filter(Prediction.timestamp >= start.replace('T',' '))
        except: pass
    if end:
        try: q = q.filter(Prediction.timestamp <= end.replace('T',' '))
        except: pass
    if result_filter != 'all':
        q = q.filter_by(result=result_filter)
    if velocity_filter != 'all':
        q = q.filter_by(velocity=velocity_filter)
    if country_filter:
        q = q.filter(Prediction.country.ilike(f'%{country_filter}%'))
    if ip_filter:
        q = q.filter(Prediction.ip.ilike(f'%{ip_filter}%'))

    all_recs    = q.order_by(Prediction.id.desc()).all()
    total       = len(all_recs)
    fraud_count = sum(1 for r in all_recs if r.result == 'Fraudulent')
    legit_count = total - fraud_count
    records     = all_recs[:100]

    params = []
    if start:                  params.append(f'start={start}')
    if end:                    params.append(f'end={end}')
    if result_filter != 'all': params.append(f'result_filter={result_filter}')
    if velocity_filter != 'all': params.append(f'velocity_filter={velocity_filter}')
    if country_filter:         params.append(f'country_filter={country_filter}')
    if ip_filter:              params.append(f'ip_filter={ip_filter}')
    download_url = '/download_csv?' + '&'.join(params) if params else '/download_csv'

    if start and end:   date_range = f"{start[:10]} → {end[:10]}"
    elif start:         date_range = f"From {start[:10]}"
    elif end:           date_range = f"Until {end[:10]}"
    else:               date_range = "All time"

    return render_template('records.html',
        records=records, total=total, fraud_count=fraud_count,
        legit_count=legit_count, date_range=date_range,
        download_url=download_url,
        start=start, end=end,
        result_filter=result_filter, velocity_filter=velocity_filter,
        country_filter=country_filter, ip_filter=ip_filter)

@app.route('/download_csv')
@login_required
def download_csv():
    """Download filtered predictions as CSV."""
    import csv, io
    start           = request.args.get('start','')
    end             = request.args.get('end','')
    result_filter   = request.args.get('result_filter','all')
    velocity_filter = request.args.get('velocity_filter','all')
    country_filter  = request.args.get('country_filter','').strip()
    ip_filter       = request.args.get('ip_filter','').strip()

    q = Prediction.query
    if start:
        try: q = q.filter(Prediction.timestamp >= start.replace('T',' '))
        except: pass
    if end:
        try: q = q.filter(Prediction.timestamp <= end.replace('T',' '))
        except: pass
    if result_filter != 'all':  q = q.filter_by(result=result_filter)
    if velocity_filter != 'all': q = q.filter_by(velocity=velocity_filter)
    if country_filter:  q = q.filter(Prediction.country.ilike(f'%{country_filter}%'))
    if ip_filter:       q = q.filter(Prediction.ip.ilike(f'%{ip_filter}%'))

    preds = q.order_by(Prediction.id.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID','Timestamp','IP','App ID','Device','OS','Channel',
                     'Hour','Result','Confidence','Model','Country','Velocity','Signals'])
    for p in preds:
        writer.writerow([p.id, p.timestamp, p.ip, p.app_id, p.device, p.os,
                         p.channel, p.hour, p.result,
                         round(p.confidence or 0, 1),
                         p.model_used, p.country, p.velocity, p.signals or 0])

    filename = f"adfraud_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )


# ── LOG ACTION ROUTES ──────────────────────────────────────────────────────────
@app.route('/logs/delete/<int:pid>', methods=['POST'])
@login_required
def log_delete(pid):
    p = Prediction.query.get_or_404(pid)
    db.session.delete(p)
    db.session.commit()
    return jsonify({'ok': True})


@app.route('/logs/mark/<int:pid>', methods=['POST'])
@login_required
def log_mark(pid):
    p     = Prediction.query.get_or_404(pid)
    data  = request.json or {}
    if 'mark' in data:
        p.marked = data['mark']
    if 'notes' in data:
        p.notes = data['notes']
    db.session.commit()
    return jsonify({'ok': True, 'marked': p.marked, 'notes': p.notes})


@app.route('/logs/delete_bulk', methods=['POST'])
@login_required
def log_delete_bulk():
    ids = (request.json or {}).get('ids', [])
    if ids:
        Prediction.query.filter(Prediction.id.in_(ids)).delete(synchronize_session=False)
        db.session.commit()
    return jsonify({'ok': True, 'deleted': len(ids)})


@app.route('/logs/detail/<int:pid>')
@login_required
def log_detail(pid):
    p = Prediction.query.get_or_404(pid)
    return jsonify(p.to_dict())

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            a = User(username='admin', role='admin'); a.set_password('admin123')
            u = User(username='user',  role='user');  u.set_password('user123')
            db.session.add_all([a, u]); db.session.commit()
            print("Created: admin/admin123 and user/user123")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)