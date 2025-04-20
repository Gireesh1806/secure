import os
import stripe
from dotenv import load_dotenv
import numpy as np
from PIL import Image as img
import cv2
import exifread
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from wtforms import StringField, PasswordField, BooleanField, SubmitField, FileField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import pyotp
import qrcode
import io
import base64
from tensorflow.keras.models import load_model
import jwt
import requests
import re

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
instance_path = os.path.join(os.path.dirname(__file__), 'instance')
os.makedirs(instance_path, exist_ok=True)

# Determine environment (development or production)
ENV = os.getenv('FLASK_ENV', 'production')  # Default to development if not set
IS_PRODUCTION = ENV == 'production'

app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY', 'your-secret-key'),
    SQLALCHEMY_DATABASE_URI='postgresql://security_k124_user:JajMBFVjs2oFdXDYGewK2uEN9VYkEMUE@dpg-d026ik3uibrs73aiqci0-a/security_k124',
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    UPLOAD_FOLDER='static/uploads',
    ALLOWED_EXTENSIONS={'jpg', 'jpeg'},
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    JWT_SECRET_KEY=os.getenv('JWT_SECRET_KEY', 'your-jwt-secret-key'),
    RECAPTCHA_SITE_KEY=os.getenv('RECAPTCHA_SITE_KEY') if IS_PRODUCTION else None,
    RECAPTCHA_SECRET_KEY=os.getenv('RECAPTCHA_SECRET_KEY') if IS_PRODUCTION else None
)

# Stripe configuration
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
STRIPE_PUBLISHABLE_KEY = os.getenv('STRIPE_PUBLISHABLE_KEY')

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
csrf = CSRFProtect(app)
login_manager.login_view = 'login'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Models
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    mfa_secret = db.Column(db.String(32))
    mfa_enabled = db.Column(db.Boolean, default=False)
    last_login = db.Column(db.DateTime)
    images = db.relationship('Image', backref='user', lazy=True)
    subscriptions = db.relationship('Subscription', backref='user', lazy=True)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def is_premium_user(self):
        active_subscription = Subscription.query.filter_by(user_id=self.id, active=True).first()
        if active_subscription and active_subscription.end_date > datetime.utcnow():
            return True
        if active_subscription and active_subscription.end_date <= datetime.utcnow():
            active_subscription.active = False
            db.session.commit()
        return False

class Image(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    original_filename = db.Column(db.String(128), nullable=False)
    filename = db.Column(db.String(128), nullable=False)
    filepath = db.Column(db.String(256), nullable=False)
    ela_filepath = db.Column(db.String(256))
    analysis_result = db.Column(db.Text)
    weather_result = db.Column(db.Text)
    location = db.Column(db.String(256))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    is_outdoor = db.Column(db.Boolean, default=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)

class Subscription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    plan_name = db.Column(db.String(64), nullable=False)
    price = db.Column(db.Float, nullable=False)
    start_date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    end_date = db.Column(db.DateTime, nullable=False)
    active = db.Column(db.Boolean, default=True)
    stripe_subscription_id = db.Column(db.String(255))

class Config(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    free_image_limit = db.Column(db.Integer, default=5)

# Image processing functions
def prepare_image_for_ela(image_path):
    original = img.open(image_path).convert('RGB')
    temp_path = 'temp.jpg'
    original.save(temp_path, 'JPEG', quality=90)
    temp = img.open(temp_path)
    ela_image = img.new('RGB', original.size)
    for x in range(original.size[0]):
        for y in range(original.size[1]):
            r1, g1, b1 = original.getpixel((x, y))
            r2, g2, b2 = temp.getpixel((x, y))
            ela_image.putpixel((x, y), (
                abs(r1 - r2) * 2,
                abs(g1 - g2) * 2,
                abs(b1 - b2) * 2
            ))
    os.remove(temp_path)
    img_data = cv2.imread(image_path)
    img_data = cv2.resize(img_data, (128, 128))
    np_img = np.expand_dims(img_data, axis=0) / 255.0
    return np_img, ela_image

def image_coordinates(image_path):
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
        date_time_str = str(tags.get('EXIF DateTimeOriginal', ''))
        date_time = datetime.strptime(date_time_str, '%Y:%m:%d %H:%M:%S') if date_time_str else datetime.now()
        lat = tags.get('GPS GPSLatitude')
        lon = tags.get('GPS GPSLongitude')
        if lat and lon:
            lat = sum(float(x)/y for x, y in lat.values) / len(lat.values)
            lon = sum(float(x)/y for x, y in lon.values) / len(lon.values)
            return date_time, lat, lon, True
        return date_time, None, None, False
    except Exception:
        return datetime.now(), None, None, False

def get_weather(date_time, lat, lon):
    geolocator = Nominatim(user_agent="image_tampering_detection")
    location = geolocator.reverse((lat, lon))
    return location.address if location else "Unknown", date_time.strftime('%Y-%m-%d'), "Clear"

# Forms
class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember = BooleanField('Remember Me')
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(3, 64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=12)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    recaptcha = StringField('reCAPTCHA', validators=[DataRequired()] if IS_PRODUCTION else [])
    submit = SubmitField('Register')

    def validate_username(self, username):
        if User.query.filter_by(username=username.data).first():
            raise ValidationError('Username already taken.')

    def validate_email(self, email):
        if User.query.filter_by(email=email.data).first():
            raise ValidationError('Email already registered.')

    def validate_password(self, password):
        if not re.match(r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{12,}$', password.data):
            raise ValidationError('Password must be at least 12 characters long and include at least one uppercase letter, one lowercase letter, one number, and one special character (@$!%*?&).')

class MFAForm(FlaskForm):
    token = StringField('MFA Code', validators=[DataRequired(), Length(6, 6)])
    submit = SubmitField('Verify')

class ImageForm(FlaskForm):
    image = FileField('Image', validators=[DataRequired()])
    is_outdoor = BooleanField('Outdoor Image')
    submit = SubmitField('Analyze')

# User loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# reCAPTCHA verification (only in production)
def verify_recaptcha(token):
    if not IS_PRODUCTION:
        return True  # Bypass in development
    response = requests.post(
        'https://www.google.com/recaptcha/api/siteverify',
        data={
            'secret': app.config['RECAPTCHA_SECRET_KEY'],
            'response': token
        }
    )
    result = response.json()
    return result.get('success', False) and result.get('score', 0) >= 0.5

# Routes with current_year added to all render_template calls
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html', current_year=datetime.utcnow().year)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            if IS_PRODUCTION:
                token = jwt.encode({
                    'user_id': user.id,
                    'exp': datetime.utcnow() + timedelta(hours=24)
                }, app.config['JWT_SECRET_KEY'], algorithm='HS256')
                session['jwt_token'] = token
            session['mfa_user_id'] = user.id
            return redirect(url_for('mfa_verify'))
        flash('Invalid email or password.', 'danger')
    return render_template('login.html', form=form, current_year=datetime.utcnow().year)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    form = RegisterForm()
    if form.validate_on_submit():
        if IS_PRODUCTION and not verify_recaptcha(form.recaptcha.data):
            flash('reCAPTCHA verification failed. Please try again.', 'danger')
            return render_template('register.html', form=form, recaptcha_site_key=app.config['RECAPTCHA_SITE_KEY'], is_production=IS_PRODUCTION, current_year=datetime.utcnow().year)
        user = User(
            username=form.username.data,
            email=form.email.data,
            password_hash=generate_password_hash(form.password.data),
            mfa_secret=pyotp.random_base32()
        )
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form, recaptcha_site_key=app.config['RECAPTCHA_SITE_KEY'], is_production=IS_PRODUCTION, current_year=datetime.utcnow().year)

@app.route('/mfa/setup', methods=['GET', 'POST'])
@login_required
def mfa_setup():
    if current_user.mfa_enabled:
        flash('MFA is already enabled.', 'info')
        return redirect(url_for('dashboard'))
    form = MFAForm()
    if form.validate_on_submit():
        if pyotp.TOTP(current_user.mfa_secret).verify(form.token.data):
            current_user.mfa_enabled = True
            db.session.commit()
            flash('MFA enabled successfully.', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid MFA code.', 'danger')
    uri = pyotp.TOTP(current_user.mfa_secret).provisioning_uri(
        current_user.email, issuer_name="Image Tampering Detection")
    qr = qrcode.make(uri)
    buffer = io.BytesIO()
    qr.save(buffer)
    qr_code = base64.b64encode(buffer.getvalue()).decode()
    return render_template('mfa.html', form=form, qr_code=qr_code, setup=True, current_year=datetime.utcnow().year)

@app.route('/mfa/verify', methods=['GET', 'POST'])
def mfa_verify():
    user_id = session.get('mfa_user_id')
    if not user_id:
        return redirect(url_for('login'))
    user = User.query.get(user_id)
    if not user:
        session.pop('mfa_user_id', None)
        return redirect(url_for('login'))

    if not user.mfa_enabled:
        login_user(user)
        user.last_login = datetime.utcnow()
        db.session.commit()
        session.pop('mfa_user_id', None)
        flash('Logged in successfully. Consider enabling MFA for extra security.', 'success')
        return redirect(url_for('dashboard'))

    form = MFAForm()
    if form.validate_on_submit():
        if pyotp.TOTP(user.mfa_secret).verify(form.token.data):
            login_user(user)
            user.last_login = datetime.utcnow()
            db.session.commit()
            session.pop('mfa_user_id', None)
            flash('MFA verified successfully.', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid MFA code.', 'danger')
    return render_template('mfa.html', form=form, setup=False, current_year=datetime.utcnow().year)

@app.route('/logout')
@login_required
def logout():
    if IS_PRODUCTION:
        session.pop('jwt_token', None)
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if IS_PRODUCTION:
        token = session.get('jwt_token')
        if not token:
            logout_user()
            flash('Session expired. Please log in again.', 'warning')
            return redirect(url_for('login'))
        try:
            jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            logout_user()
            flash('Session expired. Please log in again.', 'warning')
            return redirect(url_for('login'))
        except jwt.InvalidTokenError:
            logout_user()
            flash('Invalid session. Please log in again.', 'warning')
            return redirect(url_for('login'))

    form = ImageForm()
    config = Config.query.first() or Config(free_image_limit=5)
    if form.validate_on_submit():
        if not current_user.is_premium_user() and len(current_user.images) >= config.free_image_limit:
            flash('Free user limit reached. Please subscribe.', 'warning')
            return redirect(url_for('subscription'))

        file = form.image.data
        if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']):
            flash('Only JPG/JPEG images allowed.', 'danger')
            return redirect(url_for('dashboard'))

        filename = secure_filename(file.filename)
        unique_filename = f"{datetime.now().timestamp()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        image = Image(
            user_id=current_user.id,
            original_filename=filename,
            filename=unique_filename,
            filepath=filepath,
            is_outdoor=form.is_outdoor.data
        )

        model = load_model('models/model_ela.h5')
        np_img, ela_img = prepare_image_for_ela(filepath)
        prediction = model.predict(np_img, verbose=0)
        class_ela = ['Real', 'Tampered']
        predicted_class = class_ela[np.argmax(prediction[0])]
        confidence = round(np.max(prediction[0]) * 100)
        image.analysis_result = f"Model indicates {confidence}% confidence that image is {predicted_class}"

        ela_filename = f"ela_{unique_filename}"
        ela_filepath = os.path.join(app.config['UPLOAD_FOLDER'], ela_filename)
        ela_img.save(ela_filepath)
        image.ela_filepath = ela_filepath

        if form.is_outdoor.data:
            date_time, lat, lon, is_valid = image_coordinates(filepath)
            if is_valid and lat and lon:
                location, date, weather = get_weather(date_time, lat, lon)
                image.weather_result = f"Image taken at {location} on {date} with {weather}"
                image.latitude = lat
                image.longitude = lon
                image.location = location

        db.session.add(image)
        db.session.commit()
        flash('Image analyzed successfully.', 'success')

    images = Image.query.filter_by(user_id=current_user.id).order_by(Image.upload_date.desc()).limit(10).all()
    return render_template('dashboard.html', form=form, images=images, is_premium=current_user.is_premium_user(), current_year=datetime.utcnow().year)

@app.route('/subscription')
@login_required
def subscription():
    plans = [
        {'name': 'Weekly', 'price': 4.99, 'duration': '7 days', 'duration_days': 7},
        {'name': 'Monthly', 'price': 12.99, 'duration': '30 days', 'duration_days': 30},
        {'name': 'Yearly', 'price': 99.99, 'duration': '365 days', 'duration_days': 365}
    ]
    return render_template('subscription.html', plans=plans, is_premium=current_user.is_premium_user(), key=STRIPE_PUBLISHABLE_KEY, current_year=datetime.utcnow().year)

@app.route('/create-checkout-session', methods=['POST'])
@login_required
def create_checkout_session():
    plans = {
        'Weekly': {'price': 4.99, 'duration': timedelta(days=7)},
        'Monthly': {'price': 12.99, 'duration': timedelta(days=30)},
        'Yearly': {'price': 99.99, 'duration': timedelta(days=365)}
    }

    plan_name = request.form.get('plan_name')
    if plan_name not in plans:
        flash('Invalid plan selected.', 'danger')
        return redirect(url_for('subscription'))

    if current_user.is_premium_user():
        flash('You already have an active subscription.', 'info')
        return redirect(url_for('dashboard'))

    try:
        plan = plans[plan_name]
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'unit_amount': int(plan['price'] * 100),
                    'product_data': {
                        'name': f'{plan_name} Subscription',
                    },
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=url_for('subscribe_success', plan_name=plan_name, _external=True),
            cancel_url=url_for('subscription', _external=True),
            metadata={'user_id': current_user.id, 'plan_name': plan_name}
        )
        return jsonify({'id': checkout_session.id})
    except Exception as e:
        flash(f'Error creating checkout session: {str(e)}', 'danger')
        return redirect(url_for('subscription'))

@app.route('/subscribe/<plan_name>')
@login_required
def subscribe(plan_name):
    flash('Please use the payment gateway to subscribe.', 'warning')
    return redirect(url_for('subscription'))

@app.route('/subscribe-success/<plan_name>')
@login_required
def subscribe_success(plan_name):
    plans = {
        'Weekly': {'price': 4.99, 'duration': timedelta(days=7)},
        'Monthly': {'price': 12.99, 'duration': timedelta(days=30)},
        'Yearly': {'price': 99.99, 'duration': timedelta(days=365)}
    }

    if plan_name not in plans:
        flash('Invalid plan selected.', 'danger')
        return redirect(url_for('subscription'))

    if current_user.is_premium_user():
        flash('You already have an active subscription.', 'info')
        return redirect(url_for('dashboard'))

    plan = plans[plan_name]
    subscription = Subscription(
        user_id=current_user.id,
        plan_name=plan_name,
        price=plan['price'],
        start_date=datetime.utcnow(),
        end_date=datetime.utcnow() + plan['duration'],
        active=True,
        stripe_subscription_id=None
    )
    db.session.add(subscription)
    db.session.commit()

    flash(f'Subscribed to {plan_name} plan successfully!', 'success')
    return redirect(url_for('dashboard'))

# Initialize database
with app.app_context():
    db.create_all()
    if not Config.query.first():
        db.session.add(Config(free_image_limit=5))
        db.session.commit()

if __name__ == '__main__':
    app.run(port=5000, debug=True)
