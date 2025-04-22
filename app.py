import os
import stripe
from dotenv import load_dotenv
import numpy as np
from PIL import Image as img
import cv2
import exifread
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
import jwt
import requests
import re
import logging

# Configure logging to output to both file and console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log script start and process ID
logger.debug(f"Script starting... Process ID: {os.getpid()}")

# Load environment variables
logger.debug("Loading environment variables...")
load_dotenv()
logger.debug(f"FLASK_ENV: {os.getenv('FLASK_ENV')}")

# Initialize Flask app
logger.debug("Creating Flask app...")
try:
    app = Flask(__name__)
    instance_path = os.path.join(os.path.dirname(__file__), 'instance')
    os.makedirs(instance_path, exist_ok=True)
    logger.debug(f"Instance path created/exists: {instance_path}")
except Exception as e:
    logger.error(f"Failed to create Flask app or instance path: {str(e)}", exc_info=True)
    raise

# Determine environment (development or production)
ENV = os.getenv('FLASK_ENV', 'production')
IS_PRODUCTION = ENV == 'production'
logger.debug(f"Environment: {ENV}, IS_PRODUCTION: {IS_PRODUCTION}")

# Configure app
logger.debug("Configuring Flask app...")
try:
    app.config.update(
        SECRET_KEY="eb5309bed65e17d8c9ac293f2e245f6ab30979d69e338dd3",
        SQLALCHEMY_DATABASE_URI='postgresql://secure_sx88_user:I6Wdds9mxg256ADuFpii9QXP7SAIuZop@dpg-d03j2mje5dus73abjdu0-a.oregon-postgres.render.com/secure_sx88',
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER='static/uploads',
        ALLOWED_EXTENSIONS={'jpg', 'jpeg'},
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
        JWT_SECRET_KEY="7127b70009319336c4c86b81a0c971efb2a650a83f2b4099",
        RECAPTCHA_SITE_KEY="6LcFtRQrAAAAAHo-4F6DvBTyodOon_yq8j25LrU2" if IS_PRODUCTION else None,
        RECAPTCHA_SECRET_KEY="6LcFtRQrAAAAAHJ8rA3_T-CUK0sfdbwTHZKhoWuh" if IS_PRODUCTION else None,
        WTF_CSRF_FIELD_NAME='csrf_token',
        WTF_CSRF_HEADERS=['X-CSRF-Token']
    )
    logger.debug(f"App config: {app.config}")
except Exception as e:
    logger.error(f"Failed to configure Flask app: {str(e)}", exc_info=True)
    raise

# Stripe configuration
logger.debug("Configuring Stripe...")
try:
    stripe.api_key = "sk_test_51RC1RnRAboKx3Wwpmr6xRe8zQHPCcwFhCDFvjruWLNh9Dd2qqcrcwKLYCUYEwO68doRsS5foR0onqAnzuYvGSi5U00VVU21Yd3"
    STRIPE_PUBLISHABLE_KEY = "pk_test_51RC1RnRAboKx3WwpoNrHgwMBOgJEC8oWYUwOW3DzRwCXVqITQYoj8xGs4myx8EsiVe5AeMm2XqHoVJTBMwPeJvPX00K2jj2zMV"
    logger.debug("Stripe configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Stripe: {str(e)}", exc_info=True)
    raise

# Initialize extensions
logger.debug("Initializing Flask extensions...")
try:
    db = SQLAlchemy(app)
    migrate = Migrate(app, db)
    login_manager = LoginManager(app)
    csrf = CSRFProtect(app)
    login_manager.login_view = 'login'
    logger.debug("Extensions initialized: SQLAlchemy, Migrate, LoginManager, CSRFProtect")
except Exception as e:
    logger.error(f"Failed to initialize extensions: {str(e)}", exc_info=True)
    raise

# Ensure upload folder exists
logger.debug("Creating upload folder...")
try:
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    logger.debug(f"Upload folder created/exists: {app.config['UPLOAD_FOLDER']}")
except Exception as e:
    logger.error(f"Failed to create upload folder: {str(e)}", exc_info=True)
    raise

# Models
logger.debug("Defining models...")
try:
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
        website_access_count = db.Column(db.Integer, default=0)
        website_access_reset_date = db.Column(db.DateTime, default=datetime.utcnow)

        def check_password(self, password):
            return check_password_hash(self.password_hash, password)

        def is_premium_user(self):
            logger.debug(f"Checking premium status for user {self.id}")
            active_subscription = Subscription.query.filter_by(user_id=self.id, active=True).first()
            if active_subscription and active_subscription.end_date > datetime.utcnow():
                return True
            if active_subscription and active_subscription.end_date <= datetime.utcnow():
                active_subscription.active = False
                db.session.commit()
            return False

        def can_access_website(self, free_access_limit):
            logger.debug(f"Checking website access for user {self.id}, current count: {self.website_access_count}")
            now = datetime.utcnow()
            if self.website_access_reset_date.month != now.month or self.website_access_reset_date.year != now.year:
                self.website_access_count = 0
                self.website_access_reset_date = now
                db.session.commit()
            if self.is_premium_user():
                return True
            return self.website_access_count < free_access_limit

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
        free_website_access_limit = db.Column(db.Integer, default=5)

    logger.debug("Models defined successfully")
except Exception as e:
    logger.error(f"Failed to define models: {str(e)}", exc_info=True)
    raise

# Image processing functions
def prepare_image_for_ela(image_path):
    logger.debug(f"Preparing ELA for image: {image_path}")
    try:
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
        logger.debug("ELA preparation completed")
        return np_img, ela_image
    except Exception as e:
        logger.error(f"Failed to prepare ELA for image {image_path}: {str(e)}", exc_info=True)
        raise

def image_coordinates(image_path):
    logger.debug(f"Extracting coordinates from image: {image_path}")
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
            logger.debug(f"Coordinates extracted: lat={lat}, lon={lon}")
            return date_time, lat, lon, True
        else:
            logger.debug("No GPS coordinates found")
            return date_time, None, None, False
    except Exception as e:
        logger.error(f"Failed to extract coordinates from {image_path}: {str(e)}", exc_info=True)
        return datetime.now(), None, None, False

def get_weather(date_time, lat, lon):
    logger.debug(f"Fetching weather for lat={lat}, lon={lon}, date={date_time}")
    try:
        geolocator = Nominatim(user_agent="image_tampering_detection")
        location = geolocator.reverse((lat, lon))
        logger.debug(f"Weather data: location={location.address if location else 'Unknown'}")
        return location.address if location else "Unknown", date_time.strftime('%Y-%m-%d'), "Clear"
    except Exception as e:
        logger.error(f"Failed to fetch weather: {str(e)}", exc_info=True)
        return "Unknown", date_time.strftime('%Y-%m-%d'), "Clear"

# Forms
logger.debug("Defining forms...")
try:
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

    logger.debug("Forms defined successfully")
except Exception as e:
    logger.error(f"Failed to define forms: {str(e)}", exc_info=True)
    raise

# User loader
logger.debug("Setting up user loader...")
try:
    @login_manager.user_loader
    def load_user(user_id):
        logger.debug(f"Loading user with ID: {user_id}")
        return User.query.get(int(user_id))
except Exception as e:
    logger.error(f"Failed to set up user loader: {str(e)}", exc_info=True)
    raise

# reCAPTCHA verification
def verify_recaptcha(token):
    logger.debug("Verifying reCAPTCHA...")
    try:
        if not IS_PRODUCTION:
            logger.debug("reCAPTCHA skipped in development mode")
            return True
        response = requests.post(
            'https://www.google.com/recaptcha/api/siteverify',
            data={
                'secret': app.config['RECAPTCHA_SECRET_KEY'],
                'response': token
            }
        )
        result = response.json()
        success = result.get('success', False) and result.get('score', 0) >= 0.5
        logger.debug(f"reCAPTCHA verification result: {success}")
        return success
    except Exception as e:
        logger.error(f"Failed to verify reCAPTCHA: {str(e)}", exc_info=True)
        return False

# Routes
logger.debug("Defining routes...")
try:
    @app.route('/')
    def index():
        logger.debug("Handling index route")
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        return render_template('index.html', current_year=datetime.utcnow().year)

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        logger.debug("Handling login route")
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        form = LoginForm()
        if form.validate_on_submit():
            logger.debug(f"Login attempt for email: {form.email.data}")
            user = User.query.filter_by(email=form.email.data).first()
            if user and user.check_password(form.password.data):
                if IS_PRODUCTION:
                    token = jwt.encode({
                        'user_id': user.id,
                        'exp': datetime.utcnow() + timedelta(hours=24)
                    }, app.config['JWT_SECRET_KEY'], algorithm='HS256')
                    session['jwt_token'] = token
                    logger.debug(f"JWT token created for user {user.id}")
                session['mfa_user_id'] = user.id
                return redirect(url_for('mfa_verify'))
            flash('Invalid email or password.', 'danger')
            logger.warning(f"Failed login attempt for email: {form.email.data}")
        return render_template('login.html', form=form, current_year=datetime.utcnow().year)

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        logger.debug("Handling register route")
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        form = RegisterForm()
        if form.validate_on_submit():
            logger.debug(f"Register attempt for username: {form.username.data}")
            if IS_PRODUCTION and not verify_recaptcha(form.recaptcha.data):
                flash('reCAPTCHA verification failed. Please try again.', 'danger')
                logger.warning("reCAPTCHA verification failed")
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
            logger.info(f"User registered: {form.username.data}")
            return redirect(url_for('login'))
        return render_template('register.html', form=form, recaptcha_site_key=app.config['RECAPTCHA_SITE_KEY'], is_production=IS_PRODUCTION, current_year=datetime.utcnow().year)

    @app.route('/mfa/setup', methods=['GET', 'POST'])
    @login_required
    def mfa_setup():
        logger.debug("Handling MFA setup route")
        if current_user.mfa_enabled:
            flash('MFA is already enabled.', 'info')
            logger.info(f"MFA already enabled for user {current_user.id}")
            return redirect(url_for('dashboard'))
        form = MFAForm()
        if form.validate_on_submit():
            logger.debug(f"Verifying MFA token for user {current_user.id}")
            if pyotp.TOTP(current_user.mfa_secret).verify(form.token.data):
                current_user.mfa_enabled = True
                db.session.commit()
                flash('MFA enabled successfully.', 'success')
                logger.info(f"MFA enabled for user {current_user.id}")
                return redirect(url_for('dashboard'))
            flash('Invalid MFA code.', 'danger')
            logger.warning(f"Invalid MFA code for user {current_user.id}")
        uri = pyotp.TOTP(current_user.mfa_secret).provisioning_uri(
            current_user.email, issuer_name="Image Tampering Detection")
        qr = qrcode.make(uri)
        buffer = io.BytesIO()
        qr.save(buffer)
        qr_code = base64.b64encode(buffer.getvalue()).decode()
        return render_template('mfa.html', form=form, qr_code=qr_code, setup=True, current_year=datetime.utcnow().year)

    @app.route('/mfa/verify', methods=['GET', 'POST'])
    def mfa_verify():
        logger.debug("Handling MFA verify route")
        user_id = session.get('mfa_user_id')
        if not user_id:
            logger.warning("No MFA user ID in session")
            return redirect(url_for('login'))
        user = User.query.get(user_id)
        if not user:
            session.pop('mfa_user_id', None)
            logger.warning(f"User not found for ID: {user_id}")
            return redirect(url_for('login'))

        if not user.mfa_enabled:
            login_user(user)
            user.last_login = datetime.utcnow()
            db.session.commit()
            session.pop('mfa_user_id', None)
            flash('Logged in successfully. Consider enabling MFA for extra security.', 'success')
            logger.info(f"User {user.id} logged in without MFA")
            return redirect(url_for('dashboard'))

        form = MFAForm()
        if form.validate_on_submit():
            logger.debug(f"Verifying MFA token for user {user.id}")
            if pyotp.TOTP(user.mfa_secret).verify(form.token.data):
                login_user(user)
                user.last_login = datetime.utcnow()
                db.session.commit()
                session.pop('mfa_user_id', None)
                flash('MFA verified successfully.', 'success')
                logger.info(f"MFA verified for user {user.id}")
                return redirect(url_for('dashboard'))
            flash('Invalid MFA code.', 'danger')
            logger.warning(f"Invalid MFA code for user {user.id}")
        return render_template('mfa.html', form=form, setup=False, current_year=datetime.utcnow().year)

    @app.route('/logout')
    @login_required
    def logout():
        logger.debug("Handling logout route")
        if IS_PRODUCTION:
            session.pop('jwt_token', None)
        logout_user()
        flash('Logged out successfully.', 'success')
        logger.info(f"User {current_user.id} logged out")
        return redirect(url_for('index'))

    @app.route('/dashboard', methods=['GET', 'POST'])
    @login_required
    def dashboard():
        logger.debug(f"Handling dashboard route for user {current_user.id}")
        if IS_PRODUCTION:
            token = session.get('jwt_token')
            if not token:
                logout_user()
                flash('Session expired. Please log in again.', 'warning')
                logger.warning(f"Session expired for user {current_user.id}")
                return redirect(url_for('login'))
            try:
                jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
            except jwt.ExpiredSignatureError:
                logout_user()
                flash('Session expired. Please log in again.', 'warning')
                logger.warning(f"JWT expired for user {current_user.id}")
                return redirect(url_for('login'))
            except jwt.InvalidTokenError:
                logout_user()
                flash('Invalid session. Please log in again.', 'warning')
                logger.warning(f"Invalid JWT for user {current_user.id}")
                return redirect(url_for('login'))

        form = ImageForm()
        config = Config.query.first() or Config(free_image_limit=5, free_website_access_limit=5)
        if form.validate_on_submit():
            logger.debug(f"Processing image upload for user {current_user.id}")
            if not current_user.is_premium_user() and len(current_user.images) >= config.free_image_limit:
                flash('Free user limit reached. Please subscribe.', 'warning')
                logger.warning(f"Image limit reached for user {current_user.id}")
                return redirect(url_for('subscription'))

            file = form.image.data
            if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']):
                flash('Only JPG/JPEG images allowed.', 'danger')
                logger.warning(f"Invalid file type uploaded by user {current_user.id}")
                return redirect(url_for('dashboard'))

            filename = secure_filename(file.filename)
            unique_filename = f"{datetime.now().timestamp()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            logger.debug(f"Image saved: {filepath}")

            image = Image(
                user_id=current_user.id,
                original_filename=filename,
                filename=unique_filename,
                filepath=filepath,
                is_outdoor=form.is_outdoor.data
            )

            # ELA processing without model
            logger.debug(f"Performing ELA analysis for image: {filepath}")
            _, ela_img = prepare_image_for_ela(filepath)
            ela_filename = f"ela_{unique_filename}"
            ela_filepath = os.path.join(app.config['UPLOAD_FOLDER'], ela_filename)
            ela_img.save(ela_filepath)
            image.ela_filepath = ela_filepath
            image.analysis_result = "Image analysis disabled (model not loaded)."

            if form.is_outdoor.data:
                logger.debug("Processing outdoor image coordinates")
                date_time, lat, lon, is_valid = image_coordinates(filepath)
                if is_valid and lat and lon:
                    location, date, weather = get_weather(date_time, lat, lon)
                    image.weather_result = f"Image taken at {location} on {date} with {weather}"
                    image.latitude = lat
                    image.longitude = lon
                    image.location = location

            db.session.add(image)
            db.session.commit()
            flash('Image uploaded and processed successfully.', 'success')
            logger.info(f"Image processed successfully for user {current_user.id}")

        images = Image.query.filter_by(user_id=current_user.id).order_by(Image.upload_date.desc()).limit(10).all()
        return render_template('dashboard.html', form=form, images=images, is_premium=current_user.is_premium_user(), current_user=current_user, current_year=datetime.utcnow().year)

    @app.route('/embedded-website')
    @login_required
    def embedded_website():
        logger.debug(f"Handling embedded website route for user {current_user.id}")
        logger.debug(f"Session data: {session}")
        logger.debug(f"User {current_user.id} access count: {current_user.website_access_count}, is_premium: {current_user.is_premium_user()}")

        website_url = 'https://imagetamperingdetection.streamlit.app/?embedded=true'

        # Check if the website allows iframe embedding or causes redirects
        try:
            response = requests.head(website_url, timeout=5, allow_redirects=True)
            x_frame_options = response.headers.get('X-Frame-Options', '').lower()
            logger.debug(f"Response headers: {response.headers}")
            logger.debug(f"X-Frame-Options: {x_frame_options}")

            if x_frame_options in ['deny', 'sameorigin']:
                logger.warning(f"Iframe blocked by X-Frame-Options: {x_frame_options}")
                flash('This website cannot be embedded due to security restrictions. Opening in a new tab.', 'warning')
                return redirect(website_url)

            # Check for excessive redirects
            if len(response.history) > 10:
                logger.warning(f"Excessive redirects detected: {len(response.history)}")
                flash('The website is causing too many redirects. Opening in a new tab.', 'warning')
                return redirect(website_url)
        except requests.TooManyRedirects:
            logger.error("Too many redirects detected when checking website")
            flash('The website is causing a redirect loop. Opening in a new tab.', 'warning')
            return redirect(website_url)
        except requests.RequestException as e:
            logger.error(f"Failed to check iframe compatibility: {str(e)}")
            flash('Unable to verify website compatibility. Opening in a new tab.', 'warning')
            return redirect(website_url)

        # JWT validation (only in production)
        if IS_PRODUCTION:
            token = session.get('jwt_token')
            logger.debug(f"JWT token: {token}")
            if not token:
                logout_user()
                flash('Session expired. Please log in again.', 'warning')
                logger.warning(f"Session expired for user {current_user.id}")
                return redirect(url_for('login'))
            try:
                jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
                logger.debug("JWT token validated successfully")
            except jwt.ExpiredSignatureError:
                logout_user()
                flash('Session expired. Please log in again.', 'warning')
                logger.warning(f"JWT expired for user {current_user.id}")
                return redirect(url_for('login'))
            except jwt.InvalidTokenError:
                logout_user()
                flash('Invalid session. Please log in again.', 'warning')
                logger.warning(f"Invalid JWT for user {current_user.id}")
                return redirect(url_for('login'))

        # Access limit check
        config = Config.query.first() or Config(free_image_limit=5, free_website_access_limit=5)
        free_access_limit = config.free_website_access_limit
        logger.debug(f"Checking website access: count={current_user.website_access_count}, limit={free_access_limit}")

        if current_user.can_access_website(free_access_limit):
            current_user.website_access_count += 1
            db.session.commit()
            logger.info(f"Website access granted for user {current_user.id}, new count: {current_user.website_access_count}")
            try:
                return render_template('embedded_website.html',
                                    website_url=website_url,
                                    is_premium=current_user.is_premium_user(),
                                    current_user=current_user,
                                    free_access_limit=free_access_limit)
            except Exception as e:
                logger.error(f"Failed to render embedded website template: {str(e)}", exc_info=True)
                flash('Unable to load embedded website. Opening in a new tab.', 'warning')
                return redirect(website_url)
        else:
            flash('You have reached the website access limit for free users. Please upgrade to premium.', 'warning')
            logger.warning(f"Website access denied for user {current_user.id}, limit reached")
            return redirect(url_for('subscription'))

    @app.route('/subscription')
    @login_required
    def subscription():
        logger.debug(f"Handling subscription route for user {current_user.id}")
        plans = [
            {'name': 'Weekly', 'price': 4.99, 'duration': '7 days', 'duration_days': 7},
            {'name': 'Monthly', 'price': 12.99, 'duration': '30 days', 'duration_days': 30},
            {'name': 'Yearly', 'price': 99.99, 'duration': '365 days', 'duration_days': 365}
        ]
        return render_template('subscription.html', plans=plans, is_premium=current_user.is_premium_user(), key=STRIPE_PUBLISHABLE_KEY, current_year=datetime.utcnow().year)

    @app.route('/create-checkout-session', methods=['POST'])
    @login_required
    def create_checkout_session():
        logger.debug(f"Handling create-checkout-session route for user {current_user.id}, data: {request.form}")
        plans = {
            'Weekly': {'price': 4.99, 'duration': timedelta(days=7)},
            'Monthly': {'price': 12.99, 'duration': timedelta(days=30)},
            'Yearly': {'price': 99.99, 'duration': timedelta(days=365)}
        }

        try:
            plan_name = request.form.get('plan_name')
            logger.debug(f"Selected plan: {plan_name}")
            if not plan_name or plan_name not in plans:
                logger.error(f"Invalid plan selected: {plan_name}")
                return jsonify({'error': 'Invalid plan selected'}), 400

            if current_user.is_premium_user():
                logger.info(f"User {current_user.id} already has active subscription")
                return jsonify({'error': 'You already have an active subscription'}), 400

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
            logger.info(f"Created checkout session: {checkout_session.id}")
            return jsonify({'id': checkout_session.id})
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error: {str(e)}")
            return jsonify({'error': f'Stripe error: {str(e)}'}), 400
        except Exception as e:
            logger.error(f"Server error: {str(e)}", exc_info=True)
            return jsonify({'error': f'Server error: {str(e)}'}), 500

    @app.route('/subscribe/<plan_name>')
    @login_required
    def subscribe(plan_name):
        logger.debug(f"Handling subscribe route for plan: {plan_name}")
        flash('Please use the payment gateway to subscribe.', 'warning')
        return redirect(url_for('subscription'))

    @app.route('/subscribe-success/<plan_name>')
    @login_required
    def subscribe_success(plan_name):
        logger.debug(f"Handling subscribe-success route for plan: {plan_name}")
        plans = {
            'Weekly': {'price': 4.99, 'duration': timedelta(days=7)},
            'Monthly': {'price': 12.99, 'duration': timedelta(days=30)},
            'Yearly': {'price': 99.99, 'duration': timedelta(days=365)}
        }

        if plan_name not in plans:
            flash('Invalid plan selected.', 'danger')
            logger.error(f"Invalid plan: {plan_name}")
            return redirect(url_for('subscription'))

        if current_user.is_premium_user():
            flash('You already have an active subscription.', 'info')
            logger.info(f"User {current_user.id} already subscribed")
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
        logger.info(f"User {current_user.id} subscribed to {plan_name}")
        return redirect(url_for('dashboard'))

    logger.debug("Routes defined successfully")
except Exception as e:
    logger.error(f"Failed to define routes: {str(e)}", exc_info=True)
    raise

# Initialize database
logger.debug("Initializing database...")
try:
    with app.app_context():
        db.create_all()
        if not Config.query.first():
            db.session.add(Config(free_image_limit=5, free_website_access_limit=5))
            db.session.commit()
        logger.debug("Database initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize database: {str(e)}", exc_info=True)
    raise

# Run the app
if __name__ == '__main__':
    logger.debug("Starting Flask app...")
    try:
        app.run(port=5000, debug=True, use_reloader=False)  # Disable reloader to prevent multiple runs
    except Exception as e:
        logger.error(f"Failed to start Flask app: {str(e)}", exc_info=True)
        raise
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
import jwt
import requests
import re
import logging

# Configure logging to output to both file and console
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log script start and process ID
logger.debug(f"Script starting... Process ID: {os.getpid()}")

# Load environment variables
logger.debug("Loading environment variables...")
load_dotenv()
logger.debug(f"FLASK_ENV: {os.getenv('FLASK_ENV')}")

# Initialize Flask app
logger.debug("Creating Flask app...")
try:
    app = Flask(__name__)
    instance_path = os.path.join(os.path.dirname(__file__), 'instance')
    os.makedirs(instance_path, exist_ok=True)
    logger.debug(f"Instance path created/exists: {instance_path}")
except Exception as e:
    logger.error(f"Failed to create Flask app or instance path: {str(e)}", exc_info=True)
    raise

# Determine environment (development or production)
ENV = os.getenv('FLASK_ENV', 'development')
IS_PRODUCTION = ENV == 'production'
logger.debug(f"Environment: {ENV}, IS_PRODUCTION: {IS_PRODUCTION}")

# Configure app
logger.debug("Configuring Flask app...")
try:
    app.config.update(
        SECRET_KEY="eb5309bed65e17d8c9ac293f2e245f6ab30979d69e338dd3",
        SQLALCHEMY_DATABASE_URI='sqlite:///db.app',
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        UPLOAD_FOLDER='static/uploads',
        ALLOWED_EXTENSIONS={'jpg', 'jpeg'},
        MAX_CONTENT_LENGTH=16 * 1024 * 1024,
        JWT_SECRET_KEY="7127b70009319336c4c86b81a0c971efb2a650a83f2b4099",
        RECAPTCHA_SITE_KEY="6LcFtRQrAAAAAHo-4F6DvBTyodOon_yq8j25LrU2" if IS_PRODUCTION else None,
        RECAPTCHA_SECRET_KEY="6LcFtRQrAAAAAHJ8rA3_T-CUK0sfdbwTHZKhoWuh" if IS_PRODUCTION else None,
        WTF_CSRF_FIELD_NAME='csrf_token',
        WTF_CSRF_HEADERS=['X-CSRF-Token']
    )
    logger.debug(f"App config: {app.config}")
except Exception as e:
    logger.error(f"Failed to configure Flask app: {str(e)}", exc_info=True)
    raise

# Stripe configuration
logger.debug("Configuring Stripe...")
try:
    stripe.api_key = "sk_test_51RC1RnRAboKx3Wwpmr6xRe8zQHPCcwFhCDFvjruWLNh9Dd2qqcrcwKLYCUYEwO68doRsS5foR0onqAnzuYvGSi5U00VVU21Yd3"
    STRIPE_PUBLISHABLE_KEY = "pk_test_51RC1RnRAboKx3WwpoNrHgwMBOgJEC8oWYUwOW3DzRwCXVqITQYoj8xGs4myx8EsiVe5AeMm2XqHoVJTBMwPeJvPX00K2jj2zMV"
    logger.debug("Stripe configured successfully")
except Exception as e:
    logger.error(f"Failed to configure Stripe: {str(e)}", exc_info=True)
    raise

# Initialize extensions
logger.debug("Initializing Flask extensions...")
try:
    db = SQLAlchemy(app)
    migrate = Migrate(app, db)
    login_manager = LoginManager(app)
    csrf = CSRFProtect(app)
    login_manager.login_view = 'login'
    logger.debug("Extensions initialized: SQLAlchemy, Migrate, LoginManager, CSRFProtect")
except Exception as e:
    logger.error(f"Failed to initialize extensions: {str(e)}", exc_info=True)
    raise

# Ensure upload folder exists
logger.debug("Creating upload folder...")
try:
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    logger.debug(f"Upload folder created/exists: {app.config['UPLOAD_FOLDER']}")
except Exception as e:
    logger.error(f"Failed to create upload folder: {str(e)}", exc_info=True)
    raise

# Models
logger.debug("Defining models...")
try:
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
        website_access_count = db.Column(db.Integer, default=0)
        website_access_reset_date = db.Column(db.DateTime, default=datetime.utcnow)

        def check_password(self, password):
            return check_password_hash(self.password_hash, password)

        def is_premium_user(self):
            logger.debug(f"Checking premium status for user {self.id}")
            active_subscription = Subscription.query.filter_by(user_id=self.id, active=True).first()
            if active_subscription and active_subscription.end_date > datetime.utcnow():
                return True
            if active_subscription and active_subscription.end_date <= datetime.utcnow():
                active_subscription.active = False
                db.session.commit()
            return False

        def can_access_website(self, free_access_limit):
            logger.debug(f"Checking website access for user {self.id}, current count: {self.website_access_count}")
            now = datetime.utcnow()
            if self.website_access_reset_date.month != now.month or self.website_access_reset_date.year != now.year:
                self.website_access_count = 0
                self.website_access_reset_date = now
                db.session.commit()
            if self.is_premium_user():
                return True
            return self.website_access_count < free_access_limit

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
        free_website_access_limit = db.Column(db.Integer, default=5)

    logger.debug("Models defined successfully")
except Exception as e:
    logger.error(f"Failed to define models: {str(e)}", exc_info=True)
    raise

# Image processing functions
def prepare_image_for_ela(image_path):
    logger.debug(f"Preparing ELA for image: {image_path}")
    try:
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
        logger.debug("ELA preparation completed")
        return np_img, ela_image
    except Exception as e:
        logger.error(f"Failed to prepare ELA for image {image_path}: {str(e)}", exc_info=True)
        raise

def image_coordinates(image_path):
    logger.debug(f"Extracting coordinates from image: {image_path}")
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
            logger.debug(f"Coordinates extracted: lat={lat}, lon={lon}")
            return date_time, lat, lon, True
        logger.debug("No GPS coordinates found")
        return date_time, None, None, False
    except Exception as e:
        logger.error(f"Failed to extract coordinates from {image_path}: {str(e)}", exc_info=True)
        return datetime.now(), None, None, False

def get_weather(date_time, lat, lon):
    logger.debug(f"Fetching weather for lat={lat}, lon={lon}, date={date_time}")
    try:
        geolocator = Nominatim(user_agent="image_tampering_detection")
        location = geolocator.reverse((lat, lon))
        logger.debug(f"Weather data: location={location.address if location else 'Unknown'}")
        return location.address if location else "Unknown", date_time.strftime('%Y-%m-%d'), "Clear"
    except Exception as e:
        logger.error(f"Failed to fetch weather: {str(e)}", exc_info=True)
        return "Unknown", date_time.strftime('%Y-%m-%d'), "Clear"

# Forms
logger.debug("Defining forms...")
try:
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

    logger.debug("Forms defined successfully")
except Exception as e:
    logger.error(f"Failed to define forms: {str(e)}", exc_info=True)
    raise

# User loader
logger.debug("Setting up user loader...")
try:
    @login_manager.user_loader
    def load_user(user_id):
        logger.debug(f"Loading user with ID: {user_id}")
        return User.query.get(int(user_id))
except Exception as e:
    logger.error(f"Failed to set up user loader: {str(e)}", exc_info=True)
    raise

# reCAPTCHA verification
def verify_recaptcha(token):
    logger.debug("Verifying reCAPTCHA...")
    try:
        if not IS_PRODUCTION:
            logger.debug("reCAPTCHA skipped in development mode")
            return True
        response = requests.post(
            'https://www.google.com/recaptcha/api/siteverify',
            data={
                'secret': app.config['RECAPTCHA_SECRET_KEY'],
                'response': token
            }
        )
        result = response.json()
        success = result.get('success', False) and result.get('score', 0) >= 0.5
        logger.debug(f"reCAPTCHA verification result: {success}")
        return success
    except Exception as e:
        logger.error(f"Failed to verify reCAPTCHA: {str(e)}", exc_info=True)
        return False

# Routes
logger.debug("Defining routes...")
try:
    @app.route('/')
    def index():
        logger.debug("Handling index route")
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        return render_template('index.html', current_year=datetime.utcnow().year)

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        logger.debug("Handling login route")
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        form = LoginForm()
        if form.validate_on_submit():
            logger.debug(f"Login attempt for email: {form.email.data}")
            user = User.query.filter_by(email=form.email.data).first()
            if user and user.check_password(form.password.data):
                if IS_PRODUCTION:
                    token = jwt.encode({
                        'user_id': user.id,
                        'exp': datetime.utcnow() + timedelta(hours=24)
                    }, app.config['JWT_SECRET_KEY'], algorithm='HS256')
                    session['jwt_token'] = token
                    logger.debug(f"JWT token created for user {user.id}")
                session['mfa_user_id'] = user.id
                return redirect(url_for('mfa_verify'))
            flash('Invalid email or password.', 'danger')
            logger.warning(f"Failed login attempt for email: {form.email.data}")
        return render_template('login.html', form=form, current_year=datetime.utcnow().year)

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        logger.debug("Handling register route")
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        form = RegisterForm()
        if form.validate_on_submit():
            logger.debug(f"Register attempt for username: {form.username.data}")
            if IS_PRODUCTION and not verify_recaptcha(form.recaptcha.data):
                flash('reCAPTCHA verification failed. Please try again.', 'danger')
                logger.warning("reCAPTCHA verification failed")
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
            logger.info(f"User registered: {form.username.data}")
            return redirect(url_for('login'))
        return render_template('register.html', form=form, recaptcha_site_key=app.config['RECAPTCHA_SITE_KEY'], is_production=IS_PRODUCTION, current_year=datetime.utcnow().year)

    @app.route('/mfa/setup', methods=['GET', 'POST'])
    @login_required
    def mfa_setup():
        logger.debug("Handling MFA setup route")
        if current_user.mfa_enabled:
            flash('MFA is already enabled.', 'info')
            logger.info(f"MFA already enabled for user {current_user.id}")
            return redirect(url_for('dashboard'))
        form = MFAForm()
        if form.validate_on_submit():
            logger.debug(f"Verifying MFA token for user {current_user.id}")
            if pyotp.TOTP(current_user.mfa_secret).verify(form.token.data):
                current_user.mfa_enabled = True
                db.session.commit()
                flash('MFA enabled successfully.', 'success')
                logger.info(f"MFA enabled for user {current_user.id}")
                return redirect(url_for('dashboard'))
            flash('Invalid MFA code.', 'danger')
            logger.warning(f"Invalid MFA code for user {current_user.id}")
        uri = pyotp.TOTP(current_user.mfa_secret).provisioning_uri(
            current_user.email, issuer_name="Image Tampering Detection")
        qr = qrcode.make(uri)
        buffer = io.BytesIO()
        qr.save(buffer)
        qr_code = base64.b64encode(buffer.getvalue()).decode()
        return render_template('mfa.html', form=form, qr_code=qr_code, setup=True, current_year=datetime.utcnow().year)

    @app.route('/mfa/verify', methods=['GET', 'POST'])
    def mfa_verify():
        logger.debug("Handling MFA verify route")
        user_id = session.get('mfa_user_id')
        if not user_id:
            logger.warning("No MFA user ID in session")
            return redirect(url_for('login'))
        user = User.query.get(user_id)
        if not user:
            session.pop('mfa_user_id', None)
            logger.warning(f"User not found for ID: {user_id}")
            return redirect(url_for('login'))

        if not user.mfa_enabled:
            login_user(user)
            user.last_login = datetime.utcnow()
            db.session.commit()
            session.pop('mfa_user_id', None)
            flash('Logged in successfully. Consider enabling MFA for extra security.', 'success')
            logger.info(f"User {user.id} logged in without MFA")
            return redirect(url_for('dashboard'))

        form = MFAForm()
        if form.validate_on_submit():
            logger.debug(f"Verifying MFA token for user {user.id}")
            if pyotp.TOTP(user.mfa_secret).verify(form.token.data):
                login_user(user)
                user.last_login = datetime.utcnow()
                db.session.commit()
                session.pop('mfa_user_id', None)
                flash('MFA verified successfully.', 'success')
                logger.info(f"MFA verified for user {user.id}")
                return redirect(url_for('dashboard'))
            flash('Invalid MFA code.', 'danger')
            logger.warning(f"Invalid MFA code for user {user.id}")
        return render_template('mfa.html', form=form, setup=False, current_year=datetime.utcnow().year)

    @app.route('/logout')
    @login_required
    def logout():
        logger.debug("Handling logout route")
        if IS_PRODUCTION:
            session.pop('jwt_token', None)
        logout_user()
        flash('Logged out successfully.', 'success')
        logger.info(f"User {current_user.id} logged out")
        return redirect(url_for('index'))

    @app.route('/dashboard', methods=['GET', 'POST'])
    @login_required
    def dashboard():
        logger.debug(f"Handling dashboard route for user {current_user.id}")
        if IS_PRODUCTION:
            token = session.get('jwt_token')
            if not token:
                logout_user()
                flash('Session expired. Please log in again.', 'warning')
                logger.warning(f"Session expired for user {current_user.id}")
                return redirect(url_for('login'))
            try:
                jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
            except jwt.ExpiredSignatureError:
                logout_user()
                flash('Session expired. Please log in again.', 'warning')
                logger.warning(f"JWT expired for user {current_user.id}")
                return redirect(url_for('login'))
            except jwt.InvalidTokenError:
                logout_user()
                flash('Invalid session. Please log in again.', 'warning')
                logger.warning(f"Invalid JWT for user {current_user.id}")
                return redirect(url_for('login'))

        form = ImageForm()
        config = Config.query.first() or Config(free_image_limit=5, free_website_access_limit=5)
        if form.validate_on_submit():
            logger.debug(f"Processing image upload for user {current_user.id}")
            if not current_user.is_premium_user() and len(current_user.images) >= config.free_image_limit:
                flash('Free user limit reached. Please subscribe.', 'warning')
                logger.warning(f"Image limit reached for user {current_user.id}")
                return redirect(url_for('subscription'))

            file = form.image.data
            if not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']):
                flash('Only JPG/JPEG images allowed.', 'danger')
                logger.warning(f"Invalid file type uploaded by user {current_user.id}")
                return redirect(url_for('dashboard'))

            filename = secure_filename(file.filename)
            unique_filename = f"{datetime.now().timestamp()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            logger.debug(f"Image saved: {filepath}")

            image = Image(
                user_id=current_user.id,
                original_filename=filename,
                filename=unique_filename,
                filepath=filepath,
                is_outdoor=form.is_outdoor.data
            )

            # ELA processing without model
            logger.debug(f"Performing ELA analysis for image: {filepath}")
            _, ela_img = prepare_image_for_ela(filepath)
            ela_filename = f"ela_{unique_filename}"
            ela_filepath = os.path.join(app.config['UPLOAD_FOLDER'], ela_filename)
            ela_img.save(ela_filepath)
            image.ela_filepath = ela_filepath
            image.analysis_result = "Image analysis disabled (model not loaded)."

            if form.is_outdoor.data:
                logger.debug("Processing outdoor image coordinates")
                date_time, lat, lon, is_valid = image_coordinates(filepath)
                if is_valid and lat and lon:
                    location, date, weather = get_weather(date_time, lat, lon)
                    image.weather_result = f"Image taken at {location} on {date} with {weather}"
                    image.latitude = lat
                    image.longitude = lon
                    image.location = location

            db.session.add(image)
            db.session.commit()
            flash('Image uploaded and processed successfully.', 'success')
            logger.info(f"Image processed successfully for user {current_user.id}")

        images = Image.query.filter_by(user_id=current_user.id).order_by(Image.upload_date.desc()).limit(10).all()
        return render_template('dashboard.html', form=form, images=images, is_premium=current_user.is_premium_user(), current_user=current_user, current_year=datetime.utcnow().year)

    @app.route('/embedded-website')
    @login_required
    def embedded_website():
        logger.debug(f"Handling embedded website route for user {current_user.id}")
        if IS_PRODUCTION:
            token = session.get('jwt_token')
            if not token:
                logout_user()
                flash('Session expired. Please log in again.', 'warning')
                logger.warning(f"Session expired for user {current_user.id}")
                return redirect(url_for('login'))
            try:
                jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
            except jwt.ExpiredSignatureError:
                logout_user()
                flash('Session expired. Please log in again.', 'warning')
                logger.warning(f"JWT expired for user {current_user.id}")
                return redirect(url_for('login'))
            except jwt.InvalidTokenError:
                logout_user()
                flash('Invalid session. Please log in again.', 'warning')
                logger.warning(f"Invalid JWT for user {current_user.id}")
                return redirect(url_for('login'))

        config = Config.query.first() or Config(free_image_limit=5, free_website_access_limit=5)
        free_access_limit = config.free_website_access_limit
        logger.debug(f"Checking website access: count={current_user.website_access_count}, limit={free_access_limit}")
        if current_user.can_access_website(free_access_limit):
            current_user.website_access_count += 1
            db.session.commit()
            logger.info(f"Website access granted for user {current_user.id}, new count: {current_user.website_access_count}")
            return render_template('embedded_website.html', website_url='https://imagetamperingdetection.streamlit.app/', is_premium=current_user.is_premium_user(), current_user=current_user, free_access_limit=free_access_limit)
        else:
            flash('You have reached the website access limit for free users. Please upgrade to premium.', 'warning')
            logger.warning(f"Website access denied for user {current_user.id}, limit reached")
            return redirect(url_for('subscription'))

    @app.route('/subscription')
    @login_required
    def subscription():
        logger.debug(f"Handling subscription route for user {current_user.id}")
        plans = [
            {'name': 'Weekly', 'price': 4.99, 'duration': '7 days', 'duration_days': 7},
            {'name': 'Monthly', 'price': 12.99, 'duration': '30 days', 'duration_days': 30},
            {'name': 'Yearly', 'price': 99.99, 'duration': '365 days', 'duration_days': 365}
        ]
        return render_template('subscription.html', plans=plans, is_premium=current_user.is_premium_user(), key=STRIPE_PUBLISHABLE_KEY, current_year=datetime.utcnow().year)

    @app.route('/create-checkout-session', methods=['POST'])
    @login_required
    def create_checkout_session():
        logger.debug(f"Handling create-checkout-session route for user {current_user.id}, data: {request.form}")
        plans = {
            'Weekly': {'price': 4.99, 'duration': timedelta(days=7)},
            'Monthly': {'price': 12.99, 'duration': timedelta(days=30)},
            'Yearly': {'price': 99.99, 'duration': timedelta(days=365)}
        }

        try:
            plan_name = request.form.get('plan_name')
            logger.debug(f"Selected plan: {plan_name}")
            if not plan_name or plan_name not in plans:
                logger.error(f"Invalid plan selected: {plan_name}")
                return jsonify({'error': 'Invalid plan selected'}), 400

            if current_user.is_premium_user():
                logger.info(f"User {current_user.id} already has active subscription")
                return jsonify({'error': 'You already have an active subscription'}), 400

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
            logger.info(f"Created checkout session: {checkout_session.id}")
            return jsonify({'id': checkout_session.id})
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error: {str(e)}")
            return jsonify({'error': f'Stripe error: {str(e)}'}), 400
        except Exception as e:
            logger.error(f"Server error: {str(e)}", exc_info=True)
            return jsonify({'error': f'Server error: {str(e)}'}), 500

    @app.route('/subscribe/<plan_name>')
    @login_required
    def subscribe(plan_name):
        logger.debug(f"Handling subscribe route for plan: {plan_name}")
        flash('Please use the payment gateway to subscribe.', 'warning')
        return redirect(url_for('subscription'))

    @app.route('/subscribe-success/<plan_name>')
    @login_required
    def subscribe_success(plan_name):
        logger.debug(f"Handling subscribe-success route for plan: {plan_name}")
        plans = {
            'Weekly': {'price': 4.99, 'duration': timedelta(days=7)},
            'Monthly': {'price': 12.99, 'duration': timedelta(days=30)},
            'Yearly': {'price': 99.99, 'duration': timedelta(days=365)}
        }

        if plan_name not in plans:
            flash('Invalid plan selected.', 'danger')
            logger.error(f"Invalid plan: {plan_name}")
            return redirect(url_for('subscription'))

        if current_user.is_premium_user():
            flash('You already have an active subscription.', 'info')
            logger.info(f"User {current_user.id} already subscribed")
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
        logger.info(f"User {current_user.id} subscribed to {plan_name}")
        return redirect(url_for('dashboard'))

    logger.debug("Routes defined successfully")
except Exception as e:
    logger.error(f"Failed to define routes: {str(e)}", exc_info=True)
    raise

# Initialize database
logger.debug("Initializing database...")
try:
    with app.app_context():
        db.create_all()
        if not Config.query.first():
            db.session.add(Config(free_image_limit=5, free_website_access_limit=5))
            db.session.commit()
        logger.debug("Database initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize database: {str(e)}", exc_info=True)
    raise

# Run the app
if __name__ == '__main__':
    logger.debug("Starting Flask app...")
    try:
        app.run(port=5000, debug=True, use_reloader=False)  # Disable reloader to prevent multiple runs
    except Exception as e:
        logger.error(f"Failed to start Flask app: {str(e)}", exc_info=True)
        raise
